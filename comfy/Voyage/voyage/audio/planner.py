"""Audio slow-loop planner (DESIGN §§35/40): 30-60s music takes covering the video timeline.

The planner is pure logic over a persisted takes ledger (`audio/takes.jsonl`
under the run dir): given the video time consumed so far and the director's
current music caption, it decides whether to (a) keep the current take, (b)
render a fresh take, or (c) repaint the not-yet-consumed region of a take
whose caption changed. Music is *ahead* of video: takes are rendered when
coverage drops below `take_seconds + ahead_seconds`, so the GPU swap for a
take render happens rarely (once per ~45s of video instead of per segment).

Ledger records are immutable and versioned: a repaint never mutates a take
file, it appends a new take. Assembly groups by the take path in effect at
each segment, so already-committed segments are never rewritten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voyage.atomic import fsync_dir
from voyage.concepts import token_set_similarity
from voyage.errors import StateError
from voyage.paths import resolve_stored_path

TAKES_FILENAME = "takes.jsonl"


@dataclass
class AudioTake:
    """One rendered music take (§35)."""

    take_id: str
    path: str
    caption: str
    seed: int
    covers_from: float  # video-time (seconds) the take starts covering
    duration: float  # take length in seconds
    segment_index: int  # first video segment this take may serve
    bpm: float | None = None  # grid BPM this take was rendered at (None = legacy)

    def covers_until(self) -> float:
        """Exclusive video-time end of this take's coverage."""
        return self.covers_from + self.duration

    def resolved_path(self, run_dir: Path) -> Path:
        """Usable take file path (issue 016 consumer side).

        Stored form is run-relative POSIX; legacy entries are absolute.
        An existing path is used as-is, otherwise a relative entry
        resolves against `run_dir` — so a relocated run keeps serving
        takes instead of failing on a stale absolute path. HOOK FOR THE
        SUPERVISOR TRACK: `supervisor._ensure_audio_coverage` should call
        `serving.resolved_path(self._run_dir)` (and resolve
        `current.path` for the repaint `reference_audio` payload) instead
        of `Path(serving.path)` / `current.path` verbatim.
        """
        return resolve_stored_path(run_dir, self.path)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable ledger form."""
        record: dict[str, Any] = {
            "take_id": self.take_id,
            "path": self.path,
            "caption": self.caption,
            "seed": self.seed,
            "covers_from": self.covers_from,
            "duration": self.duration,
            "segment_index": self.segment_index,
        }
        if self.bpm is not None:
            record["bpm"] = self.bpm
        return record

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AudioTake:
        """Rebuild from a ledger line (pre-BPM lines carry no bpm key).

        Validates geometry at the boundary (issue 104): corrupt/hand-edited
        ledger lines (nan/inf/negative/zero durations, negative covers_from,
        non-finite bpm) fail loud here with StateError instead of becoming
        live takes that poison the slice walk downstream. Tiny-but-positive
        durations still load — the walk bound owns those, not the loader.
        """
        try:
            take_id = str(raw["take_id"])
            take_path = str(raw["path"])
            caption = str(raw["caption"])
            seed = int(raw["seed"])
            covers_from_value = float(raw["covers_from"])
            duration_value = float(raw["duration"])
            segment_index = int(raw["segment_index"])
            bpm_raw = raw.get("bpm")
            beat_rate: float | None = None
            if bpm_raw is not None:
                beat_rate = float(bpm_raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise StateError(f"corrupt audio take record {raw!r}: {exc}") from exc
        if not math.isfinite(covers_from_value) or covers_from_value < 0:
            raise StateError(
                f"corrupt audio take {take_id!r}: covers_from must be finite >= 0 "
                f"(got {covers_from_value})"
            )
        if not math.isfinite(duration_value) or duration_value <= 0:
            raise StateError(
                f"corrupt audio take {take_id!r}: duration must be finite > 0 "
                f"(got {duration_value})"
            )
        if beat_rate is not None and (not math.isfinite(beat_rate) or beat_rate <= 0):
            raise StateError(
                f"corrupt audio take {take_id!r}: bpm must be finite > 0 (got {beat_rate})"
            )
        return cls(
            take_id=take_id,
            path=take_path,
            caption=caption,
            seed=seed,
            covers_from=covers_from_value,
            duration=duration_value,
            segment_index=segment_index,
            bpm=beat_rate,
        )


@dataclass
class PlanDecision:
    """What the supervisor must do for the upcoming video time."""

    action: str  # "keep" | "render" | "repaint"
    take: AudioTake | None = None  # the take to render/repaint (render/repaint)
    current: AudioTake | None = None  # take already covering now (keep/repaint)
    reason: str = ""


@dataclass
class AudioPlanner:
    """Decides music-take coverage (§35/§40). Pure logic; I/O stays in the caller."""

    take_seconds: float = 45.0
    ahead_seconds: float = 20.0
    takes: list[AudioTake] = field(default_factory=list)
    # Repaint gate (Stage B): a caption change repaints the unconsumed
    # region only when the new caption is genuinely different (Jaccard
    # token-set similarity below this threshold). A mere LLM rewording
    # (boba baseline: 0.72–1.00) falls through to chained/keep, so the
    # music evolves at take joints instead of restarting mid-take.
    repaint_similarity_threshold: float = 0.5
    # When set, fresh-take durations snap to whole multiples of one
    # segment so takes chain on segment-aligned boundaries (beat-grid
    # downbeats stay on segment boundaries across take joints). None
    # keeps the legacy unquantized length (old runs, unit tests).
    segment_seconds: float | None = None

    def coverage_until(self) -> float:
        """Video-time covered by rendered takes (0.0 when the ledger is empty)."""
        if not self.takes:
            return 0.0
        return max(take.covers_until() for take in self.takes)

    def take_for_time(self, video_time: float) -> AudioTake | None:
        """Newest take covering `video_time`, or None when uncovered."""
        covering = [
            take for take in self.takes if take.covers_from <= video_time < take.covers_until()
        ]
        if not covering:
            return None
        return max(covering, key=lambda take: take.segment_index)

    def plan(
        self,
        video_time: float,
        caption: str,
        seed: int,
        segment_index: int,
    ) -> PlanDecision:
        """Decide the next audio action for the upcoming video time.

        - No take covers now → "render" a fresh take starting at `video_time`.
        - Coverage runs out within `ahead_seconds` → "render" chained at the
          coverage end (audio-ahead: the swap happens before video needs it).
        - Caption changed and the current take still has an unconsumed region
          → "repaint" that region (continuation, §37) instead of a fresh take.
          ACE repaint output is timeline-aligned with its source (the head
          before repaint_start is preserved near bit-exact), so the new take
          inherits the source's `covers_from` — anchoring it at `video_time`
          would replay the preserved head in the next slice (duplicated
          music at the segment boundary). The repaint gate (Stage B) skips
          this when the captions are merely reworded (similarity at or above
          `repaint_similarity_threshold`): the music then evolves at the next
          take joint, carried by the chained take's fresh caption.
        - Otherwise → "keep" serving from the current take.
        """
        current = self.take_for_time(video_time)
        if current is None:
            return PlanDecision(
                action="render",
                take=self._fresh_take(video_time, caption, seed, segment_index),
                reason="no take covers current video time",
            )
        if current.caption != caption and video_time < current.covers_until():
            similarity = token_set_similarity(current.caption, caption)
            if similarity < self.repaint_similarity_threshold:
                return PlanDecision(
                    action="repaint",
                    take=self._fresh_take(current.covers_from, caption, seed, segment_index),
                    current=current,
                    reason=(
                        "caption changed with unconsumed region remaining "
                        f"(similarity {similarity:.3f})"
                    ),
                )
            # Rewording, not a new direction (Stage B gate): fall through
            # to chained/keep so the music evolves at the next take joint,
            # carried by the chained take's fresh caption.
        if current.covers_until() - video_time <= self.ahead_seconds:
            chained = self._fresh_take(current.covers_until(), caption, seed + 1, segment_index)
            return PlanDecision(
                action="render",
                take=chained,
                current=current,
                reason="coverage within ahead window; chaining next take",
            )
        return PlanDecision(action="keep", current=current, reason="covered")

    def record(self, take: AudioTake) -> None:
        """Append a rendered take to the in-memory ledger."""
        self.takes.append(take)

    def _fresh_take(
        self, covers_from: float, caption: str, seed: int, segment_index: int
    ) -> AudioTake:
        """Skeleton for a take the caller will render (path filled in after)."""
        take_id = f"take_{len(self.takes):04d}"
        duration = self.take_seconds
        if self.segment_seconds is not None:
            from voyage.audio.beat import quantize_take_seconds

            duration = quantize_take_seconds(self.take_seconds, self.segment_seconds)
        if not duration > self.ahead_seconds:
            raise ValueError(
                f"quantized take duration ({duration}) must exceed ahead_seconds "
                f"({self.ahead_seconds}): a take no longer than the audio-ahead "
                "window forces a full video-audio GPU swap on EVERY segment — "
                "keep take_seconds >> ahead_seconds (issue 121 plan-site guard)"
            )
        return AudioTake(
            take_id=take_id,
            path="",
            caption=caption,
            seed=seed,
            covers_from=covers_from,
            duration=duration,
            segment_index=segment_index,
        )


def load_takes(ledger: Path) -> list[AudioTake]:
    """Read the persisted takes ledger (missing file → empty)."""
    import json

    if not ledger.exists():
        return []
    takes = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            takes.append(AudioTake.from_dict(json.loads(line)))
    return takes


def append_take(ledger: Path, take: AudioTake) -> None:
    """Durably append one take to the ledger (flush + fsync + fsync_dir).

    File fsync persists content; the directory sync persists the namespace
    entry (issue 101, in-tree contract in voyage/atomic.py): without it a
    crash can lose the ledger tail while the rendered take file survives,
    and the next run re-renders audio it already has.
    """
    import json
    import os

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(take.to_dict()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(ledger.parent)

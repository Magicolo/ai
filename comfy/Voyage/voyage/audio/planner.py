"""Audio slow-loop planner (§35/§40): 30-60s music takes covering the video timeline.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        """Rebuild from a ledger line (pre-BPM lines carry no bpm key)."""
        bpm_raw = raw.get("bpm")
        return cls(
            take_id=str(raw["take_id"]),
            path=str(raw["path"]),
            caption=str(raw["caption"]),
            seed=int(raw["seed"]),
            covers_from=float(raw["covers_from"]),
            duration=float(raw["duration"]),
            segment_index=int(raw["segment_index"]),
            bpm=float(bpm_raw) if bpm_raw is not None else None,
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
          music at the segment boundary).
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
            return PlanDecision(
                action="repaint",
                take=self._fresh_take(current.covers_from, caption, seed, segment_index),
                current=current,
                reason="caption changed with unconsumed region remaining",
            )
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
    """Durably append one take to the ledger (flush + fsync)."""
    import json
    import os

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(take.to_dict()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

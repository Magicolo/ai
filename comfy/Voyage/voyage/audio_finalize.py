"""Finalize-time ACE music rendering for deferred backends (DESIGN §140).

Why: ltxv/causvid commit no ACE takes (full move to finalize — no
per-segment `_with_audio_gpu` evict/render/evict/rebuild on the 4060).
Commit writes a timeline-exact silent stub (`write_deferred_stub_audio`);
finalize replays the stored director decisions in segment order
(`ensure_deferred_takes`) and renders takes through the injected
`render_take_fn` seam (production: ACE-Step `SubprocessWorker`, same
spawn pattern as `render_sfx_bed`; tests: stdlib sine). ltx25/ltx23
joint-audio never enters this module (`is_deferred_backend` gate).

Ledger contract mirrors `audio/planner.py` (`flush+fsync+fsync_dir`,
issue 101): a take is appended only after its file renders, so a
failed render raises `MediaError` with nothing appended.

Continuity preservation (same module, second job): the deferred branch
skips the commit-time audio swap whose rebuild derived `video_tail.mp4`
as a side effect, so `derive_conditioning_tail` derives it at commit
instead — otherwise every segment goes fresh (121f) instead of
continuing (96f).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voyage.audio.beat import beats_for_segment
from voyage.audio.planner import (
    TAKES_FILENAME,
    AudioPlanner,
    AudioTake,
    append_take,
    load_takes,
)
from voyage.errors import MediaError
from voyage.media_audio import probed_take_seconds
from voyage.seeds import audio_seed
from voyage.segment_manifest import load_metrics, load_transition
from voyage.supervisor_proposal import effective_music_caption

#: Backends whose ACE music renders at finalize, never at commit.
DEFERRED_AUDIO_BACKENDS = frozenset({"ltxv", "causvid"})

#: Commit stub sample format (matches the take path: 48 kHz stereo s16le).
_STUB_CODEC = "pcm_s16le"

#: Causvid default overlap (`workers/video_causvid.py:93
#: DEFAULT_OVERLAP_FRAMES`; nothing outside the worker overrides it).
#: Resume re-encodes `4 * (overlap - 1) + 1` committed frames, so at the
#: default overlap 3 the window is 9 frames and the shared 25-frame floor
#: rules; a larger overlap raises the tail to match.
_CAUSVID_DEFAULT_OVERLAP_FRAMES = 3

#: ACE renderer ceiling (mirrors the commit path's explicit cap).
_ACE_MAX_BPM = 300.0

#: Commit-path take sizing replayed verbatim at finalize.
_TAKE_SECONDS = 45.0
_AHEAD_SECONDS = 20.0
_BEATS_PER_SEGMENT = 4


def is_deferred_backend(backend: str) -> bool:
    """True for take-based CUDA backends whose music moves to finalize."""
    return backend in DEFERRED_AUDIO_BACKENDS


def write_deferred_stub_audio(
    segment: Path, duration_seconds: float, sample_rate: int, channels: int
) -> Path:
    """Write a timeline-exact silent `audio.wav` into a segment dir.

    The stub satisfies `validate_audio` (format) + the 0.6 s A/V gate so
    commit/validate/finalize triage pass unchanged; real music arrives at
    finalize via `ensure_deferred_takes`. Raises `MediaError` when ffmpeg
    fails or the stub probes empty.
    """
    from voyage.media_audio import run_capture

    out = segment / "audio.wav"
    layout = "stereo" if channels == 2 else "mono" if channels == 1 else None
    if layout is None:
        raise MediaError(f"deferred stub supports mono/stereo, got {channels}ch")
    proc = run_capture(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl={layout}",
            "-t",
            f"{duration_seconds:.6f}",
            "-c:a",
            _STUB_CODEC,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(out),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"deferred stub render failed for {segment}: {proc.stderr[-2000:]}")
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError(f"deferred stub render produced no audio: {out}")
    return out


def deferred_tail_frames(
    backend: str, *, overlap_frames: int = _CAUSVID_DEFAULT_OVERLAP_FRAMES
) -> int:
    """Conditioning-tail length matching each deferred worker's resume derive.

    ltxv resumes onto `CONDITIONING_TAIL_FRAMES` (25); causvid resumes
    onto `max(25, reencode_window(overlap))`. The commit-time derive must
    write exactly this many frames: a short tail would be adopted
    untouched by resume (existing tail wins) and silently anchor the
    next segment on too few frames.     Raises `MediaError` for unknown
    backends — guessing a length is worse than failing loud.
    """
    # Local import (§12 GPU ban): `workers.video_common` pulls torch at
    # module scope, which the supervisor/CLI side must never import.
    from voyage.workers import video_common

    if backend == "ltxv":
        return video_common.DERIVED_TAIL_FRAMES
    if backend == "causvid":
        window = 4 * (overlap_frames - 1) + 1
        return max(video_common.DERIVED_TAIL_FRAMES, window)
    raise MediaError(f"deferred tail derive has no frame count for backend {backend!r}")


def derive_conditioning_tail(segment: Path, tail_frames: int) -> bool:
    """Derive this segment's `video_tail.mp4` for the next segment to chain.

    Why: the deferred branch skips the commit-time audio swap whose
    rebuild derived the tail as a side effect. Without this derive the
    next segment's resident tail path is missing and the worker goes
    fresh (121f) instead of continuing (96f). Returns True when derived,
    False when an existing tail is adopted untouched. Raises
    `MediaError` when the segment video is missing or ffmpeg fails —
    a missing tail must never pass silently.
    """
    # Local import (§12 GPU ban): see `deferred_tail_frames`.
    from voyage.workers import video_common

    if tail_frames < 1:
        raise MediaError(f"deferred tail derive needs tail_frames >= 1, got {tail_frames}")
    video = segment / "video.mp4"
    if not video.is_file():
        raise MediaError(f"deferred tail derive needs {video} (segment video missing)")
    tail = segment / video_common.TAIL_FILENAME
    if tail.is_file():
        return False
    try:
        video_common.derive_tail_from_segment_video(video, tail, tail_frames)
    except (OSError, ValueError, RuntimeError) as exc:
        raise MediaError(f"deferred tail derive failed for {segment}: {exc}") from exc
    return True


def _take_path(audio_dir: Path, run_dir: Path, take_id: str) -> tuple[Path, str]:
    """Take file location + run-relative stored form (issue 016)."""
    take_file = audio_dir / f"{take_id}.wav"
    return take_file, take_file.relative_to(run_dir).as_posix()


def _load_existing_takes(run_dir: Path) -> list[AudioTake]:
    """Ledger truth for finalize replays (SFX last-wins philosophy).

    A re-finalize must no-op when every take already rendered — starting
    from an empty list would re-render and double-append. A missing
    ledger is a fresh finalize, not an error.
    """
    ledger = run_dir / "audio" / TAKES_FILENAME
    if not ledger.exists():
        return []
    return load_takes(ledger)


def _segment_stretched_seconds(segment: Path, source_fps: float, stretch: float) -> float:
    """Stretched content seconds for one usable segment (fail-loud)."""
    metrics = load_metrics(segment)
    frames = metrics.get("frames", 0)
    if not isinstance(frames, int) or frames < 1:
        raise MediaError(f"deferred replay needs frames for {segment}")
    return frames / source_fps * stretch


def _segment_music_inputs(
    segment: Path, music_style: str, explicit_caption: str | None
) -> tuple[str, float]:
    """Director caption + clamped energy for one segment's replay."""
    transition = load_transition(segment)
    raw_decision = transition.get("decision", {})
    audio = raw_decision.get("audio", {}) if isinstance(raw_decision, dict) else {}
    caption = effective_music_caption(
        explicit_caption,
        str(audio.get("music_caption", "")),
        music_style,
    )
    energy_raw = audio.get("energy", 0.5)
    energy = float(energy_raw) if isinstance(energy_raw, (int, float)) else 0.5
    return caption, min(1.0, max(0.0, energy))


def deferred_render_pending(
    *,
    run_dir: Path,
    usable: list[Path],
    source_fps: float,
    run_seed: int,
    music_style: str = "",
    explicit_caption: str | None = None,
    stretch: float = 1.0,
    take_seconds: float = _TAKE_SECONDS,
    ahead_seconds: float = _AHEAD_SECONDS,
    beats_per_segment: int = _BEATS_PER_SEGMENT,
    sample_rate: int = 48000,
    channels: int = 2,
) -> bool:
    """Pure dry walk: True when replay would render at least one take.

    No GPU, no filesystem writes — the finalize gate uses this to skip
    spawning the ACE worker when the ledger already covers the timeline
    (re-finalize), and the parallel path uses it to serialize only when
    rendering is actually pending. `sample_rate`/`channels` ride the
    shared sizing dict but never affect coverage (format, not timeline).
    """
    takes = _load_existing_takes(run_dir)
    cursor = 0.0
    for segment in usable:
        number = int(segment.name)
        stretched = _segment_stretched_seconds(segment, source_fps, stretch)
        caption, _ = _segment_music_inputs(segment, music_style, explicit_caption)
        planner = AudioPlanner(
            take_seconds=take_seconds,
            ahead_seconds=ahead_seconds,
            takes=list(takes),
            segment_seconds=stretched,
        )
        end = cursor + stretched
        while True:
            seed = audio_seed(run_seed, number, len(takes))
            plan = planner.plan(cursor, caption, seed, number)
            if plan.action == "keep":
                break
            return True
        cursor = end
    return False


def ensure_deferred_takes(
    *,
    run_dir: Path,
    usable: list[Path],
    source_fps: float,
    run_seed: int,
    render_take_fn: Callable[[dict[str, Any], Path], None],
    music_style: str = "",
    explicit_caption: str | None = None,
    stretch: float = 1.0,
    take_seconds: float = _TAKE_SECONDS,
    ahead_seconds: float = _AHEAD_SECONDS,
    beats_per_segment: int = _BEATS_PER_SEGMENT,
    sample_rate: int = 48000,
    channels: int = 2,
) -> list[dict[str, Any]]:
    """Replay director decisions and render ACE takes at finalize.

    Walks `usable` segment dirs in order, replays each stored decision
    through `AudioPlanner` (same seed formula, take quantization, and
    beat grid as the commit path), and renders takes via
    `render_take_fn` (payload mirrors the commit ACE payload). Take
    coverage targets the *stretched* timeline (`stretch` = presentation
    factor, e.g. 1.5 for 2x interp at 32 fps) so slices exist for the
    slow-mo mix. Appends each rendered take to `audio/takes.jsonl`
    before rendering the next; any render failure raises `MediaError`
    with nothing appended for that take.
    """
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger = audio_dir / TAKES_FILENAME
    takes: list[AudioTake] = _load_existing_takes(run_dir)
    rendered: list[dict[str, Any]] = []
    cursor = 0.0
    for segment in usable:
        number = int(segment.name)
        stretched = _segment_stretched_seconds(segment, source_fps, stretch)
        caption, energy = _segment_music_inputs(segment, music_style, explicit_caption)
        planner = AudioPlanner(
            take_seconds=take_seconds,
            ahead_seconds=ahead_seconds,
            takes=list(takes),
            segment_seconds=stretched,
        )
        _, grid_bpm = beats_for_segment(stretched, beats_per_segment, max_bpm=_ACE_MAX_BPM)
        take_bpm = int(round(grid_bpm))
        end = cursor + stretched
        while True:
            seed = audio_seed(run_seed, number, len(takes))
            plan = planner.plan(cursor, caption, seed, number)
            if plan.action == "keep":
                cursor = end
                break
            take = plan.take
            if take is None:
                raise MediaError(f"deferred plan for {segment.name} rendered no take")
            take.bpm = float(take_bpm)
            take_file, stored = _take_path(audio_dir, run_dir, take.take_id)
            payload: dict[str, Any] = {
                "segment_id": segment.name,
                "style": caption,
                "energy": energy,
                "seed": take.seed,
                "output_path": str(take_file),
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_seconds": take.duration,
                "bpm": take_bpm,
            }
            if plan.action == "repaint" and plan.current is not None:
                current = plan.current
                payload["task_type"] = "repaint"
                payload["reference_audio"] = str(current.resolved_path(run_dir))
                payload["repaint_start"] = cursor - current.covers_from
                payload["repaint_end"] = current.duration
            try:
                render_take_fn(payload, take_file)
            except Exception as exc:
                raise MediaError(f"deferred take {take.take_id} render failed: {exc}") from exc
            shortfall = take.duration - probed_take_seconds(take_file)
            if shortfall > 1e-3:
                take.duration = probed_take_seconds(take_file)
            take.path = stored
            planner.record(take)
            takes.append(take)
            append_take(ledger, take)
            rendered.append(take.to_dict())
            if planner.coverage_until() >= end - 1e-6:
                cursor = end
                break
    return rendered


def ensure_deferred_for_finalize(
    *,
    run_dir: Path,
    usable: list[Path],
    source_fps: float,
    stretch: float,
    run_seed: int,
    models_dir: Path | str | None,
    device: str = "cuda:0",
    music_style: str = "",
    explicit_caption: str | None = None,
    take_seconds: float = _TAKE_SECONDS,
    ahead_seconds: float = _AHEAD_SECONDS,
    beats_per_segment: int = _BEATS_PER_SEGMENT,
    sample_rate: int = 48000,
    channels: int = 2,
) -> bool:
    """Render pending deferred takes, spawning ACE only when needed.

    Pure `deferred_render_pending` dry walk first — a complete ledger
    (re-finalize, or the parallel path's Thread A after the pre-fork
    ensure) returns False with no worker spawned. Otherwise one shared
    `SubprocessWorker` renders every pending take and shuts down
    best-effort. Returns True when at least one take rendered.
    """
    sizing: dict[str, Any] = {
        "music_style": music_style,
        "explicit_caption": explicit_caption,
        "stretch": stretch,
        "take_seconds": take_seconds,
        "ahead_seconds": ahead_seconds,
        "beats_per_segment": beats_per_segment,
        "sample_rate": sample_rate,
        "channels": channels,
    }
    if not deferred_render_pending(
        run_dir=run_dir,
        usable=usable,
        source_fps=source_fps,
        run_seed=run_seed,
        **sizing,
    ):
        return False
    render_fn, shutdown = spawn_ace_render_fn(run_dir, models_dir, device)
    try:
        ensure_deferred_takes(
            run_dir=run_dir,
            usable=usable,
            source_fps=source_fps,
            run_seed=run_seed,
            render_take_fn=render_fn,
            **sizing,
        )
    finally:
        shutdown()
    return True


def spawn_ace_render_fn(
    run_dir: Path, models_dir: Path | str | None, device: str = "cuda:0"
) -> tuple[Callable[[dict[str, Any], Path], None], Callable[[], None]]:
    """Start the ACE-Step worker and return `(render_fn, shutdown_fn)`.

    Production `render_take_fn` for `ensure_deferred_takes` (DESIGN §140):
    one `SubprocessWorker` around `audio_worker_module("acestep")`
    (same spawn pattern as `render_sfx_bed`), so all finalize takes
    share one GPU residency instead of one process per take. The worker
    writes `payload["output_path"]` (rewritten to the take file); any
    worker error propagates and `ensure_deferred_takes` wraps it
    fail-loud. `shutdown_fn` stops the worker best-effort (never masks
    the render result). Raises `MediaError` when `models_dir` is
    missing — deferred finalize cannot render without weights.
    """
    from voyage.rpc import SubprocessWorker
    from voyage.supervisor_routing import audio_worker_module

    if models_dir is None:
        raise MediaError("deferred ACE render needs a models_dir (no weights)")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    worker = SubprocessWorker(
        audio_worker_module("acestep"),
        run_dir,
        run_dir / "logs" / "ace-finalize.log",
        init_op="init",
        init_payload={"models_dir": str(models_dir), "device": device},
    )
    worker.start()

    def _render(payload: dict[str, Any], output_path: Path) -> None:
        request = dict(payload)
        request["output_path"] = str(output_path)
        worker.call("generate_audio", request)

    def _shutdown() -> None:
        with contextlib.suppress(Exception):
            worker.stop()

    return _render, _shutdown

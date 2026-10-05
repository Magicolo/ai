"""Finalize-time ACE music rendering for always-deferred finalize (DESIGN §140).

Why: every backend commits video only — no per-segment audio coverage, no
`audio.wav` previews. Finalize replays the stored director decisions in
segment order (`ensure_deferred_takes`) and renders takes through the
injected `render_take_fn` seam (production: ACE-Step `SubprocessWorker`
or the fake sine worker via `audio_backend` routing; tests: stdlib sine).

Ledger contract mirrors `audio/planner.py` (`flush+fsync+fsync_dir`,
issue 101): a take is appended only after its file renders, so a
failed render raises `MediaError` with nothing appended.

Continuity preservation (same module, second job): the commit path holds
no audio worker whose rebuild derived `video_tail.mp4` as a side effect,
so `derive_conditioning_tail` derives it at commit instead — otherwise
every segment goes fresh (121f) instead of continuing (96f).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voyage.audio.beat import beats_for_segment
from voyage.audio.planner import (
    TAKES_FILENAME,
    AudioPlanner,
    AudioTake,
    append_take,
    load_takes,
)
from voyage.console import optional_bar, optional_stage
from voyage.errors import MediaError
from voyage.media_audio import probed_take_seconds, run_capture
from voyage.seeds import audio_seed
from voyage.segment_manifest import load_metrics, load_transition
from voyage.supervisor_proposal import effective_music_caption

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

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

#: Continuation overlap between chained takes (seconds): take N+1 opens
#: with take N's tail (rendered as an ACE repaint over a tail+silence
#: source) and extends it instead of restarting from silence. Takes keep
#: the full quantized duration, so the overlap region is covered twice
#: and newest-wins serving plays the continuing take there. Tuned to one
#: stretched segment (6.0s at the ltx25 slow-mo cadence) so the joint the
#: listener hears is a same-material continuation, not a crossfade between
#: two different generations.
_CHAIN_OVERLAP_SECONDS = 6.0


def deferred_tail_frames(
    backend: str, *, overlap_frames: int = _CAUSVID_DEFAULT_OVERLAP_FRAMES
) -> int:
    """Conditioning-tail length for each streaming worker's resume derive.

    ltxv/ltx25/ltx23 resume onto `CONDITIONING_TAIL_FRAMES` (25); causvid
    resumes onto `max(25, reencode_window(overlap))`. The commit-time
    derive must write exactly this many frames: a short tail would be
    adopted untouched by resume (existing tail wins) and silently anchor
    the next segment on too few frames. Gated on
    `supervisor_routing.STREAMING_VIDEO_BACKENDS` (lazy import, §12 GPU
    ban — the supervisor owns the eager import): raises `MediaError`
    for non-streaming backends (`fake` commits no tail because its
    worker is stateless) — guessing a length is worse than failing loud.
    """
    # Local imports (§12 GPU ban): `workers.video_common` pulls torch at
    # module scope, which the supervisor/CLI side must never import;
    # `supervisor_routing` is stdlib-only but stays lazy for symmetry.
    from voyage.supervisor_routing import STREAMING_VIDEO_BACKENDS
    from voyage.workers import video_common

    if backend not in STREAMING_VIDEO_BACKENDS:
        raise MediaError(f"deferred tail derive has no frame count for backend {backend!r}")
    if backend in ("ltxv", "ltx25", "ltx23"):
        return video_common.DERIVED_TAIL_FRAMES
    if backend == "causvid":
        window = 4 * (overlap_frames - 1) + 1
        return max(video_common.DERIVED_TAIL_FRAMES, window)
    raise MediaError(f"deferred tail derive has no frame count for backend {backend!r}")


def derive_conditioning_tail(segment: Path, tail_frames: int) -> bool:
    """Derive this segment's `video_tail.mp4` for the next segment to chain.

    Why: the commit path holds no audio worker whose rebuild derived the
    tail as a side effect. Without this derive the next segment's
    resident tail path is missing and the worker goes fresh (121f)
    instead of continuing (96f). Returns True when derived, False when
    an existing tail is adopted untouched. Raises `MediaError` when the
    segment video is missing or ffmpeg fails — a missing tail must
    never pass silently.
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


def _build_continuation_src(
    prev_take_file: Path,
    take_duration: float,
    overlap: float,
    sample_rate: int,
    channels: int,
    dest: Path,
) -> Path:
    """Build the ACE repaint source for a chained continuation take.

    Output is `take_duration` seconds of canonical WAV: the previous
    take's last `overlap` seconds followed by `(take_duration - overlap)`
    seconds of silence. Rendering the chained take as a repaint over this
    source (repaint region `[overlap, take_duration]`) preserves the tail
    head near bit-exact while the new material grows out of it —
    continuation instead of restart. Fails loud when the previous file is
    missing, shorter than the overlap, or ffmpeg fails; the source file
    is never ledgered (the caller deletes it best-effort after a
    successful render, keeps it for forensics after a failure).
    """
    if not prev_take_file.is_file():
        raise MediaError(f"continuation source needs previous take file {prev_take_file} (missing)")
    if not overlap > 0.0:
        raise MediaError(f"continuation overlap must be positive (got {overlap})")
    if not take_duration > overlap:
        raise MediaError(
            f"continuation take duration ({take_duration}) must exceed overlap ({overlap})"
        )
    prev_seconds = probed_take_seconds(prev_take_file)
    if prev_seconds < overlap - 1e-6:
        raise MediaError(
            f"previous take {prev_take_file} is {prev_seconds:.3f}s, "
            f"shorter than the {overlap:.3f}s continuation overlap"
        )
    channel_layout = "stereo" if channels == 2 else "mono"
    tail_start = prev_seconds - overlap
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(prev_take_file),
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl={channel_layout}",
            "-filter_complex",
            f"[0:a]atrim=start={tail_start:.6f},asetpts=PTS-STARTPTS[tail];"
            f"[1:a]atrim=end={take_duration - overlap:.6f},"
            "asetpts=PTS-STARTPTS[sil];"
            "[tail][sil]concat=n=2:v=0:a=1[out]",
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(
            f"continuation source build failed for {prev_take_file}: {proc.stderr[-2000:]}"
        )
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"continuation source build produced empty output for {dest}")
    return dest


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
    """Director caption + clamped energy for one segment's replay.

    Production manifests store the decision flattened at `transition` (no
    `decision` wrapper — `supervisor` persists `decision.model_dump()`
    directly); the legacy/test shape nests it under `transition.decision`.
    Read the wrapper when present, else the section itself — reading only
    the wrapper silently pinned every take to `music_style` (kaolin: 73
    identical captions instead of the evolving per-segment prompts).
    """
    transition = load_transition(segment)
    raw_decision = transition.get("decision")
    if not isinstance(raw_decision, dict) or not raw_decision:
        raw_decision = transition
    audio = raw_decision.get("audio", {}) if isinstance(raw_decision, dict) else {}
    caption = effective_music_caption(
        explicit_caption,
        str(audio.get("music_caption", "")),
        music_style,
    )
    energy_raw = audio.get("energy", 0.5)
    energy = float(energy_raw) if isinstance(energy_raw, (int, float)) else 0.5
    return caption, min(1.0, max(0.0, energy))


def _replay_segments(
    usable: list[Path],
    source_fps: float,
    stretch: float,
    music_style: str,
    explicit_caption: str | None,
) -> list[tuple[float, float, int, str, float]]:
    """Per-segment replay row: (start, stretched, number, caption, energy).

    Starts accumulate exactly like the replay walks below, so a take's
    `covers_from` maps back to the segment where its coverage begins.
    Both the dry walk and the render walk read these rows (never their
    own accumulators), so the gate and the render can never disagree on
    video time or captions.
    """
    replay: list[tuple[float, float, int, str, float]] = []
    cursor = 0.0
    for segment in usable:
        number = int(segment.name)
        stretched = _segment_stretched_seconds(segment, source_fps, stretch)
        caption, energy = _segment_music_inputs(segment, music_style, explicit_caption)
        replay.append((cursor, stretched, number, caption, energy))
        cursor += stretched
    return replay


def _coverage_start_position(
    replay: list[tuple[float, float, int, str, float]], covers_from: float
) -> int:
    """Index of the segment containing `covers_from` (start-segment rule).

    A take begins serving where its coverage begins, so that segment's
    director caption is its ACE prompt — not the caption of the segment
    being processed when the take chained (up to `ahead_seconds` earlier
    on the timeline). Clamps to the first segment for float dust at 0.0
    and to the last for chained overhang past the timeline end.
    """
    position = 0
    for index, (start, _stretched, _number, _caption, _energy) in enumerate(replay):
        if start <= covers_from + 1e-6:
            position = index
        else:
            break
    return position


def _resolve_take_caption(
    replay: list[tuple[float, float, int, str, float]], covers_from: float
) -> tuple[str, int]:
    """(caption, segment_index) a fresh/chained take carries in the ledger.

    The caption (and segment index) of the segment where coverage begins.
    Repaints never reach here — they keep the current segment's caption
    (the new caption is the repaint's entire purpose); see the caller.
    """
    position = _coverage_start_position(replay, covers_from)
    _start, _stretched, number, caption, _energy = replay[position]
    return caption, number


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
    chain_overlap_seconds: float = _CHAIN_OVERLAP_SECONDS,
) -> bool:
    """Pure dry walk: True when replay would render at least one take.

    No GPU, no filesystem writes — the finalize gate uses this to skip
    spawning the audio worker when the ledger already covers the timeline
    (re-finalize). `sample_rate`/`channels` ride the shared sizing dict
    but never affect coverage (format, not timeline). Reads the same
    replay rows as `ensure_deferred_takes`, so the gate and the render
    can never disagree on video time or captions.
    """
    takes = _load_existing_takes(run_dir)
    replay = _replay_segments(usable, source_fps, stretch, music_style, explicit_caption)
    for start, stretched, number, caption, _energy in replay:
        planner = AudioPlanner(
            take_seconds=take_seconds,
            ahead_seconds=ahead_seconds,
            takes=list(takes),
            segment_seconds=stretched,
            chain_overlap_seconds=chain_overlap_seconds,
        )
        while True:
            seed = audio_seed(run_seed, number, len(takes))
            plan = planner.plan(start, caption, seed, number)
            if plan.action == "keep":
                break
            return True
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
    progress: VoyageConsole | None = None,
    chain_overlap_seconds: float = _CHAIN_OVERLAP_SECONDS,
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
    with nothing appended for that take. Each rendered take carries the
    music caption of the segment where its coverage begins (repaints keep
    the current segment's caption — the new caption is their purpose).
    Chained takes overlap the previous take by `chain_overlap_seconds`
    and render as ACE repaints continuing its tail (not restarts).
    """
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger = audio_dir / TAKES_FILENAME
    takes: list[AudioTake] = _load_existing_takes(run_dir)
    rendered: list[dict[str, Any]] = []
    replay = _replay_segments(usable, source_fps, stretch, music_style, explicit_caption)
    take_bar = optional_bar(progress, "ace takes")
    with take_bar as tracker:
        for position, segment in enumerate(usable):
            start, stretched, number, caption, energy = replay[position]
            planner = AudioPlanner(
                take_seconds=take_seconds,
                ahead_seconds=ahead_seconds,
                takes=list(takes),
                segment_seconds=stretched,
                chain_overlap_seconds=chain_overlap_seconds,
            )
            _, grid_bpm = beats_for_segment(stretched, beats_per_segment, max_bpm=_ACE_MAX_BPM)
            take_bpm = int(round(grid_bpm))
            end = start + stretched
            while True:
                seed = audio_seed(run_seed, number, len(takes))
                plan = planner.plan(start, caption, seed, number)
                if plan.action == "keep":
                    break
                take = plan.take
                if take is None:
                    raise MediaError(f"deferred plan for {segment.name} rendered no take")
                if plan.action == "repaint":
                    take.caption = caption
                else:
                    take.caption, take.segment_index = _resolve_take_caption(
                        replay, take.covers_from
                    )
                take.bpm = float(take_bpm)
                take_file, stored = _take_path(audio_dir, run_dir, take.take_id)
                payload: dict[str, Any] = {
                    "segment_id": segment.name,
                    "style": take.caption,
                    "energy": energy,
                    "seed": take.seed,
                    "output_path": str(take_file),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration_seconds": take.duration,
                    "bpm": take_bpm,
                }
                continuation_src: Path | None = None
                if plan.action == "repaint" and plan.current is not None:
                    current = plan.current
                    payload["task_type"] = "repaint"
                    payload["reference_audio"] = str(current.resolved_path(run_dir))
                    payload["repaint_start"] = start - current.covers_from
                    payload["repaint_end"] = current.duration
                elif (
                    plan.action == "render"
                    and plan.current is not None
                    and chain_overlap_seconds > 0.0
                ):
                    previous = plan.current
                    _src_file, _ = _take_path(audio_dir, run_dir, f"{take.take_id}_src")
                    continuation_src = _build_continuation_src(
                        previous.resolved_path(run_dir),
                        take.duration,
                        chain_overlap_seconds,
                        sample_rate,
                        channels,
                        _src_file,
                    )
                    payload["task_type"] = "repaint"
                    payload["reference_audio"] = str(continuation_src)
                    payload["repaint_start"] = chain_overlap_seconds
                    payload["repaint_end"] = take.duration
                try:
                    render_take_fn(payload, take_file)
                except Exception as exc:
                    raise MediaError(f"deferred take {take.take_id} render failed: {exc}") from exc
                if continuation_src is not None:
                    with contextlib.suppress(OSError):
                        continuation_src.unlink()
                shortfall = take.duration - probed_take_seconds(take_file)
                if shortfall > 1e-3:
                    take.duration = probed_take_seconds(take_file)
                take.path = stored
                planner.record(take)
                takes.append(take)
                append_take(ledger, take)
                rendered.append(take.to_dict())
                if tracker is not None:
                    tracker.update()
                if planner.coverage_until() >= end - 1e-6:
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
    audio_backend: str = "acestep",
    music_style: str = "",
    explicit_caption: str | None = None,
    take_seconds: float = _TAKE_SECONDS,
    ahead_seconds: float = _AHEAD_SECONDS,
    beats_per_segment: int = _BEATS_PER_SEGMENT,
    sample_rate: int = 48000,
    channels: int = 2,
    progress: VoyageConsole | None = None,
    chain_overlap_seconds: float = _CHAIN_OVERLAP_SECONDS,
) -> bool:
    """Render pending takes, spawning the audio worker only when needed.

    Pure `deferred_render_pending` dry walk first — a complete ledger
    (re-finalize) returns False with no worker spawned. Otherwise one
    shared `SubprocessWorker` renders every pending take and shuts down
    best-effort: the fake sine worker when `audio_backend == "fake"`
    (no weights, `models_dir` ignored), else the ACE-Step worker via
    `spawn_ace_render_fn`. Returns True when at least one take rendered.
    Raises `MediaError` for an unknown `audio_backend` — guessing a
    renderer is worse than failing loud. `sizing` carries the shared
    take geometry (including `chain_overlap_seconds`) into both walks
    so the gate and the render can never disagree.
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
        "chain_overlap_seconds": chain_overlap_seconds,
    }
    if not deferred_render_pending(
        run_dir=run_dir,
        usable=usable,
        source_fps=source_fps,
        run_seed=run_seed,
        **sizing,
    ):
        return False
    if audio_backend == "fake":
        render_fn, shutdown = spawn_fake_render_fn(run_dir)
    elif audio_backend == "acestep":
        render_fn, shutdown = spawn_ace_render_fn(run_dir, models_dir, device)
    else:
        raise MediaError(f"unknown audio backend {audio_backend!r} (known: fake, acestep)")
    stage_cm = optional_stage(progress, "music takes", audio_backend)
    try:
        with stage_cm:
            ensure_deferred_takes(
                run_dir=run_dir,
                usable=usable,
                source_fps=source_fps,
                run_seed=run_seed,
                render_take_fn=render_fn,
                progress=progress,
                **sizing,
            )
    finally:
        shutdown()
    return True


def spawn_fake_render_fn(
    run_dir: Path,
) -> tuple[Callable[[dict[str, Any], Path], None], Callable[[], None]]:
    """Start the fake sine worker and return `(render_fn, shutdown_fn)`.

    Offline `render_take_fn` for `ensure_deferred_takes`: one
    `SubprocessWorker` around `audio_worker_module("fake")` (same
    `generate_audio` contract as the ACE worker — deterministic sine,
    no GPU, no weights, so `models_dir` is neither taken nor needed).
    The worker writes `payload["output_path"]`; any worker error
    propagates and `ensure_deferred_takes` wraps it fail-loud.
    `shutdown_fn` stops the worker best-effort (never masks the render
    result).
    """
    from voyage.rpc import SubprocessWorker
    from voyage.supervisor_routing import audio_worker_module

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    worker = SubprocessWorker(
        audio_worker_module("fake"),
        run_dir,
        run_dir / "logs" / "fake-finalize.log",
        init_op="init",
        init_payload={},
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
    missing — ACE finalize cannot render without weights.
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
        # ACE-Step venv (DESIGN §140 audio continuity): the ACE stack is
        # isolated in /opt/venvs/acestep on voyage-ltx; unset (video
        # image, tests) falls back to the supervisor interpreter.
        executable=os.environ.get("VOYAGE_ACESTEP_PYTHON"),
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

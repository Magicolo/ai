"""Finalize-time SFX pass (slice 3, three-caption doctrine).

Runs AFTER `finalize_run` publishes the music-only final: windows
condition on the shipped pixels (continuous frames, so junctions hear
both sides), join with manual fades (never acrossfade — same two
incidents as the music path), and amix lays the bed under the music
at -6 dB (Zoomy parity). Stems persist under `audio/sfx/` with an
`sfx.jsonl` ledger (immutable, versioned, fsynced — takes-philosophy);
`validate_run` extends read-only.

No torch/GPU imports here: workers are subprocesses behind the JSONL
loop; this module only plans, spawns, blends, and mixes.
"""

from __future__ import annotations

import contextlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import fsync_dir
from voyage.augment import augment_devices
from voyage.errors import MediaError
from voyage.media import (
    AV_ALIGNMENT_TOLERANCE_SECONDS,
    _audio_duration_seconds,
    _blend_pair,
    probe,
    run_capture,
)
from voyage.paths import resolve_stored_path

SFX_WINDOW_SECONDS = 8.0
"""Native MMAudio window (upstream: 1.23 s per 8 s clip on small)."""

SFX_WINDOW_OVERLAP = 1.0
"""Overlap between consecutive windows, killed by manual fades on join."""

MIN_SFX_WINDOW_SECONDS = 1.0
"""Shortest renderable window: below ~0.64 s the sync branch yields fewer
than 16 frames and the synchformer segments an empty list (same crash
class the music path floors with pad-to-17 — here the planner merges a
stub tail into its predecessor instead, since counts must stay exact)."""

SFX_VOLUME = 0.5
"""Bed level under the music (-6 dB, Zoomy parity)."""

SFX_LEDGER_NAME = "sfx.jsonl"
SFX_STEMS_DIRNAME = "sfx"

SFX_WORKER_MODULES = {
    "fake": "voyage.workers.sfx",
    "mmaudio": "voyage.workers.sfx_mmaudio",
}
"""Backend name → worker module (mirrors AUDIO_WORKER_MODULES)."""

SFX_DUAL_MODEL_SIZE = "small_44k"
"""Dual-shard runs small on both GPUs (quality-consistent windows —
a large/small mix would step quality at window joints)."""


@dataclass
class SfxWindow:
    """One effects window: half-open [start, start+duration) of the final."""

    window_id: str
    start: float
    duration: float
    caption: str
    seed: int


def plan_sfx_windows(
    timeline_seconds: float,
    segment_bounds: list[tuple[float, float, str]],
    seed_base: int = 0,
    window_seconds: float = SFX_WINDOW_SECONDS,
    overlap_seconds: float = SFX_WINDOW_OVERLAP,
) -> list[SfxWindow]:
    """Tile [0, timeline) into overlapping caption-carrying windows.

    Step is window - overlap; the last window clamps to the timeline. A
    window fully inside one segment carries that segment's SFX caption;
    a junction window joins both sides with "; " so the render hears the
    transition (the per-segment-SFX incoherence the user flagged).
    Seeds derive deterministically (base + index) so re-finalize
    re-renders identical bytes. Empty captions stay empty — the worker
    falls back to its neutral default.
    """
    if not timeline_seconds > 0.0:
        raise ValueError(f"timeline must be positive (got {timeline_seconds})")
    step = window_seconds - overlap_seconds
    if step <= 0.0:
        raise ValueError(f"overlap {overlap_seconds} must be < window {window_seconds}")
    if timeline_seconds < MIN_SFX_WINDOW_SECONDS:
        raise ValueError(
            f"timeline {timeline_seconds:.2f}s below renderable minimum "
            f"{MIN_SFX_WINDOW_SECONDS:.2f}s"
        )
    starts: list[float] = []
    cursor = 0.0
    while cursor < timeline_seconds - 1e-6:
        starts.append(cursor)
        cursor += step
    # Stub tail: a remainder under the renderable minimum merges into its
    # predecessor (extended to the timeline, still exact counts) instead
    # of rendering a starved window the model cannot condition.
    if len(starts) > 1 and timeline_seconds - starts[-1] < MIN_SFX_WINDOW_SECONDS:
        starts.pop()
    windows: list[SfxWindow] = []
    for index, start in enumerate(starts):
        end = min(start + window_seconds, timeline_seconds)
        window_end = start + window_seconds
        captions: list[str] = []
        for seg_start, seg_end, caption in segment_bounds:
            if seg_start < window_end - 1e-6 and seg_end > start + 1e-6:
                cleaned = " ".join(caption.split())
                if cleaned and cleaned not in captions:
                    captions.append(cleaned)
        windows.append(
            SfxWindow(
                window_id=f"w{index:04d}",
                start=start,
                duration=end - start,
                caption="; ".join(captions),
                seed=seed_base + index,
            )
        )
    return windows


def segment_sfx_bounds(
    run_dir: Path,
    usable: list[Path],
    fps: int,
    caption_override: str | None = None,
) -> list[tuple[float, float, str]]:
    """Per-segment (start, end, sfx_caption) over the finalized timeline.

    Bounds come from probed segment durations (same walk as the music
    path's `_segment_timeline`); captions read each segment's committed
    `transition.json` audio.sfx_caption best-effort (missing/torn/legacy
    → "" — a history read must never break a finalize). A
    `caption_override` replaces every segment caption (old runs whose
    decisions predate SFX captions, or a deliberate single-caption dub).
    """
    from voyage.models import EvolutionDecision

    bounds: list[tuple[float, float, str]] = []
    cursor = 0.0
    for segment in usable:
        try:
            info = probe(segment / "video.mp4")
            duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        except (MediaError, ValueError, KeyError, TypeError):
            duration = 0.0
        if duration <= 0.0:
            try:
                streams = probe(segment / "video.mp4").get("streams", [{}])
                first = streams[0] if streams else {}
                frames = int(first.get("nb_frames", 0) if isinstance(first, dict) else 0)
                duration = frames / fps if frames > 0 else 0.0
            except (MediaError, ValueError, KeyError, TypeError, IndexError):
                duration = 0.0
        caption = ""
        try:
            raw = json.loads((segment / "transition.json").read_text(encoding="utf-8"))
            caption = EvolutionDecision.model_validate(raw).audio.sfx_caption
        except (OSError, ValueError):
            caption = ""
        if caption_override is not None:
            caption = caption_override
        bounds.append((cursor, cursor + duration, caption or ""))
        cursor += duration
    return bounds


def append_sfx_window(ledger: Path, window: SfxWindow, path: str, model_size: str) -> None:
    """Durably append one rendered window (flush + fsync + fsync_dir, takes pattern).

    File fsync persists content; the directory sync persists the namespace
    entry (issue 101 twin of `append_take`, in-tree contract in
    `voyage/atomic.py`). Callers must serialize appends: `render_sfx_bed`
    collects worker results and appends in plan order after the pool joins,
    so the ledger is deterministic and never interleaved (issue 054).
    """
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "window_id": window.window_id,
        "start": window.start,
        "duration": window.duration,
        "caption": window.caption,
        "seed": window.seed,
        "path": path,
        "model_size": model_size,
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(ledger.parent)


def load_sfx_ledger(ledger: Path) -> list[dict[str, Any]]:
    """Read the persisted SFX ledger (missing file → empty)."""
    if not ledger.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def validate_sfx_ledger(run_dir: Path, timeline_seconds: float) -> list[str]:
    """Read-only SFX checks: files exist, windows tile the timeline.

    No ledger (SFX never ran / old run) is clean — the pass is optional.
    Duplicate `window_id` lines (re-render appends) dedupe last-wins and
    the walk sorts by start, so ledger order can never false-positive
    coverage (153, mirroring the `existing` dict in `render_sfx_bed`).
    """
    ledger = run_dir / "audio" / SFX_STEMS_DIRNAME / SFX_LEDGER_NAME
    if not ledger.exists():
        return []
    errors: list[str] = []
    try:
        records = load_sfx_ledger(ledger)
    except (OSError, ValueError) as exc:
        return [f"sfx ledger unreadable: {exc}"]
    deduped = {str(record.get("window_id", "")): record for record in records}
    ordered = sorted(deduped.values(), key=lambda record: float(record.get("start", 0.0)))
    cursor = 0.0
    covered_until = 0.0
    for record in ordered:
        try:
            stem = resolve_stored_path(run_dir, str(record.get("path", "")))
        except MediaError as exc:
            errors.append(f"sfx {record.get('window_id')} escapes the run dir: {exc}")
            stem = None
        if stem is None or not stem.exists():
            errors.append(f"sfx {record.get('window_id')} missing {record.get('path')}")
        start = float(record.get("start", -1.0))
        if abs(start - cursor) > SFX_WINDOW_OVERLAP + 0.01:
            errors.append(
                f"sfx coverage gap: window {record.get('window_id')} starts at "
                f"{start:.2f}s, expected ~{cursor:.2f}s"
            )
        covered_until = start + float(record.get("duration", 0.0))
        cursor = covered_until - SFX_WINDOW_OVERLAP
    if covered_until < timeline_seconds - AV_ALIGNMENT_TOLERANCE_SECONDS:
        errors.append(
            f"sfx coverage {covered_until:.2f}s short of timeline {timeline_seconds:.2f}s"
        )
    return errors


def _sfx_worker_module(backend: str) -> str:
    try:
        return SFX_WORKER_MODULES[backend]
    except KeyError:
        known = ", ".join(sorted(SFX_WORKER_MODULES))
        raise MediaError(f"unknown sfx backend {backend!r} (known: {known})") from None


def _stem_cache_hit(record: dict[str, Any], window: SfxWindow, model_size: str) -> bool:
    """Whether a ledger record satisfies a planned window (153, pure).

    Duration matches fuzzily within `AV_ALIGNMENT_TOLERANCE_SECONDS` (the
    same budget the bed/timeline checks use) so a re-finalize after float
    rounding drift still hits; caption/seed/model_size must match exactly
    so real plan changes always re-render. Malformed durations miss.
    """
    if record.get("caption") != window.caption:
        return False
    if record.get("seed") != window.seed:
        return False
    if record.get("model_size") != model_size:
        return False
    try:
        logged = float(record.get("duration", float("nan")))
    except (TypeError, ValueError):
        return False
    return abs(logged - window.duration) <= AV_ALIGNMENT_TOLERANCE_SECONDS


def _prune_stale_partials(sfx_dir: Path) -> int:
    """Remove crashed-render `*.partial.wav` leftovers (153, best-effort).

    Runs at plan time so a killed render never orphans bytes the next
    finalize mistakes for stems. Returns the pruned count. The temp name
    keeps the `.wav` suffix so every backend's ffmpeg/soundfile format
    inference keeps working on the temp path.
    """
    pruned = 0
    if sfx_dir.is_dir():
        for partial in sorted(sfx_dir.glob("*.partial.wav")):
            with contextlib.suppress(OSError):
                partial.unlink()
                pruned += 1
    return pruned


def render_sfx_bed(
    run_dir: Path,
    final_video: Path,
    timeline_seconds: float,
    bounds: list[tuple[float, float, str]],
    tmpdir: Path,
    backend: str,
    models_dir: str,
    device: str,
    model_size: str,
    seed_base: int,
    sample_rate: int,
    channels: int,
    num_workers: int = 1,
) -> Path:
    """Render every window (reusing ledger-matching stems) and join the bed.

    Windows shard round-robin over one worker, or two same-model workers
    on cuda:0/cuda:1 when num_workers=2 (small on both — see
    SFX_DUAL_MODEL_SIZE; two workers need two visible GPUs, gated below
    via `augment_devices`). Stems persist under audio/sfx/ with ledger
    entries; the bed joins stems with manual-fade `_blend_pair`s and
    verifies timeline-exactness before returning.
    """
    from voyage.rpc import SubprocessWorker

    if num_workers not in (1, 2):
        raise MediaError(f"sfx workers must be 1 or 2 (got {num_workers})")
    module = _sfx_worker_module(backend)
    if backend == "mmaudio" and device == "cuda:1" and model_size != "small_44k":
        raise MediaError(
            f"sfx model {model_size} cannot fit cuda:1 (6 GB) — the ladder measured "
            "medium OOM there; use --sfx-model-size small_44k or --sfx-device cuda:0"
        )
    if num_workers == 2:
        visible = augment_devices()
        if len(visible) < 2:
            seen = ", ".join(visible) if visible else "none"
            raise MediaError(
                f"--sfx-workers 2 needs 2 visible GPUs, saw {len(visible)} "
                f"({seen}); use --sfx-workers 1 on a single-GPU box"
            )
    sizes = [model_size] * num_workers
    devices = [device] * num_workers
    if num_workers == 2:
        sizes = [SFX_DUAL_MODEL_SIZE, SFX_DUAL_MODEL_SIZE]
        devices = ["cuda:0", "cuda:1"]
    windows = plan_sfx_windows(timeline_seconds, bounds, seed_base=seed_base)
    sfx_dir = run_dir / "audio" / SFX_STEMS_DIRNAME
    ledger = sfx_dir / SFX_LEDGER_NAME
    sfx_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_partials(sfx_dir)
    existing = {record["window_id"]: record for record in load_sfx_ledger(ledger)}
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    stems: list[Path] = []
    workers: list[Any] = []
    try:
        for slot in range(num_workers):
            worker = SubprocessWorker(
                module,
                run_dir,
                run_dir / "logs" / f"sfx-{slot}.log",
                init_op="init",
                init_payload={
                    "models_dir": models_dir,
                    "device": devices[slot],
                    "model_size": sizes[slot],
                },
            )
            worker.start()
            workers.append(worker)

        def _render_one(index: int, window: SfxWindow) -> tuple[Path, SfxWindow | None, str, str]:
            """Render one window; ledger append is deferred to the serial join (054).

            Returns `(stem, logged, stored, model_size)` where `logged` is the
            window to ledger-append or None on a cache hit. Stems land via
            atomic replace in the worker threads (distinct files, safe in
            parallel); the ledger itself is appended serially in plan order
            after the pool joins, so two workers can never interleave lines
            or race the order. No threading.Lock needed by construction.
            """
            stem = sfx_dir / f"{window.window_id}.wav"
            stored = f"audio/{SFX_STEMS_DIRNAME}/{window.window_id}.wav"
            slot = index % num_workers
            record = existing.get(window.window_id)
            if (
                record is not None
                and _stem_cache_hit(record, window, sizes[slot])
                and resolve_stored_path(run_dir, str(record.get("path", ""))).exists()
            ):
                return (resolve_stored_path(run_dir, str(record["path"])), None, "", "")
            # Render-to-temp + atomic replace (153): a failed render leaves
            # the old stem and ledger line intact — validate never sees a
            # half-written window, and the old stem stays the valid fallback.
            # The temp keeps the `.wav` suffix (format inference in workers).
            tmp_stem = sfx_dir / f"{window.window_id}.partial.wav"
            try:
                result = workers[slot].call(
                    "generate_sfx",
                    {
                        "window_id": window.window_id,
                        "caption": window.caption,
                        "video_path": str(final_video),
                        "start_seconds": window.start,
                        "duration_seconds": window.duration,
                        "seed": window.seed,
                        "output_path": str(tmp_stem),
                        "sample_rate": sample_rate,
                        "channels": channels,
                    },
                )
            except Exception:
                with contextlib.suppress(OSError):
                    tmp_stem.unlink()
                raise
            if not isinstance(result, dict):
                with contextlib.suppress(OSError):
                    tmp_stem.unlink()
                raise MediaError(f"sfx {window.window_id}: worker returned no result")
            os.replace(tmp_stem, stem)
            # Truncated tail (EOF edge): the ledger records reality so the
            # bed join and coverage math follow the stem, not the plan.
            resolved = result.get("duration_seconds", window.duration)
            logged = SfxWindow(
                window.window_id,
                window.start,
                float(resolved),
                window.caption,
                window.seed,
            )
            return (stem, logged, stored, sizes[slot])

        if num_workers == 1:
            pending = [_render_one(index, window) for index, window in enumerate(windows)]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                pending = list(pool.map(_render_one, range(len(windows)), windows))
        stems = []
        for stem, logged, stored, size in pending:
            if logged is not None:
                append_sfx_window(ledger, logged, stored, size)
            stems.append(stem)
    finally:
        for worker in workers:
            # Best-effort teardown: a stop failure must never mask the
            # render result (same discipline as the audio GPU swap).
            with contextlib.suppress(Exception):
                worker.stop()
    bed = tmpdir / "sfx_bed.wav"
    if len(stems) == 1:
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(stems[0]),
                "-c:a",
                "pcm_s16le",
                str(bed),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"sfx bed copy failed: {proc.stderr[-2000:]}")
        return bed
    accum = stems[0]
    for index in range(1, len(stems)):
        step = tmpdir / f"sfx_blend_{index:02d}.wav"
        _blend_pair(accum, stems[index], step, SFX_WINDOW_OVERLAP)
        accum = step
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(accum),
            "-c:a",
            "pcm_s16le",
            str(bed),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"sfx bed convert failed: {proc.stderr[-2000:]}")
    bed_seconds = _audio_duration_seconds(bed)
    if abs(bed_seconds - timeline_seconds) > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(f"sfx bed {bed_seconds:.2f}s drifts from timeline {timeline_seconds:.2f}s")
    return bed


def mix_music_and_sfx(
    music: Path, sfx_bed: Path, dest: Path, sample_rate: int, channels: int
) -> Path:
    """Lay the bed under the music (amix, no auto-gain) and verify length.

    amix with normalize=0 keeps both stems' levels (bed pre-scaled to
    -6 dB); duration=longest plus an explicit length check so a short
    bed can never silently truncate the music.
    """
    music_seconds = _audio_duration_seconds(music)
    filter_graph = (
        f"[1:a]volume={SFX_VOLUME}[bed];"
        "[0:a][bed]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[aout]"
    )
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(music),
            "-i",
            str(sfx_bed),
            "-filter_complex",
            filter_graph,
            "-map",
            "[aout]",
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
        raise MediaError(f"sfx mix failed: {proc.stderr[-2000:]}")
    mixed_seconds = _audio_duration_seconds(dest)
    if abs(mixed_seconds - music_seconds) > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(f"sfx mix {mixed_seconds:.2f}s drifts from music {music_seconds:.2f}s")
    return dest


def finalize_sfx_pass(
    run_dir: Path,
    final_path: Path,
    backend: str,
    models_dir: str,
    device: str,
    model_size: str,
    seed: int,
    sample_rate: int,
    channels: int,
    num_workers: int,
    fps: int,
    caption_override: str | None = None,
) -> Path:
    """Full post-pass: bed over the shipped pixels, mixed, remuxed in place.

    Demuxes the music from the published final, renders + joins the SFX
    bed conditioned on the final video, mixes, and muxes video-copy +
    mixed audio back over the same path (atomic replace — a failed pass
    never strands a half-written final). Returns the final path.
    """
    import tempfile

    info = probe(final_path)
    timeline = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    if timeline <= 0.0:
        raise MediaError(f"sfx pass: unprobable final duration for {final_path}")
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    usable = sorted(
        d for d in segments_root.iterdir() if d.is_dir() and (d / paths.DONE_MARKER).exists()
    )
    bounds = segment_sfx_bounds(run_dir, usable, fps, caption_override)
    with tempfile.TemporaryDirectory(prefix="voyage-sfx-final-") as tmp:
        tmpdir = Path(tmp)
        bed = render_sfx_bed(
            run_dir,
            final_path,
            timeline,
            bounds,
            tmpdir,
            backend,
            models_dir,
            device,
            model_size,
            seed,
            sample_rate,
            channels,
            num_workers,
        )
        music = tmpdir / "final_music.wav"
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(final_path),
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s16le",
                str(music),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"sfx pass music demux failed: {proc.stderr[-2000:]}")
        mixed = tmpdir / "final_mixed.wav"
        mix_music_and_sfx(music, bed, mixed, sample_rate, channels)
        remuxed = tmpdir / "final_sfx.mp4"
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(final_path),
                "-i",
                str(mixed),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(remuxed),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"sfx pass remux failed: {proc.stderr[-2000:]}")
        from voyage.atomic import atomic_write_bytes

        atomic_write_bytes(final_path, remuxed.read_bytes())
    return final_path

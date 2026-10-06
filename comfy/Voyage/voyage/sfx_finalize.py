"""Finalize-time SFX pass (slice 3, three-caption doctrine).

DESIGN §7, §56, §140 SFX-slice as-built.

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
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from voyage import paths
from voyage.atomic import fsync_dir
from voyage.augment import augment_devices
from voyage.console import optional_bar, optional_stage
from voyage.errors import MediaError
from voyage.media import (
    AV_ALIGNMENT_TOLERANCE_SECONDS,
    _audio_duration_seconds,
    _join_audio_single_graph,
    probe,
    run_capture,
)
from voyage.paths import resolve_stored_path
from voyage.segment_manifest import load_transition

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

SFX_WINDOW_SECONDS = 8.0
"""Native MMAudio window (upstream: 1.23 s per 8 s clip on small)."""

SFX_WINDOW_OVERLAP = 1.0
"""Overlap between consecutive windows, killed by manual fades on join."""

SFX_ORPHAN_ADOPT_TOLERANCE = 0.05
"""Max |probed - requested| seconds for adopting a ledger-less stem.

Stems orphaned by a killed batch (ledger used to append only after the
pool joined) are adopted when their probed duration matches the current
plan window this closely; anything further off re-renders instead."""

MIN_SFX_WINDOW_SECONDS = 1.0
"""Shortest renderable window: below ~0.64 s the sync branch yields fewer
than 16 frames and the synchformer segments an empty list (same crash
class the music path floors with pad-to-17 — here the planner merges a
stub tail into its predecessor instead, since counts must stay exact)."""

SFX_VOLUME = 0.5
"""Bed level under the music (-6 dB, Zoomy parity)."""

BOUNDS_RESCALE_IDENTITY_TOLERANCE = 1e-6
"""Timelines within this (seconds) count as identical — float noise from
probing must not rescale the SFX bounds (byte-identical legacy path)."""

SFX_REQUEST_MATCH_TOLERANCE = 1e-6
"""Request-identity match budget (seconds) for the stem cache hit-test.

The ledger `duration` is the *requested* plan duration, so a re-finalize
only hits when the plan asks for (almost) exactly what it asked before —
float noise from probing still hits, but a stale window from a shorter
timeline (tenths of a second off) always re-renders. Reality lives in
`probed_duration` (ffprobe of the stem file right after the atomic
replace); coverage math follows the stem, not the plan.
"""

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

SFX_MAX_WORKERS = 2
"""Only the 1- and 2-GPU shapes exist: 2 is the video-aug-cuda:0/SFX-cuda:1
pairing, and anything above fail-fasts without 2 visible GPUs (issue 158)."""

ATEMPO_MIN_FACTOR = 0.5
ATEMPO_MAX_FACTOR = 2.0
"""ffmpeg atempo accepts a single factor in [0.5, 2.0] — larger stretches
chain filters (1.5x slow-mo is one filter; 4x is two)."""

SFX_CONDITIONING_SHIPPED = "shipped"
SFX_CONDITIONING_PROXY = "proxy"
"""Conditioning-pixel identity for ledger records (parallel finalize).

The bed can condition on the shipped final pixels (`shipped`, the legacy
post-pass) or on a stream-copy concat of the committed segments (`proxy`,
the parallel path — same source timeline, no upscale/interp/morph).
Stems only cache-hit within one identity: a re-finalize on shipped pixels
must never reuse proxy-conditioned stems or the SFX would silently track
the wrong motion. Legacy lines predate the key and read as `shipped`."""


@dataclass
class SfxWindow:
    """One effects window: half-open [start, start+duration) of the final."""

    window_id: str
    start: float
    duration: float
    caption: str
    seed: int


class _RenderedWindow(NamedTuple):
    """One planned window's render outcome (serial-join handoff, 054).

    `logged` is the request window to ledger-append (None on a cache
    hit); `probed` is the stem file's probed duration — probed on both
    paths so the join below reuses it instead of re-spawning ffprobe
    (issue 152 probe budget). Named fields beat the old positional
    tuple now that the probe rides along.
    """

    stem: Path
    logged: SfxWindow | None
    stored: str
    model_size: str
    probed: float


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
    manifest transition audio.sfx_caption best-effort (missing/torn/legacy
    → "" — a history read must never break a finalize). A
    `caption_override` replaces every segment caption (old runs whose
    decisions predate SFX captions, or a deliberate single-caption dub).

    The duration fallback searches the *video* stream (issue 191: the old
    `streams[0]` read took whatever ffprobe listed first — routinely the
    audio stream on muxed segments — and divided its frame count by the
    video fps). A segment with no container duration and no video frame
    count raises MediaError like the music path's `_segment_timeline`
    instead of tiling a zero-length bound that shifts every later caption.
    """
    from voyage.models import EvolutionDecision

    if fps <= 0:
        raise MediaError(f"sfx bounds need positive fps (got {fps})")
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
                streams = probe(segment / "video.mp4").get("streams", [])
                video = next(
                    (
                        stream
                        for stream in streams
                        if isinstance(stream, dict) and stream.get("codec_type") == "video"
                    ),
                    None,
                )
                frames = int(video.get("nb_frames", 0) or 0) if video is not None else 0
                duration = frames / fps if frames > 0 else 0.0
            except (MediaError, ValueError, KeyError, TypeError, IndexError):
                duration = 0.0
        if duration <= 0.0:
            raise MediaError(
                f"segment {segment.name} has unprobable duration "
                "(no container duration, no video frame count)"
            )
        caption = ""
        try:
            raw = load_transition(segment)
            if raw:
                caption = EvolutionDecision.model_validate(raw).audio.sfx_caption
        except (OSError, ValueError):
            caption = ""
        if caption_override is not None:
            caption = caption_override
        bounds.append((cursor, cursor + duration, caption or ""))
        cursor += duration
    return bounds


def _scale_bounds_to_timeline(
    bounds: list[tuple[float, float, str]], timeline_seconds: float
) -> list[tuple[float, float, str]]:
    """Rescale source-timeline bounds onto a retimed shipped timeline (pure).

    The SFX bounds walk source segments (unchanged by any video retime),
    while the finalize SFX pass probes the shipped video duration as its
    timeline. Identical timelines (within float noise) return the bounds
    untouched — byte-identical to today's path; a stretched slow-mo
    timeline scales every bound uniformly so captions stay aligned with
    the retimed video.
    """
    if not bounds:
        raise MediaError("cannot scale empty sfx bounds onto a timeline")
    source_end = bounds[-1][1]
    if source_end <= 0.0 or timeline_seconds <= 0.0:
        raise MediaError(
            f"sfx bounds scaling needs positive durations (got {source_end}/{timeline_seconds})"
        )
    if abs(timeline_seconds - source_end) <= BOUNDS_RESCALE_IDENTITY_TOLERANCE:
        return list(bounds)
    factor = timeline_seconds / source_end
    return [(start * factor, end * factor, caption) for start, end, caption in bounds]


def append_sfx_window(
    ledger: Path,
    window: SfxWindow,
    path: str,
    model_size: str,
    probed_duration: float | None = None,
    conditioning_source: str = SFX_CONDITIONING_SHIPPED,
    conditioning_timeline: float | None = None,
) -> None:
    """Durably append one rendered window (flush + fsync + fsync_dir, takes pattern).

    `window.duration` is the requested plan duration (cache-hit identity);
    `probed_duration` is the stem file's probed duration (reality — the
    bed join and coverage math follow the stem, not the plan). Legacy
    lines without `probed_duration` read back as requested duration,
    which is exact for them: pre-split ledgers always recorded the plan.

    `conditioning_source` pins which pixels the stem conditioned on
    (`shipped` vs `proxy`); `conditioning_timeline` is the timeline the
    bounds tiled (source duration for proxy beds, shipped duration for
    the legacy pass). Legacy lines carry neither and validate against
    the passed timeline exactly as before.

    File fsync persists content; the directory sync persists the namespace
    entry (issue 101 twin of `append_take`, in-tree contract in
    `voyage/atomic.py`). Appends are lock-guarded per window from the
    render path, so a kill/cancel/failure can never orphan a completed
    stem: the line lands as soon as its window completes. Ledger order is
    completion order and irrelevant — resume reads key by window_id and
    validate dedupes last-wins (issue 054).
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
        "conditioning_source": conditioning_source,
    }
    if probed_duration is not None:
        record["probed_duration"] = probed_duration
    if conditioning_timeline is not None:
        record["conditioning_timeline"] = conditioning_timeline
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(ledger.parent)


def load_sfx_ledger(ledger: Path) -> list[dict[str, Any]]:
    """Read the persisted SFX ledger (missing file → empty).

    Torn trailing lines (a crash mid-append between `write` and the
    newline+fsync) are skipped, not fatal — mirroring the sidecar
    `load_chunk_ledger` contract: the interrupted window simply has no
    record and is re-rendered on the next pass.
    """
    if not ledger.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _record_covered_duration(record: dict[str, Any]) -> float:
    """Stem reality for coverage math: probed duration first, request fallback.

    Lines written with a stem probe carry `probed_duration` (what the bed
    join actually consumed); legacy lines predate the probe and recorded
    the request, so the fallback is exact for them.
    """
    if record.get("probed_duration") is not None:
        return float(record["probed_duration"])
    return float(record.get("duration", 0.0))


def validate_sfx_ledger(run_dir: Path, timeline_seconds: float) -> list[str]:
    """Read-only SFX checks: files exist, windows tile the timeline.

    No ledger (SFX never ran / old run) is clean — the pass is optional.
    Duplicate `window_id` lines (re-render appends) dedupe last-wins and
    the walk sorts by start, so ledger order can never false-positive
    coverage (153, mirroring the `existing` dict in `render_sfx_bed`).
    Records group by conditioning source only (proxy vs shipped; legacy
    lines read as shipped): `conditioning_timeline` is provenance, not
    identity — `_stem_cache_hit` deliberately ignores it, so a timeline
    extension reuses head stems (append-only proxy: head pixels unchanged)
    and appends only the new tail. Splitting the walk per timeline would
    then demand the tail group tile from zero, which it never can (kaolin:
    fatal `w0117 starts at 819.00s, expected ~0.00s` on a complete bed).
    The union per source must tile from zero (gaps stay fatal) and reach
    the passed source timeline (shortfall stays healable via
    `is_healable_sfx_shortfall`). Production always passes the source
    timeline, which every conditioning timeline meets or precedes.
    """
    ledger = run_dir / "audio" / SFX_STEMS_DIRNAME / SFX_LEDGER_NAME
    if not ledger.exists():
        return []
    errors: list[str] = []
    try:
        records = load_sfx_ledger(ledger)
    except (OSError, ValueError) as exc:
        return [f"sfx ledger unreadable: {exc}"]
    deduped = {}
    for record in records:
        key = (
            str(record.get("conditioning_source", SFX_CONDITIONING_SHIPPED)),
            str(record.get("window_id", "")),
        )
        deduped[key] = record
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (_source, _window_id), record in deduped.items():
        grouped.setdefault(_source, []).append(record)
    for _source, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda record: float(record.get("start", 0.0)))
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
            covered_until = start + _record_covered_duration(record)
            cursor = covered_until - SFX_WINDOW_OVERLAP
        if covered_until < timeline_seconds - AV_ALIGNMENT_TOLERANCE_SECONDS:
            errors.append(
                f"sfx coverage {covered_until:.2f}s short of timeline {timeline_seconds:.2f}s"
            )
    return errors


#: Matches exactly the pure-shortfall line above (two `%.2f` seconds).
#: Generate's pre-finalize gate filters this line — and only this line —
#: because the finalize SFX pass heals it (`render_sfx_bed` cache-hits old
#: windows and renders the new ones; render failures raise). Gaps, missing
#: stems, unreadable ledgers, and escapes stay fatal: nothing heals those.
_SFX_SHORTFALL_RE = re.compile(r"^sfx coverage \d+\.\d{2}s short of timeline \d+\.\d{2}s$")


def is_healable_sfx_shortfall(error: str) -> bool:
    """Whether an sfx error line is the pure tail shortfall finalize heals."""
    return bool(_SFX_SHORTFALL_RE.match(error))


def _sfx_worker_module(backend: str) -> str:
    try:
        return SFX_WORKER_MODULES[backend]
    except KeyError:
        known = ", ".join(sorted(SFX_WORKER_MODULES))
        raise MediaError(f"unknown sfx backend {backend!r} (known: {known})") from None


def _stem_cache_hit(
    record: dict[str, Any],
    window: SfxWindow,
    model_size: str,
    conditioning_source: str = SFX_CONDITIONING_SHIPPED,
) -> bool:
    """Whether a ledger record satisfies a planned window (153, pure).

    The ledger `duration` is the requested plan duration, so the hit-test
    is request identity within float noise (`SFX_REQUEST_MATCH_TOLERANCE`):
    a re-finalize after probe rounding still hits, while a stale window
    from a shorter timeline (tenths of a second off) always re-renders.
    Caption/seed/model_size must match exactly so real plan changes always
    re-render. The conditioning source must match too: shipped-pixel stems
    and proxy-pixel stems are never interchangeable (parallel finalize).
    Legacy lines predate the key and read as `shipped`. Malformed
    durations miss. Stem reality (`probed_duration`) never participates:
    an honest encode shortfall (hundredths of a second) must not force a
    re-render every finalize.
    """
    if record.get("conditioning_source", SFX_CONDITIONING_SHIPPED) != conditioning_source:
        return False
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
    return abs(logged - window.duration) <= SFX_REQUEST_MATCH_TOLERANCE


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
    *,
    blend_timings: list[float] | None = None,
    progress: VoyageConsole | None = None,
    conditioning_source: str = SFX_CONDITIONING_SHIPPED,
    conditioning_timeline: float | None = None,
) -> Path:
    """Render every window (reusing ledger-matching stems) and join the bed.

    Windows shard round-robin over one worker, or two same-model workers
    on cuda:0/cuda:1 when num_workers=2 (small on both — see
    SFX_DUAL_MODEL_SIZE; two workers need two visible GPUs, gated below
    via `augment_devices`). Stems persist under audio/sfx/ with ledger
    entries; the bed joins stems in one single-graph spawn (issue 152:
    chained pairwise stages with s32 barriers, byte-identical to the old
    fold, each stem decoded once) and verifies timeline-exactness before
    returning. Each stem is probed once and threaded through the join;
    `blend_timings` collects the single join wall-milliseconds for soak
    trending. `conditioning_source` pins the pixel identity the stems
    condition on (and ledger under); the timeline the bounds tile rides
    in `conditioning_timeline` (defaults to `timeline_seconds`).
    """
    from voyage.logrotate import append_line
    from voyage.rpc import SubprocessWorker

    started = time.perf_counter()

    if num_workers not in (1, SFX_MAX_WORKERS):
        raise MediaError(f"sfx workers must be 1 or 2 (got {num_workers})")
    module = _sfx_worker_module(backend)
    if backend == "mmaudio" and device == "cuda:1" and model_size != "small_44k":
        raise MediaError(
            f"sfx model {model_size} cannot fit cuda:1 (6 GB) — the ladder measured "
            "medium OOM there; use --sfx-model-size small_44k or --sfx-device cuda:0"
        )
    if num_workers == SFX_MAX_WORKERS:
        visible = augment_devices()
        if len(visible) < SFX_MAX_WORKERS:
            seen = ", ".join(visible) if visible else "none"
            raise MediaError(
                f"--sfx-workers 2 needs 2 visible GPUs, saw {len(visible)} "
                f"({seen}); use --sfx-workers 1 on a single-GPU box"
            )
    sizes = [model_size] * num_workers
    devices = [device] * num_workers
    if num_workers == SFX_MAX_WORKERS:
        sizes = [SFX_DUAL_MODEL_SIZE, SFX_DUAL_MODEL_SIZE]
        devices = ["cuda:0", "cuda:1"]
    windows = plan_sfx_windows(timeline_seconds, bounds, seed_base=seed_base)
    sfx_dir = run_dir / "audio" / SFX_STEMS_DIRNAME
    ledger = sfx_dir / SFX_LEDGER_NAME
    sfx_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_partials(sfx_dir)
    existing = {record["window_id"]: record for record in load_sfx_ledger(ledger)}
    ledger_timeline = (
        conditioning_timeline if conditioning_timeline is not None else timeline_seconds
    )
    ledger_lock = threading.Lock()
    # Orphan-stem adoption (SFX resume): batches killed before the old
    # serial-join append left completed stems with no ledger lines. Adopt
    # stem files that match the current plan identity — window in plan, no
    # ledger line yet, file present with a probed duration within tolerance
    # of the request — and ledger them under the current conditioning
    # source, so a rerun heals instead of re-rendering. Each adoption is
    # logged loudly; anything not matching re-renders through the normal
    # path below.
    for adopt_index, adopt_window in enumerate(windows):
        if adopt_window.window_id in existing:
            continue
        orphan = sfx_dir / f"{adopt_window.window_id}.wav"
        if not orphan.exists():
            continue
        try:
            probed_duration = _audio_duration_seconds(orphan)
        except Exception:  # noqa: BLE001 - unreadable orphan stem re-renders below; adoption must never fail finalize
            continue
        if abs(probed_duration - adopt_window.duration) > SFX_ORPHAN_ADOPT_TOLERANCE:
            continue
        adopt_size = sizes[adopt_index % num_workers]
        adopt_stored = f"audio/{SFX_STEMS_DIRNAME}/{adopt_window.window_id}.wav"
        append_sfx_window(
            ledger,
            adopt_window,
            adopt_stored,
            adopt_size,
            probed_duration=probed_duration,
            conditioning_source=conditioning_source,
            conditioning_timeline=ledger_timeline,
        )
        message = (
            f"sfx orphan adopted: {adopt_window.window_id} "
            f"(no ledger line, stem {probed_duration:.3f}s ~= request "
            f"{adopt_window.duration:.3f}s)"
        )
        if progress is not None:
            progress.warn(message)
        else:
            print(message, file=sys.stderr)
        existing[adopt_window.window_id] = {
            "window_id": adopt_window.window_id,
            "start": adopt_window.start,
            "duration": adopt_window.duration,
            "caption": adopt_window.caption,
            "seed": adopt_window.seed,
            "path": adopt_stored,
            "model_size": adopt_size,
            "conditioning_source": conditioning_source,
            "probed_duration": probed_duration,
            "conditioning_timeline": ledger_timeline,
        }
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    stems: list[Path] = []
    workers: list[Any] = []
    try:
        with optional_stage(progress, "load sfx workers", f"{num_workers} worker(s)"):
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
                        "scratch_dir": str(paths.ensure_scratch_dir(run_dir)),
                    },
                    # MMAudio venv (DESIGN §140 SFX continuity): the MMAudio
                    # stack is isolated from the LTX freeze; unset (video
                    # image, tests) falls back to the supervisor interpreter.
                    executable=os.environ.get("VOYAGE_SFX_PYTHON"),
                )
                worker.start()
                workers.append(worker)

        def _render_one(index: int, window: SfxWindow) -> _RenderedWindow:
            """Render one window; the ledger line lands as it completes.

            Returns a `_RenderedWindow` whose `logged` is always None (the
            line is already durable by return time) and whose `probed` is
            the stem file's probed duration (probed on both paths so the
            join reuses it). Stems land via atomic replace in the worker
            threads (distinct files, safe in parallel); the ledger append
            is lock-guarded, and order is irrelevant — resume reads key by
            window_id and validate dedupes last-wins. A kill/cancel/failure
            later in the batch can never orphan this window.
            """
            stem = sfx_dir / f"{window.window_id}.wav"
            stored = f"audio/{SFX_STEMS_DIRNAME}/{window.window_id}.wav"
            slot = index % num_workers
            record = existing.get(window.window_id)
            if (
                record is not None
                and _stem_cache_hit(record, window, sizes[slot], conditioning_source)
                and resolve_stored_path(run_dir, str(record.get("path", ""))).exists()
            ):
                hit = resolve_stored_path(run_dir, str(record["path"]))
                return _RenderedWindow(hit, None, "", "", _audio_duration_seconds(hit))
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
            # Truncated tail (EOF edge): the ledger records the request for
            # identity plus the stem file's probed duration for reality, so
            # the bed join and coverage math follow the stem, not the plan.
            # The probe (not the worker's nested `sfx.duration_seconds`) is
            # authoritative: backends echo the request or report yielded
            # frames, while the file is what the join consumes.
            probed = _audio_duration_seconds(stem)
            # Durable per-window append (SFX resume): the line lands under a
            # lock as soon as this window completes, so a kill, a cancel, or
            # a sibling window's failure can never orphan it.
            with ledger_lock:
                append_sfx_window(
                    ledger,
                    window,
                    stored,
                    sizes[slot],
                    probed_duration=probed,
                    conditioning_source=conditioning_source,
                    conditioning_timeline=ledger_timeline,
                )
            return _RenderedWindow(stem, None, "", "", probed)

        with optional_bar(progress, "sfx windows", total=len(windows)) as tracker:
            if num_workers == 1:
                pending = []
                for index, window in enumerate(windows):
                    pending.append(_render_one(index, window))
                    if tracker is not None:
                        tracker.update()
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    pending = []
                    for row in pool.map(_render_one, range(len(windows)), windows):
                        pending.append(row)
                        if tracker is not None:
                            tracker.update()
        stems = []
        for row in pending:
            stems.append(row.stem)
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
        _emit_sfx_pass_completed(
            run_dir, windows, backend, device, model_size, started, append_line
        )
        return bed
    # Stem durations ride along from `_render_one` (probed on both the
    # render and cache-hit paths) so the join never re-spawns ffprobe
    # per stem (issue 152 probe budget).
    stem_seconds = [row.probed for row in pending]
    joined = tmpdir / "sfx_joined.wav"
    _join_audio_single_graph(
        stems,
        joined,
        SFX_WINDOW_OVERLAP,
        durations=stem_seconds,
        timing_ms=blend_timings,
    )
    accum = joined
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
    _emit_sfx_pass_completed(run_dir, windows, backend, device, model_size, started, append_line)
    return bed


def _emit_sfx_pass_completed(
    run_dir: Path,
    windows: list[SfxWindow],
    backend: str,
    device: str,
    model_size: str,
    started: float,
    append_line: Any,
) -> None:
    """Emit `sfx_pass_completed` (bed windows + wall seconds, DESIGN §140).

    Single home so both the sequential `finalize_sfx_pass` and the
    parallel-finalize Thread B (both funnel through `render_sfx_bed`)
    report the SFX leg like `finalize_completed` reports the model pass.
    """
    append_line(
        run_dir / paths.LOGS_DIRNAME / "metrics.jsonl",
        json.dumps(
            {
                "ts": time.time(),
                "event": "sfx_pass_completed",
                "windows": len(windows),
                "backend": backend,
                "device": device,
                "model_size": model_size,
                "sfx_pass_s": round(time.perf_counter() - started, 3),
            }
        ),
    )


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


def demux_music_audio(source_video: Path, dest_wav: Path) -> Path:
    """Demux the first audio stream of `source_video` to pcm_s16le WAV.

    Single home for the SFX dub's music read (the sequential pass and
    the parallel finalize path demux identically, so their mixes stay
    byte-identical).
    """
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source_video),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            str(dest_wav),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"sfx pass music demux failed: {proc.stderr[-2000:]}")
    return dest_wav


def remux_video_with_audio(source_video: Path, mixed_audio: Path, dest_mp4: Path) -> Path:
    """Mux `source_video` (stream copy) + `mixed_audio` (AAC 256k) to `dest_mp4`.

    Single home for the SFX dub's publish encode (sequential and parallel
    paths remux identically — video is never re-encoded here).
    """
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source_video),
            "-i",
            str(mixed_audio),
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
            str(dest_mp4),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"sfx pass remux failed: {proc.stderr[-2000:]}")
    return dest_mp4


def build_proxy_reference(run_dir: Path, usable: list[Path], tmpdir: Path) -> tuple[Path, float]:
    """Concat the committed segments into a source-timeline proxy (no re-encode).

    The parallel finalize path conditions the SFX bed on this reference
    instead of the shipped final: stream-copy concat keeps the exact source
    duration 1:1, so the worker's (start, duration) contract and the
    unscaled source bounds stay exact with no seeking math. Documented
    approximations vs shipped pixels: hard cuts instead of morph joints,
    duplicated frames instead of FILM-smooth slow-mo, no upscale (all
    near-irrelevant at the worker's 384px conditioning resample).
    Returns the proxy path and its probed source duration.
    """
    from voyage.media import write_concat_list

    if not usable:
        raise MediaError("sfx proxy: no usable segments to reference")
    sources = [segment / "video.mp4" for segment in usable]
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise MediaError(f"sfx proxy: missing segment videos: {', '.join(missing)}")
    concat_list = write_concat_list(sources, tmpdir / "sfx_proxy_concat.txt")
    proxy = tmpdir / "sfx_proxy_ref.mp4"
    proc = run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "copy",
            str(proxy),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"sfx proxy concat failed: {proc.stderr[-2000:]}")
    source_seconds = float(probe(proxy).get("format", {}).get("duration", 0.0) or 0.0)
    if source_seconds <= 0.0:
        raise MediaError(f"sfx proxy: unprobable reference duration for {proxy}")
    return proxy, source_seconds


def atempo_chain_for_stretch(stretch: float) -> str | None:
    """Build an atempo filter chain for a uniform retime factor (pure).

    Returns None when `stretch` is identity within float noise (the dub
    then skips the stretch leg byte-identically). Otherwise decomposes
    the factor into atempo filters each inside the [0.5, 2.0] accepted
    range (halving/doubling the remainder until it fits). Non-finite or
    non-positive factors fail loud — a bogus stretch must never
    silently ship unstretched audio.
    """
    if not math.isfinite(stretch) or stretch <= 0.0:
        raise MediaError(f"sfx stretch needs a positive factor (got {stretch})")
    if abs(stretch - 1.0) <= BOUNDS_RESCALE_IDENTITY_TOLERANCE:
        return None
    factors: list[float] = []
    remaining = stretch
    while remaining > ATEMPO_MAX_FACTOR:
        factors.append(ATEMPO_MAX_FACTOR)
        remaining /= ATEMPO_MAX_FACTOR
    while remaining < ATEMPO_MIN_FACTOR:
        factors.append(ATEMPO_MIN_FACTOR)
        remaining /= ATEMPO_MIN_FACTOR
    factors.append(remaining)
    parts = []
    for factor in factors:
        quantized = round(factor, 6)
        if not ATEMPO_MIN_FACTOR <= quantized <= ATEMPO_MAX_FACTOR:
            raise MediaError(f"sfx stretch {stretch} leaves atempo out of range ({quantized})")
        parts.append(f"atempo={quantized}")
    return ",".join(parts)


def stretch_and_dub_sfx_bed(
    staged_video: Path,
    proxy_bed: Path,
    music_wav: Path,
    dest_mp4: Path,
    sample_rate: int,
    channels: int,
    stretch: float,
) -> Path:
    """Dub a proxy-conditioned bed onto the staged final (parallel path).

    The bed tiles the source timeline; `stretch` (probed shipped duration
    over probed source duration) retimes it once via an atempo chain, then
    the dub reuses the `mix_music_and_sfx` + `remux_video_with_audio`
    single-homes — same mix levels and publish encode as the sequential
    pass. The dubbed file is verified against the staged video within
    `AV_ALIGNMENT_TOLERANCE_SECONDS` (not frame-exact: the remux `-shortest`
    trims to the audio, same semantics as the sequential dub).
    """
    chain = atempo_chain_for_stretch(stretch)
    work_bed = proxy_bed
    if chain is not None:
        stretched = dest_mp4.parent / "sfx_bed_stretched.wav"
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(proxy_bed),
                "-af",
                chain,
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-c:a",
                "pcm_s16le",
                str(stretched),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"sfx bed stretch failed: {proc.stderr[-2000:]}")
        work_bed = stretched
    mixed = dest_mp4.parent / "sfx_dub_mixed.wav"
    mix_music_and_sfx(music_wav, work_bed, mixed, sample_rate, channels)
    remux_video_with_audio(staged_video, mixed, dest_mp4)
    staged_seconds = float(probe(staged_video).get("format", {}).get("duration", 0.0) or 0.0)
    dubbed_seconds = float(probe(dest_mp4).get("format", {}).get("duration", 0.0) or 0.0)
    if abs(dubbed_seconds - staged_seconds) > AV_ALIGNMENT_TOLERANCE_SECONDS:
        raise MediaError(f"sfx dub {dubbed_seconds:.2f}s drifts from staged {staged_seconds:.2f}s")
    return dest_mp4


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
    progress: VoyageConsole | None = None,
) -> Path:
    """Full post-pass: bed over the shipped pixels, mixed, remuxed in place.

    Demuxes the music from the published final, renders + joins the SFX
    bed conditioned on the final video, mixes, and muxes video-copy +
    mixed audio back over the same path (atomic replace — a failed pass
    never strands a half-written final). Returns the final path.

    The `sfx_pass_completed` timing event fires inside `render_sfx_bed`
    (shared with the parallel-finalize path).
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
    # Slow-mo retime (DESIGN §140): the bounds walk the source timeline
    # but the shipped video may be stretched — rescale onto the probed
    # shipped duration so captions stay aligned. Identical timelines are
    # returned untouched (byte-identical legacy path).
    bounds = _scale_bounds_to_timeline(bounds, timeline)
    with tempfile.TemporaryDirectory(
        prefix="voyage-sfx-final-", dir=paths.ensure_scratch_dir(run_dir)
    ) as tmp:
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
            progress=progress,
        )
        with optional_stage(progress, "dub sfx onto final"):
            music = tmpdir / "final_music.wav"
            demux_music_audio(final_path, music)
            mixed = tmpdir / "final_mixed.wav"
            mix_music_and_sfx(music, bed, mixed, sample_rate, channels)
            remuxed = tmpdir / "final_sfx.mp4"
            remux_video_with_audio(final_path, mixed, remuxed)
            from voyage.atomic import atomic_write_bytes

            atomic_write_bytes(final_path, remuxed.read_bytes())
    return final_path

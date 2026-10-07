"""Durable finalize model pass over sidecar plan dirs (DESIGN §§56-57, §140).

Finalize-side orchestration for the independent augment workers: poll the
upscale worker (2060 ESRGAN) and the interp worker (4060 FILM, picking up
the ledgered upscale chunks) to completion, drain one intermediate per
usable segment, and concat them in segment order. Replaces the all-or-nothing `TemporaryDirectory`
`run_finalize_model_pass` flow — every expensive chunk persists under
`run_dir/augment/<plan-hash>/` with a ledger record, so a crashed finalize
retries only the missing chunks.

Stdlib only at module scope (supervisor §12 GPU ban): the pollers, drain,
and ffmpeg enter through injected seams or function-local lazy imports.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voyage.augment_joints import ensure_joint_units
from voyage.augment_morph import MORPH_BRIDGE
from voyage.augment_sidecar import plan_dir_for_segment
from voyage.console import optional_bar, optional_stage
from voyage.errors import MediaError
from voyage.hashing import sha256_file, sha256_text

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

FINAL_INTERMEDIATE_FILENAME = "model_intermediate.mp4"
"""Concat output in the caller's work dir (cheap stream copy, redone per finalize)."""

_MAX_POLL_PASSES = 10
"""Poll-loop cap: each pass must finish chunks or the run is stuck (fail-loud)."""

_JOINT_BAR_LEG_COUNT = 2
"""Joint-bar legs: upscale + interp. The fix stage renders inside
`_refresh_joint_units` without bar callbacks, so only the two uniform
legs advance the combined "joint frames" bar; its total is this count
times `MORPH_BRIDGE` source frames per joint."""


_WEIGHTS_KEY_CACHE: dict[tuple[str, str, int, int, int, int], str] = {}
"""Ledger weights keys keyed by (interp, realesrgan, interp_mtime_ns,
interp_size, realesrgan_mtime_ns, realesrgan_size).

Issue 250: `weights_key_for` re-read ~90 MB of weights on every call
(once per finalize plus once per commit via the background plan).
Weights never change mid-run; the identity keeps the memo honest
across swaps. Per-process only; stat failures compute uncached.
"""


def _weights_identity(path: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) for a weights file, None when unstated."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def weights_key_for(weights: Any, interp_backend: str = "rife") -> str:
    """Ledger key covering both model legs, legacy `sha|sha` shape.

    The key intentionally carries NO backend prefix: the interp-leg sha
    differs between backends (FILM and RIFE weights are different files),
    so a backend switch already misses old records by construction --
    while pre-backend runs (e.g. kaolin's FILM ledgers) keep hitting
    with zero re-render. Final freshness across a backend switch rides
    the skip-key backend component instead. Results memoize per file
    identity (issue 250) — repeat calls with unchanged weights hash
    nothing.
    """
    from voyage.workers import augment_worker

    backend = augment_worker.validate_interp_backend(interp_backend)
    # Legacy fakes carry the FILM leg as `.film` (no `interp_leg` method).
    interp = weights.interp_leg(backend) if hasattr(weights, "interp_leg") else weights.film
    realesrgan = weights.realesrgan
    if interp is None or realesrgan is None:
        raise ValueError(
            "durable model pass needs the interp leg and the upscale leg provisioned "
            f"(backend={backend}, interp={interp!r}, realesrgan={realesrgan!r})"
        )
    interp_path = Path(interp) if not isinstance(interp, Path) else interp
    realesrgan_path = Path(realesrgan) if not isinstance(realesrgan, Path) else realesrgan
    interp_identity = _weights_identity(interp_path)
    realesrgan_identity = _weights_identity(realesrgan_path)
    if interp_identity is not None and realesrgan_identity is not None:
        memo_key = (
            str(interp_path),
            str(realesrgan_path),
            interp_identity[0],
            interp_identity[1],
            realesrgan_identity[0],
            realesrgan_identity[1],
        )
        cached = _WEIGHTS_KEY_CACHE.get(memo_key)
        if cached is not None:
            return cached
        key = f"{sha256_file(interp)}|{sha256_file(realesrgan)}"
        _WEIGHTS_KEY_CACHE[memo_key] = key
        return key
    return f"{sha256_file(interp)}|{sha256_file(realesrgan)}"


def _interp_weights_path(weights: Any, interp_backend: str) -> Path | None:
    """Active interp-leg weights path for `interp_backend`.

    Delegates to the typed `augment.interp_leg_path` (backend validated
    inside; legacy fakes fall back to `.film`).
    """
    from voyage.augment import interp_leg_path

    return interp_leg_path(weights, interp_backend)


_AUGMENT_PLAN_FINGERPRINT_VERSION = "augment-plan-v1"
"""Fingerprint namespace: bump when the hashed field set changes (old markers miss)."""

_COVERAGE_INTERP_BACKENDS = frozenset({"film", "rife"})
"""Backends the coverage marker accepts (mirrors the worker vocabulary)."""

_AUGMENT_COVERAGE_REQUIRED_KEYS = frozenset(
    {
        "fingerprint",
        "segments",
        "joints",
        "jointed_timeline_sha",
        "interp_backend",
        "weights_key",
    }
)
"""Exact marker keys: missing or extra keys fail validation (typo-proof)."""


def _require_key_list(name: str, values: Sequence[str]) -> list[str]:
    """Validated copy of an ordered key list (non-empty strings, order kept)."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings (got {type(values).__name__})")
    cleaned: list[str] = []
    for entry in values:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{name} entries must be non-empty strings (got {entry!r})")
        cleaned.append(entry)
    return cleaned


def _require_frame_counts(name: str, values: Sequence[int]) -> list[int]:
    """Validated copy of per-segment frame counts (positive ints, order kept)."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of ints (got {type(values).__name__})")
    cleaned: list[int] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 1:
            raise ValueError(f"{name} entries must be ints >= 1 (got {entry!r})")
        cleaned.append(entry)
    return cleaned


def augment_plan_fingerprint(
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    multiplier: int,
    crf: int,
    preset: str,
    interp_backend: str,
    segment_source_keys: Sequence[str],
    joint_fix_keys: Sequence[str],
) -> str:
    """Deterministic plan fingerprint over every input that forks plan dirs (DESIGN §§56-57, §140).

    Covers `weights_key`, output geometry (`out_width`/`out_height`/`out_fps`
    as the integer source-fps key), the upscale factor, the interp
    `multiplier`, the chunk recipe (`crf`/`preset`), the `interp_backend`,
    and the ordered segment `source_key` list plus the ordered joint fix-key
    list. Order matters (presentation order): reordering segments or joints
    forks the fingerprint and forces re-evaluation. Pure function, no I/O:
    callers pass already-computed keys in (fix keys already derive from
    video shas upstream) — this function never hashes video files.

    `chunk_frames` is intentionally absent: plan dirs do not hash it (the
    `ChunkKey` inside does), so a chunking change still misses via the
    ledger exact-match in `augment_work_complete`, not via this hash.
    """
    from voyage.augment import CRF_MAXIMUM, CRF_MINIMUM

    weights_key = _require_text("weights_key", weights_key)
    out_width = _require_box("out_width", out_width)
    out_height = _require_box("out_height", out_height)
    out_fps = _require_box("out_fps", out_fps)
    upscale_factor = _require_factor(upscale_factor)
    multiplier = _require_multiplier(multiplier)
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise TypeError(f"crf must be an int (got {type(crf).__name__})")
    if not CRF_MINIMUM <= crf <= CRF_MAXIMUM:
        raise ValueError(f"crf must be in [{CRF_MINIMUM}, {CRF_MAXIMUM}] (got {crf!r})")
    preset = _require_text("preset", preset)
    interp_backend = _require_text("interp_backend", interp_backend)
    if interp_backend not in _COVERAGE_INTERP_BACKENDS:
        raise ValueError(f"interp_backend must be one of {sorted(_COVERAGE_INTERP_BACKENDS)}")
    ordered_segments = _require_key_list("segment_source_keys", segment_source_keys)
    ordered_fixes = _require_key_list("joint_fix_keys", joint_fix_keys)
    payload = {
        "crf": crf,
        "interp_backend": interp_backend,
        "joint_fix_keys": ordered_fixes,
        "multiplier": multiplier,
        "out_fps": out_fps,
        "out_height": out_height,
        "out_width": out_width,
        "preset": preset,
        "segment_source_keys": ordered_segments,
        "upscale_factor": upscale_factor,
        "version": _AUGMENT_PLAN_FINGERPRINT_VERSION,
        "weights_key": weights_key,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def stamp_augment_coverage(marker: dict[str, Any]) -> dict[str, Any]:
    """Validate + stamp an augment coverage marker (DESIGN §56, §140).

    Schema: `{"fingerprint": str, "segments": int >= 1, "joints": int >= 0,
    "jointed_timeline_sha": str, "interp_backend": "film" | "rife",
    "weights_key": str}`. Exact keys only (missing or extra fails), fresh
    copy returned (never the caller's dict). The drain records
    `jointed_timeline_sha` via `hashing.sha256_file(final)`; persistence
    wiring (reading/writing the marker beside the run state) belongs to the
    orchestrator — this module only constructs and validates.
    """
    if not isinstance(marker, dict):
        raise TypeError(f"marker must be a dict (got {type(marker).__name__})")
    if set(marker.keys()) != set(_AUGMENT_COVERAGE_REQUIRED_KEYS):
        raise ValueError(
            f"marker keys must be exactly {sorted(_AUGMENT_COVERAGE_REQUIRED_KEYS)} "
            f"(got {sorted(marker.keys())})"
        )
    fingerprint = marker["fingerprint"]
    segments = marker["segments"]
    joints = marker["joints"]
    timeline_sha = marker["jointed_timeline_sha"]
    interp_backend = marker["interp_backend"]
    weights_key = marker["weights_key"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"fingerprint must be a non-empty string (got {fingerprint!r})")
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
        raise ValueError(f"segments must be an int >= 1 (got {segments!r})")
    if isinstance(joints, bool) or not isinstance(joints, int) or joints < 0:
        raise ValueError(f"joints must be an int >= 0 (got {joints!r})")
    if not isinstance(timeline_sha, str) or not timeline_sha:
        raise ValueError(f"jointed_timeline_sha must be a non-empty string (got {timeline_sha!r})")
    if not isinstance(interp_backend, str) or interp_backend not in _COVERAGE_INTERP_BACKENDS:
        raise ValueError(
            f"interp_backend must be one of {sorted(_COVERAGE_INTERP_BACKENDS)} "
            f"(got {interp_backend!r})"
        )
    if not isinstance(weights_key, str) or not weights_key:
        raise ValueError(f"weights_key must be a non-empty string (got {weights_key!r})")
    return {
        "fingerprint": fingerprint,
        "segments": segments,
        "joints": joints,
        "jointed_timeline_sha": timeline_sha,
        "interp_backend": interp_backend,
        "weights_key": weights_key,
    }


def augment_work_complete(
    run_dir: Path,
    fingerprint: str,
    expected_segments: int,
    expected_joints: int,
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int,
    crf: int,
    preset: str,
    interp_backend: str,
    segment_source_keys: Sequence[str],
    segment_frame_counts: Sequence[int],
    joint_fix_keys: Sequence[str],
) -> bool:
    """Cheap ledger-truth completeness check for joints + model-pass work (DESIGN §§56-57, §140).

    Returns True only when every joint fix record matches its expected
    fix key with a non-empty `joint.mp4` beside it, and every expected
    chunk in every segment and joint plan dir carries a `chunk_mp4`
    record plus a probed-complete mp4 (`chunk_mp4_complete`, the one
    ffprobe-backed truth in this path — joint videos check exists only to
    stay fast on 127 dirs). No GPU, no ffmpeg render: ledger reads plus
    exists checks, plus the `chunk_mp4_complete` probe which reads
    fail-open (True) when ffprobe cannot parse. Any gap or any exception
    reads as False (fail-open: incomplete work takes the normal poll
    path) — joints work is therefore never re-done once complete,
    cancellable/resumable via the ledger, and only a configuration change
    (fingerprint mismatch from `augment_plan_fingerprint`) triggers
    re-evaluation.
    """
    try:
        if not isinstance(run_dir, Path):
            return False
        if not isinstance(fingerprint, str) or not fingerprint:
            return False
        if (
            isinstance(expected_segments, bool)
            or not isinstance(expected_segments, int)
            or expected_segments < 0
        ):
            return False
        if (
            isinstance(expected_joints, bool)
            or not isinstance(expected_joints, int)
            or expected_joints < 0
        ):
            return False
        ordered_segments = _require_key_list("segment_source_keys", segment_source_keys)
        ordered_counts = _require_frame_counts("segment_frame_counts", segment_frame_counts)
        ordered_fixes = _require_key_list("joint_fix_keys", joint_fix_keys)
        if len(ordered_segments) != expected_segments:
            return False
        if len(ordered_fixes) != expected_joints:
            return False
        if len(ordered_counts) != expected_segments:
            return False
        recomputed = augment_plan_fingerprint(
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=out_fps,
            upscale_factor=upscale_factor,
            multiplier=multiplier,
            crf=crf,
            preset=preset,
            interp_backend=interp_backend,
            segment_source_keys=ordered_segments,
            joint_fix_keys=ordered_fixes,
        )
        if recomputed != fingerprint:
            return False
        chunk_frames = _require_count("chunk_frames", chunk_frames)
        from voyage.augment import augment_plan
        from voyage.augment_joints import (
            JOINT_DIR_TEMPLATE,
            JOINT_RECORD_FILENAME,
            JOINT_VIDEO_FILENAME,
            joint_sources_root,
        )
        from voyage.augment_sidecar import (
            CHUNKS_LEDGER_FILENAME,
            ChunkKey,
            chunk_mp4_complete,
            load_chunk_ledger,
        )

        joint_root = joint_sources_root(run_dir)
        joint_shas: list[str] = []
        for joint_position, expected_fix in enumerate(ordered_fixes):
            joint_dir = joint_root / JOINT_DIR_TEMPLATE.format(
                left=joint_position, right=joint_position + 1
            )
            record_path = joint_dir / JOINT_RECORD_FILENAME
            try:
                record_text = record_path.read_text(encoding="utf-8")
            except OSError:
                return False
            try:
                record = json.loads(record_text)
            except ValueError:
                return False
            if not isinstance(record, dict):
                return False
            if record.get("source_key") != expected_fix:
                return False
            joint_sha = record.get("joint_sha")
            if not isinstance(joint_sha, str) or not joint_sha:
                return False
            joint_video = joint_dir / JOINT_VIDEO_FILENAME
            try:
                if not joint_video.is_file() or joint_video.stat().st_size == 0:
                    return False
            except OSError:
                return False
            joint_shas.append(joint_sha)
        for source_key, total_frames in zip(ordered_segments, ordered_counts, strict=True):
            plan_dir = plan_dir_for_segment(
                run_dir,
                source_key=source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=out_fps,
                upscale_factor=upscale_factor,
                crf=crf,
                preset=preset,
            )
            records = load_chunk_ledger(plan_dir / CHUNKS_LEDGER_FILENAME)
            for chunk in augment_plan(total_frames, chunk=chunk_frames, multiplier=multiplier):
                key = ChunkKey(
                    chunk_index=chunk.index,
                    start_frame=chunk.start_frame,
                    source_frames=chunk.source_frames,
                    expected_frames=chunk.expected_frames,
                    upscale_factor=upscale_factor,
                    multiplier=multiplier,
                    crf=crf,
                    preset=preset,
                    source_key=source_key,
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    out_fps=out_fps,
                    chunk_frames=chunk_frames,
                )
                if not chunk_mp4_complete(plan_dir, records, key):
                    return False
        for joint_sha in joint_shas:
            joint_plan = plan_dir_for_segment(
                run_dir,
                source_key=joint_sha,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=out_fps,
                upscale_factor=upscale_factor,
                crf=crf,
                preset=preset,
            )
            joint_records = load_chunk_ledger(joint_plan / CHUNKS_LEDGER_FILENAME)
            for chunk in augment_plan(MORPH_BRIDGE, chunk=chunk_frames, multiplier=multiplier):
                key = ChunkKey(
                    chunk_index=chunk.index,
                    start_frame=chunk.start_frame,
                    source_frames=chunk.source_frames,
                    expected_frames=chunk.expected_frames,
                    upscale_factor=upscale_factor,
                    multiplier=multiplier,
                    crf=crf,
                    preset=preset,
                    source_key=joint_sha,
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    out_fps=out_fps,
                    chunk_frames=chunk_frames,
                )
                if not chunk_mp4_complete(joint_plan, joint_records, key):
                    return False
    except Exception:  # noqa: BLE001 - fail-open: incomplete reads as not-complete
        return False
    return True


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    """True when `func` declares keyword `name` (injected-fake tolerance).

    Production pollers accept the backend keyword; older injected test
    doubles may not — probing keeps them working instead of TypeErroring.
    Mirrors the `_supports_on_pair` probe in `augment_interp_poller`.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _require_run_dir(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"run_dir must be a Path (got {type(value).__name__})")
    return value


def _require_usable(value: list[Path]) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"usable needs at least one segment dir (got {value!r})")
    for segment in value:
        if not isinstance(segment, Path):
            raise TypeError(f"usable entries must be Paths (got {type(segment).__name__})")
    return value


def _require_box(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int (got {value!r})")
    return value


def _require_fps(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number (got {type(value).__name__})")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{name} must be finite and > 0 (got {value!r})")
    return rate


def _require_multiplier(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"multiplier must be an int >= 1 (got {value!r})")
    return value


def _require_factor(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 4):
        raise ValueError(f"upscale_factor must be one of (1, 2, 4) (got {value!r})")
    return value


def _require_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1 (got {value!r})")
    return value


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string (got {value!r})")
    return value


def _expected_frame_totals(run_dir: Path, *, chunk_frames: int, multiplier: int) -> tuple[int, int]:
    """Upfront (source_frames, output_frames) totals over committed segments.

    Best-effort: any failure (no segments yet, unreadable manifest) yields
    `(0, 0)` and the bars fall back to end-of-pass accounting. Source
    frames tile via `chunk_windows`; output frames apply the unchunked
    per-window `(n-1)*m+1` recipe, so the finish line can show both units.
    """
    try:
        from voyage.augment import chunk_windows, interpolated_chunk_frame_count
        from voyage.augment_upscale_poller import committed_segment_sources

        sources, _skipped = committed_segment_sources(run_dir)
        source_total = 0
        output_total = 0
        for source in sources:
            for index, (_start, count) in enumerate(
                chunk_windows(source.total_frames, chunk_frames)
            ):
                source_total += count
                output_total += interpolated_chunk_frame_count(index, count, multiplier)
    except (OSError, ValueError, TypeError):
        return (0, 0)
    else:
        return (source_total, output_total)


def _poll_to_completion(
    run_dir: Path,
    *,
    weights: Any,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps: float,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int,
    device: str,
    crf: int,
    preset: str,
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
    upscale_device: str | None = None,
    interp_device: str | None = None,
    include_interp: bool = True,
    include_upscale: bool = True,
    interp_backend: str = "rife",
    joint_interp_fn: Callable[..., Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Run both pollers segment-interleaved until a pass finishes nothing.

    Each pass enumerates the committed segments once, then runs every
    segment's upscale immediately followed by its interp — interp starts
    on committed frames instead of waiting for the full upscale sweep
    (the old full-sweep-then-full-sweep stalls long finalizes by ~1h).
    Both legs default to `device` (legacy single-device behavior);
    `upscale_device`/`interp_device` pin legs to cards (interp moved to
    the 2060, cuda:1, sharing the card with upscale — RIFE's 0.65 GiB
    peak fits beside the llama sidecar). `include_interp=False` runs the
    upscale leg only; `include_upscale=False` runs the interp leg only.

    `progress` renders one leg bar per segment (`upscale frames`, then
    `interp frames` — sequential, never two concurrent Live displays).
    Bars count source frames, so both legs share one comparable unit
    (not the multiplied interp output); totals come from each segment
    (elapsed + ETA from the first second) and reconcile at scoped-call
    end, rendered chunks advance live via the pollers' `on_chunk_frames`
    callbacks, and interp chunks additionally advance per finished
    interp pair via `on_pair_frames`.     Counts still accumulate into
    `timings` for the elapsed-time report. Joints follow fix, then the
    uniform pass (seam fix -> upscale -> interpolate): the fix stage
    (`ensure_joint_units`) renders every adjacent pair's 4 source-res
    morph bridges once, then each pass polls the joint units through the
    SAME upscale + interp pollers as the segments (via the pollers'
    `sources=` override — the joints re-interpolate by design). Joint
    units rebuild when the committed segment list changes mid-poll
    (cheap manifest-key comparison; the fix itself is ledger-hit).
    `joint_interp_fn`, when given, drives the fix-stage bridge render
    (tests stub it); else the resident interp leg.
    """
    if upscale_poll_fn is None:
        from voyage.augment_upscale_poller import upscale_poll_once

        upscale_poll_fn = upscale_poll_once
    if interp_poll_fn is None:
        from voyage.augment_interp_poller import interp_poll_once

        interp_poll_fn = interp_poll_once
    source_fps_key = int(round(source_fps))
    up_dev = upscale_device or device
    ip_dev = interp_device or device
    verbose = bool(progress is not None and progress.verbose)
    expected_source, expected_output = _expected_frame_totals(
        run_dir, chunk_frames=chunk_frames, multiplier=multiplier
    )
    # Fix stage (seam fix -> upscale -> interpolate): every adjacent pair's
    # 4 source-res morph bridges render once here; each pass below then
    # polls the joint units through the same upscale + interp pollers as
    # the segments. The unit list rebuilds only when the committed
    # segment list changes mid-poll (manifest-key comparison — no file
    # hashing per pass; the fix itself is ledger-hit idempotent).
    from voyage.augment_upscale_poller import committed_segment_sources

    interp_weights = _interp_weights_path(weights, interp_backend)
    joint_units: list[Any] = []
    joint_signature: tuple[tuple[str, str], ...] | None = None

    def _refresh_joint_units(ordered: list[Any]) -> list[Any]:
        nonlocal joint_units, joint_signature
        signature = tuple((source.segment_id, source.source_key) for source in ordered)
        if joint_signature is None or signature != joint_signature:
            # Track C shared cache: the drain reuses this exact list via
            # `joint_units_signature` (no second checksum pass).
            cached = _JOINT_UNIT_CACHE.get(joint_units_signature(ordered))
            if cached is not None:
                joint_units = list(cached)
            else:
                joint_units = ensure_joint_units(
                    run_dir,
                    ordered,
                    source_fps=source_fps_key,
                    crf=crf,
                    preset=preset,
                    interp_fn=joint_interp_fn,
                    weights=interp_weights,
                    device=ip_dev,
                    interp_backend=interp_backend,
                )
                _JOINT_UNIT_CACHE[joint_units_signature(ordered)] = tuple(joint_units)
            joint_signature = signature
        return joint_units

    for _ in range(_MAX_POLL_PASSES):
        up_done = 0
        ip_done = 0
        ip_waiting = 0
        up_per_segment: dict[str, list[int]] = {}
        ip_per_segment: dict[str, list[int]] = {}
        # Segment-interleaved pipeline: enumerate once per pass, then run
        # each segment's upscale immediately followed by its interp, so
        # interp starts on committed frames instead of waiting for the
        # full upscale sweep. Pass totals below preserve the old settle
        # semantics exactly (a pass that finishes nothing returns, unless
        # interp chunks are still waiting on missing upscale outputs).
        segment_sources, _skipped_sources = committed_segment_sources(run_dir)
        ordered_sources = sorted(segment_sources, key=lambda source: source.segment_id)
        up_takes_segments = _accepts_keyword(upscale_poll_fn, "segment_ids")
        ip_takes_segments = _accepts_keyword(interp_poll_fn, "segment_ids")
        interp_weights = _interp_weights_path(weights, interp_backend)
        for source in ordered_sources:
            if should_stop is not None and should_stop():
                return
            segment_id = source.segment_id
            segment_frames = source.total_frames or None
            if include_upscale:
                upscale_start = time.monotonic()
                with optional_bar(progress, "upscale frames", segment_frames) as up_tracker:

                    def _up_chunk(
                        segment_id: str,
                        index: int,
                        _total: int,
                        _into: dict[str, list[int]] = up_per_segment,
                    ) -> None:
                        _into.setdefault(segment_id, []).append(index)

                    def _up_frames(segment_id: str, frames: int) -> None:
                        del segment_id
                        if up_tracker is not None:
                            up_tracker.update(frames)

                    upscale_kwargs: dict[str, Any] = {
                        "weights_path": weights.realesrgan,
                        "weights_key": weights_key,
                        "out_width": out_width,
                        "out_height": out_height,
                        "out_fps": source_fps_key,
                        "upscale_factor": upscale_factor,
                        "chunk_frames": chunk_frames,
                        "device": up_dev,
                        "crf": crf,
                        "preset": preset,
                        "on_chunk": _up_chunk,
                        "on_chunk_frames": _up_frames,
                    }
                    if _accepts_keyword(upscale_poll_fn, "interp_multiplier"):
                        upscale_kwargs["interp_multiplier"] = multiplier
                    if _accepts_keyword(upscale_poll_fn, "shared_segment_decode"):
                        upscale_kwargs["shared_segment_decode"] = True
                    if up_takes_segments:
                        upscale_kwargs["segment_ids"] = [segment_id]
                    upscale_result = upscale_poll_fn(run_dir, **upscale_kwargs)
                    upscale_seconds = time.monotonic() - upscale_start
                    scoped_done = int(getattr(upscale_result, "chunks_done", 0) or 0)
                    scoped_frames = int(getattr(upscale_result, "frames_done", 0) or 0)
                    scoped_skipped = int(getattr(upscale_result, "frames_skipped", 0) or 0)
                    up_done += scoped_done
                    if up_tracker is not None:
                        scoped_total = scoped_frames + scoped_skipped
                        if scoped_total > 0:
                            up_tracker.set_total(scoped_total)
                        if scoped_skipped > 0:
                            up_tracker.update(scoped_skipped)
                        if scoped_frames > 0 and upscale_seconds > 0:
                            up_tracker.set_extra(f"{scoped_frames / upscale_seconds:.1f} frames/s")
                if timings is not None:
                    timings["upscale_poll_s"] = timings.get("upscale_poll_s", 0.0) + upscale_seconds
                    timings["upscale_chunks_done"] = timings.get(
                        "upscale_chunks_done", 0.0
                    ) + float(scoped_done)
                    timings["upscale_frames_done"] = timings.get(
                        "upscale_frames_done", 0.0
                    ) + float(scoped_frames)
            if include_interp:
                interp_start = time.monotonic()
                with optional_bar(progress, "interp frames", segment_frames) as ip_tracker:
                    _ip_carry: list[float] = [0.0]

                    def _ip_chunk(
                        segment_id: str,
                        index: int,
                        _total: int,
                        _into: dict[str, list[int]] = ip_per_segment,
                    ) -> None:
                        _into.setdefault(segment_id, []).append(index)

                    def _ip_pair(
                        segment_id: str,
                        fraction_done: float,
                        _carry: list[float] = _ip_carry,
                    ) -> None:
                        del segment_id
                        _carry[0] += fraction_done
                        whole, _carry[0] = divmod(_carry[0], 1.0)
                        if ip_tracker is not None and whole >= 1:
                            ip_tracker.update(int(whole))

                    interp_kwargs: dict[str, Any] = {
                        "weights_path": interp_weights,
                        "weights_key": weights_key,
                        "out_width": out_width,
                        "out_height": out_height,
                        "out_fps": source_fps_key,
                        "upscale_factor": upscale_factor,
                        "chunk_frames": chunk_frames,
                        "multiplier": multiplier,
                        "device": ip_dev,
                        "crf": crf,
                        "preset": preset,
                        "on_chunk": _ip_chunk,
                        "on_pair_frames": _ip_pair,
                    }
                    if ip_takes_segments:
                        interp_kwargs["segment_ids"] = [segment_id]
                    if _accepts_keyword(interp_poll_fn, "interp_backend"):
                        interp_kwargs["interp_backend"] = interp_backend
                    interp_result = interp_poll_fn(run_dir, **interp_kwargs)
                    interp_seconds = time.monotonic() - interp_start
                    scoped_done = int(getattr(interp_result, "chunks_done", 0) or 0)
                    scoped_frames = int(getattr(interp_result, "frames_done", 0) or 0)
                    scoped_skipped = int(getattr(interp_result, "frames_skipped", 0) or 0)
                    scoped_waiting = int(getattr(interp_result, "chunks_waiting", 0) or 0)
                    ip_done += scoped_done
                    ip_waiting += scoped_waiting
                    if ip_tracker is not None:
                        scoped_total = scoped_frames + scoped_skipped + scoped_waiting
                        if scoped_total > 0:
                            ip_tracker.set_total(scoped_total)
                        advanced = scoped_skipped + scoped_waiting
                        if advanced > 0:
                            ip_tracker.update(advanced)
                        if scoped_frames > 0 and interp_seconds > 0:
                            ip_tracker.set_extra(
                                f"{scoped_frames / interp_seconds:.1f} frames/s "
                                f"({expected_output} out frames)"
                            )
                        else:
                            ip_tracker.set_extra(f"{expected_output} out frames")
                if timings is not None:
                    timings["interp_poll_s"] = timings.get("interp_poll_s", 0.0) + interp_seconds
                    timings["interp_chunks_done"] = timings.get("interp_chunks_done", 0.0) + float(
                        scoped_done
                    )
                    timings["interp_frames_done"] = timings.get("interp_frames_done", 0.0) + float(
                        scoped_frames
                    )
        # Uniform pass over the fix-stage joints: the joint units poll
        # through the same legs as the segments (sources= override), so
        # bridge frames upscale + re-interpolate by design. Joints are new
        # units with no prior upscale, so an interp-only poll still runs
        # their upscale — otherwise they would strand as waiting forever.
        joint_here = _refresh_joint_units(ordered_sources)
        joint_sources = [unit.as_source() for unit in joint_here]

        joint_up = include_upscale or (include_interp and bool(joint_sources))
        joint_order = [unit.segment_id for unit in joint_sources]
        joint_position = {segment_id: position for position, segment_id in enumerate(joint_order)}
        joint_fired = [0, 0]  # upscale / interp source frames fired live
        joint_bar_cm: Any = None
        joint_tracker: Any = None
        if joint_sources and (joint_up or include_interp):
            joint_bar_cm = optional_bar(
                progress, "joint frames", _JOINT_BAR_LEG_COUNT * MORPH_BRIDGE * len(joint_sources)
            )
            joint_tracker = joint_bar_cm.__enter__()

        def _joint_status(
            leg: str,
            segment_id: str,
            _tracker: Any = joint_tracker,
            _positions: dict[str, int] = joint_position,
            _order: list[str] = joint_order,
        ) -> None:
            if _tracker is None:
                return
            position = _positions.get(segment_id, 0)
            _tracker.set_extra(f"{leg} {segment_id} ({position + 1}/{len(_order)})")

        def _joint_up_chunk(
            segment_id: str,
            index: int,
            _total: int,
            _into: dict[str, list[int]] = up_per_segment,
        ) -> None:
            _into.setdefault(segment_id, []).append(index)

        def _joint_up_frames(
            segment_id: str,
            frames: int,
            _tracker: Any = joint_tracker,
            _fired: list[int] = joint_fired,
        ) -> None:
            if _tracker is not None and frames > 0:
                _fired[0] += frames
                _tracker.update(frames)
                _joint_status("upscaling", segment_id)

        def _joint_ip_chunk(
            segment_id: str,
            index: int,
            _total: int,
            _into: dict[str, list[int]] = ip_per_segment,
        ) -> None:
            _into.setdefault(segment_id, []).append(index)

        _joint_ip_carry: list[float] = [0.0]

        def _joint_ip_pair(
            segment_id: str,
            fraction_done: float,
            _carry: list[float] = _joint_ip_carry,
            _tracker: Any = joint_tracker,
            _fired: list[int] = joint_fired,
        ) -> None:
            _carry[0] += fraction_done
            whole, _carry[0] = divmod(_carry[0], 1.0)
            if _tracker is not None and whole >= 1:
                _fired[1] += int(whole)
                _tracker.update(int(whole))
                _joint_status("interpolating", segment_id)

        if joint_up and joint_sources:
            joint_up_start = time.monotonic()
            joint_up_kwargs: dict[str, Any] = {
                "weights_path": weights.realesrgan,
                "weights_key": weights_key,
                "out_width": out_width,
                "out_height": out_height,
                "out_fps": source_fps_key,
                "upscale_factor": upscale_factor,
                "chunk_frames": chunk_frames,
                "device": up_dev,
                "crf": crf,
                "preset": preset,
                "on_chunk": _joint_up_chunk,
                "on_chunk_frames": _joint_up_frames,
                "sources": joint_sources,
            }
            if _accepts_keyword(upscale_poll_fn, "interp_multiplier"):
                joint_up_kwargs["interp_multiplier"] = multiplier
            if up_takes_segments:
                joint_up_kwargs["segment_ids"] = [unit.segment_id for unit in joint_sources]
            joint_up_result = upscale_poll_fn(run_dir, **joint_up_kwargs)
            joint_up_seconds = time.monotonic() - joint_up_start
            up_done += int(getattr(joint_up_result, "chunks_done", 0) or 0)
            if timings is not None:
                timings["upscale_poll_s"] = timings.get("upscale_poll_s", 0.0) + joint_up_seconds
                timings["upscale_chunks_done"] = timings.get("upscale_chunks_done", 0.0) + float(
                    getattr(joint_up_result, "chunks_done", 0) or 0
                )
                timings["upscale_frames_done"] = timings.get("upscale_frames_done", 0.0) + float(
                    getattr(joint_up_result, "frames_done", 0) or 0
                )
        if include_interp and joint_sources:
            joint_ip_start = time.monotonic()
            joint_ip_kwargs: dict[str, Any] = {
                "weights_path": interp_weights,
                "weights_key": weights_key,
                "out_width": out_width,
                "out_height": out_height,
                "out_fps": source_fps_key,
                "upscale_factor": upscale_factor,
                "chunk_frames": chunk_frames,
                "multiplier": multiplier,
                "device": ip_dev,
                "crf": crf,
                "preset": preset,
                "on_chunk": _joint_ip_chunk,
                "on_pair_frames": _joint_ip_pair,
                "sources": joint_sources,
            }
            if ip_takes_segments:
                joint_ip_kwargs["segment_ids"] = [unit.segment_id for unit in joint_sources]
            if _accepts_keyword(interp_poll_fn, "interp_backend"):
                joint_ip_kwargs["interp_backend"] = interp_backend
            joint_ip_result = interp_poll_fn(run_dir, **joint_ip_kwargs)
            joint_ip_seconds = time.monotonic() - joint_ip_start
            ip_done += int(getattr(joint_ip_result, "chunks_done", 0) or 0)
            ip_waiting += int(getattr(joint_ip_result, "chunks_waiting", 0) or 0)
            if timings is not None:
                timings["interp_poll_s"] = timings.get("interp_poll_s", 0.0) + joint_ip_seconds
                timings["interp_chunks_done"] = timings.get("interp_chunks_done", 0.0) + float(
                    getattr(joint_ip_result, "chunks_done", 0) or 0
                )
                timings["interp_frames_done"] = timings.get("interp_frames_done", 0.0) + float(
                    getattr(joint_ip_result, "frames_done", 0) or 0
                )
        if joint_tracker is not None:
            # Ledger-hit chunks never fire: bump the remainder so the bar
            # always finishes at its total (mirrors the per-segment bars).
            joint_tracker.update(
                max(
                    0,
                    _JOINT_BAR_LEG_COUNT * MORPH_BRIDGE * len(joint_sources)
                    - joint_fired[0]
                    - joint_fired[1],
                )
            )
        if joint_bar_cm is not None:
            joint_bar_cm.__exit__(None, None, None)
        if verbose and up_per_segment and progress is not None:
            for segment_id in sorted(up_per_segment):
                progress.info(
                    f"upscale {segment_id}: chunks {_format_ranges(up_per_segment[segment_id])}"
                )
        if verbose and ip_per_segment and progress is not None:
            for segment_id in sorted(ip_per_segment):
                progress.info(
                    f"interp {segment_id}: chunks {_format_ranges(ip_per_segment[segment_id])}"
                )
        if include_interp:
            if up_done == 0 and ip_done == 0:
                if ip_waiting > 0:
                    raise MediaError(
                        f"augment polling stuck: {ip_waiting} chunks waiting "
                        "with no progress (rerun the pollers, then finalize again)"
                    )
                # Track C settle output-truth: ledger counts alone once
                # settled torn encodes — verify PNG/mp4 outputs before
                # returning (a gap here fails loud instead of draining
                # short silently).
                settled, missing = verify_settle_output_truth(
                    run_dir,
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    source_fps_key=source_fps_key,
                    upscale_factor=upscale_factor,
                    multiplier=multiplier,
                    chunk_frames=chunk_frames,
                    crf=crf,
                    preset=preset,
                )
                if not settled:
                    raise MediaError(
                        f"augment polling settled by counts but outputs missing: {missing} "
                        "(rerun the pollers, then finalize again)"
                    )
                return
        elif up_done == 0:
            settled, missing = verify_settle_output_truth(
                run_dir,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                source_fps_key=source_fps_key,
                upscale_factor=upscale_factor,
                multiplier=1,
                chunk_frames=chunk_frames,
                crf=crf,
                preset=preset,
            )
            if not settled:
                raise MediaError(
                    f"upscale polling settled by counts but outputs missing: {missing}"
                )
            return
    raise MediaError(
        f"augment polling made no settling pass in {_MAX_POLL_PASSES} rounds "
        "(rerun the pollers, then finalize again)"
    )


def verify_settle_output_truth(
    run_dir: Path,
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps_key: int,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int,
    crf: int,
    preset: str,
) -> tuple[bool, list[str]]:
    """Verify settle via output-truth, not ledger-only (Track C).

    Checks every LEDGERED chunk has output-truth completion (durable mp4
    with frame-count truth, else complete interp PNGs with matching
    geometry). Returns `(ok, missing_descriptions)` — settle may return
    only when ok; otherwise the poll loop fails loud when no progress is
    possible. Ledger counts alone once settled torn encodes with intact
    records; this is the gate that catches them. Sources/plan dirs with
    no ledger records verify vacuously (injected test fakes render
    without writing sidecars — counts settle those, not this gate).
    """
    from voyage.augment import (
        augment_plan,
        chunk_windows,
    )
    from voyage.augment import (
        chunk_frames_match_size as _size_match,
    )
    from voyage.augment_sidecar import (
        STAGE_CHUNK_MP4,
        ChunkKey,
        chunk_mp4_complete,
        chunk_output_complete,
        load_chunk_ledger,
        plan_dir_for_segment,
    )
    from voyage.augment_upscale_poller import committed_segment_sources

    try:
        sources, _ = committed_segment_sources(run_dir)
    except (OSError, ValueError, TypeError):
        return (False, ["unreadable segment sources"])
    missing: list[str] = []
    for source in sorted(sources, key=lambda item: item.segment_id):
        try:
            plan_dir = plan_dir_for_segment(
                run_dir,
                source_key=source.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=source_fps_key,
                upscale_factor=upscale_factor,
                crf=crf,
                preset=preset,
            )
        except (OSError, ValueError, TypeError):
            continue
        ledger = plan_dir / "chunks.jsonl"
        records = load_chunk_ledger(ledger)
        if not records:
            continue  # injected fakes render without sidecars — counts settle those
        try:
            windows = list(chunk_windows(source.total_frames, chunk_frames))
        except (TypeError, ValueError):
            continue
        plan = augment_plan(source.total_frames, chunk=chunk_frames, multiplier=multiplier)
        # Only ledgered chunks verify: unrecorded windows are the
        # pollers' business (counts settle them); recorded-but-incomplete
        # outputs are this gate's catch (torn encodes with intact records).
        recorded_indexes = {
            record.get("chunk_index")
            for record in records
            if isinstance(record.get("chunk_index"), int)
            and not isinstance(record.get("chunk_index"), bool)
        }
        for chunk in plan:
            if chunk.index not in recorded_indexes:
                continue
            key = ChunkKey(
                chunk_index=chunk.index,
                start_frame=chunk.start_frame,
                source_frames=chunk.source_frames,
                expected_frames=chunk.expected_frames,
                upscale_factor=upscale_factor,
                multiplier=multiplier,
                crf=crf,
                preset=preset,
                source_key=source.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=source_fps_key,
                chunk_frames=chunk_frames,
            )
            # Settled = durable mp4 (with frame-count truth) OR complete
            # interp PNGs with matching geometry. Anything else is missing.
            if chunk_mp4_complete(plan_dir, records, key):
                continue
            interp_dir = plan_dir / f"interpolated_{chunk.index:02d}"
            if not chunk_output_complete(interp_dir, chunk.expected_frames):
                missing.append(f"{source.segment_id}:chunk{chunk.index}")
                continue
            if not _size_match(interp_dir, (out_width, out_height)):
                missing.append(f"{source.segment_id}:chunk{chunk.index}:geometry")
                continue
        _ = windows
        _ = STAGE_CHUNK_MP4
    return (not missing, missing)


def _format_ranges(indexes: list[int]) -> str:
    """Compact `0, 1, 2, 5` chunk indexes as `0-2, 5` (verbose lines)."""
    ordered = sorted(set(indexes))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _ensure_model_pass_timings(timings: dict[str, float] | None) -> None:
    """Zero-initialize every model-pass timings key (miners never KeyError).

    Called on entry by both `run_durable_model_pass` and
    `_drain_to_intermediate` — the drain is also invoked directly by the
    bidirectional parallel branch (media.py), which never runs the
    durable entry's init, so the drain must not assume pre-initialized
    keys (live kaolin `KeyError: 'drain_s'`, 2026-10-06).

    Mastering keys (`mastering_poll_s` / `mastering_chunks_done` /
    `mastering_frames_done`, Track C) are zero-initialized here for the
    future `voyage/mastering.py` consumer (Track B owns that module) —
    miners and `_model_pass_stage_rows` never KeyError, even before any
    mastering work lands. Unified Track C/E vocabulary (drain/concat/
    seam/morph rows) is zeroed here too, so the timing table never
    KeyErrors on a leg this pass did not run.
    """
    if timings is None:
        return
    for key in (
        "upscale_poll_s",
        "interp_poll_s",
        "drain_s",
        "concat_s",
        "upscale_chunks_done",
        "interp_chunks_done",
        "upscale_frames_done",
        "interp_frames_done",
        "chunks_drained",
        "seam_s",
        "seams_done",
        "morph_s",
        "morphs_done",
        "mastering_poll_s",
        "mastering_chunks_done",
        "mastering_frames_done",
    ):
        timings.setdefault(key, 0.0)


_JOINT_UNIT_CACHE: dict[tuple[str, ...], tuple[Any, ...]] = {}
"""Shared joint-unit list cache (Track C redundant-hashing fix).

`_poll_to_completion` and `_drain_to_intermediate` both build the fix-stage
joint units from the same committed-segment manifests — each build re-hashes
joint videos (`ensure_joint_units` checksums anchors). The cache keys on the
ordered `(segment_id, source_key)` signature: poll stores, drain reuses, so
one finalize hashes each joint once. Per-process only; signature mismatch
rebuilds (never stale across trims/renders).
"""


def joint_units_signature(ordered: list[Any]) -> tuple[str, ...]:
    """Cache signature for an ordered source list (ids + source keys)."""
    signature: list[str] = []
    for source in ordered:
        signature.append(str(getattr(source, "segment_id", "")))
        signature.append(str(getattr(source, "source_key", "")))
    return tuple(signature)


def _live_plan_dirs_for_sweep(
    run_dir: Path,
    usable: list[Path],
    *,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    crf: int,
    preset: str,
) -> set[Path]:
    """Live `plan_dir_for_segment` set for the current usable segments.

    Track C sweep-liveness: the finalize-start sweep encodes only live
    dirs (orphans are GC's, never re-encoded). Joint units join via the
    poll's own live set; this covers segment dirs.
    """
    from voyage.augment_sidecar import plan_dir_for_segment
    from voyage.augment_upscale_poller import committed_segment_sources

    live: set[Path] = set()
    try:
        sources, _ = committed_segment_sources(run_dir)
    except (OSError, ValueError, TypeError):
        return live
    usable_names = {segment.name for segment in usable}
    for source in sources:
        if source.segment_id not in usable_names:
            continue
        try:
            live.add(
                plan_dir_for_segment(
                    run_dir,
                    source_key=source.source_key,
                    weights_key=weights_key,
                    out_width=out_width,
                    out_height=out_height,
                    out_fps=out_fps,
                    upscale_factor=upscale_factor,
                    crf=crf,
                    preset=preset,
                )
            )
        except (OSError, ValueError, TypeError):
            continue
    return live


def _resume_completed_pass(
    run_dir: Path,
    segments: list[Path],
    *,
    weights: Any,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps: float,
    source_fps_key: int,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int,
    crf: int,
    preset: str,
    device: str,
    work_dir: Path,
    interp_backend: str,
    drain_fn: Callable[..., Any] | None,
    concat_fn: Callable[[list[Path], Path], Path] | None,
    assemble_fn: Callable[..., Any] | None,
    joint_interp_fn: Callable[..., Any] | None,
    timings: dict[str, float] | None,
    progress: VoyageConsole | None,
) -> tuple[Path, int] | None:
    """Resume a fully-ledgered model pass without re-polling (stage-skip gate).

    Computes the current `augment_plan_fingerprint` from the same recipe the
    pollers use and compares it against the stored `augment_coverage`
    marker: fingerprint match + segment/joint counts match + every fix
    record and chunk mp4 verified by `augment_work_complete` means the
    expensive poll stage is already done, so it is skipped entirely. Any
    config change forks the fingerprint (weights, geometry, fps, factors,
    crf/preset, backend, or any source/joint key) and re-evaluates by
    construction; any gap returns None and the caller runs the full
    poll+drain path.

    Fix keys are derived without file hashing: `a_sha`/`b_sha` reuse the
    poller `source_key`, which is the manifest `video.mp4` checksum —
    `sha256_file` of the committed video recorded at commit, the exact
    same bytes `joint_video_sha` hashes when `ensure_joint_units`
    derives the key (committed segments are immutable). Geometry reuses
    the plan derivation (`out // upscale_factor`); a non-divisible box
    bails to the full path. A wrong derivation can only return None
    (records never match), never a false skip.

    Drain still runs when the timeline is not reusable: the lone-segment
    intermediate lives in the transient work dir, and the joint timeline
    is only reused when its sha still matches the marker. After a
    drain, the marker is re-stamped so the next finalize skips again.
    Returns the `(final, lifted_fps)` pair on the drain contract, else
    None.
    """
    from voyage.augment_joints import joint_fix_key
    from voyage.augment_upscale_poller import committed_segment_sources
    from voyage.persistence import read_augment_coverage, record_augment_coverage

    if out_width % upscale_factor or out_height % upscale_factor:
        return None
    sources, _skipped = committed_segment_sources(run_dir)
    by_id = {source.segment_id: source for source in sources}
    ordered: list[Any] = []
    for segment in segments:
        source = by_id.get(segment.name)
        if source is None:
            raise MediaError(
                f"usable segment {segment.name} has no pollable source "
                "(DONE + manifest with video checksum/frames required)"
            )
        ordered.append(source)
    src_width = out_width // upscale_factor
    src_height = out_height // upscale_factor
    fix_keys = [
        joint_fix_key(
            a_sha=ordered[index].source_key,
            b_sha=ordered[index + 1].source_key,
            width=src_width,
            height=src_height,
            fps_key=source_fps_key,
        )
        for index in range(len(ordered) - 1)
    ]
    source_keys = [source.source_key for source in ordered]
    current = augment_plan_fingerprint(
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        crf=crf,
        preset=preset,
        interp_backend=interp_backend,
        segment_source_keys=source_keys,
        joint_fix_keys=fix_keys,
    )
    try:
        stored = read_augment_coverage(run_dir)
        marker = stamp_augment_coverage(stored) if stored is not None else None
    except (TypeError, ValueError):
        return None
    if (
        marker is None
        or marker["fingerprint"] != current
        or marker["segments"] != len(ordered)
        or marker["joints"] != len(fix_keys)
    ):
        return None
    if not augment_work_complete(
        run_dir,
        current,
        len(ordered),
        len(fix_keys),
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        crf=crf,
        preset=preset,
        interp_backend=interp_backend,
        segment_source_keys=source_keys,
        segment_frame_counts=[source.total_frames for source in ordered],
        joint_fix_keys=fix_keys,
    ):
        return None
    if len(ordered) > 1:
        timeline = run_dir / "augment" / "joint_timeline" / "jointed_timeline.mp4"
        try:
            if timeline.is_file() and sha256_file(timeline) == marker["jointed_timeline_sha"]:
                return (timeline, round(source_fps * multiplier))
        except OSError:
            pass
    final, lifted_fps = _drain_to_intermediate(
        run_dir,
        segments,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        source_fps_key=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        crf=crf,
        preset=preset,
        device=device,
        work_dir=work_dir,
        interp_backend=interp_backend,
        drain_fn=drain_fn,
        concat_fn=concat_fn,
        assemble_fn=assemble_fn,
        joint_interp_fn=joint_interp_fn,
        timings=timings,
        progress=progress,
    )
    record_augment_coverage(
        run_dir,
        stamp_augment_coverage(
            {
                "fingerprint": current,
                "segments": len(ordered),
                "joints": len(fix_keys),
                "jointed_timeline_sha": sha256_file(final),
                "interp_backend": interp_backend,
                "weights_key": weights_key,
            }
        ),
    )
    return (final, lifted_fps)


def run_durable_model_pass(
    run_dir: Path,
    usable: list[Path],
    *,
    out_width: int,
    out_height: int,
    source_fps: float,
    weights: Any,
    upscale_factor: int = 2,
    multiplier: int = 4,
    chunk_frames: int = 32,
    crf: int = 15,
    preset: str = "veryfast",
    device: str = "cuda:1",
    work_dir: Path,
    interp_backend: str = "rife",
    upscale_device: str | None = None,
    interp_device: str | None = None,
    upscale_poll_fn: Callable[..., Any] | None = None,
    interp_poll_fn: Callable[..., Any] | None = None,
    chunk_encode_fn: Callable[..., Any] | None = None,
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    assemble_fn: Callable[..., Any] | None = None,
    joint_interp_fn: Callable[..., Any] | None = None,
    morph_joints: bool = False,
    morph_interp_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Path, int]:
    """Poll, drain, and concat the durable sidecar path (issue: independent workers).

    `usable` holds committed segment dirs in presentation order; `out_fps`
    for the shared plan derivation is always `source_fps` (the interp
    worker records the lifted fps inside the same plan dir). Returns the
    final intermediate + its fps (`round(source_fps * multiplier)`),
    matching the legacy `run_finalize_model_pass` contract. Pollers that
    scan `run_dir` may also finish skipped (non-usable) segments — the
    drain below only consumes `usable`, in order.

    Finalize-start cleanup: `sweep_chunk_mp4s` first encodes (+ records
    + prunes PNGs) every fully-interpolated chunk still missing its
    durable mp4 — one-time catch-up for runs rendered before per-chunk
    mp4s existed, so their gigabytes of PNGs collapse to megabytes
    before polling starts — then `prune_orphan_plan_dirs` GCs plan dirs
    no current segment or joint references. `chunk_encode_fn`, when
    given, replaces the production chunk-ffmpeg encode inside the sweep
    (tests stub it); None (default) encodes for real.

    `timings` (optional out-param) records per-phase seconds plus chunk
    counts for the elapsed-time report: `upscale_poll_s`,
    `interp_poll_s`, `drain_s`, `concat_s`, `upscale_chunks_done`,
    `interp_chunks_done`, `chunks_drained`, plus `seam_s` seconds spent
    rendering fix-stage joints and `seams_done` joints actually rendered
    (ledger hits resume without re-rendering). Keys are zero-initialized
    on entry so miners never KeyError, even when a later stage fails.

    Joints (universal, all backends): every adjacent usable pair is fixed
    once at source resolution (morph 2+2 bridges, no upscale — see
    `voyage.augment_joints`), then the 4 bridge frames flow through the
    same upscale + interp pollers as the segments and drain interleaved
    as [trimA, joint, trimB] with source-derived trims. `joint_interp_fn`
    drives the fix-stage bridge render (tests stub it); when None,
    `morph_interp_fn` is the fallback (legacy name, same seam); both
    None defaults to the resident interp leg. `morph_joints` is a legacy
    flag (validated bool): both values yield the universal joints.

    Interleaved pass (DESIGN §140): the driver runs each segment's
    upscale immediately followed by its interp on one card — the
    caller pins both `upscale_device` and `interp_device` to the 2060
    (cuda:1; RIFE fits beside the llama sidecar), and the drain below
    inherits the interp card. Both default to `device` (the legacy
    single-device behavior).
    """
    if not isinstance(morph_joints, bool):
        raise TypeError(f"morph_joints must be a bool (got {type(morph_joints).__name__})")
    run_dir = _require_run_dir(run_dir)
    segments = _require_usable(usable)
    out_width = _require_box("out_width", out_width)
    out_height = _require_box("out_height", out_height)
    source_fps = _require_fps("source_fps", source_fps)
    upscale_factor = _require_factor(upscale_factor)
    multiplier = _require_multiplier(multiplier)
    chunk_frames = _require_count("chunk_frames", chunk_frames)
    device = _require_text("device", device)
    preset = _require_text("preset", preset)
    if not isinstance(work_dir, Path):
        raise TypeError(f"work_dir must be a Path (got {type(work_dir).__name__})")
    _ensure_model_pass_timings(timings)
    weights_key = weights_key_for(weights, interp_backend)
    # Finalize-start cleanup (Track C prune-then-sweep): GC orphan plan
    # dirs FIRST (they are never re-encoded), then sweep only live dirs.
    # The old sweep-then-GC order re-encoded orphans before deleting them
    # (wasted GPU hours on stale backends); both steps stay idempotent, so
    # a kill between them converges on retry.
    from voyage.augment_drain import prune_orphan_plan_dirs, sweep_chunk_mp4s

    # Integer fps key shared by the pollers and the plan derivation below:
    # the hash formats it via str(), so float 24.0 vs int 24 would fork
    # plan dirs — one normalization keeps all three on the same dir.
    source_fps_key = int(round(source_fps))
    # Stage-skip gate (joints resume): a fully-ledgered pass for this exact
    # fingerprint returns the drained intermediate without re-polling (and
    # without re-draining when the joint timeline sha still matches); None
    # falls through to the full prune/sweep/poll/drain path below. Any
    # config change forks the fingerprint and re-evaluates by construction.
    resumed = _resume_completed_pass(
        run_dir,
        segments,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        source_fps_key=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        crf=crf,
        preset=preset,
        device=interp_device or device,
        work_dir=work_dir,
        interp_backend=interp_backend,
        drain_fn=drain_fn,
        concat_fn=concat_fn,
        assemble_fn=assemble_fn,
        joint_interp_fn=(joint_interp_fn if joint_interp_fn is not None else morph_interp_fn),
        timings=timings,
        progress=progress,
    )
    if resumed is not None:
        return resumed
    prune_orphan_plan_dirs(
        run_dir,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=source_fps_key,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
    live_dirs = _live_plan_dirs_for_sweep(
        run_dir,
        segments,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=source_fps_key,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
    sweep_chunk_mp4s(run_dir, encode_fn=chunk_encode_fn, live_dirs=live_dirs or None)
    _poll_to_completion(
        run_dir,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        device=device,
        crf=crf,
        preset=preset,
        upscale_poll_fn=upscale_poll_fn,
        interp_poll_fn=interp_poll_fn,
        timings=timings,
        progress=progress,
        upscale_device=upscale_device,
        interp_device=interp_device,
        interp_backend=interp_backend,
        joint_interp_fn=(joint_interp_fn if joint_interp_fn is not None else morph_interp_fn),
        should_stop=should_stop,
    )
    return _drain_to_intermediate(
        run_dir,
        segments,
        weights=weights,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        source_fps=source_fps,
        source_fps_key=source_fps_key,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        crf=crf,
        preset=preset,
        device=interp_device or device,
        work_dir=work_dir,
        interp_backend=interp_backend,
        drain_fn=drain_fn,
        concat_fn=concat_fn,
        assemble_fn=assemble_fn,
        joint_interp_fn=(joint_interp_fn if joint_interp_fn is not None else morph_interp_fn),
        timings=timings,
        progress=progress,
    )


def _drain_to_intermediate(
    run_dir: Path,
    segments: list[Path],
    *,
    weights: Any,
    weights_key: str,
    out_width: int,
    out_height: int,
    source_fps: float,
    source_fps_key: int,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int | None = None,
    crf: int,
    preset: str,
    device: str,
    work_dir: Path,
    interp_backend: str = "rife",
    drain_fn: Callable[..., Any] | None = None,
    concat_fn: Callable[[list[Path], Path], Path] | None = None,
    assemble_fn: Callable[..., Any] | None = None,
    joint_interp_fn: Callable[..., Any] | None = None,
    timings: dict[str, float] | None = None,
    progress: VoyageConsole | None = None,
) -> tuple[Path, int]:
    """Drain ledgered interp chunks (+ uniform-pass joints) into one intermediate.

    Shared entry for `run_durable_model_pass`: consumes
    only `usable` in order. The fix stage renders every boundary's
    source-res joints (ledger-hit when `run_durable_model_pass` already
    polled them), the joint units drain whole through the same
    `drain_fn`, and `assemble_joint_timeline` joins
    [trimA, joint, trimB, ...] with source-derived trims. All interp
    work here runs on `device` — the caller passes the 4060 interp
    device, never the 2060 upscale card.

    ``chunk_frames`` (None = legacy contiguous trims) forwards to
    ``assemble_joint_timeline`` for chunk-aware keeps; ``None`` keeps
    existing direct callers byte-identical while ``run_durable_model_pass``
    forwards its own window so production drains map trims.
    """
    if drain_fn is None:
        from voyage.augment_drain import drain_interpolated_plan

        drain_fn = drain_interpolated_plan
    if concat_fn is None:
        from voyage.augment_drain import concat_chunk_mp4s

        concat_fn = concat_chunk_mp4s
    from voyage.augment_upscale_poller import committed_segment_sources

    _ensure_model_pass_timings(timings)
    if chunk_frames is not None:
        chunk_frames = _require_count("chunk_frames", chunk_frames)
    sources, _skipped = committed_segment_sources(run_dir)
    by_id = {source.segment_id: source for source in sources}
    ordered: list[Any] = []
    for segment in segments:
        source = by_id.get(segment.name)
        if source is None:
            raise MediaError(
                f"usable segment {segment.name} has no pollable source "
                "(DONE + manifest with video checksum/frames required)"
            )
        ordered.append(source)
    # Fix stage (idempotent: the poll sweep above already rendered these;
    # a direct drain call renders them here instead). Skipped entirely
    # for a lone segment — no pair means no joint, and the interp leg is
    # never even resolved (a lone drain must not need it).
    fix_start = time.monotonic()
    joint_units: list[Any] = []
    if len(ordered) > 1:
        # Track C shared cache: the poll sweep already hashed+rendered
        # these units — reuse the list instead of re-checksumming.
        cached = _JOINT_UNIT_CACHE.get(joint_units_signature(ordered))
        if cached is not None:
            joint_units = list(cached)
        else:
            joint_units = ensure_joint_units(
                run_dir,
                ordered,
                source_fps=source_fps_key,
                crf=crf,
                preset=preset,
                interp_fn=joint_interp_fn,
                weights=_interp_weights_path(weights, interp_backend),
                device=device,
                interp_backend=interp_backend,
            )
            _JOINT_UNIT_CACHE[joint_units_signature(ordered)] = tuple(joint_units)
    if timings is not None:
        timings["seam_s"] += time.monotonic() - fix_start
        timings["seams_done"] += float(len(joint_units))
    segment_intermediates: list[Path] = []
    drain_cm = optional_bar(progress, "drain segments", total=len(segments))
    with drain_cm as drain_tracker:
        for source in ordered:
            plan_dir = plan_dir_for_segment(
                run_dir,
                source_key=source.source_key,
                weights_key=weights_key,
                out_width=out_width,
                out_height=out_height,
                out_fps=source_fps_key,
                upscale_factor=upscale_factor,
                crf=crf,
                preset=preset,
            )
            drain_start = time.monotonic()
            drained = drain_fn(plan_dir, out_fps=source_fps * multiplier)
            if timings is not None:
                timings["drain_s"] += time.monotonic() - drain_start
                timings["chunks_drained"] += float(drained.chunks_drained)
            segment_intermediates.append(drained.intermediate_mp4)
            if drain_tracker is not None:
                drain_tracker.update()
    joint_intermediates: list[Path] = []
    for unit in joint_units:
        joint_plan = plan_dir_for_segment(
            run_dir,
            source_key=unit.joint_key,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=source_fps_key,
            upscale_factor=upscale_factor,
            crf=crf,
            preset=preset,
        )
        joint_drain_start = time.monotonic()
        joint_drained = drain_fn(joint_plan, out_fps=source_fps * multiplier)
        if timings is not None:
            timings["drain_s"] += time.monotonic() - joint_drain_start
            timings["chunks_drained"] += float(joint_drained.chunks_drained)
        joint_intermediates.append(joint_drained.intermediate_mp4)
    work_dir.mkdir(parents=True, exist_ok=True)
    final = work_dir / FINAL_INTERMEDIATE_FILENAME
    if not joint_units:
        concat_start = time.monotonic()
        concat_fn(segment_intermediates, final)
        if timings is not None:
            timings["concat_s"] += time.monotonic() - concat_start
    else:
        if assemble_fn is None:
            from voyage.augment_joints import assemble_joint_timeline

            assemble_fn = assemble_joint_timeline
        joint_cm = optional_stage(progress, "joints", f"{len(joint_units)} joint(s)")
        joint_start = time.monotonic()
        with joint_cm:
            assemble_kwargs: dict[str, Any] = {
                "source_counts": [source.total_frames for source in ordered],
                "multiplier": multiplier,
                "joint_root": run_dir / "augment" / "joint_timeline",
                "fps": int(round(source_fps * multiplier)),
                "crf": crf,
                "preset": preset,
                "pix_fmt": "yuv420p",
                "concat_fn": concat_fn,
            }
            if chunk_frames is not None and _accepts_keyword(assemble_fn, "chunk_frames"):
                assemble_kwargs["chunk_frames"] = chunk_frames
            final = assemble_fn(
                segment_intermediates,
                joint_intermediates,
                **assemble_kwargs,
            )
        if timings is not None:
            timings["morph_s"] += time.monotonic() - joint_start
            timings["morphs_done"] += float(len(joint_units))
    if not final.exists() or final.stat().st_size == 0:
        raise MediaError(f"durable model pass produced empty output {final}")
    return (final, round(source_fps * multiplier))

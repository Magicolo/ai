"""GPU augment runner orchestration: chunked Real-ESRGAN upscale + FILM interpolate.

Track D spike; DESIGN §§56-57, §140 finalize-augmentation-floors as-built.

Pure orchestration — stdlib only, never torch (supervisor section 12 GPU
ban): chunk math, ffmpeg chunk decode/encode (arg-lists, verified outputs),
device selection, and chunk-level parallelism.

Reuse check (2026-09-29): `voyage.media` has `plan_augmentation`, but it
computes presentation geometry/fps floors (Track B: out box, minterpolate
vs plain fps) — a different concern from chunk windows and `(n-1)*m+1`
frame counts, so there is nothing to import here. Single-home rule
(issue 083, resolved): `interpolated_frame_count` + the CRF ladder live
here; `voyage.media` re-exports both (`media.interpolated_frame_count`
is this function, `FINALIZE_CRF_*` alias `CRF_*`) instead of duplicating
them — the old TODO to move frame-count math into `media.py` is closed.

Parallelism contract (SFX pairing, DESIGN §140 GPU defaults): the
model-pass chunks are pinned to cuda:1 via `model_pass_devices` while
the MMAudio SFX stack (when present) renders on cuda:0 — the two stages
share nothing but chunk boundaries, so a 2-GPU box runs them side by
side and a 1-GPU box runs both serially on cuda:0. `augment_plan`
stamps each chunk with its device round-robin so the pairing is visible
in the plan; `run_augment_chunks` owns the serial-vs-ThreadPoolExecutor(2)
switch, warming the first chunk serially so resident model caches
populate before threads spawn (issue 157).
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from voyage.errors import MediaError

DEFAULT_CHUNK_FRAMES = 32
"""Source frames per chunk: mirrors VHS_BatchManager `frames_per_batch` (video_export node 10)."""

DEFAULT_INTERP_MULTIPLIER = 4
"""FILM interpolation multiplier: matches the 4x video-export recipe."""

DEFAULT_UPSCALE_FACTOR = 2
"""Presentation upscale factor (4x model + Lanczos 0.5 downscale = the 2x recipe)."""

AUGMENT_DEVICE_PRIMARY = "cuda:0"
"""Video-augment device: always used, and the only device on a 1-GPU box."""

AUGMENT_DEVICE_SECONDARY = "cuda:1"
"""Model-pass device on a 2-GPU box (DESIGN §140 GPU defaults): the 2060
the Real-ESRGAN + FILM pass is pinned to, leaving cuda:0 to the SFX stack."""

MAX_PARALLEL_DEVICES = 2
"""Chunk fan-out cap: one worker per device (the SFX pairing needs no more)."""

CRF_MINIMUM = 0
"""Best-quality h264 CRF bound for chunk encodes."""

CRF_MAXIMUM = 51
"""Worst-quality h264 CRF bound for chunk encodes."""

CHUNK_CRF_DEFAULT = 15
"""Default chunk-encode quality: the 2x video-export recipe's CRF."""

CHUNK_PRESET_DEFAULT = "veryfast"
"""Default x264 preset for chunk encodes (issue 157): matches the
intermediates' recipe (`media.FINALIZE_PRESET_DEFAULT`). Every chunk
encodes at this preset unless overridden."""

CHUNK_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)
"""Allowed x264 presets for chunk encodes (issue 157).

Mirrors `media.FINALIZE_PRESETS` entry-for-entry (pinned by
`test_chunk_preset_vocabulary_mirrors_finalize`) — one vocabulary, two
homes, kept apart only because `media` already imports this module
(`CRF_*`, `interpolated_frame_count`), so reusing
`media.validate_preset` here would cycle the import.
"""

T = TypeVar("T")
"""Outcome type of the per-chunk worker passed to `run_augment_chunks`."""

_PNG_IO_WORKERS = 8
"""Thread pool width for the PNG bridge (issue: RIFE finalize PNG-bound).

The finalize model pass shuttles every frame through PNG files (ffmpeg
decode -> PIL load -> torch -> PIL save -> ffmpeg encode); at 2048x1152
the single-thread PIL codec dominates chunk wall (~85-90%), so both
bridge directions run on a pool. Capped per call to the frame count.
"""

_PNG_COMPRESS_LEVEL = 1
"""zlib level for intermediate PNGs (issue: RIFE finalize PNG-bound).

Level 1 saves ~2x faster than the default 6 at ~14% larger files
(measured 0.288 -> 0.140 s/frame at 2048x1152); intermediates are
transient (pruned after drain), so size trades for speed. Still
lossless — pixel-identical to level 6.
"""


def _require_count(name: str, value: int, minimum: int) -> int:
    """Validate an integer count: ints only (bools rejected), at least `minimum`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (got {type(value).__name__})")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value})")
    return value


def interpolated_frame_count(source_frames: int, multiplier: int) -> int:
    """Frames after FILM interpolation: `(n-1)*m+1` (a single frame passes through)."""
    sources = _require_count("source_frames", source_frames, 1)
    factor = _require_count("multiplier", multiplier, 1)
    return (sources - 1) * factor + 1


def chunk_windows(
    total_frames: int, chunk: int = DEFAULT_CHUNK_FRAMES
) -> Iterator[tuple[int, int]]:
    """Yield `(start, count)` source-frame windows tiling `[0, total_frames)` contiguously."""
    total = _require_count("total_frames", total_frames, 0)
    size = _require_count("chunk", chunk, 1)
    for start in range(0, total, size):
        yield (start, min(size, total - start))


def chunk_frames_match_size(output_dir: Path, expected_size: tuple[int, int]) -> bool:
    """True when every PNG frame in `output_dir` has `expected_size` (w, h).

    Geometry forensics (kaolin, 2026-10-06): donor adoption once propagated
    a mixed-geometry `upscaled_NN/` dir (some frames 1024x576, rest
    2048x1152) past every count-only output-truth gate, failing loud only
    later in the tensor load bridge. All ledger output-truth checks call
    this alongside their count checks. Header-only PIL reads (no pixel
    decode). Fail-OPEN (True) when PIL is unavailable (slim image has no
    PIL — keeps unit tests and fakes unaffected); fail-closed (False) on
    unreadable/unparseable PNGs, empty dirs, or a first mismatch.
    """
    try:
        import importlib

        Image = importlib.import_module("PIL.Image")
    except ImportError:
        return True
    try:
        frame_paths = sorted(output_dir.glob("frame_*.png"))
    except OSError:
        return False
    if not frame_paths:
        return False
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as image:
                size = image.size
        except OSError:
            return False
        if (int(size[0]), int(size[1])) != (int(expected_size[0]), int(expected_size[1])):
            return False
    return True


@dataclass(frozen=True)
class AugmentChunk:
    """One augment unit: source window + expected output frames + assigned device."""

    index: int
    start_frame: int
    source_frames: int
    expected_frames: int
    device: str


def augment_plan(
    total_frames: int,
    *,
    chunk: int = DEFAULT_CHUNK_FRAMES,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    devices: tuple[str, ...] | None = None,
) -> list[AugmentChunk]:
    """Plan chunked augmentation: windows, per-chunk output counts, round-robin devices.

    Chunk outputs do NOT sum to the unchunked `(n-1)*m+1` total: each chunk
    interpolates independently, so one boundary pair per chunk joint is
    skipped (the same trade-off VHS_BatchManager documents). Callers that
    need exact end-to-end counts must use the unchunked formula.
    """
    _require_count("multiplier", multiplier, 1)
    resolved = augment_devices() if devices is None else devices
    if not resolved:
        raise ValueError("augment_plan needs at least one device (got none)")
    plan: list[AugmentChunk] = []
    for index, (start, count) in enumerate(chunk_windows(total_frames, chunk)):
        plan.append(
            AugmentChunk(
                index=index,
                start_frame=start,
                source_frames=count,
                expected_frames=interpolated_frame_count(count, multiplier),
                device=resolved[index % len(resolved)],
            )
        )
    return plan


def run_capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg-style argv (arg-list, never shell); mirrors `media.run_capture`."""
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def validate_chunk_preset(value: str) -> str:
    """Validate a chunk-encode x264 preset (issue 157).

    Same contract as `media.validate_preset` (str in the shared
    vocabulary); a local copy because `media` imports this module.
    Returns the value so call sites read `preset=validate_chunk_preset(preset)`.
    """
    if not isinstance(value, str):
        raise TypeError(f"preset must be a str (got {value!r})")
    if value not in CHUNK_PRESETS:
        raise ValueError(f"preset must be one of {sorted(CHUNK_PRESETS)} (got {value!r})")
    return value


def _require_fps(name: str, value: float | None) -> float | None:
    """Validate an optional fps override: finite positive number, bools rejected."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number (got {type(value).__name__})")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{name} must be finite and > 0 (got {value})")
    return rate


def ffmpeg_decode_chunk(
    source_video: Path,
    dest_dir: Path,
    start_frame: int,
    frame_count: int,
    *,
    fps: float | None = None,
) -> list[Path]:
    """Decode one source window to PNG frames via a frame-accurate select filter.

    `dest_dir` must be a fresh per-chunk directory: any pre-existing
    `frame_*.png` files raise before ffmpeg spawns (a stale dir would
    silently merge old frames into the chunk). After decoding, the glob
    must hold exactly `frame_count` frames (issue 192) — over/under-
    delivery (stale survivors, short source) raises instead of mixing
    silently into the chunk. With `fps` given and
    `start_frame > 0`, an input `-ss` fast-seek skips the already-decoded
    prefix and the select filter re-bases to `between(n,0,count-1)`; without
    `fps` (or at chunk 0) the exact from-start select path is preserved.
    """
    start = _require_count("start_frame", start_frame, 0)
    count = _require_count("frame_count", frame_count, 1)
    rate = _require_fps("fps", fps)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(dest_dir.glob("frame_*.png"))
    if stale:
        raise MediaError(
            f"chunk dest_dir not fresh: {dest_dir} already holds "
            f"{len(stale)} frame_*.png file(s) (pass a fresh per-chunk directory)"
        )
    if rate is not None and start > 0:
        end = count - 1
        argv = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{start / rate:.6f}",
            "-i",
            str(source_video),
            "-vf",
            f"select='between(n\\,0\\,{end})',setpts=N/FRAME_RATE/TB",
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(dest_dir / "frame_%06d.png"),
        ]
    else:
        end = start + count - 1
        argv = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source_video),
            "-vf",
            f"select='between(n\\,{start}\\,{end})',setpts=N/FRAME_RATE/TB",
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(dest_dir / "frame_%06d.png"),
        ]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"chunk decode failed for frames {start}-{end}: {proc.stderr[-2000:]}")
    frames = sorted(dest_dir.glob("frame_*.png"))
    if not frames:
        raise MediaError(f"chunk decode produced no frames for {start}-{end}")
    for frame in frames:
        if frame.stat().st_size == 0:
            raise MediaError(f"chunk decode produced empty frame {frame}")
    if len(frames) != count:
        raise MediaError(
            f"chunk decode produced {len(frames)} frames for {start}-{end} "
            f"(expected {count}): stale files or a short source must fail loud, "
            "never mix silently into the chunk"
        )
    return frames


def ffmpeg_encode_chunk(
    frames_pattern: Path,
    dest: Path,
    fps: int,
    *,
    crf: int = CHUNK_CRF_DEFAULT,
    preset: str = CHUNK_PRESET_DEFAULT,
) -> Path:
    """Encode a chunk's PNG sequence (`frame_%06d.png` pattern) to h264.

    `preset` threads the x264 speed/quality trade-off per chunk (issue
    157, default `veryfast` to match the intermediates); it rides the
    argv as `-preset` and is recorded in the chunk metric by callers.
    """
    rate = _require_count("fps", fps, 1)
    quality = _require_count("crf", crf, CRF_MINIMUM)
    if quality > CRF_MAXIMUM:
        raise ValueError(f"crf must be <= {CRF_MAXIMUM} (got {quality})")
    speed = validate_chunk_preset(preset)
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-framerate",
        str(rate),
        "-i",
        str(frames_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(quality),
        "-preset",
        speed,
        str(dest),
    ]
    proc = run_capture(argv)
    if proc.returncode != 0:
        raise MediaError(f"chunk encode failed for {dest}: {proc.stderr[-2000:]}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"chunk encode produced empty output {dest}")
    return dest


def cuda_visible_device_count(env: Mapping[str, str] | None = None) -> int | None:
    """Entries in CUDA_VISIBLE_DEVICES, or None when unset (caller falls back to nvidia-smi)."""
    source = os.environ if env is None else env
    raw = source.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return sum(1 for entry in raw.split(",") if entry.strip() != "")


def nvidia_smi_device_count() -> int:
    """GPUs listed by `nvidia-smi -L`; 0 when the binary is missing or fails."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))


def augment_devices(
    *,
    env: Mapping[str, str] | None = None,
    smi_count: int | None = None,
) -> tuple[str, ...]:
    """Devices for chunk work: `(cuda:0,)` by default, plus `cuda:1` when visible.

    Visibility prefers CUDA_VISIBLE_DEVICES, then the `smi_count` test
    seam, then a live `nvidia-smi -L` probe. Unknown/zero visibility still
    yields `(cuda:0,)` — the caller gates on real CUDA — except an
    explicitly emptied CUDA_VISIBLE_DEVICES, which honors the admin's
    hide-GPU intent by yielding `()` (caller skips augment). Capped at two
    devices: the SFX pairing never needs more.
    """
    explicit = cuda_visible_device_count(env)
    if explicit is not None:
        if explicit == 0:
            return ()
        count = explicit
    elif smi_count is not None:
        count = smi_count
    else:
        count = nvidia_smi_device_count()
    if count >= MAX_PARALLEL_DEVICES:
        return (AUGMENT_DEVICE_PRIMARY, AUGMENT_DEVICE_SECONDARY)
    return (AUGMENT_DEVICE_PRIMARY,)


def model_pass_devices(
    *,
    devices: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Devices for the finalize model pass: cuda:1 alone when two GPUs show.

    DESIGN §140 GPU defaults: the Real-ESRGAN + FILM pass is pinned to
    the 2060 (cuda:1) so the MMAudio SFX stack owns the 4060 (cuda:0) —
    the two finalize stages share nothing, so a 2-GPU box runs them side
    by side. One visible GPU (or an admin-hidden GPU set) keeps the
    `augment_devices` selection untouched (cuda:0, or empty). The
    `devices` seam takes an explicit visibility tuple so tests pin the
    branch without a GPU; `None` probes live visibility.
    """
    visible = augment_devices() if devices is None else devices
    if len(visible) >= MAX_PARALLEL_DEVICES:
        return (AUGMENT_DEVICE_SECONDARY,)
    return visible


def upscale_pass_devices(
    *,
    devices: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Devices for the finalize upscale leg: cuda:1 alone when two GPUs show.

    DESIGN §140 A/V stream: the 2060 (cuda:1) owns upscaling only, so the
    SRVGG leg never shares a card with the 4060 (cuda:0) stream
    (music takes -> SFX bed -> FILM interp, sequential). One visible GPU
    (or an admin-hidden GPU set) keeps the `augment_devices` selection
    untouched (cuda:0, or empty). The `devices` seam takes an explicit
    visibility tuple so tests pin the branch without a GPU.
    """
    visible = augment_devices() if devices is None else devices
    if len(visible) >= MAX_PARALLEL_DEVICES:
        return (AUGMENT_DEVICE_SECONDARY,)
    return visible


def interp_pass_devices(
    *,
    devices: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Devices for the finalize interp leg: cuda:1 alone when two GPUs show.

    DESIGN §140 RIFE era: RIFE interpolation joins the 2060 (cuda:1) stream
    after the upscale leg, picking up the ledgered upscale chunks the same
    card published. At the 2048x1152 working size RIFE peaks at ~0.65 GiB,
    so it shares the 6 GB card with the resident llama director sidecar
    (~3.2 GiB) and the staggered SRVGG pass. (The FILM era needed ~5.9 GiB
    alone, which is why this used to pin the 16 GB primary card.) One
    visible GPU keeps the `augment_devices` selection untouched (cuda:0,
    or empty). The `devices` seam takes an explicit visibility tuple so
    tests pin the branch without a GPU.
    """
    visible = augment_devices() if devices is None else devices
    if len(visible) >= MAX_PARALLEL_DEVICES:
        return (AUGMENT_DEVICE_SECONDARY,)
    return visible


def run_augment_chunks(
    chunks: list[AugmentChunk],
    worker: Callable[[AugmentChunk, str], T],
) -> list[T]:
    """Run chunks on their planned devices, preserving chunk order in the outcomes.

    One unique device → serial loop; two → the first chunk runs serially
    (warming any resident model cache keyed by weights+device), then the
    remainder fans out over ThreadPoolExecutor(2), one thread per device
    (the SFX pairing's cuda:0/cuda:1 split). Without the warm-first gate
    every thread misses the cache at once and each pays a full load peak
    (issue 157); a lone failing first chunk fails fast before threads
    spawn. Executor.map keeps outcome order identical to chunk order
    either way, so callers can concatenate chunk outputs directly.
    """
    devices = list(dict.fromkeys(chunk.device for chunk in chunks))
    if len(devices) <= 1:
        return [worker(chunk, chunk.device) for chunk in chunks]
    first = worker(chunks[0], chunks[0].device)
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_DEVICES, len(devices))) as pool:
        rest = list(pool.map(lambda chunk: worker(chunk, chunk.device), chunks[1:]))
    return [first, *rest]


@dataclass(frozen=True)
class AugmentWeights:
    """Resolved model-pass weight paths for one finalize (issue 166).

    DESIGN §§56-57: the production seam between the registry (pinned
    RIFE/FILM + Real-ESRGAN weights) and the torch loaders in
    `voyage.workers.augment_worker`. Each leg is a loader-ready path or
    `None` when the weights are absent — the caller keeps the ffmpeg
    fallback then (default-off unless provisioned), so a missing stack
    reads as "skip the model pass", never as an error. The interp leg
    is selected per backend via `interp_leg` (`film` = legacy FILM,
    `rife` = default RIFE v4.25).
    """

    film: Path | None
    realesrgan: Path | None
    rife: Path | None = None

    def interp_leg(self, backend: str) -> Path | None:
        """Return the active interp-leg path for `backend` (torch-free)."""
        from voyage.workers import augment_worker

        augment_worker.validate_interp_backend(backend)
        return self.film if backend == "film" else self.rife


def interp_leg_path(weights: AugmentWeights, backend: str = "rife") -> Path | None:
    """Return the active interp-leg path, tolerating legacy fakes (torch-free).

    Production `AugmentWeights` carry `interp_leg`; older test fakes expose
    only `film`/`realesrgan` — those read as film-backend weights so existing
    single-backend tests keep passing unmodified.
    """
    getter = getattr(weights, "interp_leg", None)
    if callable(getter):
        leg = getter(backend)
        if leg is None or isinstance(leg, Path):
            return leg
        raise TypeError(f"interp_leg() must return a Path or None (got {type(leg).__name__})")
    return weights.film


def resolve_augment_weights(models_dir: Path | str) -> AugmentWeights:
    """Map `models_dir` to loader-ready augment weight paths (torch-free).

    DESIGN §§56-57 (issue 166): a leg resolves only when its file exists
    with at least the registry floor bytes — zero-byte (torn) and
    truncated downloads read as not provisioned, mirroring
    `augment_worker._require_weights`. Never imports torch (supervisor
    section 12 GPU ban holds at module scope — the registry pins are
    stdlib-only, imported lazily so this module's top level stays so);
    never raises for absent weights. The model-pass chunk worker passes
    these paths to `augment_worker.upscale_frames` / `interpolate_mids`;
    `finalize_run` itself is untouched (ffmpeg path stays the default —
    the inference wiring is a later slice, see the issue handoff).
    """
    if isinstance(models_dir, str):
        base = Path(models_dir)
    elif isinstance(models_dir, Path):
        base = models_dir
    else:
        raise TypeError(f"models_dir must be a Path or str (got {type(models_dir).__name__})")

    from voyage.registry_film import FILM_MIN_BYTES, FILM_REPO_PATH
    from voyage.registry_realesrgan import (
        REALESRGAN_ANIME_FILE,
        REALESRGAN_ANIME_MIN_BYTES,
        REALESRGAN_SUBDIR,
    )
    from voyage.registry_rife import RIFE_MIN_BYTES, RIFE_REPO_PATH

    def _present(relative: str, floor_bytes: int) -> Path | None:
        candidate = base / relative
        try:
            if candidate.is_file() and candidate.stat().st_size >= floor_bytes:
                return candidate
        except OSError:
            return None
        return None

    return AugmentWeights(
        film=_present(FILM_REPO_PATH, FILM_MIN_BYTES),
        realesrgan=_present(
            f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}", REALESRGAN_ANIME_MIN_BYTES
        ),
        rife=_present(RIFE_REPO_PATH, RIFE_MIN_BYTES),
    )


_MODEL_UPSCALE_FACTORS = (1, 2, 4)
"""Targets servable from one x4 Real-ESRGAN pass (mirrors the worker vocabulary torch-free)."""


def _require_upscale_factor(upscale_factor: int) -> int:
    """Validate the presentation upscale target torch-free (1, 2, or 4 from one x4 pass)."""
    if isinstance(upscale_factor, bool) or not isinstance(upscale_factor, int):
        raise TypeError(f"upscale_factor must be an int (got {type(upscale_factor).__name__})")
    if upscale_factor not in _MODEL_UPSCALE_FACTORS:
        raise ValueError(
            f"upscale_factor must be one of {_MODEL_UPSCALE_FACTORS} (got {upscale_factor})"
        )
    return upscale_factor


def _require_interp_multiplier(multiplier: int) -> int:
    """Validate the interpolation multiplier: ints only, at least 1 (1 = no mids)."""
    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        raise TypeError(f"multiplier must be an int (got {type(multiplier).__name__})")
    if multiplier < 1:
        raise ValueError(f"multiplier must be >= 1 (got {multiplier})")
    return multiplier


def enhance_frames(
    frames: list[Any],
    weights: AugmentWeights,
    *,
    device: str,
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    interp_backend: str = "rife",
) -> list[Any]:
    """Upscale then interpolate one chunk's frames via the provisioned legs (issue 166).

    DESIGN §§56-57: the chunk-scale inference seam — `resolve_augment_weights`
    provides the legs, this function consumes them. Callers only invoke
    this when legs are present and work is demanded (upscale > 1 or
    interpolate > 1); with no provisioned leg it returns the input frames
    unchanged without importing torch. Each non-None leg runs on `device`
    device — the cuda:0/cuda:1 SFX pairing): Real-ESRGAN upscales first (the
    2x video-export recipe), then the `interp_backend` engine interpolates
    `(multiplier - 1)` mids per adjacent pair at evenly spaced moments
    (`multiplier=4` gives 0.25/0.5/0.75, matching `(n-1)*m+1`). On the
    `film` backend, chunks whose device reports under 8 GiB free reverse
    the legs (interp at 1x first — FILM at the upscaled size would OOM
    small GPUs; DESIGN §140 GPU defaults); the `rife` backend never
    reverses (0.65 GiB peak at 2048x1152 fits everywhere). Absent legs skip
    (same frames out), so a half-provisioned stack still runs the
    available leg.
    """
    if not isinstance(weights, AugmentWeights):
        raise TypeError(f"weights must be AugmentWeights (got {type(weights).__name__})")
    if not isinstance(device, str) or not device:
        raise TypeError(f"device must be a non-empty str (got {device!r})")
    target_scale = _require_upscale_factor(upscale_factor)
    interp_factor = _require_interp_multiplier(multiplier)
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"enhance_frames needs at least one frame (got {frames!r})")
    from voyage.workers import augment_worker

    augment_worker.validate_interp_backend(interp_backend)
    interp_weights = weights.interp_leg(interp_backend)
    if interp_weights is None and weights.realesrgan is None:
        return list(frames)

    esrgan_weights = weights.realesrgan

    def _run_upscale(source: list[Any]) -> list[Any]:
        if esrgan_weights is None:
            return source
        return augment_worker.upscale_frames(
            source, esrgan_weights, scale=target_scale, device=device
        )

    def _run_interp(source: list[Any]) -> list[Any]:
        if interp_weights is None or len(source) <= 1 or interp_factor <= 1:
            return source
        moments = [(position + 1) / interp_factor for position in range(interp_factor - 1)]
        if interp_backend == "film":
            mids = augment_worker.interpolate_mids(
                source, interp_weights, moments=moments, device=device
            )
        else:
            mids = augment_worker.interpolate_rife_mids(
                source, interp_weights, moments=moments, device=device
            )
        blended: list[Any] = []
        step = len(moments)
        for position in range(len(source) - 1):
            blended.append(source[position])
            blended.extend(mids[position * step : (position + 1) * step])
        blended.append(source[-1])
        return blended

    working: list[Any] = list(frames)
    both_legs = (
        esrgan_weights is not None
        and interp_weights is not None
        and len(working) > 1
        and interp_factor > 1
    )
    if (
        both_legs
        and interp_backend == "film"
        and augment_worker.interp_first_for_small_device(device)
    ):
        # Small GPU on the FILM backend (DESIGN §140 GPU defaults): FILM
        # pairs at the upscaled size would OOM (measured ~5.9 GiB at
        # 2432x1408 on the 6 GB 2060), so chunks interpolate at 1x first
        # and upscale after (the established interp-then-upscale pipeline;
        # the tiled upscale leg keeps every frame servable).
        working = _run_upscale(_run_interp(working))
    else:
        working = _run_interp(_run_upscale(working))
    return working


def make_enhance_chunk_worker(
    source_frames: Mapping[int, list[Any]],
    weights: AugmentWeights,
    *,
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    interp_backend: str = "rife",
) -> Callable[[AugmentChunk, str], list[Any]]:
    """Build the `run_augment_chunks` worker that enhances one chunk's frames.

    The worker closes over `source_frames` (chunk index to its source frames)
    and `weights`; on each call it enhances `source_frames[chunk.index]` with
    `device=chunk.device` (never the passed-through string — they match by
    construction, but the chunk plan is the pairing contract). Missing chunk
    indices fail loud with `KeyError` so a mis-staged plan never encodes
    silence. Torch-free until a leg is present (the enhance path lazy-imports
    the worker then).
    """
    if not isinstance(source_frames, Mapping):
        raise TypeError(f"source_frames must be a mapping (got {type(source_frames).__name__})")
    if not isinstance(weights, AugmentWeights):
        raise TypeError(f"weights must be AugmentWeights (got {type(weights).__name__})")
    target_scale = _require_upscale_factor(upscale_factor)
    interp_factor = _require_interp_multiplier(multiplier)

    def _worker(chunk: AugmentChunk, worker_device: str) -> list[Any]:
        del worker_device
        if not isinstance(chunk, AugmentChunk):
            raise TypeError(f"chunk must be AugmentChunk (got {type(chunk).__name__})")
        try:
            chunk_source = source_frames[chunk.index]
        except KeyError as exc:
            raise KeyError(
                f"no source frames for chunk {chunk.index} (have {sorted(source_frames)})"
            ) from exc
        return enhance_frames(
            list(chunk_source),
            weights,
            device=chunk.device,
            upscale_factor=target_scale,
            multiplier=interp_factor,
            interp_backend=interp_backend,
        )

    return _worker


def run_model_augment_chunks(
    chunks: list[AugmentChunk],
    weights: AugmentWeights,
    source_frames: Mapping[int, list[Any]],
    *,
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    interp_backend: str = "rife",
) -> list[list[Any]]:
    """Run chunk enhancement through `run_augment_chunks`, preserving chunk order.

    Runs the provisioned legs per chunk on their planned devices. Empty
    plans return `[]` without touching the runner; callers only invoke
    this when work is demanded (upscale > 1 or interpolate > 1).
    """
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list (got {type(chunks).__name__})")
    if not isinstance(weights, AugmentWeights):
        raise TypeError(f"weights must be AugmentWeights (got {type(weights).__name__})")
    target_scale = _require_upscale_factor(upscale_factor)
    interp_factor = _require_interp_multiplier(multiplier)
    if not chunks:
        return []
    worker = make_enhance_chunk_worker(
        source_frames,
        weights,
        upscale_factor=target_scale,
        multiplier=interp_factor,
        interp_backend=interp_backend,
    )
    return run_augment_chunks(chunks, worker)


def model_pass_active(weights: AugmentWeights) -> bool:
    """Whether provisioned legs select the tensor chunk encode (issue 166).

    DESIGN §§56-57: pure torch-free selection — True when at least one
    leg resolved (half-provisioned still runs the available leg via
    `enhance_frames`; both absent keeps the ffmpeg fallback byte-identical).
    `finalize_run` consults this after `resolve_augment_weights`; False
    never touches torch.
    """
    if not isinstance(weights, AugmentWeights):
        raise TypeError(f"weights must be AugmentWeights (got {type(weights).__name__})")
    film_present = weights.film is not None
    rife_present = getattr(weights, "rife", None) is not None
    return bool(film_present or rife_present or weights.realesrgan is not None)


def _png_io_workers(count: int) -> int:
    """Thread pool width for the PNG bridge (never more workers than frames)."""
    return max(1, min(_PNG_IO_WORKERS, count))


def load_png_frames_as_tensors(frame_paths: list[Path]) -> list[Any]:
    """Read decoded PNG frames into torch float tensors in [0, 1] (issue 166).

    DESIGN §§56-57: the finalize model pass decodes segment videos to PNGs
    via ffmpeg (stdlib side), then enhances tensors via `enhance_frames`
    (torch side) — this is the bridge. Lazy PIL/numpy/torch imports so the
    module stays stdlib-only until the tensor path is selected (the ffmpeg
    fallback never imports them). All frames must share dimensions; mismatch
    fails loud instead of mixing silently into a chunk. Decodes run on a
    thread pool: the RIFE finalize is PNG-codec-bound (~85-90% of chunk
    wall), and PIL decode releases the GIL well enough to scale.
    """
    if not isinstance(frame_paths, list) or not frame_paths:
        raise ValueError(f"frame_paths needs at least one PNG path (got {frame_paths!r})")
    for frame_path in frame_paths:
        if not isinstance(frame_path, Path):
            raise TypeError(f"frame path must be a Path (got {type(frame_path).__name__})")
    try:
        import importlib

        Image = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise MediaError(f"model pass needs Pillow for PNG frames ({exc})") from exc
    try:
        import torch
    except ImportError as exc:
        raise MediaError(f"model pass needs torch for tensors ({exc})") from exc
    import numpy

    def _decode(frame_path: Path) -> Any:
        with Image.open(frame_path) as opened:
            converted = opened.convert("RGB")
            width, height = converted.size
            raw = converted.tobytes()
        flat = numpy.frombuffer(raw, dtype=numpy.uint8)
        try:
            shaped = flat.reshape((height, width, 3)).copy()
        except ValueError as exc:
            raise MediaError(f"cannot reshape PNG {frame_path} to RGB ({exc})") from exc
        return torch.from_numpy(shaped).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)

    with ThreadPoolExecutor(max_workers=_png_io_workers(len(frame_paths))) as pool:
        tensors = list(pool.map(_decode, frame_paths))
    expected_size: tuple[int, int] | None = None
    for frame_path, tensor in zip(frame_paths, tensors, strict=True):
        height, width = int(tensor.shape[1]), int(tensor.shape[2])
        if expected_size is None:
            expected_size = (width, height)
        elif (width, height) != expected_size:
            raise MediaError(
                f"frame size mismatch: {frame_path} is {(width, height)}, "
                f"expected {expected_size} (chunks need uniform geometry)"
            )
    return tensors


def write_tensors_as_png_frames(frames: list[Any], dest_dir: Path) -> list[Path]:
    """Write enhanced float tensors in [0, 1] back to PNG frames (issue 166).

    DESIGN §§56-57: the return bridge — `enhance_frames` yields CPU float32
    `(3, H, W)` tensors, the chunk encoder needs PNGs. Lazy PIL/numpy/torch
    (same stdlib-only rule as the load bridge); clamps to [0, 1] like the
    worker's native scale-4 path so bicubic-downscaled legs cannot ring past
    the range. Returns the written paths in order (`frame_%06d.png`).
    Saves run on a thread pool at fast zlib level (same PNG-bound issue
    as the load bridge — the pool + level 1 cut the save phase ~8-11x);
    `pool.map` preserves order and surfaces the first bad frame's
    `MediaError` at its position, matching the old serial semantics.
    """
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"frames needs at least one tensor (got {frames!r})")
    if not isinstance(dest_dir, Path):
        raise TypeError(f"dest_dir must be a Path (got {type(dest_dir).__name__})")
    try:
        import importlib

        Image = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise MediaError(f"model pass needs Pillow for PNG frames ({exc})") from exc
    try:
        import torch
    except ImportError as exc:
        raise MediaError(f"model pass needs torch for tensors ({exc})") from exc
    import numpy

    dest_dir.mkdir(parents=True, exist_ok=True)

    def _save(position_frame: tuple[int, Any]) -> Path:
        position, frame = position_frame
        tensor = torch.as_tensor(frame, dtype=torch.float32).clamp(0.0, 1.0)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise MediaError(
                f"enhanced frame {position} must be (3, H, W) (got shape {tuple(tensor.shape)})"
            )
        array = tensor.permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
        if not isinstance(array, numpy.ndarray):
            raise MediaError(f"enhanced frame {position} did not render to an array")
        height, width, _ = array.shape
        image = Image.frombytes("RGB", (width, height), array.tobytes())
        dest = dest_dir / f"frame_{position:06d}.png"
        image.save(dest, compress_level=_PNG_COMPRESS_LEVEL)
        if not dest.exists() or dest.stat().st_size == 0:
            raise MediaError(f"enhanced PNG write produced empty output {dest}")
        return dest

    with ThreadPoolExecutor(max_workers=_png_io_workers(len(frames))) as pool:
        return list(pool.map(_save, enumerate(frames)))


def _write_chunk_concat_list(entries: list[Path], dest: Path) -> Path:
    """Write a concat-demuxer list for chunk mp4s (issue 166).

    Mirrors `media.write_concat_list` entry-for-entry (single-quote escaping)
    without importing `media` — `media` already imports this module for the
    CRF ladder, so reusing it here would cycle the import (same reason the
    preset vocabulary is mirrored in `CHUNK_PRESETS`).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        f"file '{str(entry).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
        for entry in entries
    )
    dest.write_text(lines, encoding="utf-8")
    return dest


def run_finalize_model_pass(
    segment_videos: list[Path],
    weights: AugmentWeights,
    *,
    source_fps: float,
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR,
    multiplier: int = DEFAULT_INTERP_MULTIPLIER,
    interp_backend: str = "rife",
    crf: int = CHUNK_CRF_DEFAULT,
    preset: str = CHUNK_PRESET_DEFAULT,
    work_dir: Path,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    devices: tuple[str, ...] | None = None,
) -> tuple[Path, int]:
    """Decode segments, enhance tensors chunked, encode chunks, concat (issue 166).

    DESIGN §§56-57: the present-legs tensor path `finalize_run` selects via
    `model_pass_active` — decode (ffmpeg, per segment in order) to PNGs,
    tensors via the load bridge, `run_model_augment_chunks` (upscale via
    SRVGG + configured-backend mids on each chunk's device, OOM-halving
    preserved, VRAM-flat via `chunk_frames` windows), PNGs via the write
    bridge, `ffmpeg_encode_chunk` per chunk at `round(source_fps * multiplier)`,
    concat-demuxer stream copy to one intermediate. Returns the intermediate
    video + its fps; the caller applies the presentation vf
    (scale/pad/fps, no minterpolate — the backend already interpolated) so the
    shipped box still matches `plan_augmentation` exactly. Callers without
    work or legs never reach here (they keep the single vf encode).
    """
    if not isinstance(segment_videos, list) or not segment_videos:
        raise ValueError(f"segment_videos needs at least one video (got {segment_videos!r})")
    for segment_video in segment_videos:
        if not isinstance(segment_video, Path):
            raise TypeError(f"segment video must be a Path (got {type(segment_video).__name__})")
        if not segment_video.is_file():
            raise MediaError(f"segment video missing: {segment_video}")
    if not isinstance(weights, AugmentWeights):
        raise TypeError(f"weights must be AugmentWeights (got {type(weights).__name__})")
    if (
        weights.film is None
        and getattr(weights, "rife", None) is None
        and weights.realesrgan is None
    ):
        raise MediaError("model pass needs at least one provisioned leg (got none)")
    rate = _require_fps("source_fps", source_fps)
    if rate is None or rate <= 0.0:
        raise ValueError(f"source_fps must be a positive fps (got {source_fps!r})")
    target_scale = _require_upscale_factor(upscale_factor)
    interp_factor = _require_interp_multiplier(multiplier)
    quality = _require_count("crf", crf, CRF_MINIMUM)
    if quality > CRF_MAXIMUM:
        raise ValueError(f"crf must be <= {CRF_MAXIMUM} (got {quality})")
    speed = validate_chunk_preset(preset)
    if not isinstance(work_dir, Path):
        raise TypeError(f"work_dir must be a Path (got {type(work_dir).__name__})")
    window = _require_count("chunk_frames", chunk_frames, 1)
    resolved_devices = augment_devices() if devices is None else devices
    if not resolved_devices:
        raise MediaError("model pass needs at least one device (got none)")
    intermediate_fps = int(round(rate * interp_factor))
    if intermediate_fps < 1:
        raise ValueError(f"intermediate fps must be >= 1 (got {rate} * {interp_factor})")
    work_dir.mkdir(parents=True, exist_ok=True)

    decoded_paths: list[Path] = []
    for position, segment_video in enumerate(segment_videos):
        decode_dir = work_dir / f"decode_{position:02d}"
        decode_dir.mkdir(parents=True, exist_ok=True)
        proc = run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(segment_video),
                "-vsync",
                "0",
                str(decode_dir / "frame_%06d.png"),
            ]
        )
        if proc.returncode != 0:
            raise MediaError(f"model-pass decode failed for {segment_video}: {proc.stderr[-2000:]}")
        chunk_frames_found = sorted(decode_dir.glob("frame_*.png"))
        if not chunk_frames_found:
            raise MediaError(f"model-pass decode produced no frames for {segment_video}")
        decoded_paths.extend(chunk_frames_found)

    source_tensors = load_png_frames_as_tensors(decoded_paths)
    plan = augment_plan(
        len(source_tensors), chunk=window, multiplier=interp_factor, devices=resolved_devices
    )
    if not plan:
        raise MediaError("model pass planned zero chunks (no source frames)")
    source_by_chunk = {
        chunk.index: source_tensors[chunk.start_frame : chunk.start_frame + chunk.source_frames]
        for chunk in plan
    }
    enhanced_by_chunk = run_model_augment_chunks(
        plan,
        weights,
        source_by_chunk,
        upscale_factor=target_scale,
        multiplier=interp_factor,
        interp_backend=interp_backend,
    )
    chunk_videos: list[Path] = []
    for chunk, enhanced in zip(plan, enhanced_by_chunk, strict=True):
        enhanced_dir = work_dir / f"enhanced_{chunk.index:02d}"
        written = write_tensors_as_png_frames(list(enhanced), enhanced_dir)
        if len(written) != chunk.expected_frames:
            raise MediaError(
                f"model pass chunk {chunk.index} produced {len(written)} frames "
                f"(expected {chunk.expected_frames})"
            )
        chunk_video = work_dir / f"chunk_{chunk.index:02d}.mp4"
        ffmpeg_encode_chunk(
            enhanced_dir / "frame_%06d.png",
            chunk_video,
            intermediate_fps,
            crf=quality,
            preset=speed,
        )
        chunk_videos.append(chunk_video)
    concat_list = _write_chunk_concat_list(chunk_videos, work_dir / "chunks.txt")
    intermediate = work_dir / "model_intermediate.mp4"
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
            "-c",
            "copy",
            str(intermediate),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"model-pass chunk concat failed: {proc.stderr[-2000:]}")
    if not intermediate.exists() or intermediate.stat().st_size == 0:
        raise MediaError(f"model-pass concat produced empty output {intermediate}")
    return (intermediate, intermediate_fps)

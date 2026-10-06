"""GPU augment runner: Real-ESRGAN upscale + FILM interpolate (Track D spike, DESIGN §§56-57).

QUARANTINE (issue 083): this module grew out of a spike stand-in — the
orchestration (`voyage.augment`: chunk windows, ffmpeg chunk decode/encode,
device pairing) is the shipped path and never imports this module; treat any
vendored arch here as a placeholder until its upstream port lands (or this
module moves to `experimental/` with a spike contract). Debug augment quality
via the orchestration + registry weights first, not these vendored nets. The
ESRGAN leg is DONE (issue 166): the pinned `realesr-animevideov3.pth`
loads strict into the SRVGGNetCompact builder below (RRDB layouts stay
as fallback). The FILM leg is DONE
(issue 166): the pinned `film_net_fp16.safetensors` strict-loads into the
upstream FILM port below (extract/fuse/predict_flow, 82 keys).

`torch` loads only inside functions (behind a `find_spec` guard) — never
at module scope (supervisor section 12 GPU ban) — and weight checks run
before any torch import, so missing weights raise NotImplementedError
torch-free. The registry owns download/provisioning; this module never
fetches weights at runtime.

Architectures are vendored minimal inline — do NOT import Comfy nodes
(the worker images carry no ComfyUI tree):
- ESRGAN family: three layouts (issue 166) — SRVGGNetCompact PReLU
  weights (`body.<int>` + odd weight-only + 48-channel last conv; the
  pinned `realesr-animevideov3.pth` carries 16 body convs) build
  `_build_srvgg_net` at the measured depth, upstream-named RRDB weights
  (`body.N.rdb1/2/3` + `conv_up1/up2/hr`, EMA-wrapped) build
  `_build_upstream_rrdb_net` at the measured depth, while anything else
  falls back to the classic x4 residual-in-residual dense net below
  (state-dict shapes match the x4plus/UltraSharp ESRGAN family).
- FilmNet: plain-torch port of the upstream FILM graph (ECCV 2022,
  `extract` / `predict_flow` / `fuse` nesting reproduces the pinned
  `film_net_fp16.safetensors` key layout, strict-loaded). Call shape is
  the worker's own `(B, 2, C, H, W)` pairs + float moment so the
  `_run_stacked` OOM-halving loop is untouched; pyramid depth clamps to
  the input-feasible count (upstream runs 7 levels, needing sides
  >=64px), and sides below `FILM_MIN_SIDE` fail loud.

Precision is fp16 on CUDA, fp32 elsewhere. Batch inference starts full
and halves on out-of-memory down to single items, mirroring Comfy's
FrameInterpolate recipe. RPC ops are supervisor-track follow-up — this
module is an in-process library called with tensors.
"""

from __future__ import annotations

import gc
import importlib.util
import pickle
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voyage.errors import ModelCompatibilityError

RRDB_NUM_FEATURES = 64
"""RRDBNet width of the Real-ESRGAN x4 family (state-dict shapes must match)."""

RRDB_NUM_BLOCKS = 23
"""RRDBNet depth of the Real-ESRGAN x4 family (state-dict shapes must match)."""

RRDB_GROWTH_CHANNELS = 32
"""Dense-block growth rate of the Real-ESRGAN x4 family (shapes must match)."""

RRDB_NATIVE_SCALE = 4
"""Native upscale of the vendored RRDBNet (two x2 nearest stages)."""

RRDB_ANIME_NUM_BLOCKS = 6
"""Body depth of the legacy `RealESRGAN_x4plus_anime_6B` pth (`body.0`–`body.5`)."""

SRVGG_NUM_FEATURES = 64
"""SRVGGNetCompact width of the pinned anime-video-XS pth (shapes must match)."""

SRVGG_NATIVE_SCALE = 4
"""Native upscale of the SRVGGNetCompact leg (PixelShuffle x4 + residual base)."""

SRVGG_LAST_OUT_CHANNELS = 48
"""Last-conv output channels: num_out_ch 3 × upscale 4² (state-dict shape pin)."""

_UPSTREAM_UPCONV_KEYS = ("conv_up1", "conv_up2", "conv_hr")
"""Upconv names distinguishing upstream RRDB weights from the vendored classic net."""

_ESRGAN_WRAPPER_KEYS = ("params_ema", "params")
"""Outer keys of training-checkpoint pth files (the inference net hides one level down)."""

ALLOWED_UPSCALE_FACTORS = (1, 2, 4)
"""Targets served from one x4 pass (4 is native; 2/1 downscale the x4 output)."""

FILM_PYRAMID_LEVELS = 7
"""Upstream FILM depth: 7 image-pyramid levels (needs input sides >=64px)."""

FILM_FUSION_LEVELS = 5
"""Finest flow levels fused into the output (upstream default)."""

FILM_SPECIALIZED_LEVELS = 3
"""Fusion blocks with per-level widths (deeper blocks share one width)."""

FILM_SUB_LEVELS = 4
"""Feature-extractor sublevels per image level (upstream default)."""

FILM_FILTERS = 64
"""Base feature width (level widths double per sublevel: 64/128/256/512)."""

FILM_FLOW_CONVS = (3, 3, 3, 3)
"""Residual convs per flow predictor, coarse-to-fine (upstream default)."""

FILM_FLOW_FILTERS = (32, 64, 128, 256)
"""Flow predictor widths, finest-to-coarsest predictor (upstream default)."""

FILM_STATE_KEYS = 82
"""Pinned `film_net_fp16.safetensors` key count (extract/fuse/predict_flow)."""

FILM_MIN_SIDE = 8
"""Smallest frame side the FILM pyramid supports (4 levels down to 1px)."""

FILM_PAIR_BATCH = 1
"""Frame pairs per FILM forward in `interpolate_mids` (measured 2026-10-05 on
the idle 4060 Ti at 1216x704: batch 1 gives the full ~1.6x over the old
per-pair loop at base VRAM with bit-exact outputs — the win is
extract-once/flow-once per pair, not multi-pair batching. Batches 2/4 add
zero speed, cost 2-4x VRAM, and shift pixels deterministically (cudnn algo
selection per batch shape: mean abs 1.9e-4, 4.3% of pixels flip after PNG
rounding), so they stay opt-in via `pair_batch`, not the default)."""

FILM_CLASSIC_ORDER_MIN_FREE_BYTES = 8 * 1024**3
"""Chunk devices reporting less free VRAM than this interpolate at 1x first
(measured 2026-10-01 on the 6 GB 2060: FILM pairs at 2432x1408 need
~5.9 GiB and OOM where batch-halving bottoms out at one pair; the 8 GiB
line leaves margin over that peak, so 16 GB+ devices keep the validated
upscale-first recipe byte-for-byte)."""

UPSCALE_TILE_SIZE = 512
"""Spatial tile side for the tiled upscale path (a 512-side tile peaks ~1 GiB)."""

UPSCALE_TILE_OVERLAP = 64
"""Context margin around each upscale tile (cropped after upscale, so seams
never show — measured 2026-10-01: 32 leaves diffs up to 0.066 (the RRDB
receptive field reaches past it), 64 drops them to <= 0.0008 smooth and
1e-5 on white noise)."""

UPSCALE_TILE_BUDGET_PIXELS = 500_000
"""Frames above this pixel count upscale tiled (measured 2026-10-01 on the
6 GB 2060: 768x512 = 393,216 px peaks 3.29 GiB and fits, 1216x704 =
856,064 px needs ~5.5 GiB and OOMs in `conv_first` — batch-halving
bottoms out at one frame, so only spatial tiling saves it; every native
backend size at or below the budget keeps the direct path byte-for-byte)."""

_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    EOFError,
    pickle.UnpicklingError,
)
"""Weight-load failures meaning "weights unusable here" (issue 074).

Covers `torch.load` / `safetensors` decode / `load_state_dict` shape errors —
every one maps to `ModelCompatibilityError` at the loader boundary. Corrupt
pickles surface as `UnpicklingError` (a `PickleError`, outside the original
five); safetensors decode failures are normalized to `ValueError` in
`_load_state_dict` so they land here too.
"""

_ESRGAN_CACHE: dict[tuple[str, str], Any] = {}
"""Resident ESRGAN-family nets keyed by (weights path, device) — issue 047.

A 32-chunk augment must not pay 32x construction + disk load + H2D;
the first call warms the entry, later chunks reuse it. `evict_augment_models`
drops both caches (GPU hand-off, DESIGN §40).
"""

_FILM_CACHE: dict[tuple[str, str], Any] = {}
"""Resident FILM nets keyed by (weights path, device) — issue 047."""

_RIFE_CACHE: dict[tuple[str, str], Any] = {}
"""Resident RIFE nets keyed by (weights path, device) — same issue-047 rationale."""


def _model_cache_key(weights_path: Path, device: str) -> tuple[str, str]:
    """Cache identity for a resident net: stringified weights path + device."""
    return (str(weights_path), str(device))


_AUGMENT_MODEL_LOCK = threading.Lock()
"""Serializes resident-net load + prepare (torch best practice).

PyTorch modules are thread-safe to READ but not to WRITE (maintainer
albanD): sharing one net across inference threads is safe iff the forward
pass mutates no shared state. This lock covers the three writes —
cache check-then-set, `_prepare_model` (.half/.to/.eval), and the
warp-grid memo dicts — while steady-state forwards run lock-free.
Per-thread CUDA streams (the `cuda_stream` params) let the launches
overlap; kernel launches are async so Python threads share well.
"""

_WARP_GRID_LOCK = threading.Lock()
"""Serializes warp-grid memo builds (FILM + RIFE nets own one dict each)."""

_PREPARED_MODEL_KEYS: set[tuple[str, str]] = set()
"""Cache keys already moved/cast/eval'd — later calls skip the write path."""


def _get_prepared_model(
    cache: dict[tuple[str, str], Any],
    key: tuple[str, str],
    device: str,
    loader: Callable[[], Any],
) -> tuple[Any, Any, Any]:
    """Locked get-or-load + prepare-once; returns (model, torch_device, dtype).

    The loader (disk decode, seconds) runs under the lock so concurrent
    first calls wait instead of double-loading; `_prepare_model` runs at
    most once per key, so steady-state calls never `.half()`/`.to()`/`.eval()`
    under a live forward (torn precision / training-flag flips).
    """
    with _AUGMENT_MODEL_LOCK:
        model = cache.get(key)
        if model is None:
            model = loader()
            cache[key] = model
        if key in _PREPARED_MODEL_KEYS:
            import torch

            torch_device = _resolve_device(device)
            dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        else:
            torch_device, dtype = _prepare_model(model, device)
            model.eval()
            _PREPARED_MODEL_KEYS.add(key)
        return model, torch_device, dtype


def _inference_stream_context(torch_device: Any, cuda_stream: Any) -> Any:
    """`torch.cuda.stream(s)` for a live CUDA stream, else a null context.

    Per torch CUDA-semantics: work on a non-default stream needs the inputs
    recorded on it (callers `record_stream` right after H2D) and a join
    before CPU reads (our loops `.cpu()` per item, which synchronizes).
    CPU devices / `None` streams take the null path (current stream).
    """
    import contextlib

    import torch

    if cuda_stream is not None and torch_device.type == "cuda":
        return torch.cuda.stream(cuda_stream)
    return contextlib.nullcontext()


def evict_augment_models() -> int:
    """Drop all resident augment nets; return the evicted entry count.

    Prepared flags go with the nets (a reloaded net must be prepared
    again); the lock keeps an in-flight `_get_prepared_model` from
    observing a half-cleared cache.
    """
    with _AUGMENT_MODEL_LOCK:
        count = len(_ESRGAN_CACHE) + len(_FILM_CACHE) + len(_RIFE_CACHE)
        _ESRGAN_CACHE.clear()
        _FILM_CACHE.clear()
        _RIFE_CACHE.clear()
        _PREPARED_MODEL_KEYS.clear()
        return count


def _require_torch() -> None:
    """Fail fast with ImportError when `torch` is absent (slim image)."""
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            "augment_worker needs optional dependency 'torch' "
            "(slim image carries orchestration only)"
        )


def _require_weights(weights: Path | str, kind: str) -> Path:
    """Return the weights path, or raise NotImplementedError when absent/empty.

    Runs before any torch import so CPU tests and slim images get the
    provisioning error torch-free. Empty files count as absent (a torn
    download must read as "not provisioned", never as a model).
    """
    path = Path(weights)
    if not path.exists() or path.stat().st_size == 0:
        raise NotImplementedError(
            f"{kind} weights missing at {path}: the model registry owns "
            "download/provisioning — augment never fetches weights at runtime"
        )
    return path


def _require_frame_batch(frames: list[Any]) -> None:
    """Reject empty batches and non-RGB (3, H, W) frames before model work."""
    if not frames:
        raise ValueError("augment worker needs at least one frame (got none)")
    for index, frame in enumerate(frames):
        shape = frame.shape
        if len(shape) != 3 or int(shape[0]) != 3:
            raise ValueError(f"frame {index} must be (3, H, W) float (got shape {tuple(shape)})")


def validate_upscale_factor(factor: int) -> int:
    """Accept only targets servable from one x4 model pass (1, 2, or 4)."""
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError(f"upscale factor must be an int (got {type(factor).__name__})")
    if factor not in ALLOWED_UPSCALE_FACTORS:
        raise ValueError(f"upscale factor must be one of {ALLOWED_UPSCALE_FACTORS} (got {factor})")
    return factor


def validate_blend_time(moment: float) -> float:
    """Accept only blend moments in [0, 1] (NaN/inf fail the range check)."""
    if isinstance(moment, bool) or not isinstance(moment, (int, float)):
        raise TypeError(f"blend moment must be a number (got {type(moment).__name__})")
    value = float(moment)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"blend moment must be within [0, 1] (got {moment!r})")
    return value


def validate_pair_batch(pair_batch: int) -> int:
    """Accept only positive pair counts for `interpolate_mids` windows."""
    if isinstance(pair_batch, bool) or not isinstance(pair_batch, int):
        raise TypeError(f"pair batch must be an int (got {type(pair_batch).__name__})")
    if pair_batch < 1:
        raise ValueError(f"pair batch must be positive (got {pair_batch})")
    return pair_batch


def inference_precision(device_name: str) -> str:
    """Precision for `device_name`: fp16 on CUDA (matches the FILM fp16 weights), fp32 elsewhere."""
    return "fp16" if device_name.startswith("cuda") else "fp32"


_DEVICE_FALLBACK_WARNED = False
"""Whether the CPU-fallback line below already fired (warn once, not per chunk)."""


def _resolve_device(preferred: str) -> Any:
    """torch.device for `preferred`, falling back to CPU when CUDA is unavailable.

    Why fallback instead of fail-loud: the spike stays end-to-end runnable
    on CPU-only boxes (slow but exact); callers that need the GPU gate on
    `augment_devices` / nvidia-smi instead.

    The fallback warns once per process on stderr (issue 193): plan dumps
    stamp `cuda:0`/`cuda:1` while execution silently lands on CPU, and
    without a line anywhere the stall mis-triages as "model slow" instead
    of "torch without CUDA".
    """
    import sys

    import torch

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        global _DEVICE_FALLBACK_WARNED
        if not _DEVICE_FALLBACK_WARNED:
            _DEVICE_FALLBACK_WARNED = True
            print(
                f"augment_worker: {preferred} requested but CUDA is unavailable — "
                "running on CPU (slow; gate on `augment_devices` for GPU work)",
                file=sys.stderr,
            )
        return torch.device("cpu")
    return torch.device(preferred)


def _prepare_model(model: Any, device: str) -> tuple[Any, Any]:
    """Move `model` to `device` (half precision on CUDA); return (torch_device, dtype).

    The fp16 cast lands before the host-to-device move so CUDA loads pay
    one fp16 H2D instead of a full fp32 move plus an in-place half.
    """
    import torch

    torch_device = _resolve_device(device)
    if torch_device.type == "cuda":
        model.half()
    model.to(torch_device)
    if torch_device.type == "cuda":
        return torch_device, torch.float16
    return torch_device, torch.float32


def _build_rrdb_net() -> Any:
    """Real-ESRGAN x4 RRDBNet (vendored minimal — state-dict shapes match the x4 family)."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    # NOTE (mypy strict): torch resolves to Any in the slim gates image, so
    # each vendored base needs `# type: ignore[misc]` — the same idiom as
    # video_causvid's `_PrecomputedEncoder(module_base)`.
    class _ResidualDenseBlock(nn.Module):  # type: ignore[misc]
        """Five-layer dense block with 0.2 residual scaling (Real-ESRGAN)."""

        def __init__(
            self,
            num_features: int = RRDB_NUM_FEATURES,
            growth_channels: int = RRDB_GROWTH_CHANNELS,
        ) -> None:
            super().__init__()
            self.conv_first = nn.Conv2d(num_features, growth_channels, 3, 1, 1)
            self.conv_body = nn.ModuleList(
                [
                    nn.Conv2d(num_features + growth_channels * (step + 1), growth_channels, 3, 1, 1)
                    for step in range(3)
                ]
            )
            self.conv_last = nn.Conv2d(num_features + growth_channels * 4, num_features, 3, 1, 1)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value: Any) -> Any:
            grown: list[Any] = [value]
            dense = self.activation(self.conv_first(value))
            grown.append(dense)
            for layer in self.conv_body:
                dense = self.activation(layer(torch.cat(tuple(grown), 1)))
                grown.append(dense)
            return self.conv_last(torch.cat(tuple(grown), 1)) * 0.2 + value

    class _RRDB(nn.Module):  # type: ignore[misc]
        """Residual-in-residual wrapper over three dense blocks."""

        def __init__(
            self,
            num_features: int = RRDB_NUM_FEATURES,
            growth_channels: int = RRDB_GROWTH_CHANNELS,
        ) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                _ResidualDenseBlock(num_features, growth_channels),
                _ResidualDenseBlock(num_features, growth_channels),
                _ResidualDenseBlock(num_features, growth_channels),
            )

        def forward(self, value: Any) -> Any:
            return self.blocks(value) * 0.2 + value

    class _RRDBNet(nn.Module):  # type: ignore[misc]
        """Full x4 net: head, 23 RRDBs, body residual, two x2 nearest stages."""

        def __init__(self) -> None:
            super().__init__()
            self.conv_first = nn.Conv2d(3, RRDB_NUM_FEATURES, 3, 1, 1)
            self.body = nn.Sequential(*[_RRDB() for _ in range(RRDB_NUM_BLOCKS)])
            self.conv_body = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_up_first = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_up_second = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_high = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_last = nn.Conv2d(RRDB_NUM_FEATURES, 3, 3, 1, 1)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value: Any) -> Any:
            head = self.conv_first(value)
            trunk = head + self.conv_body(self.body(head))
            upsampled = functional.interpolate(trunk, scale_factor=2, mode="nearest")
            upsampled = self.activation(self.conv_up_first(upsampled))
            upsampled = functional.interpolate(upsampled, scale_factor=2, mode="nearest")
            upsampled = self.activation(self.conv_up_second(upsampled))
            return self.conv_last(self.activation(self.conv_high(upsampled)))

    return _RRDBNet()


def _build_upstream_rrdb_net(num_blocks: int) -> Any:
    """Upstream-named RRDBNet (xinntao Real-ESRGAN family — issue 166).

    Same dense-block math as the vendored classic (five convs, 0.2 scaling,
    two x2 nearest stages), but submodule names follow the released
    weights: `body.N.rdb1/2/3` (chained, 0.2-scaled residual) plus
    `conv_up1/up2/hr/last`. Depth is measured from the state dict (the
    legacy anime-6B pth carries `RRDB_ANIME_NUM_BLOCKS`), so this one
    builder serves every upstream-depth release.
    """
    import torch
    from torch import nn
    from torch.nn import functional as functional

    # NOTE (mypy strict): same `# type: ignore[misc]` idiom as the classic
    # builder above — torch resolves to Any in the slim gates image.
    class _UpstreamResidualDenseBlock(nn.Module):  # type: ignore[misc]
        """Five-layer dense block with 0.2 residual scaling (Real-ESRGAN)."""

        def __init__(
            self,
            num_features: int = RRDB_NUM_FEATURES,
            growth_channels: int = RRDB_GROWTH_CHANNELS,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(num_features, growth_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_features + growth_channels, growth_channels, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_features + growth_channels * 2, growth_channels, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_features + growth_channels * 3, growth_channels, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_features + growth_channels * 4, num_features, 3, 1, 1)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value: Any) -> Any:
            grown1 = self.activation(self.conv1(value))
            grown2 = self.activation(self.conv2(torch.cat((value, grown1), 1)))
            grown3 = self.activation(self.conv3(torch.cat((value, grown1, grown2), 1)))
            grown4 = self.activation(self.conv4(torch.cat((value, grown1, grown2, grown3), 1)))
            return self.conv5(torch.cat((value, grown1, grown2, grown3, grown4), 1)) * 0.2 + value

    class _UpstreamRRDB(nn.Module):  # type: ignore[misc]
        """Chained triple dense block with 0.2-scaled residual (upstream RRDB)."""

        def __init__(
            self,
            num_features: int = RRDB_NUM_FEATURES,
            growth_channels: int = RRDB_GROWTH_CHANNELS,
        ) -> None:
            super().__init__()
            self.rdb1 = _UpstreamResidualDenseBlock(num_features, growth_channels)
            self.rdb2 = _UpstreamResidualDenseBlock(num_features, growth_channels)
            self.rdb3 = _UpstreamResidualDenseBlock(num_features, growth_channels)

        def forward(self, value: Any) -> Any:
            return self.rdb3(self.rdb2(self.rdb1(value))) * 0.2 + value

    class _UpstreamRRDBNet(nn.Module):  # type: ignore[misc]
        """Full x4 net: head, N chained RRDBs, body residual, two x2 stages."""

        def __init__(self, depth: int) -> None:
            super().__init__()
            self.conv_first = nn.Conv2d(3, RRDB_NUM_FEATURES, 3, 1, 1)
            self.body = nn.Sequential(
                *[_UpstreamRRDB(RRDB_NUM_FEATURES, RRDB_GROWTH_CHANNELS) for _ in range(depth)]
            )
            self.conv_body = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_hr = nn.Conv2d(RRDB_NUM_FEATURES, RRDB_NUM_FEATURES, 3, 1, 1)
            self.conv_last = nn.Conv2d(RRDB_NUM_FEATURES, 3, 3, 1, 1)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value: Any) -> Any:
            head = self.conv_first(value)
            trunk = head + self.conv_body(self.body(head))
            upsampled = self.activation(
                self.conv_up1(functional.interpolate(trunk, scale_factor=2, mode="nearest"))
            )
            upsampled = self.activation(
                self.conv_up2(functional.interpolate(upsampled, scale_factor=2, mode="nearest"))
            )
            return self.conv_last(self.activation(self.conv_hr(upsampled)))

    if num_blocks <= 0:
        raise ValueError(f"upstream RRDB depth must be positive (got {num_blocks})")
    return _UpstreamRRDBNet(num_blocks)


def _unwrap_esrgan_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the inference state dict, stripping a training wrapper when present.

    Released inference weights (EMA-wrapped RRDB training checkpoints nest
    the net one level down under `params_ema`; raw training checkpoints may
    use `params`). A flat upstream/vendored state passes through untouched — only
    a single-key wrapper dict is ever unwrapped, so a flat net that happens
    to carry such a key alongside real weights is never truncated.
    """
    for wrapper in _ESRGAN_WRAPPER_KEYS:
        if set(state) == {wrapper} and isinstance(state[wrapper], dict):
            unwrapped: dict[str, Any] = state[wrapper]
            return unwrapped
    return state


def _upstream_block_count(state: dict[str, Any]) -> int | None:
    """Body depth when `state` uses upstream RRDB naming, else None (vendored classic).

    Upstream means every body index `0..N` appears under `rdb1/2/3` with the
    `conv_first/body/up1/up2/hr/last` head set present. Anything else (the
    vendored classic names, a foreign net) returns None so the caller falls
    back — `strict=True` at the load then fails loud on a partial match.
    """
    indices: set[int] = set()
    for key in state:
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "body":
            continue
        if not parts[1].isdigit() or parts[2] not in ("rdb1", "rdb2", "rdb3"):
            continue
        indices.add(int(parts[1]))
    if not indices:
        return None
    depth = max(indices) + 1
    if sorted(indices) != list(range(depth)):
        return None
    present = {key.split(".")[0] for key in state}
    if not {"conv_first", "conv_body", "conv_last", *_UPSTREAM_UPCONV_KEYS} <= present:
        return None
    return depth


def _is_srvgg_compact_state(state: dict[str, Any]) -> bool:
    """Whether `state` uses the SRVGGNetCompact PReLU layout (not RRDB).

    SRVGG means every key is `body.<int>.<weight|bias>` with contiguous
    indices `0..top` (top even, >= 4), odd indices weight-only (PReLU
    activations carry no bias), even indices weight+bias (convs), and
    the last conv emitting `SRVGG_LAST_OUT_CHANNELS` channels. RRDB
    states (`body.N.rdb1/2/3`, 4-part keys) and foreign nets return
    False so the caller falls back — `strict=True` at the load then
    fails loud on a partial match. Torch-free: shapes read via duck
    typing so stubbed states simply miss.
    """
    if not state:
        return False
    per_index: dict[int, set[str]] = {}
    for key in state:
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "body" or not parts[1].isdigit():
            return False
        per_index.setdefault(int(parts[1]), set()).add(parts[2])
    top = max(per_index)
    if top < 4 or top % 2 != 0:
        return False
    if sorted(per_index) != list(range(top + 1)):
        return False
    for index, names in per_index.items():
        if index % 2 == 1:
            if names != {"weight"}:
                return False
        elif names != {"weight", "bias"}:
            return False
    last_weight = state.get(f"body.{top}.weight")
    shape = getattr(last_weight, "shape", None)
    if shape is None:
        return False
    try:
        shape_tuple = tuple(int(dim) for dim in shape)
    except TypeError:
        return False
    return shape_tuple == (SRVGG_LAST_OUT_CHANNELS, SRVGG_NUM_FEATURES, 3, 3)


def _build_srvgg_net(num_conv: int) -> Any:
    """SRVGGNetCompact PReLU net (xinntao Real-ESRGAN family — issue 166).

    Plain-torch port of upstream `srvgg_arch.py`: a ModuleList body of
    first conv + PReLU, `num_conv` conv+PReLU pairs, and a last conv
    emitting 3×upscale² channels, then PixelShuffle x4 plus the nearest-
    upsampled input as residual base. ModuleList indices register
    `body.0..body.N` exactly, so the pinned anime-video-XS pth
    strict-loads (53 keys at num_conv 16). No dense blocks: 16 plain
    convs are why it runs ~11x faster than the RRDB anime-6B.
    """
    from torch import nn
    from torch.nn import functional as functional

    # NOTE (mypy strict): same `# type: ignore[misc]` idiom as the RRDB
    # builders above — torch resolves to Any in the slim gates image.
    class _SRVGGNetCompact(nn.Module):  # type: ignore[misc]
        """Compact VGG-style x4 net: body convs, PixelShuffle, residual base."""

        def __init__(self, depth: int) -> None:
            super().__init__()
            body: list[Any] = [nn.Conv2d(3, SRVGG_NUM_FEATURES, 3, 1, 1)]
            body.append(nn.PReLU(num_parameters=SRVGG_NUM_FEATURES))
            for _ in range(depth):
                body.append(nn.Conv2d(SRVGG_NUM_FEATURES, SRVGG_NUM_FEATURES, 3, 1, 1))
                body.append(nn.PReLU(num_parameters=SRVGG_NUM_FEATURES))
            body.append(
                nn.Conv2d(SRVGG_NUM_FEATURES, 3 * SRVGG_NATIVE_SCALE * SRVGG_NATIVE_SCALE, 3, 1, 1)
            )
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(SRVGG_NATIVE_SCALE)

        def forward(self, value: Any) -> Any:
            out = value
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            base = functional.interpolate(value, scale_factor=SRVGG_NATIVE_SCALE, mode="nearest")
            return out + base

    if num_conv <= 0:
        raise ValueError(f"SRVGG body depth must be positive (got {num_conv})")
    return _SRVGGNetCompact(num_conv)


def _build_film_net() -> Any:
    """Upstream FILM net (ECCV 2022 frame interpolation — issue 166).

    Plain-torch port of the `extract` / `predict_flow` / `fuse` graph whose
    submodule nesting reproduces the pinned `film_net_fp16.safetensors`
    key layout exactly (`extract.extract_sublevels.convs.i.j.conv.*`,
    `predict_flow._predictor[s].i._convs.j.conv.*`,
    `fuse.convs.i.j.conv.*` + `fuse.output_conv.*`), so `_load_film_net`
    strict-loads the 82 pinned keys. No Comfy imports (worker images carry
    no ComfyUI tree): `comfy.ops` wrappers are plain `nn.Conv2d` here,
    which register identical state-dict keys.

    Call shape is the worker's own `(B, 2, C, H, W)` pairs + float moment
    (not upstream's split-frames call) so `interpolate_pair`,
    `interpolate_triplet`, and the `_run_stacked` OOM-halving loop are
    untouched. Pyramid depth clamps to the input-feasible count: upstream
    always runs 7 levels (needs sides >=64px), while the worker's smoke
    frames are smaller — same weights, fewer coarse levels. Sides below
    `FILM_MIN_SIDE` fail loud (the fusion decoder needs 4 levels).
    """
    import math

    import torch
    from torch import nn
    from torch.nn import functional as functional

    # NOTE (mypy strict): same `# type: ignore[misc]` idiom as the RRDB
    # builders above — torch resolves to Any in the slim gates image.
    class _FilmConv(nn.Module):  # type: ignore[misc]
        """Conv2d with optional LeakyReLU and FILM-style even-kernel padding."""

        def __init__(
            self, in_channels: int, out_channels: int, size: int, activation: bool = True
        ) -> None:
            super().__init__()
            self.even_pad = size % 2 == 0
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=size, padding=size // 2 if size % 2 else 0
            )
            self.activation = nn.LeakyReLU(0.2) if activation else None

        def forward(self, value: Any) -> Any:
            if self.even_pad:
                value = functional.pad(value, (0, 1, 0, 1))
            value = self.conv(value)
            if self.activation is not None:
                value = self.activation(value)
            return value

    def _warp_core(image: Any, flow: Any, grid_x: Any, grid_y: Any) -> Any:
        dtype = image.dtype
        height = int(flow.shape[2])
        width = int(flow.shape[3])
        shift_x = flow[:, 0].float() / (width * 0.5)
        shift_y = flow[:, 1].float() / (height * 0.5)
        grid = torch.stack(
            [grid_x[None, None, :] + shift_x, grid_y[None, :, None] + shift_y], dim=3
        )
        # float sample: grid_sample in fp16 is inaccurate, so the upstream
        # casts up for the warp and back (this is the fp16-CUDA path).
        return functional.grid_sample(
            image.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
        ).to(dtype)

    def _build_image_pyramid(image: Any, levels: int) -> list[Any]:
        pyramid = [image]
        for _ in range(1, levels):
            image = functional.avg_pool2d(image, 2, 2)
            pyramid.append(image)
        return pyramid

    def _synthesize_flow_pyramid(residual_pyramid: list[Any]) -> list[Any]:
        flow = residual_pyramid[-1]
        flow_pyramid = [flow]
        for residual in residual_pyramid[:-1][::-1]:
            flow = (
                functional.interpolate(
                    flow, size=residual.shape[2:4], mode="bilinear", scale_factor=None
                )
                .mul_(2)
                .add_(residual)
            )
            flow_pyramid.append(flow)
        flow_pyramid.reverse()
        return flow_pyramid

    class _SubTreeExtractor(nn.Module):  # type: ignore[misc]
        """Shared conv tower: 2 convs per sublevel, avg-pool between."""

        def __init__(
            self,
            in_channels: int = 3,
            channels: int = FILM_FILTERS,
            n_layers: int = FILM_SUB_LEVELS,
        ) -> None:
            super().__init__()
            convs = []
            for index in range(n_layers):
                out_channels = channels << index
                convs.append(
                    nn.Sequential(
                        _FilmConv(in_channels, out_channels, 3),
                        _FilmConv(out_channels, out_channels, 3),
                    )
                )
                in_channels = out_channels
            self.convs = nn.ModuleList(convs)

        def forward(self, image: Any, depth: int) -> list[Any]:
            head = image
            pyramid = []
            for index, layer in enumerate(self.convs):
                head = layer(head)
                pyramid.append(head)
                if index < depth - 1:
                    head = functional.avg_pool2d(head, 2, 2)
            return pyramid

    class _FeatureExtractor(nn.Module):  # type: ignore[misc]
        """Cross-level feature pyramid (finer levels borrow coarser sublevels)."""

        def __init__(
            self,
            in_channels: int = 3,
            channels: int = FILM_FILTERS,
            sub_levels: int = FILM_SUB_LEVELS,
        ) -> None:
            super().__init__()
            self.extract_sublevels = _SubTreeExtractor(in_channels, channels, sub_levels)
            self.sub_levels = sub_levels

        def forward(self, image_pyramid: list[Any]) -> list[Any]:
            sub_pyramids = [
                self.extract_sublevels(
                    image_pyramid[index], min(len(image_pyramid) - index, self.sub_levels)
                )
                for index in range(len(image_pyramid))
            ]
            feature_pyramid = []
            for index in range(len(image_pyramid)):
                features = sub_pyramids[index][0]
                for depth in range(1, self.sub_levels):
                    if depth <= index:
                        features = torch.cat([features, sub_pyramids[index - depth][depth]], dim=1)
                feature_pyramid.append(features)
                if index >= self.sub_levels - 1:
                    sub_pyramids[index - self.sub_levels + 1] = None
            return feature_pyramid

    class _FlowEstimator(nn.Module):  # type: ignore[misc]
        """One pyramid level's residual-flow predictor (3x3 stack, 1x1 head)."""

        def __init__(self, in_channels: int, num_convs: int, num_filters: int) -> None:
            super().__init__()
            self._convs = nn.ModuleList()
            for _ in range(num_convs):
                self._convs.append(_FilmConv(in_channels, num_filters, 3))
                in_channels = num_filters
            self._convs.append(_FilmConv(in_channels, num_filters // 2, 1))
            self._convs.append(_FilmConv(num_filters // 2, 2, 1, activation=False))

        def forward(self, features_a: Any, features_b: Any) -> Any:
            net = torch.cat([features_a, features_b], dim=1)
            for conv in self._convs:
                net = conv(net)
            return net

    class _PyramidFlowEstimator(nn.Module):  # type: ignore[misc]
        """Coarse-to-fine flow: shared coarsest predictor, then fine specialists."""

        def __init__(
            self,
            filters: int = FILM_FILTERS,
            flow_convs: tuple[int, int, int, int] = FILM_FLOW_CONVS,
            flow_filters: tuple[int, int, int, int] = FILM_FLOW_FILTERS,
        ) -> None:
            super().__init__()
            in_channels = filters << 1
            predictors = []
            for index in range(len(flow_convs)):
                predictors.append(
                    _FlowEstimator(in_channels, flow_convs[index], flow_filters[index])
                )
                in_channels += filters << (index + 2)
            self._predictor = predictors[-1]
            self._predictors = nn.ModuleList(predictors[:-1][::-1])

        def forward(self, pyramid_a: list[Any], pyramid_b: list[Any], warp_fn: Any) -> list[Any]:
            levels = len(pyramid_a)
            flow = self._predictor(pyramid_a[-1], pyramid_b[-1])
            residuals = [flow]
            steps = [
                (index, self._predictor)
                for index in range(levels - 2, len(self._predictors) - 1, -1)
            ]
            steps += [
                (len(self._predictors) - 1 - index, predictor)
                for index, predictor in enumerate(self._predictors)
            ]
            for index, predictor in steps:
                flow = functional.interpolate(
                    flow, size=pyramid_a[index].shape[2:4], mode="bilinear"
                ).mul_(2)
                residual = predictor(pyramid_a[index], warp_fn(pyramid_b[index], flow))
                residuals.append(residual)
                flow = flow.add_(residual)
            residuals.reverse()
            return residuals

    def _fusion_in_channels(level: int, filters: int) -> int:
        # Per direction: multi-scale features + RGB image (3ch) + flow (2ch), doubled for both.
        return (sum(filters << index for index in range(level)) + 3 + 2) * 2

    class _Fusion(nn.Module):  # type: ignore[misc]
        """Coarse-to-fine decoder fusing warped pyramids + scaled flows into RGB."""

        def __init__(
            self,
            n_layers: int = FILM_SUB_LEVELS,
            specialized_layers: int = FILM_SPECIALIZED_LEVELS,
            filters: int = FILM_FILTERS,
        ) -> None:
            super().__init__()
            self.output_conv = nn.Conv2d(filters, 3, kernel_size=1)
            self.convs = nn.ModuleList()
            in_channels = _fusion_in_channels(n_layers, filters)
            increase = 0
            for index in range(n_layers)[::-1]:
                num_filters = (
                    (filters << index)
                    if index < specialized_layers
                    else (filters << specialized_layers)
                )
                self.convs.append(
                    nn.ModuleList(
                        [
                            _FilmConv(in_channels, num_filters, 2, activation=False),
                            _FilmConv(in_channels + (increase or num_filters), num_filters, 3),
                            _FilmConv(num_filters, num_filters, 3),
                        ]
                    )
                )
                in_channels = num_filters
                increase = _fusion_in_channels(index, filters) - num_filters // 2

        def forward(self, pyramid: list[Any]) -> Any:
            net = pyramid[-1]
            for block, layers in enumerate(self.convs):
                index = len(self.convs) - 1 - block
                net = layers[0](
                    functional.interpolate(net, size=pyramid[index].shape[2:4], mode="nearest")
                )
                net = layers[2](layers[1](torch.cat([pyramid[index], net], dim=1)))
            return self.output_conv(net)

    def _feasible_pyramid_levels(height: int, width: int) -> int:
        """Deepest pyramid whose smallest level stays >=1px (each level halves)."""
        return 1 + int(math.floor(math.log2(max(min(height, width), 1))))

    class _FilmNet(nn.Module):  # type: ignore[misc]
        """FILM graph with the worker's pairs + moment call shape."""

        def __init__(
            self,
            pyramid_levels: int = FILM_PYRAMID_LEVELS,
            fusion_levels: int = FILM_FUSION_LEVELS,
            specialized_levels: int = FILM_SPECIALIZED_LEVELS,
            sub_levels: int = FILM_SUB_LEVELS,
            filters: int = FILM_FILTERS,
            flow_convs: tuple[int, int, int, int] = FILM_FLOW_CONVS,
            flow_filters: tuple[int, int, int, int] = FILM_FLOW_FILTERS,
        ) -> None:
            super().__init__()
            self.pyramid_levels = pyramid_levels
            self.fusion_pyramid_levels = fusion_levels
            self.extract = _FeatureExtractor(3, filters, sub_levels)
            self.predict_flow = _PyramidFlowEstimator(filters, flow_convs, flow_filters)
            self.fuse = _Fusion(sub_levels, specialized_levels, filters)
            self._warp_grids: dict[tuple[int, int], Any] = {}

        def _build_warp_grids(self, height: int, width: int, levels: int, device: Any) -> None:
            """Pre-compute warp grids for every pyramid level of this resolution.

            Per-geometry memo WITHOUT reset (concurrency fix, same rationale
            as RIFE `_grids_for`): entries are small linspace vectors and are
            never deleted, so `warp` reads stay race-free.
            """
            with _WARP_GRID_LOCK:
                if (height, width) in self._warp_grids:
                    return
                for _ in range(levels):
                    self._warp_grids[(height, width)] = (
                        torch.linspace(
                            -(1 - 1 / width),
                            1 - 1 / width,
                            width,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.linspace(
                            -(1 - 1 / height),
                            1 - 1 / height,
                            height,
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                    height, width = height // 2, width // 2

        def warp(self, image: Any, flow: Any) -> Any:
            grid_x, grid_y = self._warp_grids[(int(flow.shape[2]), int(flow.shape[3]))]
            return _warp_core(image, flow, grid_x, grid_y)

        def extract_features(self, image: Any, levels: int) -> tuple[list[Any], list[Any]]:
            """Image + feature pyramids for one frame (cacheable across pairs)."""
            image_pyramid = _build_image_pyramid(image, levels)
            return image_pyramid, self.extract(image_pyramid)

        def forward(self, pairs: Any, moment: float = 0.5) -> Any:
            """One mid frame per pair row at blend `moment` (matches `_run_stacked`)."""
            height, width = int(pairs.shape[3]), int(pairs.shape[4])
            if min(height, width) < FILM_MIN_SIDE:
                raise ValueError(
                    f"FILM needs frame sides >= {FILM_MIN_SIDE}px (got {height}x{width}): "
                    "the fusion decoder runs 4 pyramid levels"
                )
            return self.forward_multi_timestep(pairs[:, 0], pairs[:, 1], [float(moment)])

        def forward_multi_timestep(
            self, frame_a: Any, frame_b: Any, moments: list[float], cache: Any = None
        ) -> Any:
            """Mid frames at each moment; flow is computed once (expects batch>=1)."""
            height, width = int(frame_a.shape[2]), int(frame_a.shape[3])
            levels = min(self.pyramid_levels, _feasible_pyramid_levels(height, width))
            self._build_warp_grids(height, width, levels, frame_a.device)
            if cache is not None and "img0" in cache:
                image_pyr_a, feat_pyr_a = cache["img0"]
            else:
                image_pyr_a, feat_pyr_a = self.extract_features(frame_a, levels)
            if cache is not None and "img1" in cache:
                image_pyr_b, feat_pyr_b = cache["img1"]
            else:
                image_pyr_b, feat_pyr_b = self.extract_features(frame_b, levels)
            fwd_flow = _synthesize_flow_pyramid(
                self.predict_flow(feat_pyr_a, feat_pyr_b, self.warp)
            )[: self.fusion_pyramid_levels]
            bwd_flow = _synthesize_flow_pyramid(
                self.predict_flow(feat_pyr_b, feat_pyr_a, self.warp)
            )[: self.fusion_pyramid_levels]
            fuse_levels = min(self.fusion_pyramid_levels, levels)
            warp_targets = [
                [
                    torch.cat([image, features], dim=1)
                    for image, features in zip(
                        image_pyr_a[:fuse_levels], feat_pyr_a[:fuse_levels], strict=True
                    )
                ],
                [
                    torch.cat([image, features], dim=1)
                    for image, features in zip(
                        image_pyr_b[:fuse_levels], feat_pyr_b[:fuse_levels], strict=True
                    )
                ],
            ]
            del image_pyr_a, image_pyr_b, feat_pyr_a, feat_pyr_b
            results = []
            for step in moments:
                bwd_scaled = [flow * step for flow in bwd_flow]
                fwd_scaled = [flow * (1.0 - step) for flow in fwd_flow]
                fwd_warped = [
                    self.warp(features, flow)
                    for features, flow in zip(warp_targets[0], bwd_scaled, strict=True)
                ]
                bwd_warped = [
                    self.warp(features, flow)
                    for features, flow in zip(warp_targets[1], fwd_scaled, strict=True)
                ]
                aligned = [
                    torch.cat([forward, backward, bwd, fwd], dim=1)
                    for forward, backward, bwd, fwd in zip(
                        fwd_warped, bwd_warped, bwd_scaled, fwd_scaled, strict=True
                    )
                ]
                del fwd_warped, bwd_warped, bwd_scaled, fwd_scaled
                results.append(self.fuse(aligned))
                del aligned
            return torch.cat(results, dim=0)

    return _FilmNet()


def _load_state_dict(weights_path: Path) -> dict[str, Any]:
    """Decode a weight file by suffix without ever unpickling safetensors (074).

    `.safetensors` decodes via `safetensors.torch.load_file` (no pickle machine);
    every other suffix loads via `torch.load(weights_only=True)`. Safetensors
    decode failures normalize to `ValueError` so the loader boundary maps them
    to `ModelCompatibilityError` via `_LOAD_ERRORS`; a missing `safetensors`
    package raises `ImportError` unchanged (environment issue, not weights).
    """
    if weights_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        try:
            decoded: Any = load_file(str(weights_path))
        except Exception as exc:
            raise ValueError(f"safetensors decode failed for {weights_path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise TypeError(
                f"safetensors file at {weights_path} decoded to "
                f"{type(decoded).__name__}, not a state dict"
            )
        return decoded
    import torch

    decoded_torch: Any = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(decoded_torch, dict):
        raise TypeError(
            f"torch file at {weights_path} loaded to {type(decoded_torch).__name__}, "
            "not a state dict"
        )
    return decoded_torch


def _verify_weights_size(weights_path: Path, kind: str, floor_bytes: int) -> None:
    """Fail loud when a weight file is smaller than its registry floor (074).

    Runs torch-free before any loader: a truncated download must read as
    incompatible, never as a model that fails deep in the decoder.
    """
    actual_bytes = weights_path.stat().st_size
    if actual_bytes < floor_bytes:
        raise ModelCompatibilityError(
            f"{kind} weights at {weights_path} size {actual_bytes} bytes below "
            f"floor {floor_bytes} bytes "
            "(truncated download — re-provision via `voyage models download`)"
        )


def _verify_weights_manifest(weights_path: Path) -> None:
    """Sha-verify against a nearby manifest record when one exists (074).

    Walks up from the weights file looking for `manifest.json`; when a record
    carries a sha for exactly this file (`checkpoint_sha256` + `checkpoint_file`
    or a `checkpoint_shas` dict entry, the 071 shapes), a mismatch raises
    `ModelCompatibilityError` before any loader runs. No manifest, no entry,
    or an unreadable manifest passes through (the ingest-time constants in
    `download_model` and `verify_model` are the closed gates there).
    """
    import json

    from voyage.hashing import sha256_file

    try:
        target = weights_path.resolve()
    except OSError:
        return
    for ancestor in target.parents:
        manifest_path = ancestor / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(loaded, dict):
            return
        for entry in loaded.values():
            if not isinstance(entry, dict):
                continue
            recorded = entry.get("checkpoint_sha256")
            relative = entry.get("checkpoint_file")
            if isinstance(recorded, str) and recorded and isinstance(relative, str) and relative:
                try:
                    if (ancestor / relative).resolve() == target:
                        actual = sha256_file(weights_path)
                        if actual.lower() != recorded.lower():
                            raise ModelCompatibilityError(
                                f"weights at {weights_path} hash mismatch: manifest "
                                f"records {recorded}, file hashes {actual} — refusing "
                                "to load an untrusted file"
                            )
                        return
                except OSError:
                    continue
            shas = entry.get("checkpoint_shas")
            if isinstance(shas, dict):
                for relative_path, expected in shas.items():
                    if isinstance(relative_path, str) and isinstance(expected, str) and expected:
                        try:
                            if (ancestor / relative_path).resolve() == target:
                                actual = sha256_file(weights_path)
                                if actual.lower() != expected.lower():
                                    raise ModelCompatibilityError(
                                        f"weights at {weights_path} hash mismatch: "
                                        f"manifest records {expected}, file hashes "
                                        f"{actual} — refusing an untrusted file"
                                    )
                                return
                        except OSError:
                            continue
        return
    return


def _load_esrgan_net(weights_path: Path) -> Any:
    """Build the matching ESRGAN-family net and load `weights_path`.

    Three layouts (issue 166): SRVGGNetCompact PReLU states
    (`body.<int>` + odd weight-only + 48-channel last conv — the pinned
    anime-video-XS pth) build an SRVGG net at the measured body depth;
    upstream-named RRDB weights (`body.N.rdb1/2/3` + `conv_up1/up2/hr`)
    build an upstream net at the measured depth; anything else falls
    back to the vendored x4 net. `strict=True` in all cases, so a
    partial match still fails loud instead of inferring on random init.
    """
    from voyage.model_registry import REALESRGAN_ANIME_MIN_BYTES

    _verify_weights_size(weights_path, "Real-ESRGAN", REALESRGAN_ANIME_MIN_BYTES)
    _verify_weights_manifest(weights_path)

    try:
        state = _unwrap_esrgan_state(_load_state_dict(weights_path))
        if _is_srvgg_compact_state(state):
            top_index = max(int(key.split(".")[1]) for key in state)
            model = _build_srvgg_net((top_index - 2) // 2)
        else:
            block_count = _upstream_block_count(state)
            if block_count is not None:
                model = _build_upstream_rrdb_net(block_count)
            else:
                model = _build_rrdb_net()
        model.load_state_dict(state, strict=True)
    except _LOAD_ERRORS as exc:
        raise ModelCompatibilityError(
            f"Real-ESRGAN weights at {weights_path} match neither the SRVGG-compact "
            f"layout (body.<int> PReLU, e.g. the pinned anime-video-XS) nor the "
            f"upstream RRDB layout (body.N.rdb1/2/3 + conv_up1/up2/hr) "
            f"nor the vendored x4 net: {exc}"
        ) from exc
    return model


def _load_film_net(weights_path: Path) -> Any:
    """Build the upstream FILM net and load `weights_path` (failures map below)."""
    from voyage.model_registry import FILM_MIN_BYTES

    _verify_weights_size(weights_path, "FILM", FILM_MIN_BYTES)
    _verify_weights_manifest(weights_path)

    model = _build_film_net()
    try:
        state = _load_state_dict(weights_path)
        model.load_state_dict(state, strict=True)
    except _LOAD_ERRORS as exc:
        raise ModelCompatibilityError(
            f"FILM weights at {weights_path} do not match the upstream FILM "
            f"architecture (extract/fuse/predict_flow, {FILM_STATE_KEYS} keys): {exc}"
        ) from exc
    return model


def _run_stacked(forward: Callable[[Any], Any], stacked: Any) -> list[Any]:
    """Run `forward` on the stacked batch, halving on OOM down to singles (Comfy recipe).

    Order-preserving: halves recurse left-then-right and concatenate.
    Non-OOM failures and single-item OOMs propagate unchanged.
    """
    import torch

    try:
        outputs = forward(stacked)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or int(stacked.shape[0]) <= 1:
            raise
        gc.collect()
        # Unconditional (issue 157): `empty_cache` is a no-op without
        # CUDA, so the old `is_available` guard bought nothing and
        # skipped relief on CPU-OOM recursion.
        torch.cuda.empty_cache()
        half = int(stacked.shape[0]) // 2
        return [
            *_run_stacked(forward, stacked[0:half]),
            *_run_stacked(forward, stacked[half:]),
        ]
    return [outputs[index] for index in range(int(outputs.shape[0]))]


def _axis_blocks(length: int, tile: int) -> list[tuple[int, int]]:
    """Split `[0, length)` into the fewest near-even blocks of size <= tile."""
    if length <= tile:
        return [(0, length)]
    count = (length + tile - 1) // tile
    base, extra = divmod(length, count)
    blocks: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        end = start + base + (1 if index < extra else 0)
        blocks.append((start, end))
        start = end
    return blocks


def tile_grid(
    width: int,
    height: int,
    tile: int = UPSCALE_TILE_SIZE,
) -> list[tuple[int, int, int, int]]:
    """Output-space tile blocks partitioning a WxH frame (row-major).

    Blocks are non-overlapping and cover the frame exactly; each side is
    <= `tile`. The caller expands each block by `UPSCALE_TILE_OVERLAP`
    (clipped) for model context and crops back to the block after upscale,
    so assembly is gapless and seam-free. Stdlib-pure (tested in gates
    without torch).
    """
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
    ):
        raise TypeError(f"frame size must be ints (got {width!r}x{height!r})")
    if width < 1 or height < 1:
        raise ValueError(f"frame size must be positive (got {width!r}x{height!r})")
    if isinstance(tile, bool) or not isinstance(tile, int) or tile < 1:
        raise ValueError(f"tile must be a positive int (got {tile!r})")
    return [
        (x0, y0, x1, y1)
        for (y0, y1) in _axis_blocks(height, tile)
        for (x0, x1) in _axis_blocks(width, tile)
    ]


def needs_upscale_tiling(width: int, height: int) -> bool:
    """True when a WxH frame must upscale tiled (over `UPSCALE_TILE_BUDGET_PIXELS`)."""
    return width * height > UPSCALE_TILE_BUDGET_PIXELS


def interp_first_for_small_device(device: str) -> bool:
    """True when `device` is a CUDA GPU with under 8 GiB free (interp at 1x first).

    Measured 2026-10-01: FILM pairs at 2432x1408 need ~5.9 GiB, so the
    classic upscale-first order OOMs the 6 GB 2060 there; interpolating
    the 1x frames first (then upscaling everything, tiled past the
    budget) fits comfortably. Non-CUDA devices, unparsable names, and
    probe failures all answer False — the validated order stands unless
    small VRAM is proven.
    """
    if not isinstance(device, str) or not device.startswith("cuda"):
        return False
    try:
        import torch

        parts = device.split(":")
        index = int(parts[1]) if len(parts) > 1 else 0
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
    except (ImportError, ValueError, RuntimeError):
        return False
    return bool(free_bytes < FILM_CLASSIC_ORDER_MIN_FREE_BYTES)


def _upscale_frame_tiled(
    model: Any,
    frame: Any,
    torch_device: Any,
    dtype: Any,
    tile: int,
) -> Any:
    """Upscale one CPU (3, H, W) frame through overlap-context tiles (x4 native).

    Each output block runs with an `UPSCALE_TILE_OVERLAP` context margin
    (clipped at borders) and only its own block is pasted back, so block
    boundaries never show. Peak memory is one tile's activations (~1 GiB
    at the default 512) instead of the full frame's (~5.5 GiB at
    1216x704 — the 2060 OOM the budget answers).
    """
    import torch

    moved = frame.to(torch_device, dtype=dtype)
    _, height, width = moved.shape
    try:
        out = torch.empty(
            (3, height * RRDB_NATIVE_SCALE, width * RRDB_NATIVE_SCALE),
            device=torch_device,
            dtype=dtype,
        )
        for x0, y0, x1, y1 in tile_grid(width, height, tile=tile):
            in_x0 = max(0, x0 - UPSCALE_TILE_OVERLAP)
            in_y0 = max(0, y0 - UPSCALE_TILE_OVERLAP)
            in_x1 = min(width, x1 + UPSCALE_TILE_OVERLAP)
            in_y1 = min(height, y1 + UPSCALE_TILE_OVERLAP)
            patch = moved[:, in_y0:in_y1, in_x0:in_x1].unsqueeze(0)
            up = model(patch)
            paste_x = (x0 - in_x0) * RRDB_NATIVE_SCALE
            paste_y = (y0 - in_y0) * RRDB_NATIVE_SCALE
            paste_w = (x1 - x0) * RRDB_NATIVE_SCALE
            paste_h = (y1 - y0) * RRDB_NATIVE_SCALE
            out[
                :,
                y0 * RRDB_NATIVE_SCALE : y0 * RRDB_NATIVE_SCALE + paste_h,
                x0 * RRDB_NATIVE_SCALE : x0 * RRDB_NATIVE_SCALE + paste_w,
            ] = up[:, :, paste_y : paste_y + paste_h, paste_x : paste_x + paste_w]
            del patch, up
        return out
    finally:
        del moved


def _run_frame_batches(
    forward: Callable[[Any], Any],
    frame_tensors: list[Any],
    torch_device: Any,
    dtype: Any,
) -> list[Any]:
    """Run `forward` over CPU frame tensors, halving the list before stacking (047).

    The full-batch `torch.stack` is the allocation the chunking was built to
    avoid, so it is never built: slices stack per half inside the recursion
    and the failed half's tensors release before splitting further. Falls
    back to `_run_stacked`'s tensor halving once a half is stacked, so both
    the stack and the activation transient halve together.
    """
    import torch

    if not frame_tensors:
        raise ValueError("augment worker needs at least one frame (got none)")
    if len(frame_tensors) == 1:
        stacked = torch.stack(frame_tensors).to(torch_device, dtype=dtype)
        try:
            return _run_stacked(forward, stacked)
        finally:
            del stacked
    try:
        stacked = torch.stack(frame_tensors).to(torch_device, dtype=dtype)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        gc.collect()
        # Unconditional (issue 157): same no-op-without-CUDA rationale
        # as `_run_stacked` above — never skip relief on the split path.
        torch.cuda.empty_cache()
        half = len(frame_tensors) // 2
        return [
            *_run_frame_batches(forward, frame_tensors[:half], torch_device, dtype),
            *_run_frame_batches(forward, frame_tensors[half:], torch_device, dtype),
        ]
    try:
        return _run_stacked(forward, stacked)
    finally:
        del stacked


def _finish_upscaled(single: Any, functional: Any, target_scale: int) -> Any:
    """Clamp a native-x4 device tensor, downscale to `target_scale`, move to CPU float32.

    Single home for the upscale postprocess (both the batched and the
    inline-tiled paths end here, so their bytes stay identical).
    """
    refined = single.clamp(0.0, 1.0)
    if target_scale != RRDB_NATIVE_SCALE:
        refined = functional.interpolate(
            refined.unsqueeze(0),
            scale_factor=target_scale / RRDB_NATIVE_SCALE,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    return refined.float().cpu()


def upscale_frames(
    frames: list[Any],
    weights: Path | str,
    *,
    scale: int = 2,
    device: str = "cuda:0",
    timings: dict[str, float] | None = None,
    tile: int | None = None,
    cuda_stream: Any = None,
) -> list[Any]:
    """Upscale (3, H, W) float frames in [0, 1] by `scale` (Real-ESRGAN x4 + downscale).

    One x4 model pass serves every allowed target: 4 is native, 2/1
    downscale the x4 output (bicubic). Outputs are float32 CPU tensors
    clamped to [0, 1] — the caller concatenates/encodes on CPU. The net
    is resident per (weights, device) across calls; `timings` records
    `load_ms` vs `infer_ms` separately when given (benchmark attribution).

    `tile` selects the spatial path: `None` (default) tiles only frames
    over `UPSCALE_TILE_BUDGET_PIXELS` (a single big frame OOMs small GPUs
    where batch-halving cannot help — DESIGN §140 GPU defaults), an int
    forces that tile size (tests, manual override; validated positive).

    `cuda_stream`, when a live `torch.cuda.Stream` on the inference
    device, encloses the forward loop so N worker threads on one GPU
    overlap H2D/compute across streams (torch best practice: one private
    stream per worker thread; per-frame `.cpu()` is the join). `None`
    (default) uses the current stream.
    """
    target = validate_upscale_factor(scale)
    weights_path = _require_weights(weights, "Real-ESRGAN")
    _require_torch()
    _require_frame_batch(frames)
    if tile is not None and (isinstance(tile, bool) or not isinstance(tile, int)):
        raise TypeError(f"tile must be an int or None (got {type(tile).__name__})")
    if tile is not None and tile < 1:
        raise ValueError(f"tile must be a positive int (got {tile})")
    import torch
    from torch.nn import functional as functional

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model, torch_device, dtype = _get_prepared_model(
        _ESRGAN_CACHE, key, device, lambda: _load_esrgan_net(weights_path)
    )
    load_ms = (time.monotonic() - load_started) * 1000.0
    cpu_frames = [torch.as_tensor(frame, dtype=torch.float32) for frame in frames]
    results: list[Any] = []
    infer_started = time.monotonic()
    try:
        with torch.no_grad(), _inference_stream_context(torch_device, cuda_stream):
            natives: list[Any] = []
            for cpu_frame in cpu_frames:
                _, frame_h, frame_w = cpu_frame.shape
                tile_size = tile
                if tile_size is None and needs_upscale_tiling(frame_w, frame_h):
                    tile_size = UPSCALE_TILE_SIZE
                if tile_size is None:
                    natives.extend(_run_frame_batches(model, [cpu_frame], torch_device, dtype))
                else:
                    # Postprocess inline (never accumulate device tiles: the
                    # interp-first order hands this path 4x the frames, and
                    # a full chunk of x4-native device tensors OOMs small
                    # GPUs — measured 2026-10-01 on the 6 GB 2060).
                    native = _upscale_frame_tiled(model, cpu_frame, torch_device, dtype, tile_size)
                    results.append(_finish_upscaled(native, functional, target))
        for single in natives:
            results.append(_finish_upscaled(single, functional, target))
    finally:
        del cpu_frames
    if timings is not None:
        timings["load_ms"] = load_ms
        timings["infer_ms"] = (time.monotonic() - infer_started) * 1000.0
    return results


def interpolate_pair(
    before: Any,
    after: Any,
    weights: Path | str,
    *,
    moment: float = 0.5,
    device: str = "cuda:0",
    timings: dict[str, float] | None = None,
) -> Any:
    """One mid frame between `before` and `after` at blend `moment` (upstream FILM)."""
    blend = validate_blend_time(moment)
    weights_path = _require_weights(weights, "FILM")
    _require_torch()
    _require_frame_batch([before, after])
    import torch

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model, torch_device, dtype = _get_prepared_model(
        _FILM_CACHE, key, device, lambda: _load_film_net(weights_path)
    )
    load_ms = (time.monotonic() - load_started) * 1000.0
    batched = torch.stack(
        [
            torch.as_tensor(before, dtype=torch.float32),
            torch.as_tensor(after, dtype=torch.float32),
        ]
    ).to(torch_device, dtype=dtype)
    infer_started = time.monotonic()
    try:
        with torch.no_grad():
            mids = _run_stacked(lambda batch: model(batch, blend), batched.unsqueeze(0))
    finally:
        del batched
    if timings is not None:
        timings["load_ms"] = load_ms
        timings["infer_ms"] = (time.monotonic() - infer_started) * 1000.0
    return mids[0].float().cpu()


def interpolate_triplet(
    first: Any,
    middle: Any,
    last: Any,
    weights: Path | str,
    *,
    device: str = "cuda:0",
    timings: dict[str, float] | None = None,
) -> tuple[Any, Any]:
    """Mid frames for (first, middle) and (middle, last) sharing one model load."""
    weights_path = _require_weights(weights, "FILM")
    _require_torch()
    _require_frame_batch([first, middle, last])
    import torch

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model, torch_device, dtype = _get_prepared_model(
        _FILM_CACHE, key, device, lambda: _load_film_net(weights_path)
    )
    load_ms = (time.monotonic() - load_started) * 1000.0
    first_tensor = torch.as_tensor(first, dtype=torch.float32)
    middle_tensor = torch.as_tensor(middle, dtype=torch.float32)
    last_tensor = torch.as_tensor(last, dtype=torch.float32)
    batched = torch.stack(
        [
            torch.stack([first_tensor, middle_tensor]),
            torch.stack([middle_tensor, last_tensor]),
        ]
    ).to(torch_device, dtype=dtype)
    infer_started = time.monotonic()
    try:
        with torch.no_grad():
            mids = _run_stacked(lambda batch: model(batch, 0.5), batched)
    finally:
        del batched
    if timings is not None:
        timings["load_ms"] = load_ms
        timings["infer_ms"] = (time.monotonic() - infer_started) * 1000.0
    return (mids[0].float().cpu(), mids[1].float().cpu())


def _run_pair_window(model: Any, pairs: Any, blends: list[float]) -> list[Any]:
    """Run one pair window at every blend, halving the pair count on OOM.

    Window-local `_run_stacked`: the stacked dim-0 is the pair count, so a
    failed window splits into whole-pair halves (left-then-right,
    order-preserving) and each half re-runs the full blend list. Returned
    rows are pair-major — pair 0 at every blend, then pair 1, and so on —
    so the caller can demux without tracking half boundaries. Non-OOM
    failures and single-pair OOMs propagate unchanged.
    """
    import torch

    try:
        rows = model.forward_multi_timestep(pairs[:, 0], pairs[:, 1], blends)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or int(pairs.shape[0]) <= 1:
            raise
        gc.collect()
        # Unconditional (issue 157): `empty_cache` is a no-op without
        # CUDA, so no availability guard — same rationale as `_run_stacked`.
        torch.cuda.empty_cache()
        half = int(pairs.shape[0]) // 2
        return [
            *_run_pair_window(model, pairs[0:half], blends),
            *_run_pair_window(model, pairs[half:], blends),
        ]
    window_len = int(pairs.shape[0])
    blend_count = len(blends)
    return [
        rows[blend_index * window_len + offset]
        for offset in range(window_len)
        for blend_index in range(blend_count)
    ]


def interpolate_mids(
    frames: list[Any],
    weights: Path | str,
    *,
    moments: list[float] | tuple[float, ...],
    pair_batch: int = FILM_PAIR_BATCH,
    device: str = "cuda:0",
    timings: dict[str, float] | None = None,
    on_pair: Callable[[int, int], None] | None = None,
) -> list[Any]:
    """Mid frames for every adjacent pair at each moment (batched FILM).

    Pair-major output: for pair `i`, the mids at `moments` in order, so the
    caller interleaves `frames[i]` + that slice exactly like the old
    per-pair `interpolate_pair` loop — but each forward evaluates
    `pair_batch` pairs at ALL moments with one flow computation per pair
    (flow-once via `forward_multi_timestep`; the single-pair multi-moment
    morph case drops from 4 extracts + 8 flows to 1 + 2). Outputs are
    float32 CPU tensors. `pair_batch=1` matches the `interpolate_pair`
    loop bit-exactly (same shapes → same cudnn kernels); larger batches
    are deterministic per config but shift pixels slightly (different
    kernels per batch shape — measured mean abs 1.9e-4, ~4% of pixels
    flip after PNG rounding — so only raise `pair_batch` if a future GPU
    shows a real speedup). A window that OOMs halves to single pairs, so larger
    `pair_batch` values stay safe. `on_pair`, when given, fires per
    finished pair with `(pair_index, pair_count)`.

    Validation order (before any torch import): moments, pair batch, frame
    count (>= 2 — a bare length check, so it stays torch-free), weights,
    then torch, then frame shapes and the `FILM_MIN_SIDE` floor.
    """
    if isinstance(moments, (str, bytes)) or not isinstance(moments, (list, tuple)):
        raise TypeError(
            f"moments must be a list or tuple of blend times (got {type(moments).__name__})"
        )
    if not moments:
        raise ValueError("interpolate_mids needs at least one blend moment (got none)")
    blends = [validate_blend_time(moment) for moment in moments]
    batch = validate_pair_batch(pair_batch)
    if on_pair is not None and not callable(on_pair):
        raise TypeError(f"on_pair must be callable or None (got {type(on_pair).__name__})")
    if not isinstance(frames, list) or len(frames) < 2:
        count = len(frames) if isinstance(frames, list) else type(frames).__name__
        raise ValueError(f"interpolate_mids needs at least two frames (got {count})")
    weights_path = _require_weights(weights, "FILM")
    _require_torch()
    _require_frame_batch(frames)
    for frame in frames:
        height, width = int(frame.shape[1]), int(frame.shape[2])
        if min(height, width) < FILM_MIN_SIDE:
            raise ValueError(
                f"FILM needs frame sides >= {FILM_MIN_SIDE}px (got {height}x{width}): "
                "the fusion decoder runs 4 pyramid levels"
            )
    import torch

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model, torch_device, dtype = _get_prepared_model(
        _FILM_CACHE, key, device, lambda: _load_film_net(weights_path)
    )
    load_ms = (time.monotonic() - load_started) * 1000.0
    pair_count = len(frames) - 1
    mids: list[Any] = []
    infer_started = time.monotonic()
    with torch.no_grad():
        for window_start in range(0, pair_count, batch):
            window_end = min(window_start + batch, pair_count)
            unique = [
                torch.as_tensor(frames[index], dtype=torch.float32)
                for index in range(window_start, window_end + 1)
            ]
            pairs = torch.stack(
                [
                    torch.stack([unique[offset], unique[offset + 1]])
                    for offset in range(len(unique) - 1)
                ]
            ).to(torch_device, dtype=dtype)
            try:
                mids.extend(row.float().cpu() for row in _run_pair_window(model, pairs, blends))
            finally:
                del pairs, unique
            if on_pair is not None:
                for pair_index in range(window_start, window_end):
                    on_pair(pair_index, pair_count)
    if timings is not None:
        timings["load_ms"] = load_ms
        timings["infer_ms"] = (time.monotonic() - infer_started) * 1000.0
    return mids


# ---------------------------------------------------------------------------
# RIFE interpolation (Practical-RIFE IFNet, Comfy-layout port).
# ---------------------------------------------------------------------------

RIFE_PAD_ALIGN = 64
"""Reflect-pad alignment for RIFE inference (upstream `pad_align`)."""

RIFE_MIN_SIDE = 8
"""Minimum RIFE input side — mirrors `FILM_MIN_SIDE`.

Sides below this fail loud instead of hitting torch's reflect-pad limits.
"""

INTERP_BACKENDS = ("film", "rife")
"""Selectable interpolation backends (`AugmentConfig.interp_backend`)."""


def validate_interp_backend(backend: str) -> str:
    """Validate an interpolation backend name, returning it unchanged."""
    if not isinstance(backend, str):
        raise TypeError(f"interp backend must be a string (got {type(backend).__name__})")
    if backend not in INTERP_BACKENDS:
        raise ValueError(f"unknown interp backend {backend!r} (expected one of {INTERP_BACKENDS})")
    return backend


def _build_rife_net(
    head_channels: int = 4,
    block_channels: tuple[int, ...] = (192, 128, 96, 64, 32),
) -> Any:
    """Comfy-layout RIFE IFNet (hzwer Practical-RIFE 4.x — RIFE backend).

    Plain-torch port of `Comfy/comfy_extras/frame_interpolation_models/ifnet.py`
    (`Head` / `ResConv` / `IFBlock` / `IFNet` + warp grids), so the submodule
    nesting reproduces the pinned `rife_v4.25_heavy.safetensors` key layout exactly
    (`encode.cnn{0..3}.*`, `blocks.{0..4}.conv0/convblock/lastconv.*`) and
    `_load_rife_net` strict-loads it. No Comfy imports (worker images carry
    no ComfyUI tree): `comfy.ops` wrappers are plain `nn.Conv2d` here, which
    register identical state-dict keys. Math mirrors upstream exactly
    (addcmul residual, flow div_/add_ accumulation, lerp output blend, fp32
    border warp with align_corners=True). One deliberate deviation: warp
    grids are keyed by (height, width, device) instead of shape alone, so one
    resident net can serve both cards without stale-device grids.

    Why RIFE alongside FILM: per-pair RIFE is ~16.8x faster than FILM at
    2048x1152 on the 4060 Ti (0.051s vs 0.851s per forward, issue-166 probe)
    at ~0.65 GiB peak vs ~5.6 GiB, and the line-art A/B shows no ghosting
    or thin-line shimmer (see docs/AUGMENT.md). Each timestep needs its own
    flow (no flow-once factorization), so multi-moment pairs cost one
    forward per moment — still far cheaper than FILM's single flow+fuse.
    """
    import torch
    from torch import nn
    from torch.nn import functional as functional

    # NOTE (mypy strict): same `# type: ignore[misc]` idiom as the FILM/RRDB
    # builders above — torch resolves to Any in the slim gates image.
    class _RifeHead(nn.Module):  # type: ignore[misc]
        """4-level feature head (3px in, `out_channels` feature maps out)."""

        def __init__(self, out_channels: int) -> None:
            super().__init__()
            self.cnn0 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.cnn1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
            self.cnn2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
            self.cnn3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)
            self.relu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, value: Any) -> Any:
            value = self.relu(self.cnn0(value))
            value = self.relu(self.cnn1(value))
            value = self.relu(self.cnn2(value))
            # NOTE: no relu after cnn3 (upstream layout — raw feature maps).
            return self.cnn3(value)

    class _RifeResConv(nn.Module):  # type: ignore[misc]
        """Residual conv with a learned per-channel gate (`beta`)."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
            self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=False)
            self.relu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, value: Any) -> Any:
            return self.relu(torch.addcmul(value, self.conv(value), self.beta))

    class _RifeBlock(nn.Module):  # type: ignore[misc]
        """One IFBlock: 4x down, 8x ResConv, transpose+shuffle back up."""

        def __init__(self, in_planes: int, channels: int) -> None:
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(in_planes, channels // 2, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            )
            self.convblock = nn.Sequential(*[_RifeResConv(channels) for _ in range(8)])
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(channels, 4 * 13, kernel_size=4, stride=2, padding=1),
                nn.PixelShuffle(2),
            )

        def forward(self, value: Any, flow: Any = None, scale: int = 1) -> Any:
            value = functional.interpolate(value, scale_factor=1.0 / scale, mode="bilinear")
            if flow is not None:
                flow = functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear")
                flow = flow.div_(scale)
                value = torch.cat((value, flow), 1)
            feat = self.convblock(self.conv0(value))
            tmp = functional.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

    class _RifeNet(nn.Module):  # type: ignore[misc]
        """5-block IFNet cascade over the shared feature head."""

        def __init__(self, head_ch: int, channels: tuple[int, ...]) -> None:
            super().__init__()
            self.encode = _RifeHead(head_ch)
            block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
            self.blocks = nn.ModuleList(
                [_RifeBlock(block_in[index], channels[index]) for index in range(5)]
            )
            self.scale_list = [16, 8, 4, 2, 1]
            self.warp_grids: dict[Any, Any] = {}

        def _grids_for(self, height: int, width: int, device: Any) -> Any:
            """Base grid + flow divisors for a frame geometry (cached per device).

            Per-geometry memo WITHOUT clear() (concurrency fix): the old
            `clear()` deleted every geometry, so two threads building
            different sizes corrupted each other (KeyError on return). One
            grid is ~18 MiB at 2x and a finalize holds a single geometry;
            double-checked under `_WARP_GRID_LOCK` so concurrent builds of
            the same size compute once. Keys are never deleted, so `warp`
            reads stay lock-free and race-free.
            """
            key = (height, width, str(device))
            if key not in self.warp_grids:
                with _WARP_GRID_LOCK:
                    if key not in self.warp_grids:
                        grid_y, grid_x = torch.meshgrid(
                            torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32),
                            torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32),
                            indexing="ij",
                        )
                        self.warp_grids[key] = (
                            torch.stack((grid_x, grid_y), dim=0).unsqueeze(0),
                            torch.tensor(
                                [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
            return self.warp_grids[key]

        def warp(self, image: Any, flow: Any) -> Any:
            batch, _, height, width = image.shape
            base_grid, flow_div = self._grids_for(height, width, image.device)
            flow_norm = torch.cat(
                [flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1
            ).float()
            grid = (base_grid.expand(batch, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return functional.grid_sample(
                image.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(image.dtype)

        def forward(self, first: Any, second: Any, timestep: Any = 0.5, cache: Any = None) -> Any:
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.full(
                    (first.shape[0], 1, first.shape[2], first.shape[3]),
                    timestep,
                    device=first.device,
                    dtype=first.dtype,
                )
            batch = first.shape[0]
            if cache and "img0" in cache:
                feat0 = cache["img0"].expand(batch, -1, -1, -1)
            else:
                feat0 = self.encode(first)
            if cache and "img1" in cache:
                feat1 = cache["img1"].expand(batch, -1, -1, -1)
            else:
                feat1 = self.encode(second)
            flow = mask = feat = None
            warped0, warped1 = first, second
            for index, block in enumerate(self.blocks):
                if flow is None:
                    flow, mask, feat = block(
                        torch.cat((first, second, feat0, feat1, timestep), 1),
                        None,
                        scale=self.scale_list[index],
                    )
                else:
                    flow_delta, mask, feat = block(
                        torch.cat(
                            (
                                warped0,
                                warped1,
                                self.warp(feat0, flow[:, :2]),
                                self.warp(feat1, flow[:, 2:4]),
                                timestep,
                                mask,
                                feat,
                            ),
                            1,
                        ),
                        flow,
                        scale=self.scale_list[index],
                    )
                    flow = flow.add_(flow_delta)
                warped0 = self.warp(first, flow[:, :2])
                warped1 = self.warp(second, flow[:, 2:4])
            return torch.lerp(warped1, warped0, torch.sigmoid(mask))

    return _RifeNet(head_channels, tuple(block_channels))


def _remap_rife_state(state: dict[str, Any]) -> dict[str, Any]:
    """Strip Practical-RIFE prefixes to the Comfy key layout (torch-free).

    Drops `module.` / `flownet.`, rewrites `block{i}.` to `blocks.{i}.`,
    and drops `teacher.` / `caltime.` distillation keys Comfy never loads.
    """
    remapped: dict[str, Any] = {}
    for key, value in state.items():
        short = key.replace("module.", "").replace("flownet.", "")
        for index in range(5):
            prefix = f"block{index}."
            if short.startswith(prefix):
                short = f"blocks.{index}." + short[len(prefix) :]
                break
        if short.startswith("teacher.") or short.startswith("caltime."):
            continue
        remapped[short] = value
    return remapped


def _detect_rife_config(state: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    """Read (head channels, block channels) from Comfy-layout RIFE keys (torch-free)."""
    head_channels = int(state["encode.cnn3.weight"].shape[1])
    block_channels = tuple(
        int(state[f"blocks.{index}.conv0.1.0.weight"].shape[0]) for index in range(5)
    )
    return head_channels, block_channels


_RIFE_LOAD_ERRORS: tuple[type[BaseException], ...] = (*_LOAD_ERRORS, KeyError)
"""Except-tuple for the RIFE loader (module-level so mypy verifies it)."""


def _load_rife_net(weights_path: Path) -> Any:
    """Build the Comfy-layout IFNet and load `weights_path` (failures map below)."""
    from voyage.model_registry import RIFE_MIN_BYTES

    _verify_weights_size(weights_path, "RIFE", RIFE_MIN_BYTES)
    _verify_weights_manifest(weights_path)

    try:
        state = _remap_rife_state(_load_state_dict(weights_path))
        head_channels, block_channels = _detect_rife_config(state)
        model = _build_rife_net(head_channels, block_channels)
        model.load_state_dict(state, strict=True)
    except _RIFE_LOAD_ERRORS as exc:
        raise ModelCompatibilityError(
            f"RIFE weights at {weights_path} do not match the Comfy IFNet layout "
            f"(encode + blocks.0-4, v4.25 pin): {exc}"
        ) from exc
    return model


def interpolate_rife_mids(
    frames: list[Any],
    weights: Path | str,
    *,
    moments: list[float] | tuple[float, ...],
    device: str = "cuda:0",
    timings: dict[str, float] | None = None,
    on_pair: Callable[[int, int], None] | None = None,
    cuda_stream: Any = None,
) -> list[Any]:
    """Mid frames for every adjacent pair at each moment (RIFE IFNet).

    Pair-major output with the same contract as `interpolate_mids`: for pair
    `i`, the mids at `moments` in order, so the caller interleaves
    `frames[i]` + that slice exactly like the FILM loop. Outputs are
    float32 CPU tensors. Differences from the FILM path: one forward per
    pair x moment (RIFE has no flow-once factorization — each timestep needs
    its own flow), so there is no `pair_batch`; the per-pair feature encode
    is shared across moments instead (3 forwards share 1 encode pair at
    m=4). Inputs reflect-pad to `RIFE_PAD_ALIGN` and crop back, so sizes
    that are not multiples of 64 (e.g. 832x480) just work. OOM propagates —
    a single pair is already the minimal unit (measured peaks 0.65 GiB at
    2x on the 4060 Ti, 0.17 GiB at 1x), so there is nothing to halve to.
    `on_pair`, when given, fires per finished pair with
    `(pair_index, pair_count)`.

    Validation order (before any torch import): moments, frame count
    (>= 2 — a bare length check, so it stays torch-free), weights, then
    torch, then frame shapes and the `RIFE_MIN_SIDE` floor.

    `cuda_stream` optionally pins the forward loop to a caller-owned CUDA
    stream (same threading contract as `upscale_frames`); `None` default.
    """
    if isinstance(moments, (str, bytes)) or not isinstance(moments, (list, tuple)):
        raise TypeError(
            f"moments must be a list or tuple of blend times (got {type(moments).__name__})"
        )
    if not moments:
        raise ValueError("interpolate_rife_mids needs at least one blend moment (got none)")
    blends = [validate_blend_time(moment) for moment in moments]
    if on_pair is not None and not callable(on_pair):
        raise TypeError(f"on_pair must be callable or None (got {type(on_pair).__name__})")
    if not isinstance(frames, list) or len(frames) < 2:
        count = len(frames) if isinstance(frames, list) else type(frames).__name__
        raise ValueError(f"interpolate_rife_mids needs at least two frames (got {count})")
    weights_path = _require_weights(weights, "RIFE")
    _require_torch()
    _require_frame_batch(frames)
    for frame in frames:
        height, width = int(frame.shape[1]), int(frame.shape[2])
        if min(height, width) < RIFE_MIN_SIDE:
            raise ValueError(
                f"RIFE needs frame sides >= {RIFE_MIN_SIDE}px (got {height}x{width}): "
                "torch reflect padding needs smaller pads than the input sides"
            )
    import torch
    from torch.nn import functional as functional

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model, torch_device, dtype = _get_prepared_model(
        _RIFE_CACHE, key, device, lambda: _load_rife_net(weights_path)
    )
    load_ms = (time.monotonic() - load_started) * 1000.0
    pair_count = len(frames) - 1
    mids: list[Any] = []
    infer_started = time.monotonic()
    with torch.no_grad(), _inference_stream_context(torch_device, cuda_stream):
        for pair_index in range(pair_count):
            first = torch.as_tensor(frames[pair_index], dtype=torch.float32).unsqueeze(0)
            second = torch.as_tensor(frames[pair_index + 1], dtype=torch.float32).unsqueeze(0)
            height, width = int(first.shape[2]), int(first.shape[3])
            pad_right = -width % RIFE_PAD_ALIGN
            pad_bottom = -height % RIFE_PAD_ALIGN
            if pad_right or pad_bottom:
                first = functional.pad(first, (0, pad_right, 0, pad_bottom), mode="reflect")
                second = functional.pad(second, (0, pad_right, 0, pad_bottom), mode="reflect")
            first = first.to(torch_device, dtype=dtype)
            second = second.to(torch_device, dtype=dtype)
            if cuda_stream is not None and torch_device.type == "cuda":
                first.record_stream(cuda_stream)
                second.record_stream(cuda_stream)
            try:
                # One encode per pair, shared across moments (the RIFE
                # feature-cache win — moments only re-run the flow cascade).
                cache = {"img0": model.encode(first), "img1": model.encode(second)}
                for blend in blends:
                    mid = model(first, second, timestep=float(blend), cache=cache)
                    mids.append(mid[:, :, :height, :width].float().cpu().squeeze(0))
            finally:
                del first, second
            if on_pair is not None:
                on_pair(pair_index, pair_count)
    if timings is not None:
        timings["load_ms"] = load_ms
        timings["infer_ms"] = (time.monotonic() - infer_started) * 1000.0
    return mids

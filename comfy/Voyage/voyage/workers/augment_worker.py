"""GPU augment runner: Real-ESRGAN upscale + FILM interpolate (Track D spike).

`torch` loads only inside functions (behind a `find_spec` guard) — never
at module scope (supervisor section 12 GPU ban) — and weight checks run
before any torch import, so missing weights raise NotImplementedError
torch-free. The registry owns download/provisioning; this module never
fetches weights at runtime.

Architectures are vendored minimal inline — do NOT import Comfy nodes
(the worker images carry no ComfyUI tree):
- RRDBNet: the Real-ESRGAN x4 residual-in-residual dense net (state-dict
  shapes match the x4plus/UltraSharp ESRGAN family; the compact
  `RealESRGAN_x4plus_anime_6B` SRVGG variant needs its own loader —
  follow-up, not this spike).
- FilmNetMini: a spike stand-in flow blender with FILM's semantic
  contract (two frames + time give the mid frame), batched as
  `(B, 2, C, H, W)` so OOM-halving applies. Official `film_net` weights
  will NOT load here (shape mismatch raises ModelCompatibilityError);
  the full upstream FILM port is follow-up.

Precision is fp16 on CUDA, fp32 elsewhere. Batch inference starts full
and halves on out-of-memory down to single items, mirroring Comfy's
FrameInterpolate recipe. RPC ops are supervisor-track follow-up — this
module is an in-process library called with tensors.
"""

from __future__ import annotations

import importlib.util
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

ALLOWED_UPSCALE_FACTORS = (1, 2, 4)
"""Targets served from one x4 pass (4 is native; 2/1 downscale the x4 output)."""

FILM_MINI_CHANNELS = 32
"""Feature width of the spike FILM stand-in (tiny on purpose — quality is follow-up)."""

_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    EOFError,
)
"""torch.load / load_state_dict failures meaning "weights unusable here".

Exotic failures outside this tuple (e.g. unpickling errors from a
safetensors file passed to torch.load) propagate as worker errors instead.
"""


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


def inference_precision(device_name: str) -> str:
    """Precision for `device_name`: fp16 on CUDA (matches the FILM fp16 weights), fp32 elsewhere."""
    return "fp16" if device_name.startswith("cuda") else "fp32"


def _resolve_device(preferred: str) -> Any:
    """torch.device for `preferred`, falling back to CPU when CUDA is unavailable.

    Why fallback instead of fail-loud: the spike stays end-to-end runnable
    on CPU-only boxes (slow but exact); callers that need the GPU gate on
    `augment_devices` / nvidia-smi instead.
    """
    import torch

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def _prepare_model(model: Any, device: str) -> tuple[Any, Any]:
    """Move `model` to `device` (half precision on CUDA); return (torch_device, dtype)."""
    import torch

    torch_device = _resolve_device(device)
    model.to(torch_device)
    if torch_device.type == "cuda":
        model.half()
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


def _build_film_net() -> Any:
    """Spike stand-in flow blender with FILM's semantic contract.

    NOT upstream FILM: a small two-scale encoder predicts a flow pair plus
    an occlusion mask at quarter resolution, warps both inputs toward the
    blend moment, and mixes the time blend with the warped average by the
    mask. Same call shape (two frames + moment give the mid frame) so
    chunk code written against it survives the full upstream port.
    """
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class _FilmNetMini(nn.Module):  # type: ignore[misc]
        """Quarter-res flow pair + mask, warp, mask-mixed time blend."""

        def __init__(self, channels: int = FILM_MINI_CHANNELS) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(6, channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channels, 2 * channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(2 * channels, 4 * channels, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.flow_head = nn.Conv2d(4 * channels, 4, 3, 1, 1)
            self.mask_head = nn.Conv2d(4 * channels, 1, 3, 1, 1)

        @staticmethod
        def _warp(frame: Any, flow: Any) -> Any:
            grid_height = int(frame.shape[2])
            grid_width = int(frame.shape[3])
            axis_y, axis_x = torch.meshgrid(
                torch.arange(grid_height, device=frame.device),
                torch.arange(grid_width, device=frame.device),
                indexing="ij",
            )
            base = torch.stack((axis_x, axis_y), -1).unsqueeze(0).to(frame.dtype)
            sample = base + flow.permute(0, 2, 3, 1)
            sample[..., 0] = sample[..., 0] / max(grid_width - 1, 1) * 2 - 1
            sample[..., 1] = sample[..., 1] / max(grid_height - 1, 1) * 2 - 1
            return functional.grid_sample(
                frame, sample, mode="bilinear", padding_mode="border", align_corners=True
            )

        def forward(self, pairs: Any, moment: float) -> Any:
            frame_a = pairs[:, 0]
            frame_b = pairs[:, 1]
            features = self.encoder(torch.cat((frame_a, frame_b), 1))
            flows = functional.interpolate(
                self.flow_head(features), scale_factor=4, mode="bilinear", align_corners=False
            )
            mask = torch.sigmoid(
                functional.interpolate(
                    self.mask_head(features), scale_factor=4, mode="bilinear", align_corners=False
                )
            )
            warped_a = self._warp(frame_a, flows[:, 0:2] * moment)
            warped_b = self._warp(frame_b, flows[:, 2:4] * (1.0 - moment))
            timed = warped_a * (1.0 - moment) + warped_b * moment
            average = 0.5 * (warped_a + warped_b)
            return timed * mask + average * (1.0 - mask)

    return _FilmNetMini()


def _load_rrdb_net(weights_path: Path) -> Any:
    """Build RRDBNet and load `weights_path`; shape/content failures become compatibility errors."""
    import torch

    model = _build_rrdb_net()
    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except _LOAD_ERRORS as exc:
        raise ModelCompatibilityError(
            f"Real-ESRGAN weights at {weights_path} do not match the vendored "
            f"x4 RRDBNet (compact anime_6B needs its own loader — follow-up): {exc}"
        ) from exc
    return model


def _load_film_net(weights_path: Path) -> Any:
    """Build the FILM stand-in and load `weights_path`; failures become ModelCompatibilityError."""
    import torch

    model = _build_film_net()
    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except _LOAD_ERRORS as exc:
        raise ModelCompatibilityError(
            f"FILM weights at {weights_path} do not match the spike stand-in "
            f"architecture (full upstream FILM port is follow-up): {exc}"
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = int(stacked.shape[0]) // 2
        return [
            *_run_stacked(forward, stacked[0:half]),
            *_run_stacked(forward, stacked[half:]),
        ]
    return [outputs[index] for index in range(int(outputs.shape[0]))]


def upscale_frames(
    frames: list[Any],
    weights: Path | str,
    *,
    scale: int = 2,
    device: str = "cuda:0",
) -> list[Any]:
    """Upscale (3, H, W) float frames in [0, 1] by `scale` (Real-ESRGAN x4 + downscale).

    One x4 model pass serves every allowed target: 4 is native, 2/1
    downscale the x4 output (bicubic). Outputs are float32 CPU tensors
    clamped to [0, 1] — the caller concatenates/encodes on CPU.
    """
    target = validate_upscale_factor(scale)
    weights_path = _require_weights(weights, "Real-ESRGAN")
    _require_torch()
    _require_frame_batch(frames)
    import torch
    from torch.nn import functional as functional

    model = _load_rrdb_net(weights_path)
    torch_device, dtype = _prepare_model(model, device)
    model.eval()
    stacked = torch.stack([torch.as_tensor(frame, dtype=torch.float32) for frame in frames]).to(
        torch_device, dtype=dtype
    )
    results: list[Any] = []
    with torch.no_grad():
        for single in _run_stacked(model, stacked):
            refined = single.clamp(0.0, 1.0)
            if target != RRDB_NATIVE_SCALE:
                refined = functional.interpolate(
                    refined.unsqueeze(0),
                    scale_factor=target / RRDB_NATIVE_SCALE,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            results.append(refined.float().cpu())
    return results


def interpolate_pair(
    before: Any,
    after: Any,
    weights: Path | str,
    *,
    moment: float = 0.5,
    device: str = "cuda:0",
) -> Any:
    """One mid frame between `before` and `after` at blend `moment` (FILM stand-in)."""
    time = validate_blend_time(moment)
    weights_path = _require_weights(weights, "FILM")
    _require_torch()
    _require_frame_batch([before, after])
    import torch

    model = _load_film_net(weights_path)
    torch_device, dtype = _prepare_model(model, device)
    model.eval()
    batched = torch.stack(
        [
            torch.as_tensor(before, dtype=torch.float32),
            torch.as_tensor(after, dtype=torch.float32),
        ]
    ).to(torch_device, dtype=dtype)
    with torch.no_grad():
        mids = _run_stacked(lambda batch: model(batch, time), batched.unsqueeze(0))
    return mids[0].float().cpu()


def interpolate_triplet(
    first: Any,
    middle: Any,
    last: Any,
    weights: Path | str,
    *,
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Mid frames for (first, middle) and (middle, last) sharing one model load."""
    weights_path = _require_weights(weights, "FILM")
    _require_torch()
    _require_frame_batch([first, middle, last])
    import torch

    model = _load_film_net(weights_path)
    torch_device, dtype = _prepare_model(model, device)
    model.eval()
    first_tensor = torch.as_tensor(first, dtype=torch.float32)
    middle_tensor = torch.as_tensor(middle, dtype=torch.float32)
    last_tensor = torch.as_tensor(last, dtype=torch.float32)
    batched = torch.stack(
        [
            torch.stack([first_tensor, middle_tensor]),
            torch.stack([middle_tensor, last_tensor]),
        ]
    ).to(torch_device, dtype=dtype)
    with torch.no_grad():
        mids = _run_stacked(lambda batch: model(batch, 0.5), batched)
    return (mids[0].float().cpu(), mids[1].float().cpu())

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

import gc
import importlib.util
import pickle
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
    pickle.UnpicklingError,
)
"""Weight-load failures meaning "weights unusable here" (issue 074).

Covers `torch.load` / `safetensors` decode / `load_state_dict` shape errors —
every one maps to `ModelCompatibilityError` at the loader boundary. Corrupt
pickles surface as `UnpicklingError` (a `PickleError`, outside the original
five); safetensors decode failures are normalized to `ValueError` in
`_load_state_dict` so they land here too.
"""

_RRDB_CACHE: dict[tuple[str, str], Any] = {}
"""Resident Real-ESRGAN nets keyed by (weights path, device) — issue 047.

A 32-chunk augment must not pay 32x construction + disk load + H2D;
the first call warms the entry, later chunks reuse it. `evict_augment_models`
drops both caches (GPU hand-off, DESIGN §40).
"""

_FILM_CACHE: dict[tuple[str, str], Any] = {}
"""Resident FILM stand-ins keyed by (weights path, device) — issue 047."""


def _model_cache_key(weights_path: Path, device: str) -> tuple[str, str]:
    """Cache identity for a resident net: stringified weights path + device."""
    return (str(weights_path), str(device))


def evict_augment_models() -> int:
    """Drop all resident augment nets; return the evicted entry count."""
    count = len(_RRDB_CACHE) + len(_FILM_CACHE)
    _RRDB_CACHE.clear()
    _FILM_CACHE.clear()
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


def _load_rrdb_net(weights_path: Path) -> Any:
    """Build RRDBNet and load `weights_path`; shape/content failures become compatibility errors."""
    from voyage.model_registry import REALESRGAN_ANIME_MIN_BYTES

    _verify_weights_size(weights_path, "Real-ESRGAN", REALESRGAN_ANIME_MIN_BYTES)
    _verify_weights_manifest(weights_path)

    model = _build_rrdb_net()
    try:
        state = _load_state_dict(weights_path)
        model.load_state_dict(state, strict=True)
    except _LOAD_ERRORS as exc:
        raise ModelCompatibilityError(
            f"Real-ESRGAN weights at {weights_path} do not match the vendored "
            f"x4 RRDBNet (compact anime_6B needs its own loader — follow-up): {exc}"
        ) from exc
    return model


def _load_film_net(weights_path: Path) -> Any:
    """Build the FILM stand-in and load `weights_path`; failures become ModelCompatibilityError."""
    from voyage.model_registry import FILM_MIN_BYTES

    _verify_weights_size(weights_path, "FILM", FILM_MIN_BYTES)
    _verify_weights_manifest(weights_path)

    model = _build_film_net()
    try:
        state = _load_state_dict(weights_path)
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = int(stacked.shape[0]) // 2
        return [
            *_run_stacked(forward, stacked[0:half]),
            *_run_stacked(forward, stacked[half:]),
        ]
    return [outputs[index] for index in range(int(outputs.shape[0]))]


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
        if torch.cuda.is_available():
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


def upscale_frames(
    frames: list[Any],
    weights: Path | str,
    *,
    scale: int = 2,
    device: str = "cuda:0",
    timings: dict[str, float] | None = None,
) -> list[Any]:
    """Upscale (3, H, W) float frames in [0, 1] by `scale` (Real-ESRGAN x4 + downscale).

    One x4 model pass serves every allowed target: 4 is native, 2/1
    downscale the x4 output (bicubic). Outputs are float32 CPU tensors
    clamped to [0, 1] — the caller concatenates/encodes on CPU. The net
    is resident per (weights, device) across calls; `timings` records
    `load_ms` vs `infer_ms` separately when given (benchmark attribution).
    """
    target = validate_upscale_factor(scale)
    weights_path = _require_weights(weights, "Real-ESRGAN")
    _require_torch()
    _require_frame_batch(frames)
    import torch
    from torch.nn import functional as functional

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model = _RRDB_CACHE.get(key)
    if model is None:
        model = _load_rrdb_net(weights_path)
        _RRDB_CACHE[key] = model
    load_ms = (time.monotonic() - load_started) * 1000.0
    torch_device, dtype = _prepare_model(model, device)
    model.eval()
    cpu_frames = [torch.as_tensor(frame, dtype=torch.float32) for frame in frames]
    results: list[Any] = []
    infer_started = time.monotonic()
    try:
        with torch.no_grad():
            batched = _run_frame_batches(model, cpu_frames, torch_device, dtype)
        for single in batched:
            refined = single.clamp(0.0, 1.0)
            if target != RRDB_NATIVE_SCALE:
                refined = functional.interpolate(
                    refined.unsqueeze(0),
                    scale_factor=target / RRDB_NATIVE_SCALE,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            results.append(refined.float().cpu())
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
    """One mid frame between `before` and `after` at blend `moment` (FILM stand-in)."""
    blend = validate_blend_time(moment)
    weights_path = _require_weights(weights, "FILM")
    _require_torch()
    _require_frame_batch([before, after])
    import torch

    load_started = time.monotonic()
    key = _model_cache_key(weights_path, device)
    model = _FILM_CACHE.get(key)
    if model is None:
        model = _load_film_net(weights_path)
        _FILM_CACHE[key] = model
    load_ms = (time.monotonic() - load_started) * 1000.0
    torch_device, dtype = _prepare_model(model, device)
    model.eval()
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
    model = _FILM_CACHE.get(key)
    if model is None:
        model = _load_film_net(weights_path)
        _FILM_CACHE[key] = model
    load_ms = (time.monotonic() - load_started) * 1000.0
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

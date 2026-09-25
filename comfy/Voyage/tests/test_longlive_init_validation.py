"""LongLive init validation + stream-start accounting (issues 066, 074).

CPU-only: shape/size checks run before any torch/CUDA touch, and the
`generate_blocks` wiring test drives the unbound method with fakes.
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.workers import video_longlive


def test_validate_latent_shape_accepts_five_positive_dims() -> None:
    assert video_longlive.validate_latent_shape([1, 21, 16, 60, 104]) == [1, 21, 16, 60, 104]


def test_validate_latent_shape_rejects_short_and_non_positive() -> None:
    for bad_shape in ([1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6], []):
        with pytest.raises(ValueError, match="5 positive ints"):
            video_longlive.validate_latent_shape(list(bad_shape))
    for bad_shape in ([1, 0, 16, 60, 104], [1, 21, -16, 60, 104]):
        with pytest.raises(ValueError, match="5 positive ints"):
            video_longlive.validate_latent_shape(list(bad_shape))


def test_validate_local_attn_size_keeps_sentinel() -> None:
    video_longlive.validate_local_attn_size(-1)
    video_longlive.validate_local_attn_size(8)
    video_longlive.validate_local_attn_size(16)
    for bad_size in (0, -2):
        with pytest.raises(ValueError, match="local_attn_size"):
            video_longlive.validate_local_attn_size(bad_size)


def test_validate_sink_size_rejects_negative() -> None:
    video_longlive.validate_sink_size(0)
    video_longlive.validate_sink_size(8)
    with pytest.raises(ValueError, match="sink_size"):
        video_longlive.validate_sink_size(-1)


def test_validate_attn_capacity_enforces_sink_plus_block() -> None:
    video_longlive.validate_attn_capacity(16, 8)
    video_longlive.validate_attn_capacity(8, 0)
    with pytest.raises(ValueError, match="cache too small"):
        video_longlive.validate_attn_capacity(8, 8)
    # The -1 full-context sentinel skips the floor.
    video_longlive.validate_attn_capacity(-1, 100)


def test_handle_init_rejects_bad_shape_without_gpu(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="5 positive ints"):
        video_longlive.handle_init(
            {
                "models_dir": str(tmp_path),
                "device": "cuda:0",
                "latent_shape": [1, 2],
            }
        )


def test_handle_init_rejects_bad_attn_sizes_without_gpu(tmp_path: Path) -> None:
    base: dict[str, Any] = {
        "models_dir": str(tmp_path),
        "device": "cuda:0",
        "latent_shape": [1, 8, 8, 44, 80],
    }
    with pytest.raises(ValueError, match="local_attn_size"):
        video_longlive.handle_init({**base, "local_attn_size": 0})
    with pytest.raises(ValueError, match="cache too small"):
        video_longlive.handle_init({**base, "local_attn_size": 8, "sink_size": 8})


def test_stream_start_frame_tracks_block_size() -> None:
    assert video_longlive.stream_start_frame_for_call(100, 8, 3) == 76
    assert video_longlive.stream_start_frame_for_call(100, 6, 2) == 88
    assert video_longlive.stream_start_frame_for_call(8, 8, 1) == 0


class _FakeCuda:
    def empty_cache(self) -> None:
        return None


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.uint8 = "uint8"
        self.saved: list[tuple[Any, str]] = []

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()

    def cat(self, tensors: list[Any], dim: int = 1) -> Any:
        del dim
        return tensors[0]

    def save(self, obj: Any, path: str) -> None:
        self.saved.append((obj, path))


class _FakeVaeModel:
    def clear_cache(self) -> None:
        return None


class _FakeVae:
    def __init__(self) -> None:
        self.model = _FakeVaeModel()

    def to(self, device: Any) -> _FakeVae:
        del device
        return self

    def decode_to_pixel_chunk(
        self, latents: Any, use_cache: bool = False, chunk_size: int = 0
    ) -> Any:
        del latents, use_cache, chunk_size
        return object()


class _FakeGenerator:
    def to(self, device: Any) -> _FakeGenerator:
        del device
        return self


class _FakePipe:
    num_frame_per_block = 6

    def __init__(self) -> None:
        self.vae = _FakeVae()
        self.generator = _FakeGenerator()


class _FakeLatents:
    shape = (1, 12)

    def detach(self) -> _FakeLatents:
        return self

    def cpu(self) -> _FakeLatents:
        return self


class _FakeEmbeds:
    def detach(self) -> _FakeEmbeds:
        return self

    def cpu(self) -> _FakeEmbeds:
        return self


class _FakeVideo:
    """Stub decoded video: 2 frames of 4x6, uint8-ready."""

    shape = (1, 2, 4, 6)

    def cpu(self) -> _FakeVideo:
        return self

    def to(self, dtype: Any) -> _FakeVideo:
        del dtype
        return self

    def __rmul__(self, other: Any) -> _FakeVideo:
        del other
        return self

    def __getitem__(self, key: Any) -> Any:
        del key
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        return types.SimpleNamespace(numpy=lambda: frame)


class _FakeStream:
    """Stub stream clocked in 6-frame blocks from frame 94."""

    def __init__(self) -> None:
        self.next_start_frame = 94
        self.blocks_appended = 12

    def begin_sequence(self, total_blocks: int, seed: int) -> None:
        del total_blocks, seed

    def append_block(self, prompt: str, cut: bool = False) -> _FakeLatents:
        del prompt, cut
        self.next_start_frame += 6
        self.blocks_appended += 1
        return _FakeLatents()

    def _encode(self, prompt: str) -> tuple[dict[str, _FakeEmbeds], list[Any]]:
        del prompt
        return ({"prompt_embeds": _FakeEmbeds()}, [])

    def offload_caches(self) -> None:
        return None

    def restore_caches(self) -> None:
        return None

    def noise_rng_state(self) -> bytes:
        return b"rng"


def _install_longlive_media_stubs(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    written: list[Any] = []

    class _Writer:
        def __enter__(self) -> _Writer:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def append_data(self, frame: Any) -> None:
            written.append(frame)

    imageio_mod = types.ModuleType("imageio")
    v2_mod = types.ModuleType("imageio.v2")
    v2_mod.get_writer = lambda *args, **kwargs: _Writer()  # type: ignore[attr-defined]
    imageio_mod.v2 = v2_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2_mod)
    einops_mod = types.ModuleType("einops")
    einops_mod.rearrange = lambda tensor, pattern: _FakeVideo()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "einops", einops_mod)
    return written


def test_generate_blocks_reports_pipe_block_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 074: with 6-frame blocks the start frame is next-6*count."""
    _install_longlive_media_stubs(monkeypatch)
    session_self = types.SimpleNamespace(
        _torch=_FakeTorch(),
        _pipeline=_FakePipe(),
        _stream=_FakeStream(),
        _device="cuda:0",
        _profile="longlive2-bf16-fp8",
        _latent_shape=[1, 8, 8, 8, 8],
    )
    result = video_longlive.LongLiveSession.generate_blocks(
        session_self,  # type: ignore[arg-type]
        prompts=["a meadow", "a harbor"],
        seeds=[11, 12],
        scene_cuts=[False, False],
        output_path=tmp_path / "seg.mp4",
        fps=24,
    )
    # Stream ran 94 -> 106 over two 6-frame blocks; this call started at 94.
    assert result["stream_start_frame"] == 94

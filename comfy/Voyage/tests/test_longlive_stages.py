"""CUDA-event stage timers for the longlive2 video worker (Track 1.1 audit).

CPU-only: every CUDA/torch/pipeline/media dependency is faked — no GPU,
no model downloads. Covers timer math with stubbed events, the
flag-off zero-overhead path (no events, no synchronizes, no stage keys),
and the benchmark return shape with stages on/off via a monkeypatched
session.
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage.workers import video_longlive
from voyage.workers.video_longlive import _STAGE_NAMES, _CudaStageTimer


class _FakeCuda:
    """Stub for `torch.cuda`: counts events, replays queued elapsed values."""

    def __init__(self, elapsed_values: list[float] | None = None) -> None:
        self.elapsed_values = list(elapsed_values) if elapsed_values else []
        self.events_created = 0
        self.enable_timing_flags: list[bool] = []
        self.records: list[str] = []
        self.synchronizes: list[str] = []
        self.empty_cache_calls = 0

    def Event(self, enable_timing: bool = False) -> _FakeEvent:
        self.events_created += 1
        self.enable_timing_flags.append(enable_timing)
        return _FakeEvent(self)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeEvent:
    """Stub for `torch.cuda.Event`: deterministic `elapsed_time` replay."""

    def __init__(self, cuda: _FakeCuda) -> None:
        self._cuda = cuda

    def record(self) -> None:
        self._cuda.records.append("record")

    def synchronize(self) -> None:
        self._cuda.synchronizes.append("synchronize")

    def elapsed_time(self, other: _FakeEvent) -> float:
        del other
        return self._cuda.elapsed_values.pop(0)


class _FakeTorch:
    """Stub for the session's lazy `torch` handle (CUDA side only)."""

    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda
        self.uint8 = "uint8"
        self.saved: list[tuple[Any, str]] = []

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()

    def cat(self, tensors: list[Any], dim: int = 1) -> _FakeLatents:
        del tensors, dim
        return _FakeLatents()

    def save(self, obj: Any, path: str) -> None:
        self.saved.append((obj, path))


class _FakeLatents:
    """Stub latent tensor: detach/cpu are identity, decode reads `shape`."""

    shape = (1, 8)

    def detach(self) -> _FakeLatents:
        return self

    def cpu(self) -> _FakeLatents:
        return self


class _FakeEmbeds:
    def detach(self) -> _FakeEmbeds:
        return self

    def cpu(self) -> _FakeEmbeds:
        return self


class _FakeFrame:
    def numpy(self) -> bytes:
        return b"frame"


class _FakeVideo:
    """Stub decoded video: 2 frames of 4x6, uint8-ready."""

    shape = (1, 2, 4, 6)

    def cpu(self) -> _FakeVideo:
        return self

    def to(self, dtype: Any) -> _FakeVideo:
        del dtype
        return self

    def __getitem__(self, key: Any) -> _FakeFrame:
        del key
        return _FakeFrame()

    def __rmul__(self, other: Any) -> _FakeVideo:
        del other
        return self


class _FakeVaeModel:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_cache(self) -> None:
        self.clear_calls += 1


class _FakeVae:
    def __init__(self) -> None:
        self.model = _FakeVaeModel()
        self.moved: list[str] = []

    def to(self, device: Any) -> _FakeVae:
        self.moved.append(str(device))
        return self

    def decode_to_pixel_chunk(
        self, latents: Any, use_cache: bool = False, chunk_size: int = 0
    ) -> Any:
        del latents, use_cache, chunk_size
        return object()


class _FakeGenerator:
    def __init__(self) -> None:
        self.moved: list[str] = []

    def to(self, device: Any) -> _FakeGenerator:
        self.moved.append(str(device))
        return self


class _FakePipe:
    def __init__(self) -> None:
        self.vae = _FakeVae()
        self.generator = _FakeGenerator()


class _FakeStream:
    """Stub stream: one 8-frame block per `append_block`, fixed clock."""

    def __init__(self) -> None:
        self.next_start_frame = 8
        self.blocks_appended = 1
        self.begun: list[tuple[int, int]] = []
        self.offloads = 0
        self.restores = 0

    def begin_sequence(self, total_blocks: int, seed: int) -> None:
        self.begun.append((total_blocks, seed))

    def append_block(self, prompt: str, cut: bool = False) -> _FakeLatents:
        del prompt, cut
        return _FakeLatents()

    def _encode(self, prompt: str) -> tuple[dict[str, _FakeEmbeds], list[Any]]:
        del prompt
        return ({"prompt_embeds": _FakeEmbeds()}, [])

    def offload_caches(self) -> None:
        self.offloads += 1

    def restore_caches(self) -> None:
        self.restores += 1

    def noise_rng_state(self) -> bytes:
        return b"rng"


def _install_media_stubs(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stub `imageio.v2.get_writer` + `einops.rearrange` (slim has neither)."""
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

    def _get_writer(*args: Any, **kwargs: Any) -> _Writer:
        del args, kwargs
        return _Writer()

    v2_mod.get_writer = _get_writer  # type: ignore[attr-defined]
    imageio_mod.v2 = v2_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2_mod)
    einops_mod = types.ModuleType("einops")
    einops_mod.rearrange = lambda tensor, pattern: _FakeVideo()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "einops", einops_mod)
    return written


def _session_self(cuda: _FakeCuda) -> Any:
    torch = _FakeTorch(cuda)
    return types.SimpleNamespace(
        _torch=torch,
        _pipeline=_FakePipe(),
        _stream=_FakeStream(),
        _device="cuda:0",
        _profile="longlive2-bf16-fp8",
        _latent_shape=[1, 8, 8, 8, 8],
    )


def test_stage_names_cover_all_pipeline_stages() -> None:
    assert _STAGE_NAMES == (
        "offload_vae_to_cpu_ms",
        "denoise_blocks_ms",
        "tape_encode_ms",
        "offload_for_decode_ms",
        "vae_decode_ms",
        "restore_after_decode_ms",
        "media_write_ms",
    )


def test_stage_timer_reports_event_elapsed() -> None:
    cuda = _FakeCuda(elapsed_values=[12.5])
    timer = _CudaStageTimer(_FakeTorch(cuda))
    timer.start("denoise_blocks_ms")
    timer.stop("denoise_blocks_ms")
    assert timer.stage_ms["denoise_blocks_ms"] == pytest.approx(12.5)
    assert cuda.events_created == 2
    assert cuda.enable_timing_flags == [True, True]
    assert cuda.records == ["record", "record"]
    assert cuda.synchronizes == ["synchronize"]


def test_generate_blocks_off_path_has_zero_timing_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default call creates no events, synchronizes nothing, adds no keys."""
    _install_media_stubs(monkeypatch)
    cuda = _FakeCuda()
    session_self = _session_self(cuda)
    output = tmp_path / "seg.mp4"
    result = video_longlive.LongLiveSession.generate_blocks(
        session_self,
        prompts=["a meadow"],
        seeds=[11],
        scene_cuts=[False],
        output_path=output,
        fps=24,
    )
    assert cuda.events_created == 0
    assert cuda.synchronizes == []
    assert "stage_ms" not in result
    assert result["frames"] == 2
    assert result["blocks"] == 1
    assert result["fps"] == 24
    assert result["width"] == 6
    assert result["height"] == 4


def test_generate_blocks_on_path_reports_deterministic_stage_ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in call times all seven stages with one sync each."""
    _install_media_stubs(monkeypatch)
    expected = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    cuda = _FakeCuda(elapsed_values=list(expected))
    session_self = _session_self(cuda)
    result = video_longlive.LongLiveSession.generate_blocks(
        session_self,
        prompts=["a meadow"],
        seeds=[11],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
        fps=24,
        profile_stages=True,
    )
    stage_ms = result["stage_ms"]
    assert set(stage_ms) == set(_STAGE_NAMES)
    for name, millis in zip(_STAGE_NAMES, expected, strict=True):
        assert stage_ms[name] == pytest.approx(millis)
    assert cuda.events_created == 2 * len(_STAGE_NAMES)
    assert cuda.enable_timing_flags == [True] * (2 * len(_STAGE_NAMES))
    assert cuda.synchronizes == ["synchronize"] * len(_STAGE_NAMES)


class _StubSession:
    """Stand-in for the resident session: records flags, replays video dicts."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def generate_blocks(self, **kwargs: Any) -> dict[str, Any]:
        self.seen.append(kwargs)
        # Grounded placeholder (issue 090): the stub never writes a tape,
        # so the path names a file under the call's own output dir instead
        # of a hardcoded /tmp string no test could stat.
        output_parent = Path(str(kwargs.get("output_path", "segment.mp4"))).parent
        video: dict[str, Any] = {"frames": 29, "recovery_path": str(output_parent / "recovery.pt")}
        if kwargs.get("profile_stages"):
            video["stage_ms"] = dict.fromkeys(_STAGE_NAMES, 3.25)
        return video


def test_handle_generate_blocks_passes_profile_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub = _StubSession()
    monkeypatch.setattr(video_longlive, "_SESSION", stub)
    base: dict[str, Any] = {
        "segment_id": "000000",
        "prompt": "a meadow",
        "seed": 11,
        "output_path": str(tmp_path / "seg.mp4"),
        "fps": 24,
    }
    off = video_longlive.handle_generate_blocks(dict(base))
    assert stub.seen[-1]["profile_stages"] is False
    assert "stage_ms" not in off["video"]
    on = video_longlive.handle_generate_blocks({**base, "profile_stages": True})
    assert stub.seen[-1]["profile_stages"] is True
    assert set(on["video"]["stage_ms"]) == set(_STAGE_NAMES)


def test_handle_benchmark_stage_shape_on_and_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off: existing keys byte-identical. On: per-block splits alongside."""
    stub = _StubSession()
    monkeypatch.setattr(video_longlive, "_SESSION", stub)

    cuda_calls: list[str] = []

    class _BenchCuda:
        def reset_peak_memory_stats(self) -> None:
            cuda_calls.append("reset")

        def max_memory_allocated(self) -> int:
            cuda_calls.append("peak")
            return 2 * 1024**3

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_BenchCuda()))
    existing = {
        "backend",
        "warmup_blocks",
        "measured_blocks",
        "frames_per_block",
        "block_wall_seconds",
        "blocks_per_second",
        "vram_peak_gib",
        "vram_avg_gib",
    }
    off = video_longlive.handle_benchmark({"warmup": 0, "measured": 2})
    assert set(off) == existing
    assert len(off["block_wall_seconds"]) == 2
    on = video_longlive.handle_benchmark({"warmup": 0, "measured": 2, "profile_stages": True})
    assert set(on) == existing | {"stage_ms_per_block"}
    splits = on["stage_ms_per_block"]
    assert len(splits) == len(on["block_wall_seconds"]) == 2
    for entry in splits:
        assert set(entry) == set(_STAGE_NAMES)
        for millis in entry.values():
            assert isinstance(millis, float)
            assert millis >= 0.0

"""Issue 015 fusion: VAE stays parked on CPU across tape-encode (CPU-only).

The decode prologue now returns the VAE to the device AFTER the generator
+ caches move aside (free-before-allocate), and both mid-segment
`empty_cache` calls run after `gc.collect` (fragmentation — the evict()
lesson). Every dependency is faked: asserts the device-move order, the
gc-before-empty_cache pairing, and the unchanged seven-stage timer keys.
"""

from __future__ import annotations

import contextlib
import gc
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage.workers import video_longlive
from voyage.workers.video_longlive import _STAGE_NAMES


class _FakeCuda:
    """Stub `torch.cuda`: records empty_cache calls on the shared log."""

    def __init__(self, events: list[tuple[str, ...]]) -> None:
        self._events = events

    def Event(self, enable_timing: bool = False) -> _FakeEvent:
        del enable_timing
        return _FakeEvent()

    def empty_cache(self) -> None:
        self._events.append(("empty_cache",))


class _FakeEvent:
    """Stub `torch.cuda.Event`: deterministic zero elapsed."""

    def record(self) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, other: _FakeEvent) -> float:
        del other
        return 1.0


class _FakeTorch:
    """Stub session torch handle: event-logged cache flushes, saved tapes."""

    def __init__(self, events: list[tuple[str, ...]]) -> None:
        self.cuda = _FakeCuda(events)
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
    def clear_cache(self) -> None:
        return None


class _FakeVae:
    def __init__(self, events: list[tuple[str, ...]]) -> None:
        self._events = events
        self.model = _FakeVaeModel()

    def to(self, device: Any) -> _FakeVae:
        self._events.append(("vae", str(device)))
        return self

    def decode_to_pixel_chunk(
        self, latents: Any, use_cache: bool = False, chunk_size: int = 0
    ) -> Any:
        del latents, use_cache, chunk_size
        return object()


class _FakeGenerator:
    def __init__(self, events: list[tuple[str, ...]]) -> None:
        self._events = events

    def to(self, device: Any) -> _FakeGenerator:
        self._events.append(("generator", str(device)))
        return self


class _FakePipe:
    def __init__(self, events: list[tuple[str, ...]]) -> None:
        self.vae = _FakeVae(events)
        self.generator = _FakeGenerator(events)


class _FakeStream:
    """Stub stream: one 8-frame block per `append_block`, fixed clock."""

    def __init__(self) -> None:
        self.next_start_frame = 8
        self.blocks_appended = 1

    def begin_sequence(self, total_blocks: int, seed: int) -> None:
        del total_blocks, seed

    def append_block(self, prompt: str, cut: bool = False) -> _FakeLatents:
        del prompt, cut
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


def _install_media_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `imageio.v2.get_writer` + `einops.rearrange` (slim has neither)."""

    class _Writer:
        def __enter__(self) -> _Writer:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def append_data(self, frame: Any) -> None:
            del frame

    imageio_mod = types.ModuleType("imageio")
    v2_mod = types.ModuleType("imageio.v2")
    v2_mod.get_writer = lambda *args, **kwargs: _Writer()  # type: ignore[attr-defined]
    imageio_mod.v2 = v2_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2_mod)
    einops_mod = types.ModuleType("einops")
    einops_mod.rearrange = lambda tensor, pattern: _FakeVideo()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "einops", einops_mod)


def _run_generate_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, ...]]
) -> dict[str, Any]:
    """One fused generate_blocks call with gc/collect spy + media stubs."""
    _install_media_stubs(monkeypatch)
    monkeypatch.setattr(gc, "collect", lambda: events.append(("gc",)))
    torch = _FakeTorch(events)
    session_self = types.SimpleNamespace(
        _torch=torch,
        _pipeline=_FakePipe(events),
        _stream=_FakeStream(),
        _device="cuda:0",
        _profile="longlive2-bf16-fp8",
        _latent_shape=[1, 8, 8, 8, 8],
    )
    result = video_longlive.LongLiveSession.generate_blocks(
        session_self,
        prompts=["a meadow"],
        seeds=[11],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
        fps=24,
    )
    assert torch.saved, "recovery tape must be written"
    return result


def test_vae_returns_after_generator_moves_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fusion order: vae-cpu first, generator-cpu before vae-device."""
    events: list[tuple[str, ...]] = []
    _run_generate_blocks(tmp_path, monkeypatch, events)
    moves = [event for event in events if event[0] in ("vae", "generator")]
    assert moves[0] == ("vae", "cpu")
    assert ("generator", "cpu") in moves
    assert ("vae", "cuda:0") in moves
    assert ("generator", "cuda:0") in moves
    assert moves.index(("generator", "cpu")) < moves.index(("vae", "cuda:0"))
    assert moves.index(("vae", "cuda:0")) < moves.index(("generator", "cuda:0"))


def test_vae_stays_parked_across_tape_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tape lands while the VAE is still on CPU (no early restore)."""
    events: list[tuple[str, ...]] = []
    _install_media_stubs(monkeypatch)
    monkeypatch.setattr(gc, "collect", lambda: events.append(("gc",)))
    torch = _FakeTorch(events)
    pipe = _FakePipe(events)
    session_self = types.SimpleNamespace(
        _torch=torch,
        _pipeline=pipe,
        _stream=_FakeStream(),
        _device="cuda:0",
        _profile="longlive2-bf16-fp8",
        _latent_shape=[1, 8, 8, 8, 8],
    )
    video_longlive.LongLiveSession.generate_blocks(
        session_self,
        prompts=["a meadow"],
        seeds=[11],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
        fps=24,
    )
    assert torch.saved, "recovery tape must be written"
    vae_moves = [event for event in events if event[0] == "vae"]
    assert vae_moves == [("vae", "cpu"), ("vae", "cuda:0")]


def test_gc_collect_precedes_every_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fragmentation pairing: each mid-segment flush follows a collection."""
    events: list[tuple[str, ...]] = []
    _run_generate_blocks(tmp_path, monkeypatch, events)
    flushes = [index for index, event in enumerate(events) if event == ("empty_cache",)]
    assert len(flushes) == 2
    for index in flushes:
        assert events[index - 1] == ("gc",)


def test_stage_keys_unchanged_by_fusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The seven-stage audit contract survives the reorder byte-identical."""
    events: list[tuple[str, ...]] = []
    _install_media_stubs(monkeypatch)
    monkeypatch.setattr(gc, "collect", lambda: events.append(("gc",)))
    torch = _FakeTorch(events)
    session_self = types.SimpleNamespace(
        _torch=torch,
        _pipeline=_FakePipe(events),
        _stream=_FakeStream(),
        _device="cuda:0",
        _profile="longlive2-bf16-fp8",
        _latent_shape=[1, 8, 8, 8, 8],
    )
    result = video_longlive.LongLiveSession.generate_blocks(
        session_self,
        prompts=["a meadow"],
        seeds=[11],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
        fps=24,
        profile_stages=True,
    )
    assert set(result["stage_ms"]) == set(_STAGE_NAMES)
    assert result["frames"] == 2

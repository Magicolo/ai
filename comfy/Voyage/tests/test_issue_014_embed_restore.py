"""Issue 014: embed cache survives evict/rebuild via the recovery tape.

The CPU T5-XXL encode costs minutes per segment, and the cache lived in
the session object that `evict()` destroys — every audio take forced a
cold re-encode of the identical prompt. The tape already persisted the
tail embeds; `restore_tail_embed_cache` re-seeds the LRU from it in
`resume_from_tape`.

CPU-only: `LongLiveStreamSession` is built without `__init__` (GPU) and
`torch` is stubbed in `sys.modules` (the function imports it lazily).
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage.workers import video_common, video_longlive
from voyage.workers.video_common import EmbedCache
from voyage.workers.video_longlive import LongLiveStreamSession, restore_tail_embed_cache


class _FakeTensor:
    """Recording stand-in for a torch tensor (device moves are visible)."""

    def __init__(self, label: str, frames: int = 8) -> None:
        self.label = label
        self.shape = (1, frames, 4, 4, 4)
        self.moved_to: list[str] = []
        self.cpu_calls = 0

    def to(self, device: Any) -> _FakeTensor:
        self.moved_to.append(str(device))
        return self

    def cpu(self) -> _FakeTensor:
        self.cpu_calls += 1
        return self

    def detach(self) -> _FakeTensor:
        return self


class _FakeGenerator:
    def __init__(self) -> None:
        self.state: Any = None

    def set_state(self, state: Any) -> None:
        self.state = state

    def get_state(self) -> _FakeTensor:
        return _FakeTensor("rng-state")


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorchModule(types.ModuleType):
    """Stub for the lazy `import torch` in `resume_from_tape`."""

    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = _FakeCuda()
        self.bfloat16 = "bfloat16"
        self.int64 = "int64"

    def zeros(self, shape: Any, device: Any = None, dtype: Any = None) -> _FakeTensor:
        del device, dtype
        return _FakeTensor("timestep", frames=int(shape[1]))

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()

    def Generator(self, device: Any = None) -> _FakeGenerator:
        del device
        return _FakeGenerator()

    def as_tensor(self, value: Any) -> _FakeTensor:
        fake = _FakeTensor("tape-state")
        fake.label = repr(value)
        return fake


class _FakeVae:
    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: Any) -> _FakeVae:
        self.moved_to.append(str(device))
        return self


class _FakePipe:
    def __init__(self) -> None:
        self.vae = _FakeVae()
        self.kv_cache_pos: Any = None
        self.kv_cache_neg: Any = None
        self.crossattn_cache_pos: Any = None
        self.crossattn_cache_neg: Any = None
        self.generator_calls = 0

    def _initialize_kv_cache(self, batch_size: int, dtype: Any, device: Any) -> None:
        del batch_size, dtype, device
        self.kv_cache_pos = []

    def _initialize_crossattn_cache(self, batch_size: int, dtype: Any, device: Any) -> None:
        del batch_size, dtype, device
        self.crossattn_cache_pos = []

    def generator(self, **kwargs: Any) -> None:
        del kwargs
        self.generator_calls += 1


def _stream_session(pipe: _FakePipe) -> LongLiveStreamSession:
    session = LongLiveStreamSession.__new__(LongLiveStreamSession)
    session._pipeline = pipe  # type: ignore[attr-defined]
    session._latent_shape = [1, 8, 4, 4, 4]  # type: ignore[attr-defined]
    session._device = "cpu"  # type: ignore[attr-defined]
    session._next_start_frame = 0  # type: ignore[attr-defined]
    session._blocks_appended = 0  # type: ignore[attr-defined]
    session._embed_cache = EmbedCache()  # type: ignore[attr-defined]
    session._noise_rng = None  # type: ignore[attr-defined]
    session._seq_noise = None  # type: ignore[attr-defined]
    session._seq_output = None  # type: ignore[attr-defined]
    session._seq_offset = 0  # type: ignore[attr-defined]
    return session


def _tail_tape(prompt: str = "river lanterns") -> dict[str, Any]:
    return {
        "tail_latents": _FakeTensor("tail"),
        "prompt_embeds": _FakeTensor("embeds"),
        "tail_prompt": prompt,
        "tail_conditionals": [{"conditional": _FakeTensor("conditional")}],
        "noise_rng_state": "fake-rng-state",
    }


def test_restore_helper_seeds_cache_from_tape() -> None:
    cache = EmbedCache()
    tape = _tail_tape("river lanterns")
    restored = restore_tail_embed_cache(cache, tape)
    assert restored == "river lanterns"
    cached = cache.get("river lanterns")
    assert cached is not None
    condition, conditionals = cached
    assert condition == {"prompt_embeds": tape["prompt_embeds"]}
    assert conditionals == tape["tail_conditionals"]


def test_restore_helper_ignores_pre_fix_tape() -> None:
    """Tapes without the new keys resume fine, just without the warm cache."""
    cache = EmbedCache()
    tape = {"tail_latents": _FakeTensor("tail"), "prompt_embeds": _FakeTensor("embeds")}
    assert restore_tail_embed_cache(cache, tape) is None
    assert len(cache) == 0


def test_restore_helper_rejects_blank_or_missing_prompt() -> None:
    cache = EmbedCache()
    assert restore_tail_embed_cache(cache, {**_tail_tape(), "tail_prompt": ""}) is None
    assert restore_tail_embed_cache(cache, {**_tail_tape(), "tail_prompt": 42}) is None
    assert restore_tail_embed_cache(cache, {**_tail_tape(), "prompt_embeds": None}) is None
    assert len(cache) == 0


def test_resume_from_tape_warms_embed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del tmp_path
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule())
    pipe = _FakePipe()
    session = _stream_session(pipe)
    position = session.resume_from_tape(_tail_tape("river lanterns"))
    assert position["next_start_frame"] == 8
    assert position["blocks_appended"] == 1
    assert pipe.generator_calls == 1
    cached = session._embed_cache.get("river lanterns")
    assert cached is not None
    condition, _conditionals = cached
    assert isinstance(condition, dict)


def test_resume_from_tape_without_tail_prompt_leaves_cache_cold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del tmp_path
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule())
    pipe = _FakePipe()
    session = _stream_session(pipe)
    tape = _tail_tape()
    del tape["tail_prompt"]
    session.resume_from_tape(tape)
    assert len(session._embed_cache) == 0


def test_longlive_module_still_imports_torch_free() -> None:
    assert video_longlive.NUM_FRAME_PER_BLOCK == 8
    assert video_common.EMBED_CACHE_CAPACITY == 8

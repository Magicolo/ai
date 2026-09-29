"""Issue 030: embed caches are bounded, CPU-side, and device-correct.

Every distinct prompt used to pin device tensors forever (drift-every-N
guarantees distinct prompts), and LTXV hardcoded mask placement to
`"cuda"`. Now: `EmbedCache` (LRU-8) in `video_common`, entries stored
CPU-side via `move_to_cpu`, moved to the session device on use via
`move_to_device`, and the CausVid stub encoder hoisted to module level
(covered in `test_issue_029_causvid_shuttle.py`).

CPU-only: sessions are built without `__init__` (CUDA/heavy models)
with fake tokenizers/encoders recording every device move.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import pytest

from voyage.workers import video_common
from voyage.workers.video_common import (
    EMBED_CACHE_CAPACITY,
    EmbedCache,
    move_to_cpu,
    move_to_device,
)
from voyage.workers.video_longlive import LongLiveStreamSession
from voyage.workers.video_ltxv import LTXVSession


class _FakeTensor:
    """Device-move-recording stand-in (moves return fresh copies)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.moved_to: list[str] = []
        self.cpu_calls = 0

    def to(self, device: Any) -> _FakeTensor:
        # A dtype-only `.to(bfloat16)` on an already-bf16 tensor is a
        # no-op upstream (returns self); device moves copy.
        if str(device) == "bfloat16":
            return self
        moved = _FakeTensor(self.label)
        moved.moved_to = [*self.moved_to, str(device)]
        return moved

    def cpu(self) -> _FakeTensor:
        moved = _FakeTensor(self.label)
        moved.cpu_calls = self.cpu_calls + 1
        return moved


def test_default_capacity_is_eight() -> None:
    assert EMBED_CACHE_CAPACITY == 8
    assert len(EmbedCache()) == 0
    assert EmbedCache().capacity == 8


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        EmbedCache(0)


def test_oldest_entry_evicts_past_capacity() -> None:
    cache = EmbedCache(3)
    cache.put("first", 1)
    cache.put("second", 2)
    cache.put("third", 3)
    cache.put("fourth", 4)
    assert len(cache) == 3
    assert "first" not in cache
    assert "fourth" in cache


def test_hits_refresh_recency() -> None:
    cache = EmbedCache(2)
    cache.put("first", 1)
    cache.put("second", 2)
    assert cache.get("first") == 1
    cache.put("third", 3)
    assert "first" in cache
    assert "second" not in cache


def test_miss_returns_none_and_put_overwrites() -> None:
    cache = EmbedCache(2)
    assert cache.get("missing") is None
    cache.put("key", 1)
    cache.put("key", 2)
    assert cache.get("key") == 2
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert "key" not in cache


def test_move_helpers_recurse_and_preserve_shapes() -> None:
    embeds = _FakeTensor("embeds")
    mask = _FakeTensor("mask")
    nested = {"prompt_embeds": embeds, "extra": [mask, (mask, 42, "text", None)]}
    moved = move_to_device(nested, "cuda:1")
    assert moved["prompt_embeds"].moved_to == ["cuda:1"]
    assert moved["extra"][0].moved_to == ["cuda:1"]
    assert isinstance(moved["extra"][1], tuple)
    assert moved["extra"][1][1:] == (42, "text", None)
    back = move_to_cpu(moved)
    assert back["prompt_embeds"].cpu_calls == 1
    # Originals untouched: moves always copy.
    assert embeds.moved_to == []
    assert mask.moved_to == []


def test_move_helpers_pass_plain_values_through() -> None:
    assert move_to_cpu(42) == 42
    assert move_to_device("text", "cuda:0") == "text"
    assert move_to_cpu(None) is None


class _FakeInputs:
    def __init__(self) -> None:
        self.input_ids = _FakeTensor("input-ids")
        self.attention_mask = _FakeTensor("mask")


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **kwargs: Any) -> _FakeInputs:
        del text, kwargs
        self.calls += 1
        return _FakeInputs()


class _FakeTextEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids: Any) -> list[_FakeTensor]:
        del input_ids
        self.calls += 1
        return [_FakeTensor("embeds")]


class _FakeTorch:
    bfloat16 = "bfloat16"

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()


def _ltxv_session(device: str = "cuda:1") -> tuple[LTXVSession, _FakeTokenizer, _FakeTextEncoder]:
    session = LTXVSession.__new__(LTXVSession)
    tokenizer = _FakeTokenizer()
    encoder = _FakeTextEncoder()
    session._torch = _FakeTorch()  # type: ignore[attr-defined]
    session._device = device  # type: ignore[attr-defined]
    session._tokenizer = tokenizer  # type: ignore[attr-defined]
    session._text_encoder = encoder  # type: ignore[attr-defined]
    session._embed_cache = EmbedCache()  # type: ignore[attr-defined]
    session._negative = None  # type: ignore[attr-defined]
    return session, tokenizer, encoder


def test_ltxv_encode_caches_and_moves_to_session_device() -> None:
    session, tokenizer, _encoder = _ltxv_session("cuda:1")
    first_embeds, first_mask = session._encode("lantern valley")
    second_embeds, second_mask = session._encode("lantern valley")
    assert tokenizer.calls == 1
    # Both tensors ride to the session device on every use — including hits.
    assert first_mask.moved_to == ["cuda:1"]
    assert second_mask.moved_to == ["cuda:1"]
    assert first_embeds.moved_to == ["cuda:1"]
    assert second_embeds.moved_to == ["cuda:1"]
    # ... while the stored entry stays CPU-side (never device-pinned).
    stored = session._embed_cache.get("lantern valley")
    assert stored is not None
    stored_embeds, stored_mask = stored
    assert stored_embeds.moved_to == []
    assert stored_mask.moved_to == []


def test_ltxv_cache_evicts_stale_prompts() -> None:
    session, _tokenizer, _encoder = _ltxv_session()
    for index in range(EMBED_CACHE_CAPACITY + 2):
        session._encode(f"valley {index}")
    assert len(session._embed_cache) == EMBED_CACHE_CAPACITY
    assert "valley 0" not in session._embed_cache


def _longlive_session() -> LongLiveStreamSession:
    session = LongLiveStreamSession(pipeline=None, latent_shape=[1, 8, 4, 4, 4], device="cpu")
    session._pipeline = types.SimpleNamespace(text_encoder=object())  # type: ignore[attr-defined]
    return session


def test_longlive_encode_stores_cpu_side_and_hits_without_reencode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fake_blocks(text_encoder: Any, batched: list[list[str]], count: int) -> Any:
        del text_encoder, count
        calls.append(batched[0][0])
        return {"prompt_embeds": _FakeTensor("embeds")}, [{"conditional": _FakeTensor("cond")}]

    conditioning_module = types.ModuleType("utils.prompt_conditioning")
    conditioning_module.encode_prompt_blocks = _fake_blocks  # type: ignore[attr-defined]
    package_module = types.ModuleType("utils")
    package_module.prompt_conditioning = conditioning_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils", package_module)
    monkeypatch.setitem(sys.modules, "utils.prompt_conditioning", conditioning_module)
    session = _longlive_session()
    first = session._encode("lantern valley")
    second = session._encode("lantern valley")
    assert calls == ["lantern valley"]
    stored = session._embed_cache.get("lantern valley")
    assert stored is not None
    stored_condition, stored_list = stored
    assert stored_condition["prompt_embeds"].cpu_calls == 1
    assert stored_list[0]["conditional"].cpu_calls == 1
    # Hits move the CPU entry back to the session device.
    assert first[0]["prompt_embeds"].cpu_calls == 0
    assert second[0]["prompt_embeds"].moved_to == ["cpu"]


def test_video_common_surface_intact() -> None:
    assert video_common.TAIL_FILENAME == "video_tail.mp4"
    assert video_common.TAPE_FILENAME == "recovery.pt"

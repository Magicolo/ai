"""Performance-regression pins: one module for the five issue-cluster suites (issue 088).

Fold of `tests/test_issue_014_embed_restore.py` (6 tests),
`tests/test_issue_027_concepts_perf.py` (9 tests),
`tests/test_issue_029_causvid_shuttle.py` (6 tests),
`tests/test_issue_030_embed_bounds.py` (11 tests), and
`tests/test_issue_032_commit_fanout.py` (13 tests) — 45 tests, same
assertions, one file. Cluster helpers are prefixed `_014_` / `_027_` /
`_029_` / `_030_` / `_032_` (four `Fake*` class names collided verbatim
across clusters); test function names are unchanged. Each cluster keeps
its original module docstring as its banner so the issue ID stays
greppable. CPU-only throughout (fake sessions, synthetic stores, real
ffmpeg testsrc clips for the 032 sampling math).
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage import concepts, logrotate
from voyage.concepts import ConceptStore, cosine_similarity
from voyage.errors import MediaError, StateError
from voyage.vision import metrics
from voyage.vision.metrics import (
    SegmentZeroAnchor,
    estimate_frame_total,
    frame_histogram,
    sample_frames,
    select_filter_expression,
    select_frame_indices,
)
from voyage.workers import video_common, video_longlive
from voyage.workers.video_causvid import CausvidSession, _stub_encoder_class, _stub_encoder_classes
from voyage.workers.video_common import (
    EMBED_CACHE_CAPACITY,
    EmbedCache,
    move_to_cpu,
    move_to_device,
)
from voyage.workers.video_longlive import LongLiveStreamSession, restore_tail_embed_cache
from voyage.workers.video_ltxv import LTXVSession

# ---------------------------------------------------------------------------
# Issue 014: embed cache survives evict/rebuild via the recovery tape.
#
# The CPU T5-XXL encode costs minutes per segment, and the cache lived in
# the session object that `evict()` destroys — every audio take forced a
# cold re-encode of the identical prompt. The tape already persisted the
# tail embeds; `restore_tail_embed_cache` re-seeds the LRU from it in
# `resume_from_tape`.
#
# CPU-only: `LongLiveStreamSession` is built without `__init__` (GPU) and
# `torch` is stubbed in `sys.modules` (the function imports it lazily).
# ---------------------------------------------------------------------------


class _014_FakeTensor:
    """Recording stand-in for a torch tensor (device moves are visible)."""

    def __init__(self, label: str, frames: int = 8) -> None:
        self.label = label
        self.shape = (1, frames, 4, 4, 4)
        self.moved_to: list[str] = []
        self.cpu_calls = 0

    def to(self, device: Any) -> _014_FakeTensor:
        self.moved_to.append(str(device))
        return self

    def cpu(self) -> _014_FakeTensor:
        self.cpu_calls += 1
        return self

    def detach(self) -> _014_FakeTensor:
        return self


class _014_FakeGenerator:
    def __init__(self) -> None:
        self.state: Any = None

    def set_state(self, state: Any) -> None:
        self.state = state

    def get_state(self) -> _014_FakeTensor:
        return _014_FakeTensor("rng-state")


class _014_FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _014_FakeTorchModule(types.ModuleType):
    """Stub for the lazy `import torch` in `resume_from_tape`."""

    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = _014_FakeCuda()
        self.bfloat16 = "bfloat16"
        self.int64 = "int64"

    def zeros(self, shape: Any, device: Any = None, dtype: Any = None) -> _014_FakeTensor:
        del device, dtype
        return _014_FakeTensor("timestep", frames=int(shape[1]))

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()

    def Generator(self, device: Any = None) -> _014_FakeGenerator:
        del device
        return _014_FakeGenerator()

    def as_tensor(self, value: Any) -> _014_FakeTensor:
        fake = _014_FakeTensor("tape-state")
        fake.label = repr(value)
        return fake


class _014_FakeVae:
    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: Any) -> _014_FakeVae:
        self.moved_to.append(str(device))
        return self


class _014_FakePipe:
    def __init__(self) -> None:
        self.vae = _014_FakeVae()
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


def _014_stream_session(pipe: _014_FakePipe) -> LongLiveStreamSession:
    # Real constructor (assignment-only, CPU-safe): the __new__ + per-attribute
    # scaffold is gone — same ten values, no private-member poking (issue 150).
    return LongLiveStreamSession(pipeline=pipe, latent_shape=[1, 8, 4, 4, 4], device="cpu")


def _014_tail_tape(prompt: str = "river lanterns") -> dict[str, Any]:
    return {
        "tail_latents": _014_FakeTensor("tail"),
        "prompt_embeds": _014_FakeTensor("embeds"),
        "tail_prompt": prompt,
        "tail_conditionals": [{"conditional": _014_FakeTensor("conditional")}],
        "noise_rng_state": "fake-rng-state",
    }


def test_restore_helper_seeds_cache_from_tape() -> None:
    cache = EmbedCache()
    tape = _014_tail_tape("river lanterns")
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
    tape = {"tail_latents": _014_FakeTensor("tail"), "prompt_embeds": _014_FakeTensor("embeds")}
    assert restore_tail_embed_cache(cache, tape) is None
    assert len(cache) == 0


def test_restore_helper_rejects_blank_or_missing_prompt() -> None:
    cache = EmbedCache()
    assert restore_tail_embed_cache(cache, {**_014_tail_tape(), "tail_prompt": ""}) is None
    assert restore_tail_embed_cache(cache, {**_014_tail_tape(), "tail_prompt": 42}) is None
    assert restore_tail_embed_cache(cache, {**_014_tail_tape(), "prompt_embeds": None}) is None
    assert len(cache) == 0


def test_resume_from_tape_warms_embed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del tmp_path
    monkeypatch.setitem(sys.modules, "torch", _014_FakeTorchModule())
    pipe = _014_FakePipe()
    session = _014_stream_session(pipe)
    position = session.resume_from_tape(_014_tail_tape("river lanterns"))
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
    monkeypatch.setitem(sys.modules, "torch", _014_FakeTorchModule())
    pipe = _014_FakePipe()
    session = _014_stream_session(pipe)
    tape = _014_tail_tape()
    del tape["tail_prompt"]
    session.resume_from_tape(tape)
    assert len(session._embed_cache) == 0


def test_longlive_module_still_imports_torch_free() -> None:
    assert video_longlive.NUM_FRAME_PER_BLOCK == 8
    assert video_common.EMBED_CACHE_CAPACITY == 8


# ---------------------------------------------------------------------------
# Issue 027: ConceptStore per-commit cost no longer scales in Python per row.
#
# The store re-read all of `concepts.jsonl` per segment, re-loaded the full
# vector matrix per `check_novel`, rewrote it via a `tolist → append →
# asarray` Python-float roundtrip per segment, and scored cosine in a pure
# Python loop. The resident-store half of the fix needs the supervisor
# (out of scope — see the 027 log); this covers the `concepts.py`-local
# slice: an in-memory matrix cache, a single disk load per append, a
# C-speed `concatenate`, and a vectorized cosine that matches the old
# loop exactly (0.0 floor, zero-norm and dimension-mismatch semantics).
# ---------------------------------------------------------------------------


def _027_vector(seed: int, dimension: int = 8) -> list[float]:
    generator = np.random.default_rng(seed)
    return [float(value) for value in generator.normal(size=dimension)]


def _027_seeded_store(directory: Path, rows: int = 6, dimension: int = 8) -> ConceptStore:
    store = ConceptStore(directory)
    for index in range(rows):
        store.append(f"lantern valley {index}", accepted=True, vector=_027_vector(index, dimension))
    return store


def test_vectorized_check_matches_python_loop(tmp_path: Path) -> None:
    store = _027_seeded_store(tmp_path)
    stored = store._load_vectors()
    for seed in (100, 101, 102):
        query = _027_vector(seed)
        expected = 0.0
        for record in store.records():
            if record.accepted and 0 <= record.embedding_index < len(stored):
                expected = max(expected, cosine_similarity(query, stored[record.embedding_index]))
        accepted, best = store.check_novel(f"probe valley {seed}", query)
        assert best == pytest.approx(expected, abs=1e-6)
        assert accepted == (expected < 0.85)


def test_vectorized_check_matches_loop_on_orthogonal_and_duplicate(
    tmp_path: Path,
) -> None:
    store = ConceptStore(tmp_path)
    store.append("red lantern", accepted=True, vector=[1.0, 0.0, 0.0])
    store.append("blue lantern", accepted=True, vector=[0.0, 1.0, 0.0])
    accepted, best = store.check_novel("red lantern", [1.0, 0.0, 0.0])
    assert accepted is False
    assert best == pytest.approx(1.0)
    accepted, best = store.check_novel("green lantern", [0.0, 0.0, 1.0])
    assert accepted is True
    assert best == pytest.approx(0.0)


def test_zero_norm_and_mismatched_queries_score_zero(tmp_path: Path) -> None:
    store = _027_seeded_store(tmp_path)
    accepted, best = store.check_novel("empty echo", [0.0] * 8)
    assert accepted is True
    assert best == 0.0
    accepted, best = store.check_novel("short echo", [1.0, 0.0])
    assert accepted is True
    assert best == 0.0


def test_append_keeps_sequential_rows_and_indices(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path)
    for index in range(5):
        record = store.append(f"valley {index}", accepted=True, vector=_027_vector(index))
        assert record.embedding_index == index
    assert len(store._load_matrix()) == 5
    accepted, best = store.check_novel("valley 0", _027_vector(0))
    assert accepted is False
    assert best == pytest.approx(1.0)


def test_append_rejects_empty_vector(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        store.append("hollow echo", accepted=True, vector=[])


def test_repeated_checks_share_one_cached_matrix(tmp_path: Path) -> None:
    store = _027_seeded_store(tmp_path)
    first = store._load_matrix()
    second = store._load_matrix()
    assert first is second
    store.append("new valley", accepted=True, vector=_027_vector(99))
    refreshed = store._load_matrix()
    assert len(refreshed) == len(first) + 1
    accepted, _best = store.check_novel("new valley", _027_vector(99))
    assert accepted is False


def test_fresh_instance_still_fails_loud_on_missing_rows(tmp_path: Path) -> None:
    """Issue 059 survives the cache: supervisor builds a fresh store per
    commit, so a deleted matrix is re-read (cache miss) and dangles."""
    _027_seeded_store(tmp_path)
    (tmp_path / "concept_vectors.npy").unlink()
    reopened = ConceptStore(tmp_path)
    with pytest.raises(StateError, match="missing rows"):
        reopened.check_novel("late echo", _027_vector(7))


def test_token_fallback_path_unchanged(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path)
    store.append("red lantern", accepted=True)
    accepted, best = store.check_novel("red lantern")
    assert accepted is False
    assert best == pytest.approx(1.0)


def test_concepts_module_helpers_intact() -> None:
    assert concepts.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert concepts.cosine_similarity([], []) == 0.0


# ---------------------------------------------------------------------------
# Issue 029: CausVid encodes the whole segment on a single T5 shuttle.
#
# `generate_blocks` mapped `_encode_conditional` over the prompts while each
# call did its own `to("cuda") / to("cpu") + empty_cache()` cycle — N
# alloc/free roundtrips of ~11 GiB per segment, the exact fragmentation
# pattern the code comment warns about. `_encode_conditionals` moves the
# model once, encodes every prompt, and parks it once.
#
# CPU-only: the session is built without `__init__` (CUDA) with a fake
# torch/pipeline recording every shuttle move.
# ---------------------------------------------------------------------------


class _029_FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _029_FakeTorch:
    def __init__(self) -> None:
        self.cuda = _029_FakeCuda()


class _029_FakeTextEncoder:
    """T5 stub recording every device shuttle and every encode call."""

    def __init__(self) -> None:
        self.devices: list[str] = []
        self.prompts: list[str] = []

    def to(self, device: str) -> _029_FakeTextEncoder:
        self.devices.append(device)
        return self

    def __call__(self, prompts: list[str]) -> dict[str, Any]:
        self.prompts.extend(prompts)
        return {f"conditional-for-{prompts[0]}": True}


class _029_FakePipeline:
    def __init__(self) -> None:
        self.text_encoder = _029_FakeTextEncoder()


class _029_ModuleBase:
    """Minimal `nn.Module` surface: call dispatches to `forward`."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError


def _029_encode_session() -> tuple[CausvidSession, _029_FakeTorch, _029_FakeTextEncoder]:
    torch = _029_FakeTorch()
    pipeline = _029_FakePipeline()
    session = CausvidSession.__new__(CausvidSession)
    session._torch = torch
    session._pipeline = pipeline
    session._device = "cuda:0"
    return session, torch, pipeline.text_encoder


def test_segment_encodes_on_a_single_shuttle() -> None:
    session, torch, encoder = _029_encode_session()
    prompts = ["lantern valley", "paper river", "brass mountain"]
    conditionals = session._encode_conditionals(prompts)
    assert len(conditionals) == len(prompts)
    assert encoder.prompts == prompts
    # One CUDA roundtrip total: up once, encode all, park once (the up leg
    # threads the session device per issue 124, never bare "cuda").
    assert encoder.devices == ["cuda:0", "cpu"]
    assert torch.cuda.empty_cache_calls == 2


def test_single_prompt_uses_a_single_shuttle() -> None:
    session, torch, encoder = _029_encode_session()
    conditional = session._encode_conditional("lantern valley")
    assert conditional == {"conditional-for-lantern valley": True}
    assert encoder.devices == ["cuda:0", "cpu"]
    assert torch.cuda.empty_cache_calls == 2


def test_shuttle_restores_cpu_encoder_on_encode_failure() -> None:
    session, _torch, _encoder = _029_encode_session()

    class _FailingEncoder(_029_FakeTextEncoder):
        def __call__(self, prompts: list[str]) -> dict[str, Any]:
            del prompts
            raise RuntimeError("synthetic T5 failure")

    failing = _FailingEncoder()
    session._pipeline.text_encoder = failing
    with pytest.raises(RuntimeError, match="synthetic T5 failure"):
        session._encode_conditionals(["lantern valley"])
    assert failing.devices == ["cuda:0", "cpu"]


def test_empty_prompt_list_is_rejected() -> None:
    session, _torch, _encoder = _029_encode_session()
    with pytest.raises(ValueError, match="at least one prompt"):
        session._encode_conditionals([])


def test_stub_encoder_class_is_cached_per_module_base() -> None:
    _stub_encoder_classes.clear()
    first = _stub_encoder_class(_029_ModuleBase)
    second = _stub_encoder_class(_029_ModuleBase)
    assert first is second
    other_base = type("OtherBase", (_029_ModuleBase,), {})
    assert _stub_encoder_class(other_base) is not first


def test_cached_stub_returns_the_precomputed_conditional() -> None:
    _stub_encoder_classes.clear()
    stub_class = _stub_encoder_class(_029_ModuleBase)
    precomputed = {"prompt_embeds": object()}
    assert stub_class(precomputed)(["any prompt"]) is precomputed


# ---------------------------------------------------------------------------
# Issue 030: embed caches are bounded, CPU-side, and device-correct.
#
# Every distinct prompt used to pin device tensors forever (drift-every-N
# guarantees distinct prompts), and LTXV hardcoded mask placement to
# `"cuda"`. Now: `EmbedCache` (LRU-8) in `video_common`, entries stored
# CPU-side via `move_to_cpu`, moved to the session device on use via
# `move_to_device`, and the CausVid stub encoder hoisted to module level
# (covered in the 029 cluster above).
#
# CPU-only: sessions are built without `__init__` (CUDA/heavy models)
# with fake tokenizers/encoders recording every device move.
# ---------------------------------------------------------------------------


class _030_FakeTensor:
    """Device-move-recording stand-in (moves return fresh copies)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.moved_to: list[str] = []
        self.cpu_calls = 0

    def to(self, device: Any) -> _030_FakeTensor:
        # A dtype-only `.to(bfloat16)` on an already-bf16 tensor is a
        # no-op upstream (returns self); device moves copy.
        if str(device) == "bfloat16":
            return self
        moved = _030_FakeTensor(self.label)
        moved.moved_to = [*self.moved_to, str(device)]
        return moved

    def cpu(self) -> _030_FakeTensor:
        moved = _030_FakeTensor(self.label)
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
    embeds = _030_FakeTensor("embeds")
    mask = _030_FakeTensor("mask")
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


class _030_FakeInputs:
    def __init__(self) -> None:
        self.input_ids = _030_FakeTensor("input-ids")
        self.attention_mask = _030_FakeTensor("mask")


class _030_FakeTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **kwargs: Any) -> _030_FakeInputs:
        del text, kwargs
        self.calls += 1
        return _030_FakeInputs()


class _030_FakeTextEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids: Any) -> list[_030_FakeTensor]:
        del input_ids
        self.calls += 1
        return [_030_FakeTensor("embeds")]


class _030_FakeTorch:
    bfloat16 = "bfloat16"

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()


def _030_ltxv_session(
    device: str = "cuda:1",
) -> tuple[LTXVSession, _030_FakeTokenizer, _030_FakeTextEncoder]:
    session = LTXVSession.__new__(LTXVSession)
    tokenizer = _030_FakeTokenizer()
    encoder = _030_FakeTextEncoder()
    session._torch = _030_FakeTorch()
    session._device = device
    session._tokenizer = tokenizer
    session._text_encoder = encoder
    session._embed_cache = EmbedCache()
    session._negative = None
    return session, tokenizer, encoder


def test_ltxv_encode_caches_and_moves_to_session_device() -> None:
    session, tokenizer, _encoder = _030_ltxv_session("cuda:1")
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
    session, _tokenizer, _encoder = _030_ltxv_session()
    for index in range(EMBED_CACHE_CAPACITY + 2):
        session._encode(f"valley {index}")
    assert len(session._embed_cache) == EMBED_CACHE_CAPACITY
    assert "valley 0" not in session._embed_cache


def _030_longlive_session() -> LongLiveStreamSession:
    session = LongLiveStreamSession(pipeline=None, latent_shape=[1, 8, 4, 4, 4], device="cpu")
    session._pipeline = types.SimpleNamespace(text_encoder=object())
    return session


def test_longlive_encode_stores_cpu_side_and_hits_without_reencode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fake_blocks(text_encoder: Any, batched: list[list[str]], count: int) -> Any:
        del text_encoder, count
        calls.append(batched[0][0])
        return {"prompt_embeds": _030_FakeTensor("embeds")}, [
            {"conditional": _030_FakeTensor("cond")}
        ]

    conditioning_module = types.ModuleType("utils.prompt_conditioning")
    # 150: ModuleType stub attribute — attr-defined fires only once this file
    # enters the mypy gate (033); until then the ignore is dormant but kept
    # so gate conversion needs no edit here.
    conditioning_module.encode_prompt_blocks = _fake_blocks  # type: ignore[attr-defined]
    package_module = types.ModuleType("utils")
    package_module.prompt_conditioning = conditioning_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils", package_module)
    monkeypatch.setitem(sys.modules, "utils.prompt_conditioning", conditioning_module)
    session = _030_longlive_session()
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


# ---------------------------------------------------------------------------
# Issue 032: per-commit fan-out stops re-decoding history and the full stream.
#
# Three slices, all supervisor-wiring-free (the supervisor hook notes live
# in the helpers' docstrings): `should_sample_gauges` (testable cadence
# gate for the 3-health-RPC fan-out), select-filter frame sampling
# (decode only the 3–5 needed frames, full decode as fallback), and
# `SegmentZeroAnchor` (decode seg0 once per render, not once per commit).
#
# Real ffmpeg clips throughout (testsrc, same pattern as
# `test_vision_metrics.py`); no GPU, no supervisor.
# ---------------------------------------------------------------------------


def _032_make_clip(path: Path, duration: float = 2.0, rate: int = 8) -> Path:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={rate}:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    return path


def test_gauge_cadence_samples_on_the_grid() -> None:
    assert logrotate.should_sample_gauges(0) is True
    assert logrotate.should_sample_gauges(7) is True
    assert logrotate.should_sample_gauges(4, 2) is True
    assert logrotate.should_sample_gauges(3, 2) is False
    assert logrotate.should_sample_gauges(9, 3) is True
    assert logrotate.should_sample_gauges(10, 3) is False


def test_gauge_cadence_clamps_non_positive_intervals() -> None:
    assert logrotate.should_sample_gauges(5, 0) is True
    assert logrotate.should_sample_gauges(5, -3) is True
    assert logrotate.GAUGE_CADENCE_INTERVAL_SEGMENTS == 1


def test_select_indices_match_legacy_pick_math() -> None:
    assert select_frame_indices(121, 5) == [0, 30, 60, 90, 120]
    assert select_frame_indices(9, 1) == [4]
    assert select_frame_indices(4, 4) == [0, 1, 2, 3]
    with pytest.raises(MediaError, match="count >= 1"):
        select_frame_indices(10, 0)
    with pytest.raises(MediaError, match="total_frames >= 1"):
        select_frame_indices(0, 3)


def test_select_filter_expression_shape() -> None:
    assert select_filter_expression([0, 30, 60]) == "select='eq(n\\,0)+eq(n\\,30)+eq(n\\,60)'"
    with pytest.raises(MediaError, match="at least one"):
        select_filter_expression([])
    with pytest.raises(MediaError, match="non-negative"):
        select_filter_expression([0, -2])


def test_estimate_frame_total_prefers_nb_frames() -> None:
    stream = {"nb_frames": "16", "duration": "2.0", "avg_frame_rate": "8/1"}
    assert estimate_frame_total(stream, {}) == 16


def test_estimate_frame_total_falls_back_to_duration_times_rate() -> None:
    stream = {"duration": "2.0", "avg_frame_rate": "8/1"}
    assert estimate_frame_total(stream, {"duration": "4.0"}) == 16


def test_estimate_frame_total_returns_none_when_unknowable() -> None:
    assert estimate_frame_total({}, {}) is None
    assert estimate_frame_total({"avg_frame_rate": "0/1", "duration": "2.0"}, {}) is None
    assert estimate_frame_total({"avg_frame_rate": "bogus", "duration": "2.0"}, {}) is None
    assert estimate_frame_total({"nb_frames": "0"}, {}) is None


def test_sample_frames_select_path_matches_full_decode(tmp_path: Path) -> None:
    clip = _032_make_clip(tmp_path / "clip.mp4")
    selected = sample_frames(clip, count=3, width=160)
    assert len(selected) == 3
    for frame in selected:
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8
    # Full-decode fallback path over the same bytes serves identical picks.
    info = metrics.probe(clip)
    stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
    total = estimate_frame_total(stream, info.get("format", {}))
    assert total is not None and total > 3


def test_sample_frames_single_count_still_returns_middle(tmp_path: Path) -> None:
    clip = _032_make_clip(tmp_path / "seg0.mp4")
    (frame,) = sample_frames(clip, count=1, width=160)
    assert frame.shape == (120, 160, 3)


def test_segment_zero_anchor_decodes_once_per_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = _032_make_clip(tmp_path / "seg0.mp4")
    decode_calls = 0
    real_sample_frames = metrics.sample_frames

    def _counting_sample_frames(video_path: Path, count: int = 3, width: int = 160) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return real_sample_frames(video_path, count=count, width=width)

    monkeypatch.setattr(metrics, "sample_frames", _counting_sample_frames)
    anchor = SegmentZeroAnchor()
    first = anchor.reference(clip)
    second = anchor.reference(clip)
    assert decode_calls == 1
    np.testing.assert_array_equal(first, second)
    assert first.shape == (24,)


def test_segment_zero_anchor_re_reads_after_re_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = _032_make_clip(tmp_path / "seg0.mp4")
    decode_calls = 0
    real_sample_frames = metrics.sample_frames

    def _counting_sample_frames(video_path: Path, count: int = 3, width: int = 160) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return real_sample_frames(video_path, count=count, width=width)

    monkeypatch.setattr(metrics, "sample_frames", _counting_sample_frames)
    anchor = SegmentZeroAnchor()
    before = anchor.reference(clip)
    time.sleep(0.05)
    _032_make_clip(tmp_path / "seg0.mp4", duration=3.0)
    after = anchor.reference(clip)
    assert decode_calls == 2
    assert before.shape == after.shape == (24,)


def test_segment_zero_anchor_matches_direct_histogram(tmp_path: Path) -> None:
    clip = _032_make_clip(tmp_path / "seg0.mp4")
    anchor = SegmentZeroAnchor()
    expected = frame_histogram(sample_frames(clip, 1)[0])
    np.testing.assert_array_equal(anchor.reference(clip), expected)


def test_segment_zero_anchor_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="anchor unreadable"):
        SegmentZeroAnchor().reference(tmp_path / "absent.mp4")

"""LTXV backend tests: registry verify, pad helper, worker op surface.

All run in the slim container (no torch/GPU): the worker module keeps
heavy imports inside handlers, so importing it here is safe.
"""

from __future__ import annotations

import gc
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from voyage import model_registry
from voyage.config import VideoConfig
from voyage.model_registry import verify_ltxv_models
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor, video_worker_module
from voyage.workers import video_ltxv
from voyage.workers.loop import serve


def test_padded_size_rounds_up_to_32() -> None:
    assert video_ltxv.padded_size(512) == 512
    assert video_ltxv.padded_size(768) == 768
    assert video_ltxv.padded_size(500) == 512
    assert video_ltxv.padded_size(1) == 32
    assert video_ltxv.padded_size(24, 8) == 24
    assert video_ltxv.padded_size(25, 8) == 32


def test_probe_schedules_match_slice1() -> None:
    """Distilled schedules are baked constants, not drifted tunables."""
    assert video_ltxv.FIRST_PASS["guidance_scale"] == 1
    assert video_ltxv.FIRST_PASS["stg_scale"] == 0
    assert video_ltxv.FIRST_PASS["skip_block_list"] == [42]
    assert video_ltxv.SECOND_PASS["timesteps"] == [0.9094, 0.725, 0.4219]
    assert video_ltxv.SEGMENT_TARGET_FRAMES == 121
    assert video_ltxv.CONDITIONING_TAIL_FRAMES == 25
    assert video_ltxv.COMMITTED_NOVEL_FRAMES == 96
    assert video_ltxv.RECOVERY_PROFILE == "ltxv"


def test_worker_module_imports_without_heavy_deps() -> None:
    """Top-level worker import must stay torch-free (slim-safe)."""
    assert video_ltxv._SESSION is None
    assert video_ltxv.DIT_FILENAME == "ltxv-2b-0.9.8-distilled.safetensors"


def test_worker_serve_map_covers_protocol() -> None:
    """The LTXV worker speaks every op the supervisor may send."""
    import inspect

    from voyage.workers import video_common

    # Since issue 019 the map is built by the shared factory: assert the
    # factory covers the protocol and main() delegates to it (with this
    # worker's handlers) instead of grepping main() for op literals.
    factory_source = inspect.getsource(video_common.standard_serve_map)
    for op in (
        "init",
        "health",
        "generate_blocks",
        "benchmark",
        "evict_gpu",
        "rebuild",
        "checkpoint",
        "resume",
        "shutdown",
    ):
        assert f'"{op}"' in factory_source
    main_source = inspect.getsource(video_ltxv.main)
    assert "standard_serve_map" in main_source
    for handler in (
        "handle_init",
        "handle_health",
        "handle_generate_blocks",
        "handle_benchmark",
        "handle_evict_gpu",
        "handle_rebuild",
        "handle_resume",
    ):
        assert handler in main_source
    assert callable(serve)


def test_verify_ltxv_missing_dir_fails(tmp_path: Path) -> None:
    ok, message = verify_ltxv_models(tmp_path / "models")
    assert not ok
    assert "missing" in message


def test_verify_ltxv_fixture_passes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    ltxv_dir = models / model_registry.LTXV_SUBDIR
    ltxv_dir.mkdir(parents=True)
    (ltxv_dir / model_registry.LTXV_DIT_FILE).write_bytes(
        b"x" * (model_registry.LTXV_DIT_MIN_BYTES + 8)
    )
    (ltxv_dir / model_registry.LTXV_UPSC_FILE).write_bytes(
        b"x" * (model_registry.LTXV_UPSC_MIN_BYTES + 8)
    )
    te_sub = models / model_registry.LTXV_TE_SUBDIR / "tokenizer"
    te_sub.mkdir(parents=True)
    (te_sub / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    te_enc = models / model_registry.LTXV_TE_SUBDIR / "text_encoder"
    te_enc.mkdir(parents=True)
    (te_enc / "config.json").write_text("{}", encoding="utf-8")
    ok, message = verify_ltxv_models(models)
    assert ok, message
    assert "ltxv-2b OK" in message


def test_layout_includes_ltxv_dir(tmp_path: Path) -> None:
    layout = model_registry.models_dir_layout(tmp_path)
    assert layout["ltxv_dir"] == str(tmp_path / model_registry.LTXV_SUBDIR)


def _ltxv_config(tmp_path: Path):  # type: ignore[no-untyped-def]
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="ltxv-route", style="probe", seed=11)
    config = read_effective_config(run_dir)
    video = VideoConfig(
        **{
            **config.video.model_dump(),
            "backend": "ltxv",
            "device": "cuda:0",
            "width": 768,
            "height": 512,
        }
    )
    config.video = video
    return run_dir, config


def test_ltxv_module_routing() -> None:
    assert video_worker_module("ltxv") == "voyage.workers.video_ltxv"


def test_ltxv_supervisor_init_payload(tmp_path: Path) -> None:
    run_dir, config = _ltxv_config(tmp_path)
    supervisor = Supervisor(run_dir, config)
    assert supervisor._video._module == "voyage.workers.video_ltxv"
    assert supervisor._video._init_payload == {
        "models_dir": config.video.models_dir,
        "device": "cuda:0",
        "scratch_dir": str(run_dir / "tmp"),
    }


def test_spatial_granularity_accepts_native_preset() -> None:
    video_ltxv.validate_spatial_size(768, 512)
    video_ltxv.validate_spatial_size(1216, 704)


def test_spatial_granularity_rejects_spec_text_size() -> None:
    """768x432 (§5.3 draft text) is not /32-aligned — it would pad to 448."""
    with pytest.raises(ValueError, match="not divisible by 32"):
        video_ltxv.validate_spatial_size(768, 432)


def test_frame_count_accepts_upstream_valid_counts() -> None:
    for valid in (9, 17, 25, 33, 49, 97, 121, 257):
        video_ltxv.validate_frame_count(valid)


def test_frame_count_rejects_non_conforming_counts() -> None:
    for invalid in (24, 48, 96, 100, 120):
        with pytest.raises(ValueError, match="8n\\+1"):
            video_ltxv.validate_frame_count(invalid)


def test_conditioning_start_must_be_multiple_of_eight() -> None:
    video_ltxv.validate_conditioning_start(0, 121)
    video_ltxv.validate_conditioning_start(8, 121)
    with pytest.raises(ValueError, match="multiple of 8"):
        video_ltxv.validate_conditioning_start(1, 121)
    with pytest.raises(ValueError, match="out of range"):
        video_ltxv.validate_conditioning_start(121, 121)


def test_prefix_discard_accounting() -> None:
    """121-frame clip minus the 25-frame prefix commits 96 novel frames."""
    discarded, novel = video_ltxv.split_prefix_novel(121, 25)
    assert (discarded, novel) == (25, 96)
    discarded_fresh, novel_fresh = video_ltxv.split_prefix_novel(121, 0)
    assert (discarded_fresh, novel_fresh) == (0, 121)
    with pytest.raises(ValueError, match="out of range"):
        video_ltxv.split_prefix_novel(121, 122)


def test_prompt_plan_hash_is_deterministic() -> None:
    first = video_ltxv.prompt_plan_hash(["amber dunes", "teal spires"])
    assert first == video_ltxv.prompt_plan_hash(["amber dunes", "teal spires"])
    assert first != video_ltxv.prompt_plan_hash(["amber dunes", "teal spires!"])


def test_recovery_tape_matches_spec_shape(tmp_path: Path) -> None:
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"tail-bytes")
    tape = video_ltxv.build_recovery_tape(
        source_segment_id="000123",
        conditioning_tail_path=str(tail),
        conditioning_tail_sha256=video_ltxv.sha256_file(tail),
        prompts=["amber dunes"],
        seeds=[123456],
        width=768,
        height=512,
        fps=24,
    )
    assert tape["backend"] == "ltxv"
    assert tape["state_mode"] == "reconstructable_prefix"
    assert tape["source_segment_id"] == "000123"
    assert tape["conditioning_tail_path"] == str(tail)
    assert tape["seed"] == 123456
    assert tape["model_revision"] == model_registry.LTXV_HF_REVISION
    assert tape["pipeline_revision"] == model_registry.LTXV_COMMIT
    assert len(tape["profile_hash"]) == 64
    # Round-trips through JSON (the on-disk format) and validates.
    loaded = json.loads(json.dumps(tape))
    assert video_ltxv.parse_recovery_tape(loaded) == loaded


def test_recovery_tape_rejects_legacy_torch_format(tmp_path: Path) -> None:
    legacy = {"profile": "ltxv", "tail_png": str(tmp_path / "video_tail.png")}
    with pytest.raises(ValueError, match="pre-Stream-A"):
        video_ltxv.parse_recovery_tape(legacy)


def test_recovery_tape_missing_tail_file_parses_for_derive(tmp_path: Path) -> None:
    """Run-file pruning: a missing tail file is not a parse error.

    The tape records the would-be path and resume derives it from the
    sibling segment video — only a missing tail *path* fails parsing.
    """
    from voyage.workers import video_common

    tape = video_ltxv.build_recovery_tape(
        source_segment_id="000124",
        conditioning_tail_path=str(tmp_path / "video_tail.mp4"),
        prompts=["amber dunes"],
        seeds=[7],
        width=768,
        height=512,
        fps=24,
    )
    assert "conditioning_tail_sha256" not in tape
    assert video_ltxv.parse_recovery_tape(json.loads(json.dumps(tape))) == tape
    with pytest.raises(ValueError, match="conditioning tail missing"):
        video_common.ensure_conditioning_tail(tmp_path / "video_tail.mp4")


def test_recovery_tape_rejects_missing_tail_path() -> None:
    tape = video_ltxv.build_recovery_tape(
        source_segment_id="000124",
        conditioning_tail_path="",
        prompts=["amber dunes"],
        seeds=[7],
        width=768,
        height=512,
        fps=24,
    )
    with pytest.raises(ValueError, match="no conditioning tail path"):
        video_ltxv.parse_recovery_tape(tape)


def test_load_tape_json_rejects_torch_pickle_bytes(tmp_path: Path) -> None:
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_bytes(b"\x80\x04torch-pickle-not-json")
    with pytest.raises(ValueError, match="pre-Stream-A"):
        video_ltxv._load_tape_json(str(tape_path))


# ---------------------------------------------------------------------------
# 088 fold (issue 036): tests/test_ltxv_oom_fallback.py folded verbatim here.
# Original module docstring preserved as the banner below; helper + test
# function names/bodies identical. Source file deleted; gates.sh mypy entry
# removed in the same edit.
# ---------------------------------------------------------------------------
# """LTXV OOM fallback: broad catch, gc-before-empty_cache, memory logging (issue 049).
#
# CPU-only: `_generate_block` is driven through stub sessions — the first
# `_run_multiscale` call raises a scripted OOM shape, the retry returns a
# sentinel. The `ltx_video.inference` import inside `_generate_block` is
# stubbed (the slim gates image has no ltx_video), and torch itself is faked
# (the slim image has no torch either).
# """


def _stub_ltx_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `ltx_video` + `ltx_video.inference` (fresh path needs padding only)."""
    parent = types.ModuleType("ltx_video")
    monkeypatch.setitem(sys.modules, "ltx_video", parent)
    inference = types.ModuleType("ltx_video.inference")
    inference.calculate_padding = lambda *args: (0, 0, 0, 0)  # type: ignore[attr-defined]
    inference.prepare_conditioning = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ltx_video.inference", inference)


def _record_gc(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Record `gc.collect` calls (still running the real collection)."""
    real_collect = gc.collect

    def _collect(*args: Any, **kwargs: Any) -> Any:
        events.append("gc.collect")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", _collect)


def _stub_session(
    fail_factory: Callable[[Any], BaseException],
    *,
    already_fallback: bool = False,
    cuda_available: bool = True,
) -> tuple[Any, list[str], Any]:
    """Stub LTXV session: scripted first-call failure, sentinel on retry.

    `fail_factory` builds the first-call failure from the fake torch
    namespace, so torch-shaped failures are the same class object the
    worker's `except` matches (separately-created same-named classes would
    never match — the pre-fix suite caught that artifact, not the worker).
    """
    events: list[str] = []
    sentinel = object()
    oom_cls = type("OutOfMemoryError", (RuntimeError,), {})
    calls = {"run": 0}

    def _allocated() -> int:
        events.append("cuda.memory_allocated")
        return 3 * 1024**3

    def _reserved() -> int:
        events.append("cuda.memory_reserved")
        return 4 * 1024**3

    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=lambda: events.append("cuda.empty_cache"),
        synchronize=lambda: events.append("cuda.synchronize"),
        memory_allocated=_allocated,
        memory_reserved=_reserved,
    )

    class _Generator:
        def manual_seed(self, seed: int) -> _Generator:
            events.append(f"generator.manual_seed({seed})")
            return self

    torch_ns = types.SimpleNamespace(
        OutOfMemoryError=oom_cls,
        Generator=lambda device: _Generator(),
        cuda=cuda,
    )

    def _run_multiscale(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls["run"] += 1
        if calls["run"] == 1:
            raise fail_factory(torch_ns)
        return sentinel

    session = types.SimpleNamespace()
    session._torch = torch_ns
    session._device = "cuda:0"
    session._negative = ("neg-embeds", "neg-mask")
    session._encode = lambda text: (f"embeds:{text}", f"mask:{text}")
    session._run_multiscale = _run_multiscale
    session._fp8_fallback = already_fallback

    def _quantize() -> None:
        # Mirror the real method: record the call and arm the fallback flag.
        events.append("quantize")
        session._fp8_fallback = True

    session._quantize_fp8_fallback = _quantize
    return session, events, sentinel


def _generate(session: Any) -> Any:
    """One fresh (unconditioned) block through the real `_generate_block`."""
    return video_ltxv.LTXVSession._generate_block(
        session, "amber dunes", 7, 768, 512, 121, 24, None
    )


def test_is_oom_matches_class_and_message_shapes() -> None:
    oom_cls = type("OutOfMemoryError", (RuntimeError,), {})
    assert video_ltxv.is_oom(oom_cls("alloc failed"))
    assert video_ltxv.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB."))
    assert video_ltxv.is_oom(RuntimeError("cuda Out Of Memory"))
    assert not video_ltxv.is_oom(RuntimeError("boom"))
    assert not video_ltxv.is_oom(ValueError("bad shape"))


def test_runtime_error_oom_triggers_fp8_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError-shaped OOMs must reach the fp8 fallback (the 049 gap)."""
    _stub_ltx_inference(monkeypatch)
    session, events, sentinel = _stub_session(
        lambda _torch: RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.")
    )
    assert _generate(session) is sentinel
    assert session._fp8_fallback is True
    assert "quantize" in events


def test_torch_oom_still_triggers_fp8_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing torch-OOM path keeps working (regression guard)."""
    _stub_ltx_inference(monkeypatch)
    session, events, sentinel = _stub_session(
        lambda torch_ns: torch_ns.OutOfMemoryError("CUDA out of memory")
    )
    assert _generate(session) is sentinel
    assert session._fp8_fallback is True
    assert "quantize" in events


def test_non_oom_runtime_error_propagates_without_quantize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broadened catch must not swallow genuine failures."""
    _stub_ltx_inference(monkeypatch)
    session, events, _sentinel = _stub_session(lambda _torch: RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _generate(session)
    assert "quantize" not in events
    assert session._fp8_fallback is False


def test_armed_fallback_reraises_oom_without_requantize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One fallback only: the second OOM propagates (both shapes)."""
    _stub_ltx_inference(monkeypatch)
    session, events, _sentinel = _stub_session(
        lambda _torch: RuntimeError("CUDA out of memory. Tried to allocate 1 GiB."),
        already_fallback=True,
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        _generate(session)
    assert "quantize" not in events


def test_fallback_collects_gc_before_empty_cache_and_logs_memory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """gc runs before empty_cache; synchronize + memory figures are logged."""
    _stub_ltx_inference(monkeypatch)
    session, events, sentinel = _stub_session(
        lambda torch_ns: torch_ns.OutOfMemoryError("CUDA out of memory")
    )
    _record_gc(monkeypatch, events)
    assert _generate(session) is sentinel
    assert events.index("gc.collect") < events.index("cuda.empty_cache")
    assert events.index("cuda.empty_cache") < events.index("cuda.synchronize")
    assert events.index("cuda.synchronize") < events.index("quantize")
    assert "cuda.memory_allocated" in events
    assert "cuda.memory_reserved" in events
    captured = capsys.readouterr()
    assert "fp8" in captured.err.lower()
    assert "gib" in captured.err.lower()

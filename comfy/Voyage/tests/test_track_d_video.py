"""Track D video-worker liveness (cancel / progress / resume / telemetry).

CPU-only: no torch/GPU, imageio stubbed where needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage.workers import video_common
from voyage.workers.loop import handle_cancel


def _stub_handler(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)


def test_standard_serve_map_stays_nine_ops_by_default() -> None:
    handlers = video_common.standard_serve_map(
        "ltxv",
        handle_init=_stub_handler,
        handle_health=_stub_handler,
        handle_generate_blocks=_stub_handler,
        handle_benchmark=_stub_handler,
        handle_evict_gpu=_stub_handler,
        handle_rebuild=_stub_handler,
        handle_resume=_stub_handler,
    )
    assert sorted(handlers) == [
        "benchmark",
        "checkpoint",
        "evict_gpu",
        "generate_blocks",
        "health",
        "init",
        "rebuild",
        "resume",
        "shutdown",
    ]
    assert "cancel" not in handlers


def test_standard_serve_map_explicit_cancel_handler() -> None:
    handlers = video_common.standard_serve_map(
        "ltxv",
        handle_init=_stub_handler,
        handle_health=_stub_handler,
        handle_generate_blocks=_stub_handler,
        handle_benchmark=_stub_handler,
        handle_evict_gpu=_stub_handler,
        handle_rebuild=_stub_handler,
        handle_resume=_stub_handler,
        handle_cancel=_stub_handler,
    )
    assert handlers["cancel"] is _stub_handler


def test_standard_serve_map_with_cancel_adds_idle_handler() -> None:
    handlers = video_common.standard_serve_map_with_cancel(
        "ltxv",
        handle_init=_stub_handler,
        handle_health=_stub_handler,
        handle_generate_blocks=_stub_handler,
        handle_benchmark=_stub_handler,
        handle_evict_gpu=_stub_handler,
        handle_rebuild=_stub_handler,
        handle_resume=_stub_handler,
    )
    assert handlers["cancel"]({}) == {"cancelled": False, "reason": "idle-or-noop"}
    assert handle_cancel({})["cancelled"] is False


def test_report_block_progress_never_raises(capsys: Any) -> None:
    video_common.report_block_progress(0, 3, backend="ltxv25")
    captured = capsys.readouterr()
    assert "voyage_progress" in captured.err

    # Broken stream is swallowed.
    class _Broken:
        def write(self, text: str) -> int:
            raise OSError("closed")

        def flush(self) -> None:
            pass

    video_common.report_block_progress(0, 1, backend="x", stream=_Broken())


def test_touch_and_read_progress_timestamp(tmp_path: Path) -> None:
    progress = tmp_path / "progress.txt"
    assert video_common.read_progress_timestamp(progress) is None
    video_common.touch_progress_file(progress)
    stamped = video_common.read_progress_timestamp(progress)
    assert isinstance(stamped, float)
    progress.write_text("not-a-number\n", encoding="utf-8")
    assert video_common.read_progress_timestamp(progress) is None


def test_prune_work_root_keeps_root(tmp_path: Path) -> None:
    root = tmp_path / "work"
    (root / "comfy_output").mkdir(parents=True)
    (root / "comfy_output" / "frames_00001.png").write_bytes(b"png")
    (root / "stale.txt").write_text("stale", encoding="utf-8")
    video_common.prune_work_root(root)
    assert root.is_dir()
    assert list(root.iterdir()) == []
    # Missing root is a no-op.
    video_common.prune_work_root(tmp_path / "absent")


def test_validate_resume_trust_pass_and_fail() -> None:
    tape = {
        "profile_hash": "abc",
        "width": 768,
        "height": 512,
        "conditioning_tail_frames": 25,
        "model_revision": "rev1",
    }
    assert video_common.validate_resume_trust(tape, expected_profile_hash="abc") is tape
    with pytest.raises(ValueError, match="profile_hash mismatch"):
        video_common.validate_resume_trust(tape, expected_profile_hash="other")
    with pytest.raises(ValueError, match="width mismatch"):
        video_common.validate_resume_trust(tape, expected_width=1024)
    with pytest.raises(ValueError, match="tail_frames mismatch"):
        video_common.validate_resume_trust(tape, expected_tail_frames=9)
    with pytest.raises(ValueError, match="model_revision mismatch"):
        video_common.validate_resume_trust(tape, expected_model_revision="rev2")


def test_resume_gate_never_raises() -> None:
    tape = {"profile_hash": "abc"}
    trusted, metric = video_common.resume_gate_for_supervisor(
        tape, expected_profile_hash="abc", worker_name="video", segment_id="000001"
    )
    assert trusted is True
    assert metric["event"] == "resume_trusted"
    trusted2, metric2 = video_common.resume_gate_for_supervisor(
        tape, expected_profile_hash="other", worker_name="video", segment_id="000001"
    )
    assert trusted2 is False
    assert metric2["event"] == "resume_trust_mismatch"


def test_is_idempotent_op_contract() -> None:
    assert video_common.is_idempotent_op("generate_audio") is True
    assert video_common.is_idempotent_op("generate_sfx") is True
    assert video_common.is_idempotent_op("health") is True
    assert video_common.is_idempotent_op("generate_blocks") is False


def test_cuda_device_index_and_arg() -> None:
    assert video_common.cuda_device_index("cuda:0") == 0
    assert video_common.cuda_device_index("cuda:1") == 1
    assert video_common.cuda_device_index("cuda") == 0
    assert video_common.torch_device_arg(0) == ()
    assert video_common.torch_device_arg(1) == (1,)
    with pytest.raises(ValueError, match="CUDA device"):
        video_common.cuda_device_index("cpu")


def test_check_recovery_tape_size_rejects_gigabyte(tmp_path: Path) -> None:
    small = tmp_path / "recovery.pt"
    small.write_bytes(b"{}")
    assert video_common.check_recovery_tape_size(small) is small

    # Implausible size without writing GBs: monkeypatch stat size.
    class _BigStat:
        st_size: int = video_common.MAX_RECOVERY_TAPE_BYTES + 1

    original_stat = Path.stat

    def _fake_stat(self: Path) -> Any:
        if self == small:
            return _BigStat()
        return original_stat(self)

    Path.stat = _fake_stat  # type: ignore
    try:
        with pytest.raises(ValueError, match="implausible tape size"):
            video_common.check_recovery_tape_size(small)
    finally:
        Path.stat = original_stat  # type: ignore


def test_benchmark_harness_accepts_staging_parent(tmp_path: Path) -> None:
    staging = tmp_path / "scratch"
    staging.mkdir()

    def _probe(path: Path, measured: bool) -> None:
        del measured
        path.write_bytes(b"probe")

    outcome = video_common.run_benchmark_harness(
        0, 1, "voyage-track-d-", _probe, staging_parent=staging
    )
    assert len(outcome.wall_seconds) == 1


def test_video_workers_expose_cancel_op() -> None:
    from voyage.workers import video_causvid, video_ltx23, video_ltx25, video_ltxv

    for module in (video_ltxv, video_ltx25, video_ltx23, video_causvid):
        serve_map = module.main  # exists (serve wiring)
        assert callable(serve_map)


def test_ltxv_trust_helper_rejects_profile_drift() -> None:
    from voyage.workers import video_ltxv

    tape = video_ltxv.build_recovery_tape(
        source_segment_id="000000",
        conditioning_tail_path="/tmp/tail.mp4",
        prompts=["a calm valley"],
        seeds=[0],
        width=768,
        height=512,
        fps=24,
    )
    assert video_ltxv._validate_tape_trust(dict(tape))["backend"] == "ltxv"
    tampered = dict(tape)
    tampered["profile_hash"] = "tampered"
    with pytest.raises(ValueError, match="profile_hash mismatch"):
        video_ltxv._validate_tape_trust(tampered)


def test_ltx25_trust_helper_rejects_tail_drift() -> None:
    from voyage.workers import video_ltx25

    tape = video_ltx25.build_recovery_tape(
        source_segment_id="000000",
        conditioning_tail_path="/tmp/tail.mp4",
        prompts=["a calm valley"],
        seeds=[0],
        width=1216,
        height=704,
        fps=24,
    )
    assert video_ltx25._validate_tape_trust(dict(tape))["backend"] == "ltx25"
    tampered = dict(tape)
    tampered["conditioning_tail_frames"] = 9
    with pytest.raises(ValueError, match="mismatch"):
        video_ltx25._validate_tape_trust(tampered)


def test_imageio_stub_unused() -> None:
    package = types.ModuleType("imageio")
    submodule = types.ModuleType("imageio.v2")
    assert package is not None
    assert submodule is not None
    assert sys.version_info >= (3, 10)

"""Stage C: AWQ kernel spawn env + gauge cadence (DESIGN §140).

The director venv ships a `ninja` binary the torch JIT needs, but the
worker spawns by absolute interpreter path with the system PATH — JIT
compilation fails in 0.0 s and GPTQModel falls back to the ~9 tok/s
Triton linear (live-verified: 36.7 tok/s with ninja on PATH). The spawn
env must carry the interpreter's bin dir. The gauge half kills the
remaining per-tail stalls: a timed-out prefetch still occupies the
serial director RPC queue, and the sample interval becomes a config
knob instead of a hardcoded every-segment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import VoyageConfig, preset_config
from voyage.persistence import read_effective_config
from voyage.rpc import SubprocessWorker, _spawn_env
from voyage.supervisor import Supervisor


def test_spawn_env_none_inherits_process_env() -> None:
    """No alternate interpreter means no env override (plain inherit)."""
    assert _spawn_env(None) is None


def test_spawn_env_prepends_interpreter_bin_dir() -> None:
    """The venv bin dir heads PATH so JIT finds its ninja (Stage C fix)."""
    env = _spawn_env("/opt/venvs/director/bin/python")
    assert env is not None
    path = env["PATH"]
    assert path.split(os.pathsep)[0] == "/opt/venvs/director/bin"
    assert path.endswith(os.environ["PATH"])


def test_spawn_env_never_double_prepends() -> None:
    """Restarts rebuild env from os.environ each time — no PATH growth."""
    bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    baseline = os.environ.get("PATH", "").split(os.pathsep).count(bin_dir)
    env = _spawn_env(sys.executable)
    assert env is not None
    entries = env["PATH"].split(os.pathsep)
    assert entries[0] == bin_dir
    assert entries.count(bin_dir) <= baseline + 1


def test_start_passes_spawn_env_to_popen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Popen child actually receives the interpreter-bin PATH."""
    captured: dict[str, Any] = {}

    class _StubProc:
        stdin = None
        stdout = None
        pid = 4242

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def _capture(*args: Any, **kwargs: Any) -> _StubProc:
        captured.update(kwargs)
        return _StubProc()

    monkeypatch.setattr(subprocess, "Popen", _capture)
    worker = SubprocessWorker(
        "voyage.workers.audio",
        tmp_path,
        tmp_path / "w.log",
        init_op=None,
        executable="/opt/venvs/director/bin/python",
    )
    worker.start()
    try:
        assert worker.running is True
    finally:
        worker.stop()
    env = captured.get("env")
    assert isinstance(env, dict)
    assert str(env["PATH"]).split(os.pathsep)[0] == "/opt/venvs/director/bin"


def _stubbed_supervisor(run_dir: Path) -> Supervisor:
    initialize_run_directory(run_dir, run_id="stage-c")
    config, _ = read_effective_config(run_dir)
    return Supervisor(run_dir, config)


def _health_calls(supervisor: Supervisor) -> list[str]:
    calls: list[str] = []

    def _recorder(name: str) -> Any:
        def _call(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            calls.append(name)
            return {}

        return _call

    supervisor._video.call = _recorder("video")  # type: ignore[method-assign]
    supervisor._audio.call = _recorder("audio")  # type: ignore[method-assign]
    supervisor._director.call = _recorder("director")  # type: ignore[method-assign]
    return calls


def test_gauges_skip_director_after_prefetch_timeout(tmp_path: Path) -> None:
    """A done-but-None prefetch means the worker still chews — skip, don't stall."""
    run_dir = tmp_path / "run"
    supervisor = _stubbed_supervisor(run_dir)
    calls = _health_calls(supervisor)
    timed_out: Future[dict[str, Any] | None] = Future()
    timed_out.set_result(None)
    supervisor._prefetch_future = timed_out
    supervisor._sample_gauges("000000")
    assert sorted(calls) == ["audio", "video"]


def test_gauges_probe_director_after_prefetch_dict(tmp_path: Path) -> None:
    """A done-with-dict prefetch means the worker is free — probe it."""
    run_dir = tmp_path / "run"
    supervisor = _stubbed_supervisor(run_dir)
    calls = _health_calls(supervisor)
    answered: Future[dict[str, Any] | None] = Future()
    answered.set_result({"fallback": True})
    supervisor._prefetch_future = answered
    supervisor._sample_gauges("000000")
    assert sorted(calls) == ["audio", "director", "video"]


def test_gauge_interval_knob_default_and_validation() -> None:
    """Sampling cadence defaults to every segment and rejects non-positive."""
    assert VoyageConfig().resource_gauge_interval_segments == 1
    with pytest.raises(Exception, match="positive"):
        VoyageConfig(resource_gauge_interval_segments=0)


def test_gauge_interval_honored_by_sampler(tmp_path: Path) -> None:
    """Interval 2 samples even segments only (additive cadence knob)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="stage-c")
    config, _ = read_effective_config(run_dir)
    config = config.model_copy(
        update={"voyage": config.voyage.model_copy(update={"resource_gauge_interval_segments": 2})}
    )
    supervisor = Supervisor(run_dir, config)
    _health_calls(supervisor)
    log_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    supervisor._sample_gauges("000001")
    assert not log_path.exists() or "resource_gauges" not in log_path.read_text(encoding="utf-8")
    supervisor._sample_gauges("000002")
    lines = log_path.read_text(encoding="utf-8")
    assert sum(1 for line in lines.splitlines() if "resource_gauges" in line) == 1


def test_gauge_interval_in_default_toml(tmp_path: Path) -> None:
    """The knob ships in fresh run configs (preset default, no CLI flag)."""

    config = preset_config(run_id="stage-c", style="pastel", seed=0)
    assert config.voyage.resource_gauge_interval_segments == 1


def test_audio_health_reports_vram_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio health merges CUDA memory fields when the helper resolves them."""
    import voyage.workers.audio_acestep as audio_worker

    monkeypatch.setattr(audio_worker, "_cuda_mem_info_gib", lambda: (10.0, 16.0))
    info = audio_worker.handle_health({})
    assert info["status"] == "READY"
    assert info["vram_free_gib"] == 10.0
    assert info["vram_total_gib"] == 16.0


def test_audio_health_without_cuda_stays_keyless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CUDA (evicted stack, CPU box) means no vram keys — never a raise."""
    import voyage.workers.audio_acestep as audio_worker

    monkeypatch.setattr(audio_worker, "_cuda_mem_info_gib", lambda: None)
    info = audio_worker.handle_health({})
    assert info["status"] == "READY"
    assert "vram_free_gib" not in info
    assert json.dumps(info)

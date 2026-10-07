"""Track D llama sidecar liveness (ownership / health / heal).

CPU-only: processes and HTTP probes are stubbed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import URLError

import pytest

from voyage import llama_server


class _FakeProcess:
    def __init__(self, pid: int = 4242, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive
        self.terminated = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


def _install_probe(monkeypatch: pytest.MonkeyPatch, healthy: bool) -> None:
    def _fake_urlopen(request: Any, *, timeout: float | None = None) -> Any:
        del request, timeout
        if not healthy:
            raise URLError("refused")
        return SimpleNamespace(status=200, close=lambda: None)

    monkeypatch.setattr(llama_server, "urlopen", _fake_urlopen)


def test_health_gate_probes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_probe(monkeypatch, True)
    assert llama_server.health_gate("http://127.0.0.1:8080") is True
    assert llama_server.is_sidecar_healthy("http://127.0.0.1:8080") is True
    _install_probe(monkeypatch, False)
    assert llama_server.health_gate("http://127.0.0.1:8080") is False


def test_find_free_port_is_valid() -> None:
    port = llama_server.find_free_port()
    assert 1 <= port <= 65535


def test_resolve_port_keeps_free_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llama_server, "is_port_in_use", lambda port: False)
    assert llama_server.resolve_port(18080) == 18080


def test_resolve_port_escapes_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llama_server, "is_port_in_use", lambda port: True)
    monkeypatch.setattr(llama_server, "find_free_port", lambda: 19999)
    assert llama_server.resolve_port(8080) == 19999


def test_is_owned_by_checks_pid_liveness_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(pid=111, alive=True)
    handle = llama_server.LlamaSidecar(
        process=process,  # type: ignore[arg-type]
        endpoint="http://127.0.0.1:8080",
        model_path=Path("/models/gguf"),
    )
    _install_probe(monkeypatch, True)
    assert llama_server.is_owned_by(handle, 111) is True
    assert llama_server.is_owned_by(handle, 222) is False
    assert llama_server.is_owned_by(None, 111) is False
    assert llama_server.is_owned_by(handle, None) is False
    dead = _FakeProcess(pid=111, alive=False)
    dead_handle = llama_server.LlamaSidecar(
        process=dead,  # type: ignore[arg-type]
        endpoint="http://127.0.0.1:8080",
        model_path=Path("/models/gguf"),
    )
    assert llama_server.is_owned_by(dead_handle, 111) is False


def test_metric_for_heal_shape() -> None:
    metric = llama_server.metric_for_heal(op="decide", attempt=1, budget=3, reason="heal")
    assert metric["event"] == "worker_restart"
    assert metric["worker"] == "llama-sidecar"


def test_heal_returns_healthy_handle_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(pid=111, alive=True)
    handle = llama_server.LlamaSidecar(
        process=process,  # type: ignore[arg-type]
        endpoint="http://127.0.0.1:8080",
        model_path=Path("/models/gguf"),
    )
    _install_probe(monkeypatch, True)
    stayed, metric = llama_server.heal_if_needed(handle, Path("/models"))
    assert stayed is handle
    assert metric is None


def test_heal_restarts_stale_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from voyage.model_registry import QWEN35_GGUF_FILE, QWEN35_GGUF_SUBDIR

    target = tmp_path / QWEN35_GGUF_SUBDIR / QWEN35_GGUF_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"gguf")
    dead = _FakeProcess(pid=111, alive=False)
    stale = llama_server.LlamaSidecar(
        process=dead,  # type: ignore[arg-type]
        endpoint="http://127.0.0.1:8080",
        model_path=target,
    )
    fresh_process = _FakeProcess(pid=222, alive=True)
    started: list[dict[str, Any]] = []

    def _fake_start(
        models_dir: Path | str, port: int = 8080, **kwargs: Any
    ) -> llama_server.LlamaSidecar:
        started.append({"models_dir": str(models_dir), "port": port, **kwargs})
        return llama_server.LlamaSidecar(
            process=fresh_process,  # type: ignore[arg-type]
            endpoint=f"http://127.0.0.1:{port}",
            model_path=target,
        )

    # The suite-wide conftest fixture neuters `llama_server.start` (returns
    # None) for every module except `tests.test_llama_sidecar` — override
    # it here with a recording fake so the heal path (stale → restart +
    # metric) is exercised without spawning a real server.
    monkeypatch.setattr(llama_server, "start", _fake_start)
    monkeypatch.setattr(llama_server, "resolve_port", lambda port: port)
    fresh, metric = llama_server.heal_if_needed(stale, tmp_path)
    assert fresh is not None
    assert fresh.endpoint == "http://127.0.0.1:8080"
    assert started and started[0]["port"] == 8080
    assert metric is not None
    assert metric["worker"] == "llama-sidecar"

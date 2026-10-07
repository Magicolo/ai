"""Llama sidecar hardening (issue 237).

CPU-only: endpoint/binary validation is pure, the port claim uses tmp
files + stubbed probes — no server process, no network, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import llama_server


def test_validate_endpoint_host_accepts_loopback_forms() -> None:
    assert llama_server.validate_endpoint_host("http://127.0.0.1:8080") == "127.0.0.1"
    assert llama_server.validate_endpoint_host("http://localhost:8080") == "localhost"
    assert llama_server.validate_endpoint_host("http://127.0.0.1") == "127.0.0.1"


def test_validate_endpoint_host_rejects_non_loopback() -> None:
    for endpoint in (
        "http://0.0.0.0:8080",
        "http://example.com:8080",
        "http://192.168.1.10:8080",
        "not-a-url",
    ):
        with pytest.raises(llama_server.LlamaServerError):
            llama_server.validate_endpoint_host(endpoint)


def test_port_for_endpoint_rejects_non_loopback_host() -> None:
    """The single-source spawn-port parse fails closed off-loopback."""
    with pytest.raises(llama_server.LlamaServerError):
        llama_server.port_for_endpoint("http://0.0.0.0:8080")
    assert llama_server.port_for_endpoint("http://127.0.0.1:8080") == 8080
    assert llama_server.port_for_endpoint("http://localhost:9999") == 9999


def test_server_binary_defaults_to_path_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(llama_server.LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE, raising=False)
    assert llama_server.server_binary() == llama_server.LLAMA_SERVER_BINARY_NAME


def test_server_binary_accepts_baked_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    for baked in llama_server.LLAMA_SERVER_BAKED_PATHS:
        monkeypatch.setenv(llama_server.LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE, baked)
        assert llama_server.server_binary() == baked


def test_server_binary_refuses_arbitrary_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The issue's repro (`/tmp/evil-llama`) must refuse instead of spawn."""
    monkeypatch.setenv(llama_server.LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE, "/tmp/evil-llama")
    with pytest.raises(llama_server.LlamaServerError, match="allowlist"):
        llama_server.server_binary()


def test_port_file_for_lives_beside_the_run(tmp_path: Path) -> None:
    assert llama_server.port_file_for(tmp_path) == tmp_path / "llama-sidecar.port"


def test_claim_sidecar_port_records_free_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llama_server, "is_port_in_use", lambda port: False)
    lock = tmp_path / "locks" / "llama-sidecar.lock"
    port_file = tmp_path / "run" / "llama-sidecar.port"
    assert llama_server.claim_sidecar_port(8080, lock_path=lock, port_file=port_file) == 8080
    assert port_file.read_text(encoding="utf-8") == "8080\n"


def test_claim_sidecar_port_escapes_occupied_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llama_server, "is_port_in_use", lambda port: True)
    monkeypatch.setattr(llama_server, "find_free_port", lambda: 19999)
    lock = tmp_path / "llama-sidecar.lock"
    port_file = tmp_path / "llama-sidecar.port"
    assert llama_server.claim_sidecar_port(8080, lock_path=lock, port_file=port_file) == 19999
    assert port_file.read_text(encoding="utf-8") == "19999\n"

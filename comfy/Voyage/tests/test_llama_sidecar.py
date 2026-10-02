"""llama-server sidecar contract tests (sidecar track, DESIGN §140 llama entry).

Covers the supervisor-side half of the llama-server contract whose client
lives in the director worker (`voyage/workers/director.py`: `llama_endpoint`
payload key, `/v1/chat/completions` transport, `evolution_decision`
schema): sidecar lifecycle (`voyage/llama_server.py`), the GGUF registry
row, `models_ensure` selection, `DirectorConfig` plumbing, and the
supervisor init/decide payload flow. All CPU-only: the server process and
the readiness HTTP probe are stubbed — no GPU, no network, no downloads.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.error import URLError

import pytest

from voyage import llama_server
from voyage.concepts import ConceptStore
from voyage.config import (
    DEFAULT_LLAMA_ENDPOINT,
    DirectorConfig,
    ProjectConfig,
    default_config_toml,
    load_config,
    resolve_config,
)
from voyage.model_registry import (
    MODEL_SPECS,
    QWEN35_GGUF_FILE,
    QWEN35_GGUF_HF_REPO,
    QWEN35_GGUF_HF_REVISION,
    QWEN35_GGUF_LICENSE,
    QWEN35_GGUF_MIN_BYTES,
    QWEN35_GGUF_SUBDIR,
    download_model,
    verify_model,
)
from voyage.models import StyleSpec
from voyage.models_ensure import required_specs
from voyage.supervisor import Supervisor

SIDECAR_HOST = "127.0.0.1"
SIDECAR_PORT = 8080
SIDECAR_CONTEXT_SIZE = 4096
SIDECAR_GPU_LAYERS = 99


class _FakeProcess:
    """Minimal Popen stand-in: scripted liveness, recorded signals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.command: list[str] = list(args[0]) if args else []
        self.keywords: dict[str, Any] = dict(kwargs)
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.live = True
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return None if self.live else self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int | None:
        del timeout
        self.wait_calls += 1
        return self.returncode


def _install_process_stub(
    monkeypatch: pytest.MonkeyPatch, process: _FakeProcess
) -> list[list[str]]:
    """Replace the module Popen with a recorder returning `process`."""
    commands: list[list[str]] = []

    def _fake_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        commands.append(list(args[0]))
        return process

    monkeypatch.setattr(llama_server, "Popen", _fake_popen)
    return commands


def _install_probe_stub(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[bool | BaseException]
) -> None:
    """Script readiness probes: True = HTTP 200, False/exception = refused."""

    def _fake_urlopen(request: Any, *, timeout: float | None = None) -> Any:
        del timeout
        outcome = outcomes.pop(0) if outcomes else False
        if isinstance(outcome, BaseException):
            raise outcome
        if not outcome:
            raise URLError("connection refused")
        return SimpleNamespace(status=200, close=lambda: None)

    monkeypatch.setattr(llama_server, "urlopen", _fake_urlopen)


def _install_fast_clock(monkeypatch: pytest.MonkeyPatch, ticks: list[float]) -> list[float]:
    """Replace sleep/monotonic: scripted clock, recorded sleeps."""
    sleeps: list[float] = []
    remaining: list[float] = list(ticks)
    last_tick: list[float] = [0.0]

    def _fake_monotonic() -> float:
        if remaining:
            last_tick[0] = remaining.pop(0)
        return last_tick[0]

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(llama_server, "monotonic", _fake_monotonic)
    monkeypatch.setattr(llama_server, "sleep", _fake_sleep)
    return sleeps


def _write_gguf(models_dir: Path, size_bytes: int) -> Path:
    """Sparse GGUF stand-in: full st_size without writing bytes."""
    target = models_dir / QWEN35_GGUF_SUBDIR / QWEN35_GGUF_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.truncate(size_bytes)
    return target


def _llama_config() -> ProjectConfig:
    """Base config with the llama director backend selected."""
    base = ProjectConfig(style="llama sidecar probe")
    return resolve_config(base, director="llama")


def _supervisor_with_backend(run_dir: Path, backend: str) -> Supervisor:
    """Supervisor over a config carrying one director backend value."""
    base = ProjectConfig(style="llama sidecar probe")
    config = resolve_config(base, director=backend)
    return Supervisor(run_dir, config)


# ---------------------------------------------------------------------------
# Lifecycle: command shape + readiness + stop.
# ---------------------------------------------------------------------------


def test_build_server_argv_has_exact_contract_shape(tmp_path: Path) -> None:
    """argv is the pinned contract: binary, model, host, port, ctx, layers."""
    model_path = tmp_path / "model.gguf"
    assert llama_server.build_server_argv(
        "/usr/local/bin/llama-server",
        model_path,
        port=SIDECAR_PORT,
        context_size=SIDECAR_CONTEXT_SIZE,
        gpu_layers=SIDECAR_GPU_LAYERS,
    ) == [
        "/usr/local/bin/llama-server",
        "-m",
        str(model_path),
        "--host",
        SIDECAR_HOST,
        "--port",
        str(SIDECAR_PORT),
        "--ctx-size",
        str(SIDECAR_CONTEXT_SIZE),
        "-ngl",
        str(SIDECAR_GPU_LAYERS),
        "--cache-prompt",
    ]


def test_gguf_path_for_joins_subdir_and_file(tmp_path: Path) -> None:
    """Model resolution stays inside the configured models mount."""
    assert llama_server.gguf_path_for(tmp_path) == (
        tmp_path / QWEN35_GGUF_SUBDIR / QWEN35_GGUF_FILE
    )


def test_port_for_endpoint_parses_loopback_port() -> None:
    """The spawn port derives from the configured endpoint (single source)."""
    assert llama_server.port_for_endpoint("http://127.0.0.1:8080") == SIDECAR_PORT
    with pytest.raises(llama_server.LlamaServerError):
        llama_server.port_for_endpoint("not-a-url")


def test_start_spawns_binary_and_waits_for_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First probe refused, second 200: start returns a live handle."""
    model_path = _write_gguf(tmp_path, QWEN35_GGUF_MIN_BYTES)
    process = _FakeProcess()
    commands = _install_process_stub(monkeypatch, process)
    _install_probe_stub(monkeypatch, [False, True])
    _install_fast_clock(monkeypatch, [0.0, 0.0, 0.5, 1.0])
    handle = llama_server.start(tmp_path, port=SIDECAR_PORT)
    assert commands == [
        [
            llama_server.server_binary(),
            "-m",
            str(model_path),
            "--host",
            SIDECAR_HOST,
            "--port",
            str(SIDECAR_PORT),
            "--ctx-size",
            str(llama_server.LLAMA_SERVER_CONTEXT_SIZE),
            "-ngl",
            str(llama_server.LLAMA_SERVER_GPU_LAYERS),
            "--cache-prompt",
        ]
    ]
    assert handle.endpoint == DEFAULT_LLAMA_ENDPOINT
    assert handle.model_path == model_path
    assert cast(Any, handle.process) is process
    assert not process.terminated


def test_start_missing_model_fails_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No GGUF on disk: loud error, no process spawned."""
    process = _FakeProcess()
    commands = _install_process_stub(monkeypatch, process)
    with pytest.raises(llama_server.LlamaServerError, match="missing"):
        llama_server.start(tmp_path, port=SIDECAR_PORT)
    assert commands == []


def test_start_readiness_timeout_terminates_process_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server never answers: the process is terminated, never orphaned."""
    _write_gguf(tmp_path, QWEN35_GGUF_MIN_BYTES)
    process = _FakeProcess()
    _install_process_stub(monkeypatch, process)
    _install_probe_stub(monkeypatch, [])
    _install_fast_clock(
        monkeypatch, [0.0, 0.0, llama_server.LLAMA_SERVER_READY_TIMEOUT_SECONDS + 1.0]
    )
    with pytest.raises(llama_server.LlamaServerError, match="readiness"):
        llama_server.start(tmp_path, port=SIDECAR_PORT)
    assert process.terminated
    assert process.wait_calls >= 1


def test_start_process_early_exit_raises_without_full_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server dies during load: fail fast with the exit code, no full wait."""
    _write_gguf(tmp_path, QWEN35_GGUF_MIN_BYTES)
    process = _FakeProcess()
    process.live = False
    process.returncode = 1
    _install_process_stub(monkeypatch, process)
    _install_probe_stub(monkeypatch, [])
    _install_fast_clock(monkeypatch, [0.0, 0.0, 0.5])
    with pytest.raises(llama_server.LlamaServerError, match="exited"):
        llama_server.start(tmp_path, port=SIDECAR_PORT)


def test_stop_terminates_and_joins() -> None:
    """stop terminates the process and waits for the join."""
    process = _FakeProcess()
    handle = llama_server.LlamaSidecar(
        process=process,  # type: ignore[arg-type]
        endpoint=DEFAULT_LLAMA_ENDPOINT,
        model_path=Path("/models/gguf"),
    )
    llama_server.stop(handle)
    assert process.terminated
    assert process.wait_calls == 1
    assert not process.killed


def test_stop_kills_after_grace_timeout() -> None:
    """A wedged server that ignores terminate is killed, never awaited."""

    from subprocess import TimeoutExpired

    class _WedgedProcess(_FakeProcess):
        def wait(self, timeout: float | None = None) -> int | None:
            del timeout
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise TimeoutExpired("llama-server", 10.0)
            return self.returncode

    process = _WedgedProcess()
    handle = llama_server.LlamaSidecar(
        process=process,  # type: ignore[arg-type]
        endpoint=DEFAULT_LLAMA_ENDPOINT,
        model_path=Path("/models/gguf"),
    )
    llama_server.stop(handle)
    assert process.terminated
    assert process.killed


def test_stop_none_is_noop() -> None:
    """stop(None) is safe: shutdown unwinds call it unconditionally."""
    llama_server.stop(None)


# ---------------------------------------------------------------------------
# Registry: GGUF pins + spec row + download/verify verbs.
# ---------------------------------------------------------------------------


def test_gguf_pins_present() -> None:
    """bartowski Qwen3.5-4B Q4_K_M pin: repo, file, revision, size, license."""
    assert QWEN35_GGUF_HF_REPO == "bartowski/Qwen_Qwen3.5-4B-GGUF"
    assert QWEN35_GGUF_FILE == "Qwen_Qwen3.5-4B-Q4_K_M.gguf"
    assert len(QWEN35_GGUF_HF_REVISION) == 40
    assert QWEN35_GGUF_SUBDIR == "Qwen3.5-4B-GGUF"
    assert QWEN35_GGUF_MIN_BYTES == 2_500_000_000
    assert QWEN35_GGUF_LICENSE == "Apache-2.0"


def test_gguf_spec_mirrors_awq_shape() -> None:
    """The registry row is a single pinned FileSpec with a size floor."""
    spec = MODEL_SPECS["director-qwen35-gguf"]
    assert spec.name == "director-qwen35-gguf"
    assert spec.manifest_key == "director-gguf"
    assert spec.snapshots == ()
    assert len(spec.files) == 1
    file_spec = spec.files[0]
    assert file_spec.repo_id == QWEN35_GGUF_HF_REPO
    assert file_spec.revision == QWEN35_GGUF_HF_REVISION
    assert file_spec.filename == QWEN35_GGUF_FILE
    assert file_spec.relative_dir == QWEN35_GGUF_SUBDIR
    assert spec.manifest_checkpoint == f"{QWEN35_GGUF_SUBDIR}/{QWEN35_GGUF_FILE}"


def _install_hub_stub(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    """Stub huggingface_hub: snapshot is a no-op, file fetch writes payload."""

    def _fake_snapshot_download(**kwargs: Any) -> str:
        return str(kwargs.get("local_dir", ""))

    def _fake_hub_download(**kwargs: Any) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        subfolder = str(kwargs.get("subfolder", ""))
        target = local_dir / subfolder / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    hub_module = types.ModuleType("huggingface_hub")
    hub_module.snapshot_download = _fake_snapshot_download  # type: ignore[attr-defined]
    hub_module.hf_hub_download = _fake_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)


def test_gguf_download_merges_manifest_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download fetches the single GGUF and records repo/revision/license."""
    _install_hub_stub(monkeypatch, b"gguf-bytes")
    record = download_model(tmp_path, "director-qwen35-gguf")
    entry = record["director-gguf"]
    assert isinstance(entry, dict)
    assert entry["repo"] == QWEN35_GGUF_HF_REPO
    assert entry["revision"] == QWEN35_GGUF_HF_REVISION
    assert entry["license"] == QWEN35_GGUF_LICENSE
    assert (tmp_path / "manifest.json").is_file()


def test_gguf_verify_rejects_small_file(tmp_path: Path) -> None:
    """The ~3 GiB size floor rejects truncated downloads."""
    target = tmp_path / QWEN35_GGUF_SUBDIR / QWEN35_GGUF_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"truncated")
    ok, _message = verify_model(tmp_path, "director-qwen35-gguf")
    assert not ok


def test_gguf_verify_accepts_full_size_file(tmp_path: Path) -> None:
    """A full-size GGUF with an (unhashed) manifest record verifies."""
    _write_gguf(tmp_path, QWEN35_GGUF_MIN_BYTES)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"director-gguf": {"repo": QWEN35_GGUF_HF_REPO}}), encoding="utf-8"
    )
    ok, message = verify_model(tmp_path, "director-qwen35-gguf")
    assert ok, message


# ---------------------------------------------------------------------------
# models_ensure: backend-driven spec selection.
# ---------------------------------------------------------------------------


def _spec_names(config: ProjectConfig) -> set[str]:
    """Required spec names for a stock generate config (no SFX, no augment)."""
    return {
        entry.spec for entry in required_specs(config, sfx_enabled=False, augment_enabled=False)
    }


def test_required_specs_selects_gguf_for_llama_backend() -> None:
    """llama backend ensures the GGUF sidecar file, never the AWQ stack."""
    specs = _spec_names(_llama_config())
    assert "director-qwen35-gguf" in specs
    assert "director-qwen4b-awq" not in specs
    assert "director-qwen8b" not in specs


def test_required_specs_keeps_awq_for_qwen_backend() -> None:
    """Control: the explicit qwen backend still ensures the AWQ decider."""
    base = ProjectConfig(style="llama sidecar probe")
    specs = _spec_names(resolve_config(base, director="qwen"))
    assert "director-qwen4b-awq" in specs
    assert "director-qwen35-gguf" not in specs


def test_required_specs_defaults_to_gguf_for_llama_backend() -> None:
    """Default backend (llama) ensures the sidecar GGUF, not AWQ."""
    base = ProjectConfig(style="llama sidecar probe")
    specs = _spec_names(base)
    assert "director-qwen35-gguf" in specs
    assert "director-qwen4b-awq" not in specs


def test_required_specs_has_no_director_spec_for_deterministic() -> None:
    """Control: deterministic needs no director weights at all."""
    base = ProjectConfig(style="llama sidecar probe")
    specs = _spec_names(resolve_config(base, director="deterministic"))
    assert "director-qwen35-gguf" not in specs
    assert "director-qwen4b-awq" not in specs
    assert "director-qwen8b" not in specs


# ---------------------------------------------------------------------------
# config: backend vocabulary + endpoint plumbing.
# ---------------------------------------------------------------------------


def test_director_config_accepts_llama_with_default_endpoint() -> None:
    """llama joins the backend vocabulary; the endpoint defaults loopback."""
    assert DirectorConfig().backend == "llama"
    config = DirectorConfig(backend="llama")
    assert config.backend == "llama"
    assert config.llama_endpoint == DEFAULT_LLAMA_ENDPOINT
    assert DEFAULT_LLAMA_ENDPOINT == "http://127.0.0.1:8080"


def test_director_config_rejects_unknown_backend_and_bad_endpoint() -> None:
    """Typos fail at the config layer, never deep in the worker."""
    with pytest.raises(Exception, match="llama"):
        DirectorConfig(backend="llamma")  # type: ignore[arg-type]
    with pytest.raises(Exception, match="llama_endpoint"):
        DirectorConfig(llama_endpoint="not-a-url")


def test_default_config_toml_carries_llama_endpoint(tmp_path: Path) -> None:
    """Generated TOML pins the sidecar endpoint under [director]."""
    raw = default_config_toml("llama-toml", "pastel neon", 7)
    assert 'llama_endpoint = "http://127.0.0.1:8080"' in raw
    config_path = tmp_path / "voyage.toml"
    config_path.write_text(raw, encoding="utf-8")
    config, _digest = load_config(config_path)
    assert config.director.llama_endpoint == DEFAULT_LLAMA_ENDPOINT


def test_resolve_config_director_llama_roundtrip() -> None:
    """--director llama survives the resolver into the stored config."""
    base = ProjectConfig(style="llama sidecar probe")
    resolved = resolve_config(base, director="llama")
    assert resolved.director.backend == "llama"
    assert resolved.director.llama_endpoint == DEFAULT_LLAMA_ENDPOINT


# ---------------------------------------------------------------------------
# supervisor: sidecar lifecycle + endpoint flow (init + every decide).
# ---------------------------------------------------------------------------


def test_supervisor_passes_llama_endpoint_in_director_init_payload(
    tmp_path: Path,
) -> None:
    """llama backend: init carries models_dir + device + llama_endpoint."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "llama")
    assert supervisor._director._init_payload == {
        "models_dir": supervisor._config.video.models_dir,
        "device": supervisor._config.director.device,
        "llama_endpoint": DEFAULT_LLAMA_ENDPOINT,
    }


def test_supervisor_omits_llama_endpoint_for_qwen_backend(tmp_path: Path) -> None:
    """Control: the default AWQ path sends no endpoint (client untouched)."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "qwen")
    assert "llama_endpoint" not in supervisor._director._init_payload


def _decide_state() -> SimpleNamespace:
    """Minimal commit state for the decide-payload builder."""
    return SimpleNamespace(
        decision_index=0,
        phase="ESTABLISH",
        current_concept="a calm valley",
        destination_concept="a calm valley",
        committed_segments=0,
        timeline_frames=0,
        next_segment_number=0,
    )


def test_supervisor_maps_llama_to_qwen_wire_backend_with_endpoint(tmp_path: Path) -> None:
    """Decide payload: wire backend stays qwen (worker routes on it) + endpoint."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "llama")
    store = ConceptStore(tmp_path / "novelty")
    payload = supervisor._decide_payload(
        supervisor._config,
        _decide_state(),
        store,
        StyleSpec(prompt="pastel neon line-art"),
        previous_captions="",
    )
    assert payload["backend"] == "qwen"
    assert payload["llama_endpoint"] == DEFAULT_LLAMA_ENDPOINT


def test_supervisor_decide_payload_has_no_endpoint_for_qwen(tmp_path: Path) -> None:
    """Control: default decides carry no sidecar key (AWQ path untouched)."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "qwen")
    store = ConceptStore(tmp_path / "novelty")
    payload = supervisor._decide_payload(
        supervisor._config,
        _decide_state(),
        store,
        StyleSpec(prompt="pastel neon line-art"),
        previous_captions="",
    )
    assert payload["backend"] == "qwen"
    assert "llama_endpoint" not in payload


def test_start_workers_starts_sidecar_before_director(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar readiness gates the director init: order is sidecar, then workers."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "llama")
    events: list[str] = []

    def _record_sidecar(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        events.append("sidecar")

    def _record_worker(name: str) -> None:
        events.append(name)

    monkeypatch.setattr(llama_server, "start", _record_sidecar)
    for worker_name in ("_video", "_audio", "_director"):
        worker = getattr(supervisor, worker_name)
        monkeypatch.setattr(worker, "start", lambda name=worker_name: _record_worker(name))
    supervisor.start_workers()
    assert events == ["_video", "_audio", "sidecar", "_director"]
    assert supervisor._workers_running


def test_start_workers_skips_sidecar_for_qwen_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the explicit qwen backend never touches the sidecar binary."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "qwen")

    def _fail_start(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("sidecar must not start for qwen")

    monkeypatch.setattr(llama_server, "start", _fail_start)
    for worker_name in ("_video", "_audio", "_director"):
        monkeypatch.setattr(getattr(supervisor, worker_name), "start", lambda: None)
    supervisor.start_workers()
    assert supervisor._llama_sidecar is None


def test_sidecar_readiness_failure_fails_start_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead sidecar aborts startup: never silently fall back to AWQ."""
    from voyage.errors import FatalWorkerError

    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "llama")

    def _fail_start(*args: Any, **kwargs: Any) -> Any:
        raise llama_server.LlamaServerError("connection refused")

    monkeypatch.setattr(llama_server, "start", _fail_start)
    started: list[str] = []
    for worker_name in ("_video", "_audio", "_director"):
        monkeypatch.setattr(
            getattr(supervisor, worker_name),
            "start",
            lambda name=worker_name: started.append(name),
        )
    with pytest.raises(FatalWorkerError, match="[Ll]lama"):
        supervisor.start_workers()
    assert supervisor._llama_sidecar is None


def test_stop_workers_stops_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown stops the sidecar after the director (no orphan process)."""
    run_dir = tmp_path / "run"
    supervisor = _supervisor_with_backend(run_dir, "llama")
    events: list[str] = []
    supervisor._llama_sidecar = SimpleNamespace()  # type: ignore[assignment]

    def _record_stop(handle: Any) -> None:
        del handle
        events.append("sidecar")

    def _record_worker_stop(name: str) -> None:
        events.append(name)

    monkeypatch.setattr(llama_server, "stop", _record_stop)
    for worker_name in ("_video", "_audio", "_director"):
        monkeypatch.setattr(
            getattr(supervisor, worker_name),
            "stop",
            lambda name=worker_name: _record_worker_stop(name),
        )
    supervisor.stop_workers()
    assert events.index("_director") < events.index("sidecar")
    assert supervisor._llama_sidecar is None


def test_models_download_target_for_gguf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`models download director-qwen35-gguf` parses and dispatches (no hub)."""
    import voyage.cli as cli_module
    from voyage.cli import build_parser

    args = build_parser().parse_args(["models", "download", "director-qwen35-gguf"])
    assert args.models_target == "director-qwen35-gguf"
    calls: dict[str, Any] = {}

    def _fake_download(models_dir: Path) -> dict[str, Any]:
        calls["models_dir"] = models_dir
        return {"director-gguf": {"checkpoint_bytes": 123}}

    monkeypatch.setattr(cli_module, "download_director_gguf_models", _fake_download)
    download_args = argparse.Namespace(
        models_action="download",
        models_target="director-qwen35-gguf",
        models_dir=str(tmp_path),
    )
    assert cli_module.cmd_models(download_args) == 0
    assert calls["models_dir"] == tmp_path


def test_init_parser_accepts_llama_director() -> None:
    """`init --director llama` parses (backend threaded through the CLI)."""
    from voyage.cli import build_parser

    args = build_parser().parse_args(
        ["init", "--output", "out", "--style", "s", "--director", "llama"]
    )
    assert args.director == "llama"

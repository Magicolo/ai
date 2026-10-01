"""llama-server sidecar lifecycle (DESIGN §140 llama entry, §§8-9 director).

Why this module exists: the Qwen director moves off the in-process
transformers AWQ stack (~23 s/directive on cuda:1) onto a `llama-server`
sidecar serving the Qwen3.5-4B Q4_K_M GGUF on the loopback interface.
This module owns the server process and nothing else — spawn with the
pinned contract argv, wait for OpenAI-compatible readiness, terminate on
shutdown. No inference logic lives here: the HTTP client is the director
worker's `_qwen_generate` llama branch (`voyage/workers/director.py`,
`llama_endpoint` payload key), which reads the endpoint this module's
handle carries.

Contract (pinned with the director track — do not reshape unilaterally):
the sidecar listens on 127.0.0.1:8080 (fixed, isolated to containers) and
serves `bartowski/Qwen_Qwen3.5-4B-GGUF` file
`Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~3 GiB, registry row
`director-qwen35-gguf` in `voyage/model_registry.py`). Spawn argv is
`llama-server -m <gguf> --host 127.0.0.1 --port <port> --ctx-size 4096
-ngl 99 --cache-prompt` — every flag verified real against the llama.cpp
server options (`-m/--model`, `--host`, `--port`, `-c/--ctx-size`,
`-ngl/--gpu-layers/--n-gpu-layers`, `--cache-prompt/--no-cache-prompt`
default-enabled). Readiness is `GET /v1/models` (the OpenAI-compatible
index every llama-server exposes) with a bounded deadline; a timeout or
an early process exit fails loud — the supervisor converts it to a fatal
run error and never silently falls back to AWQ (silent backend swaps
corrupt experiments).

Bind note: the server always binds the loopback host above, even if a
custom endpoint names another host — the endpoint selects where the
readiness probe (and later the worker client) connects, never a wider
bind. No shell anywhere: the spawn is an argv list (`subprocess.Popen`
without `shell=True`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL, Popen, TimeoutExpired
from time import monotonic, sleep
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

LLAMA_SERVER_HOST = "127.0.0.1"
"""Loopback bind host (fixed by contract — the sidecar never leaves the box)."""

LLAMA_SERVER_PORT = 8080
"""Fixed sidecar port (contract — isolated to containers)."""

LLAMA_SERVER_CONTEXT_SIZE = 4096
"""Prompt context tokens per server slot (contract)."""

LLAMA_SERVER_GPU_LAYERS = 99
"""Offloaded layer count: 99 exceeds the 4B layer total, so all of them
ride the GPU (the `-ngl` exact-number form, same idiom as the upstream
CUDA server examples)."""

LLAMA_SERVER_BINARY_NAME = "llama-server"
"""Binary name on PATH (installed by worker/Dockerfile.video)."""

LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE = "VOYAGE_LLAMA_SERVER_BIN"
"""Explicit binary override (tests/dev): an absolute path wins over PATH."""

LLAMA_SERVER_READY_TIMEOUT_SECONDS = 120.0
"""Bounded readiness deadline: a cold CUDA init plus a ~3 GiB weight
upload lands in well under a minute; 120 s fails a wedged server
without hanging a segment commit behind it."""

LLAMA_SERVER_READY_POLL_INTERVAL_SECONDS = 0.5
"""Probe cadence while waiting for readiness (prompt but quiet)."""

LLAMA_SERVER_READY_PROBE_TIMEOUT_SECONDS = 5.0
"""Per-probe socket timeout: one hung probe never eats the whole budget."""

LLAMA_SERVER_STOP_TIMEOUT_SECONDS = 10.0
"""Grace period between terminate and kill at shutdown."""


class LlamaServerError(Exception):
    """Sidecar lifecycle failure: spawn, readiness, or teardown.

    The supervisor converts this to a fatal run error (never a silent
    backend swap) — see `Supervisor._start_llama_sidecar`.
    """


@dataclass(frozen=True)
class LlamaSidecar:
    """A running sidecar: its process plus where to reach it."""

    process: Popen[bytes]
    endpoint: str
    model_path: Path


def server_binary() -> str:
    """Resolve the server binary: explicit env path or the PATH name."""
    override = os.environ.get(LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE, "").strip()
    return override or LLAMA_SERVER_BINARY_NAME


def endpoint_for(port: int) -> str:
    """Base URL for a sidecar port (the loopback contract endpoint)."""
    return f"http://{LLAMA_SERVER_HOST}:{port}"


def port_for_endpoint(endpoint: str) -> int:
    """Parse the spawn port out of the configured endpoint (single source).

    The supervisor stores one `llama_endpoint` string; the spawn port
    derives from it so the two can never disagree.
    """
    parsed = urlparse(endpoint)
    if not parsed.hostname:
        raise LlamaServerError(f"llama endpoint {endpoint!r} has no host")
    if parsed.port is None:
        return LLAMA_SERVER_PORT
    if not 1 <= parsed.port <= 65535:
        raise LlamaServerError(f"llama endpoint {endpoint!r} has an invalid port")
    return parsed.port


def gguf_path_for(models_dir: str | Path) -> Path:
    """Resolve the sidecar weight file inside the models mount."""
    from voyage.model_registry import QWEN35_GGUF_FILE, QWEN35_GGUF_SUBDIR

    return Path(models_dir) / QWEN35_GGUF_SUBDIR / QWEN35_GGUF_FILE


def build_server_argv(
    binary: str,
    model_path: str | Path,
    *,
    port: int,
    context_size: int,
    gpu_layers: int,
) -> list[str]:
    """Build the pinned spawn argv (pure: unit-tested without a process)."""
    return [
        binary,
        "-m",
        str(model_path),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "-ngl",
        str(gpu_layers),
        "--cache-prompt",
    ]


def _probe_models_endpoint(endpoint: str) -> bool:
    """True once `GET <endpoint>/v1/models` answers HTTP 200."""
    request = Request(endpoint.rstrip("/") + "/v1/models")
    try:
        response = urlopen(request, timeout=LLAMA_SERVER_READY_PROBE_TIMEOUT_SECONDS)
        try:
            return bool(getattr(response, "status", None) == 200)
        finally:
            response.close()
    except (URLError, OSError):
        return False


def wait_ready(process: Popen[bytes], endpoint: str, timeout_seconds: float) -> None:
    """Block until readiness or fail loud (timeout or early process exit)."""
    deadline = monotonic() + timeout_seconds
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            raise LlamaServerError(
                f"llama-server exited during startup (code {exit_code}) — "
                "the sidecar never became ready"
            )
        if _probe_models_endpoint(endpoint):
            return
        if monotonic() >= deadline:
            raise LlamaServerError(
                f"llama-server readiness timeout after {timeout_seconds:.0f}s "
                f"at {endpoint} — refusing to run director traffic at a dead server"
            )
        sleep(LLAMA_SERVER_READY_POLL_INTERVAL_SECONDS)


def start(
    models_dir: str | Path,
    port: int = LLAMA_SERVER_PORT,
    context_size: int = LLAMA_SERVER_CONTEXT_SIZE,
    gpu_layers: int = LLAMA_SERVER_GPU_LAYERS,
) -> LlamaSidecar:
    """Spawn the sidecar and wait for readiness (fail loud, never orphan).

    Raises `LlamaServerError` when the weight file is missing, the binary
    cannot spawn, the process exits early, or readiness times out — a
    failed start terminates the child before raising, so no half-started
    server is ever left behind.
    """
    model_path = gguf_path_for(models_dir)
    if not model_path.is_file():
        raise LlamaServerError(
            f"llama sidecar weight missing: {model_path} "
            "(provision with `voyage models download director-qwen35-gguf`)"
        )
    binary = server_binary()
    command = build_server_argv(
        binary, model_path, port=port, context_size=context_size, gpu_layers=gpu_layers
    )
    try:
        process = Popen(command, stdout=DEVNULL, stderr=DEVNULL)
    except OSError as exc:
        raise LlamaServerError(
            f"cannot spawn llama-server {binary!r}: {exc} "
            "(the video image installs it — see worker/Dockerfile.video)"
        ) from exc
    handle = LlamaSidecar(process=process, endpoint=endpoint_for(port), model_path=model_path)
    try:
        wait_ready(process, handle.endpoint, LLAMA_SERVER_READY_TIMEOUT_SECONDS)
    except Exception:  # teardown must run for any wait failure, then re-raise
        stop(handle)
        raise
    return handle


def stop(handle: LlamaSidecar | None) -> None:
    """Terminate the sidecar and join it (`None` is a safe no-op).

    Shutdown unwinds call this unconditionally, so `None` (never started)
    must be silent. A wedged server that ignores terminate is killed
    after the grace period — shutdown never stalls. An `OSError` from an
    already-reaped child reads as already stopped, never as a failure
    that masks the original shutdown error.
    """
    if handle is None:
        return
    handle.process.terminate()
    try:
        handle.process.wait(timeout=LLAMA_SERVER_STOP_TIMEOUT_SECONDS)
    except TimeoutExpired:
        handle.process.kill()
        handle.process.wait()
    except OSError:
        pass

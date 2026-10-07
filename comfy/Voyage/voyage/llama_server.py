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
import re
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
"""Binary name on PATH (installed by worker/Dockerfile.video and worker/Dockerfile.ltx)."""

LLAMA_SERVER_BAKED_PATHS = (
    "/opt/llama.cpp/bin/llama-server",
    "/usr/local/bin/llama-server",
)
"""Allowlisted absolute sidecar binaries (issue 237).

Both worker images install the same source build twice: the real path
plus a `/usr/local/bin` symlink (see the `ln -sfn` step in each
Dockerfile). The `VOYAGE_LLAMA_SERVER_BIN` override must name one of
these or spawn refuses — an arbitrary absolute path (e.g. a planted
`/tmp` binary) never executes. Build-side integrity is the existing
source-tarball sha256 gate in the Dockerfiles; this list is the
run-side half of the same trust boundary.
"""

LLAMA_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
"""Endpoint hosts the sidecar may serve (issue 237, fail-closed).

The server always binds the loopback interface, so a non-loopback
endpoint would probe (and later route director traffic at) a server this
module never started — the "never leaves the box" contract inverts
silently. Anything outside this set raises instead of connecting.
"""

LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE = "VOYAGE_LLAMA_SERVER_BIN"
"""Explicit binary override (tests/dev): an absolute path wins over PATH."""

LLAMA_SERVER_VISIBLE_DEVICES_ENVIRONMENT_VARIABLE = "CUDA_VISIBLE_DEVICES"
"""Child-env key masking the sidecar to one GPU (the literal CUDA name)."""

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
    """Resolve the server binary: explicit env path or the PATH name.

    The override is allowlisted (issue 237): it must name one of
    `LLAMA_SERVER_BAKED_PATHS` or spawn refuses with `LlamaServerError`.
    The bare `PATH` name keeps its exact previous behavior.
    """
    override = os.environ.get(LLAMA_SERVER_BINARY_ENVIRONMENT_VARIABLE, "").strip()
    if not override:
        return LLAMA_SERVER_BINARY_NAME
    if override not in LLAMA_SERVER_BAKED_PATHS:
        raise LlamaServerError(
            f"refusing llama-server binary override {override!r} — "
            f"not in the baked allowlist {list(LLAMA_SERVER_BAKED_PATHS)} "
            "(tests/dev must use a baked path; production never overrides)"
        )
    return override


def validate_endpoint_host(endpoint: str) -> str:
    """Fail-closed loopback check for a sidecar endpoint (issue 237).

    Returns the normalized hostname when it is in `LLAMA_ALLOWED_HOSTS`;
    raises `LlamaServerError` for anything else (including a missing
    host), so a non-loopback endpoint can never be probed or served.
    """
    hostname = urlparse(endpoint).hostname or ""
    normalized = hostname.strip().lower()
    if normalized not in LLAMA_ALLOWED_HOSTS:
        raise LlamaServerError(
            f"llama endpoint {endpoint!r} names non-loopback host {hostname!r} — "
            f"refusing (allowed: {sorted(LLAMA_ALLOWED_HOSTS)})"
        )
    return normalized


def endpoint_for(port: int) -> str:
    """Base URL for a sidecar port (the loopback contract endpoint)."""
    return f"http://{LLAMA_SERVER_HOST}:{port}"


def port_for_endpoint(endpoint: str) -> int:
    """Parse the spawn port out of the configured endpoint (single source).

    The supervisor stores one `llama_endpoint` string; the spawn port
    derives from it so the two can never disagree. The host is validated
    fail-closed first (issue 237): a non-loopback endpoint raises instead
    of probing a server this module never started.
    """
    parsed = urlparse(endpoint)
    if not parsed.hostname:
        raise LlamaServerError(f"llama endpoint {endpoint!r} has no host")
    validate_endpoint_host(endpoint)
    if parsed.port is None:
        return LLAMA_SERVER_PORT
    if not 1 <= parsed.port <= 65535:
        raise LlamaServerError(f"llama endpoint {endpoint!r} has an invalid port")
    return parsed.port


def visible_devices_for(device: str) -> str | None:
    """Physical GPU index masking the sidecar, or None for no mask.

    `cuda:N` (and bare `cuda`, torch's default-device alias for index 0)
    maps to the index string the CUDA runtime expects in
    `CUDA_VISIBLE_DEVICES`, so the server process sees exactly one card.
    Anything else (`cpu`, or garbage — unreachable via the config
    validator, total anyway) returns None, keeping today's
    inherited-visibility behavior: the sidecar is GPU-only by design
    (`-ngl 99`), so a non-cuda device carries no index to pin. Masking
    beats `--tensor-split`: the process never opens a context on the
    other card at all, instead of merely holding no layers there.
    """
    match = re.fullmatch(r"cuda(?::(\d+))?", device.strip())
    if match is None:
        return None
    return match.group(1) or "0"


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


def is_sidecar_healthy(endpoint: str) -> bool:
    """Single-probe health check (Track D gate before decide/enhance).

    True when the sidecar answers `GET <endpoint>/v1/models` with HTTP
    200. Never raises, never waits — a refused connection reads as
    unhealthy so the caller heals instead of routing director traffic at
    a dead server.
    """
    return _probe_models_endpoint(endpoint)


def health_gate(endpoint: str) -> bool:
    """Health gate helper before decide/enhance (Track A wire point).

    Same single-probe contract as `is_sidecar_healthy` under the gate
    name Track A calls: True proceeds to decide/enhance, False heals via
    `heal_if_needed` first (start-once-never-healed fix — a sidecar that
    dies mid-run must not serve stale traffic).
    """
    return is_sidecar_healthy(endpoint)


def is_port_in_use(port: int) -> bool:
    """True when something answers TCP on loopback `port` (Track D)."""
    import socket as _socket

    try:
        with _socket.create_connection((LLAMA_SERVER_HOST, port), timeout=1.0):
            return True
    except OSError:
        return False


def find_free_port() -> int:
    """Random free loopback port (Track D collision escape)."""
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
        sock.bind((LLAMA_SERVER_HOST, 0))
        return int(sock.getsockname()[1])


def resolve_port(port: int) -> int:
    """Free port for the sidecar: `port` when free, else a random one.

    Probe + free-port (never adopt-or-kill an unknown owner — issue 237:
    a fixed 8080 collides with a stale server from a previous run and
    misroutes traffic). The caller propagates the resolved port via
    `handle.endpoint` (endpoint propagation — never assume 8080).
    """
    if 1 <= port <= 65535 and not is_port_in_use(port):
        return port
    return find_free_port()


def port_file_for(run_directory: str | Path) -> Path:
    """Per-run port record beside the run (issue 237).

    The resolved sidecar port is written here by `claim_sidecar_port` so
    operators (and a restarted supervisor) can see which port the run's
    server actually owns instead of assuming the fixed default.
    """
    return Path(run_directory) / "llama-sidecar.port"


def claim_sidecar_port(port: int, *, lock_path: str | Path, port_file: str | Path) -> int:
    """Resolve the sidecar port under a shared file lock (issue 237).

    `resolve_port` alone races: two concurrent supervisors can both probe
    a free 8080 and both spawn onto it. Serializing the probe + record
    on `lock_path` (one shared file, e.g. under the models mount every
    run on the box already shares) closes the race; the winner records
    its port in its own `port_file` (see `port_file_for`). Returns the
    claimed port. The supervisor wiring (claim, then `start(port=claimed)`,
    then route director traffic at the claimed endpoint) is the follow-up —
    this helper plus `validate_endpoint_host` and the binary allowlist are
    the in-scope halves that need no caller change to take effect.
    """
    import fcntl as file_lock

    resolved_path = Path(lock_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "a+") as lock_handle:
        file_lock.flock(lock_handle.fileno(), file_lock.LOCK_EX)
        try:
            claimed = resolve_port(port)
            target = Path(port_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{claimed}\n", encoding="utf-8")
            return claimed
        finally:
            file_lock.flock(lock_handle.fileno(), file_lock.LOCK_UN)


def is_owned_by(handle: LlamaSidecar | None, pid: int | None) -> bool:
    """Pid/port ownership check (Track D): True when `handle` is live.

    `pid` is the expected owner (the supervisor's recorded sidecar pid);
    None reads as unowned. A live process with a matching pid plus a
    healthy endpoint is owned; anything else (dead process, pid mismatch,
    unhealthy endpoint) is stale and must heal, never serve traffic.
    """
    if handle is None or pid is None:
        return False
    try:
        live_pid: int | None = int(handle.process.pid)
    except (AttributeError, TypeError, ValueError):
        return False
    if live_pid != pid:
        return False
    try:
        alive = handle.process.poll() is None
    except (AttributeError, OSError):
        return False
    if not alive:
        return False
    return is_sidecar_healthy(handle.endpoint)


def metric_for_heal(*, op: str, attempt: int, budget: int, reason: str) -> dict[str, str | int]:
    """`worker_restart`-style metric payload for sidecar heals (Track D).

    Returns the code (data only — the caller emits via its own
    `_log_metric`, so this module never touches supervisor logging).
    """
    return {
        "event": "worker_restart",
        "worker": "llama-sidecar",
        "op": op,
        "attempt": attempt,
        "budget": budget,
        "reason": reason,
    }


def heal_if_needed(
    handle: LlamaSidecar | None,
    models_dir: str | Path,
    *,
    port: int = LLAMA_SERVER_PORT,
    op: str = "decide",
    attempt: int = 1,
    budget: int = 3,
    visible_devices: str | None = None,
) -> tuple[LlamaSidecar | None, dict[str, str | int] | None]:
    """Heal a dead sidecar before decide/enhance (Track D never-healed fix).

    Probes `handle` (None or unhealthy/dead → stale); a healthy handle is
    returned untouched with no metric. A stale handle is stopped
    (best-effort) and a fresh sidecar starts on a resolved port
    (`resolve_port` — endpoint propagation via the returned handle).
    Returns `(handle, metric_or_None)`: the metric is the
    `worker_restart`-style code for the caller to emit (return code, caller
    emits — this module never logs). Raises `LlamaServerError` when the
    restart itself fails (fail loud, never silent fallback to AWQ).
    """
    if handle is not None:
        try:
            alive = handle.process.poll() is None
        except (AttributeError, OSError):
            alive = False
        if alive and is_sidecar_healthy(handle.endpoint):
            return handle, None
        stop(handle)
    resolved = resolve_port(port)
    fresh = start(models_dir, port=resolved, visible_devices=visible_devices)
    metric = metric_for_heal(op=op, attempt=attempt, budget=budget, reason="sidecar-heal")
    return fresh, metric


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
    visible_devices: str | None = None,
) -> LlamaSidecar:
    """Spawn the sidecar and wait for readiness (fail loud, never orphan).

    `visible_devices` masks the child to one physical GPU via
    `CUDA_VISIBLE_DEVICES` (the parent env is otherwise inherited whole);
    None passes `env=None`, which Popen documents as plain inheritance.
    The supervisor derives it from the director device, so the default
    cuda:1 run pins the server to the 2060 and it can never straddle the
    video card.

    Raises `LlamaServerError` when the weight file is missing, the binary
    cannot spawn, the process exits early, or readiness times out — a
    failed start terminates the child before raising, so no half-started
    server is ever left behind.
    """
    model_path = gguf_path_for(models_dir)
    if not model_path.is_file():
        raise LlamaServerError(
            f"llama sidecar weight missing: {model_path} "
            "(provision via the `configure` ensure-path — "
            '`model_registry.download_model(models_dir, "director-qwen35-gguf")`)'
        )
    binary = server_binary()
    command = build_server_argv(
        binary, model_path, port=port, context_size=context_size, gpu_layers=gpu_layers
    )
    child_env: dict[str, str] | None = None
    if visible_devices is not None:
        child_env = {
            **os.environ,
            LLAMA_SERVER_VISIBLE_DEVICES_ENVIRONMENT_VARIABLE: visible_devices,
        }
    try:
        # env=None inherits the parent env (Popen contract), so the masked
        # and unmasked paths share one call shape.
        process = Popen(command, stdout=DEVNULL, stderr=DEVNULL, env=child_env)
    except OSError as exc:
        raise LlamaServerError(
            f"cannot spawn llama-server {binary!r}: {exc} "
            "(the CUDA images install it — see worker/Dockerfile.video and worker/Dockerfile.ltx)"
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

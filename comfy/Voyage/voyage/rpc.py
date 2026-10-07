"""Typed JSONL RPC protocol (DESIGN §§45-46, task group D).

Transport: supervisor launches a worker subprocess; requests go to the
worker's stdin, responses come from stdout, diagnostics from stderr.
Stdout is reserved for RPC — never log there from a worker.
"""

from __future__ import annotations

import contextlib
import math
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from voyage.atomic import JsonValue
from voyage.errors import FatalWorkerError, RecoverableWorkerError, VoyageError
from voyage.logrotate import rotate_log
from voyage.models import WorkerErrorDetail, WorkerRequest, WorkerResponse

RpcPayload = dict[str, JsonValue]
"""One worker request payload: JSON-shaped, never bare `Any` (issue 036).

Per-op shapes stay `dict`-level for now — full `TypedDict`s per op would
reach into the worker implementations (out of scope); this alias is the
ratchet point (see issues/036).
"""

RpcResult = dict[str, JsonValue]
"""One worker result: same boundary contract as the payload."""

OPS = (
    "init",
    "health",
    "generate_blocks",
    "generate_audio",
    "decide",
    "checkpoint",
    "resume",
    "shutdown",
    "benchmark",
    "evict_gpu",
    "rebuild",
    "cancel",
)

CANCEL_OP = "cancel"
"""Worker op name for cooperative cancel (DESIGN §45 liveness).

Mid-render workers are blocked inside `generate_blocks` and cannot serve
it — the op exists so an idle worker answers deterministically and so
Track A has a stable wire name. True mid-render interrupt is client-side
abandon-and-restart (`SubprocessWorker.cancel` / `call_cancel`), never a
served op.
"""

DECIDE_TIMEOUT_MIN_SECONDS = 60.0
"""Floor for the synchronous decide RPC budget (Track D pin)."""

DECIDE_TIMEOUT_MAX_SECONDS = 120.0
"""Ceiling for the synchronous decide RPC budget (Track D pin)."""

WEDGED_DEFAULT_THRESHOLD_SECONDS = 300.0
"""Default no-progress threshold for `is_wedged` (5 min, Track D watchdog)."""

#: Hard cap on one worker response line (issue 001). A worker sending GBs
#: without a newline is the same DoS as a partial line that never ends —
#: the reader must stop accumulating at some point and fail the call.
MAX_RESPONSE_LINE_BYTES = 8 * 1024 * 1024

#: Bytes per os.read while accumulating a response line. Small enough to
#: notice the `\\n`/cap/deadline promptly, large enough to avoid a syscall
#: per byte on multi-megabyte results; any value works, this one just
#: avoids re-tuning the loop.
RESPONSE_READ_CHUNK_BYTES = 65536

#: Default per-call timeout in seconds. Mirrors the `[voyage]`
#: `rpc_timeout_seconds` config default — the config stays the single
#: source of truth wherever a supervisor exists; this covers bare
#: SubprocessWorker use (tests, probes) with the same budget.
DEFAULT_RPC_TIMEOUT_SECONDS = 600.0

#: Grace given to a worker to exit after its stdin closes before SIGKILL.
#: wait() returns once the kill lands, so this bounds stop() instead of
#: hanging the supervisor on an unresponsive child.
WORKER_STOP_GRACE_SECONDS = 10.0


def encode_request(request: WorkerRequest) -> str:
    return request.model_dump_json() + "\n"


def decode_request(line: str) -> WorkerRequest:
    return WorkerRequest.model_validate_json(line)


def encode_response(response: WorkerResponse) -> str:
    return response.model_dump_json() + "\n"


def decode_response(line: str) -> WorkerResponse:
    return WorkerResponse.model_validate_json(line)


def _request_sequence_number(request_id: str) -> int | None:
    """Numeric suffix of a `req-NNNNNN` id, else None (issue 137).

    Never raises: an id that is not ours (future, garbage, another
    protocol's) yields None so the caller treats it as a genuine
    protocol break instead of a discardable stale line.
    """
    prefix, _, suffix = request_id.partition("-")
    if prefix != "req" or not suffix.isdigit():
        return None
    return int(suffix)


def _is_stale_response(response_id: str, request_id: str) -> bool:
    """True when a response line belongs to an earlier request (issue 137).

    Only a strictly older sequence number counts as stale: a future or
    unparseable id is not a late line from a timed-out call but a
    protocol break, and the caller keeps it Fatal.
    """
    seen = _request_sequence_number(response_id)
    expected = _request_sequence_number(request_id)
    return seen is not None and expected is not None and seen < expected


def _require_finite_positive_timeout(module: str, operation: str, timeout_value: float) -> None:
    """Reject timeouts that would break deadline math (issue 100).

    `select.select` needs a finite non-negative timeout: NaN raises raw
    `ValueError`, inf raises raw `OverflowError`, and neither is a
    `VoyageError` — so both would bypass the supervisor's
    branch-on-class restart path. Reject every non-finite or
    non-positive value here as `RecoverableWorkerError` (the same class
    a genuinely expired deadline raises), before any deadline is
    computed. Mirrors `audio.beat._require_finite`. `bool` is rejected
    explicitly (issue 282): `True == 1` would otherwise pass as a 1 s
    deadline, mirroring the `workers/loop.py` `checked_request` guard.
    """
    if (
        isinstance(timeout_value, bool)
        or not isinstance(timeout_value, (int, float))
        or not math.isfinite(timeout_value)
        or timeout_value <= 0
    ):
        raise RecoverableWorkerError(
            f"worker {module} refusing non-finite or non-positive timeout "
            f"({timeout_value!r} on {operation})"
        )


def success(request_id: str, result: RpcResult) -> WorkerResponse:
    return WorkerResponse(id=request_id, ok=True, result=result)


def cuda_visible_devices_for_session_device(session_device: str | None) -> str | None:
    """Physical GPU index masking a worker to its session device (Track D).

    `cuda:N` (and bare `cuda`, torch's default-device alias for index 0)
    maps to the index string the CUDA runtime expects in
    `CUDA_VISIBLE_DEVICES`, so the child process never opens a context on
    another card. `cpu`, empty, or non-CUDA strings return None (no mask —
    the worker is CPU-only or the device is validated elsewhere). Never
    raises: callers with garbage get no mask, and the worker's own device
    validator fails loud.
    """
    if not isinstance(session_device, str):
        return None
    text = session_device.strip()
    if text == "cuda":
        return "0"
    prefix, _, index_text = text.partition(":")
    if prefix != "cuda":
        return None
    index_text = index_text.strip()
    if not index_text.isdigit():
        return None
    return index_text


def _spawn_env(executable: str | None, session_device: str | None = None) -> dict[str, str] | None:
    """Child env carrying the interpreter's bin dir on PATH (Stage C).

    JIT compilers (torch cpp_extension for ExLlamaV2/Marlin, ninja-driven)
    resolve their toolchain via `shutil.which` at compile time inside the
    worker — but workers spawn by absolute interpreter path with the
    supervisor's system PATH, so a venv's own `ninja` binary is invisible
    and every JIT fails in 0.0 s (live-verified: ExLlamaV2 fell back to
    the ~9 tok/s Triton linear; with ninja on PATH it compiles in ~12 s
    and serves ~37 tok/s). Prepending `dirname(executable)` fixes any
    venv interpreter without hardcoding paths. None means plain inherit
    (the default interpreter needs nothing). Never double-prepends: the
    env is rebuilt from `os.environ` on every start/restart.

    Track D spawn mask: when `session_device` names a CUDA device, the
    child is masked to that physical index via `CUDA_VISIBLE_DEVICES`
    (one card visible, never straddling). An explicitly set
    `CUDA_VISIBLE_DEVICES` in the parent env wins (explicit override —
    tests, single-GPU debugging, launcher pinning); otherwise the mask
    derives from the session device. `session_device=None` keeps the
    historical behavior exactly (no mask key added).
    """
    if executable is None and session_device is None:
        return None
    if executable is None:
        base: dict[str, str] = dict(os.environ)
    else:
        bin_dir = os.path.dirname(os.path.abspath(executable))
        system_path = os.environ.get("PATH", "")
        if system_path.split(os.pathsep)[0] == bin_dir:
            base = dict(os.environ)
        else:
            base = {**os.environ, "PATH": bin_dir + os.pathsep + system_path}
    if session_device is None:
        return base
    masked = cuda_visible_devices_for_session_device(session_device)
    if masked is None:
        return base
    # Explicit override wins: a caller-pinned CUDA_VISIBLE_DEVICES (even
    # empty, which hides every GPU) is never clobbered by the session mask.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return base
    return {**base, "CUDA_VISIBLE_DEVICES": masked}


def failure(request_id: str, code: str, message: str, retryable: bool = True) -> WorkerResponse:
    return WorkerResponse(
        id=request_id,
        ok=False,
        error=WorkerErrorDetail(code=code, message=message, retryable=retryable),
    )


class SubprocessWorker:
    """Supervisor-side handle for one JSONL worker process.

    `init_op`/`init_payload` are (re)sent after every (re)start so a
    restarted worker rebuilds its session before the next real op — the
    streaming video workers' resident sessions depend on this.
    """

    def __init__(
        self,
        module: str,
        workdir: Path,
        log_path: Path,
        init_op: str | None = "init",
        init_payload: RpcPayload | None = None,
        timeout: float = DEFAULT_RPC_TIMEOUT_SECONDS,
        executable: str | None = None,
    ) -> None:
        self._module = module
        self._workdir = workdir
        self._log_path = log_path
        self._init_op = init_op
        self._init_payload = dict(init_payload) if init_payload else {}
        self._timeout = timeout
        # Alternate interpreter for the worker (the director runs in the
        # unified image's director venv via VOYAGE_DIRECTOR_PYTHON); None
        # keeps the supervisor's own interpreter.
        self._executable = executable or sys.executable
        # Raw alternate path for spawn-env purposes (None-able): only a
        # venv interpreter needs its bin dir on PATH (Stage C `_spawn_env`
        # for JIT toolchains); default-interpreter workers inherit the
        # supervisor env untouched so tool resolution never shifts.
        self._alternate_executable = executable
        self._proc: subprocess.Popen[str] | None = None
        # Supervisor-side log handle, opened per start() (issue 058): kept
        # so stop() can close it instead of leaking one fd per restart.
        self._log_file: TextIO | None = None
        self._counter = 0
        # Restart accounting (Track D refresh path): every `restart()` and
        # every `restart_with_budget()` charges exactly one unit here, so a
        # session refresh is countable without the supervisor holding a
        # second counter. Legacy `__new__` doubles (tests) miss the field —
        # all readers use `getattr(..., 0)` and never assume presence.
        self._restart_count = 0
        # Last init handshake result (None until the first successful
        # start): workers report e.g. `load_seconds` in their init reply,
        # and the supervisor reads it for `worker_started` metrics. Kept
        # as data only — rpc never interprets worker payloads.
        self.last_init_result: RpcResult | None = None
        # Serializes concurrent callers (the supervisor's parallel director
        # prefetch shares the director worker with the commit path — the
        # JSONL stream cannot interleave two requests). Single-threaded
        # behavior is unchanged.
        self._call_lock = threading.Lock()

    def start(self) -> None:
        """Launch the worker and run the init handshake.

        A failed start leaves no stale handle (issue 170): the init call
        can raise (timeout, broken pipe, fatal desync), and without a
        fence `_proc` would keep pointing at the dead or half-initialized
        child with the log fd open. The teardown below reaps the child,
        clears `_proc`, and closes the log, so a failed start rests
        exactly as before the call (`_proc` None, `running` False) and
        the next `start()` mounts a fresh handle.
        """
        rotate_log(self._log_path)
        self._close_log_file()
        log_file = self._log_path.open("a", encoding="utf-8")
        self._log_file = log_file
        try:
            raw_device = self._init_payload.get("device")
            session_device = raw_device if isinstance(raw_device, str) else None
            self._proc = subprocess.Popen(
                [self._executable, "-m", self._module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
                cwd=str(self._workdir),
                env=_spawn_env(self._alternate_executable, session_device),
            )
            if self._init_op is not None:
                self.last_init_result = self.call(self._init_op, dict(self._init_payload))
            else:
                self.last_init_result = None
        except (VoyageError, OSError):
            # Narrow on purpose: `call()` only raises the worker taxonomy
            # and `Popen` only raises `OSError` — anything else is a bug
            # that should surface with the handle untouched by cleanup.
            self.stop()
            raise

    def _close_log_file(self) -> None:
        """Close the supervisor-side log handle, if any (best-effort)."""
        handle, self._log_file = self._log_file, None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            self._close_log_file()
            return
        if proc.stdin:
            # Peer already dead (e.g. SIGKILL); still reap below.
            with contextlib.suppress(BrokenPipeError, OSError):
                proc.stdin.close()
        try:
            proc.wait(timeout=WORKER_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            # Reap after the kill (issue 058): without this wait the child
            # stays a zombie — wait() returns once SIGKILL lands, so this
            # cannot block beyond a short grace.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=WORKER_STOP_GRACE_SECONDS)
        finally:
            self._close_log_file()

    def restart(self) -> None:
        """Stop (reaping zombies) and start a fresh worker process."""
        self.stop()
        self.start()
        current = int(getattr(self, "_restart_count", 0))
        self._restart_count = current + 1

    @property
    def restart_count(self) -> int:
        """Restarts charged on this handle (Track D refresh accounting)."""
        return int(getattr(self, "_restart_count", 0))

    def cancel(self, timeout: float = WORKER_STOP_GRACE_SECONDS) -> bool:
        """Abandon a mid-render call from another thread (Track D cancel).

        Closes stdin (the worker's `for line in stdin` loop sees EOF) and
        then `stop()`s with the same grace/kill discipline — safe to call
        from the main thread while a second thread blocks in `call()`.
        The blocked `call()` then fails Recoverable (broken pipe / closed
        stdout / timeout) instead of hanging past its deadline, and the
        caller restarts through `restart_with_budget` (one charge). Never
        sends a wire op (a blocked worker cannot serve `cancel`); returns
        False when no process was running. `timeout` is accepted for
        signature symmetry and currently unused beyond `stop()`'s grace.
        """
        del timeout
        proc = self._proc
        if proc is None:
            return False
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            with contextlib.suppress(BrokenPipeError, OSError, ValueError):
                stdin.close()
        self.stop()
        return True

    def safe_stop(self) -> bool:
        """Best-effort `stop()` for start-up unwind (Track D).

        Never raises: an `OSError` from an already-reaped child reads as
        already stopped. Returns True when a process was reaped, False
        when there was nothing running. Track A calls this in
        `start_workers` unwind paths where a stop failure must not mask
        the original start error.
        """
        try:
            had_proc = self._proc is not None
            self.stop()
        except OSError:
            return False
        else:
            return had_proc

    def restart_with_budget(
        self, budget_remaining: int, *, op: str = "", worker_name: str = ""
    ) -> int:
        """Restart charging exactly one budget unit (Track D refresh path).

        Single-charge semantics (issue 008): the outer call and its resume
        hook share this one attempt — a hook failure does not charge
        again here (the supervisor's `_call_with_restart` already folds
        hook retries into the same attempt; this helper is the countable
        primitive Track A calls for refreshes and finalize retries).
        Raises `FatalWorkerError` without restarting when
        `budget_remaining <= 0`; otherwise restarts and returns
        `budget_remaining - 1`. `op`/`worker_name` ride along for metrics
        only (the caller emits `worker_restart` with them).
        """
        del op, worker_name
        if not isinstance(budget_remaining, int) or isinstance(budget_remaining, bool):
            raise FatalWorkerError(
                f"worker {self._module} refusing non-int budget ({budget_remaining!r})"
            )
        if budget_remaining <= 0:
            raise FatalWorkerError(f"worker {self._module} restart budget exhausted (0 remaining)")
        self.restart()
        return budget_remaining - 1

    def call_with_retry_once(
        self,
        op: str,
        payload: RpcPayload,
        timeout: float | None = None,
        restart_hook: Any | None = None,
    ) -> RpcResult:
        """One call with a single restart-and-retry (Track D finalize path).

        `transport_from_restarting_call`-compatible shape: same retry-once
        discipline the supervisor uses, but bounded to exactly one restart
        so ACE/SFX finalize (idempotent window renders — output overwrite,
        atomic replace, no partials) can ride it without the full run
        budget. On `RecoverableWorkerError` the worker restarts once, the
        optional `restart_hook` runs (resume/init replay, best-effort —
        a hook failure propagates instead of charging again), and the call
        retries once. Fatal errors and a second Recoverable propagate.
        """
        try:
            return self.call(op, payload, timeout=timeout)
        except RecoverableWorkerError:
            self.restart()
            if restart_hook is not None:
                restart_hook()
            return self.call(op, payload, timeout=timeout)

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None else None

    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _read_response_line(
        self, proc: subprocess.Popen[str], op: str, effective_timeout: float
    ) -> str:
        """Read one `\\n`-terminated response line, bounded by the deadline.

        Never calls blocking `readline()` after `select`: the fd goes
        non-blocking and bytes accumulate until `\\n`, EOF, the line cap,
        or the deadline — whichever comes first. Partial dribbles and
        newline-less floods both end as RecoverableWorkerError at ~the
        deadline, never as an unbounded hang.
        """
        stdout = proc.stdout
        if stdout is None:
            # Unreachable via call() (it rejects a missing stdout first),
            # but explicit: asserts vanish under `python -O` (issue 034).
            raise FatalWorkerError(f"worker {self._module} has no stdout")
        _require_finite_positive_timeout(self._module, op, effective_timeout)
        # Test doubles pass a raw fd int as stdout; production passes a
        # TextIO. Both select and os.read accept either form.
        raw_stdout: Any = stdout
        if hasattr(raw_stdout, "fileno"):
            raw_fd = int(raw_stdout.fileno())
            readable: Any = raw_stdout
        else:
            raw_fd = int(raw_stdout)
            readable = raw_fd
        # Best-effort only: select() already gated this fd readable, so
        # a single os.read still returns promptly even if the flag
        # change fails (odd fds in tests) — and the deadline bounds us
        # regardless.
        with contextlib.suppress(OSError, ValueError):
            os.set_blocking(raw_fd, False)
        deadline = time.monotonic() + effective_timeout
        buffer = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecoverableWorkerError(
                    f"worker {self._module} timed out after {effective_timeout}s on {op}"
                )
            ready, _, _ = select.select([readable], [], [], remaining)
            if not ready:
                raise RecoverableWorkerError(
                    f"worker {self._module} timed out after {effective_timeout}s on {op}"
                )
            try:
                chunk = os.read(raw_fd, RESPONSE_READ_CHUNK_BYTES)
            except BlockingIOError:
                continue  # Spurious readiness; the deadline still bounds us.
            except OSError as exc:
                raise RecoverableWorkerError(
                    f"worker {self._module} stdout unreadable: {exc}"
                ) from exc
            if not chunk:
                raise RecoverableWorkerError(f"worker {self._module} closed stdout")
            buffer.extend(chunk)
            if len(buffer) > MAX_RESPONSE_LINE_BYTES:
                raise RecoverableWorkerError(
                    f"worker {self._module} response line exceeded "
                    f"{MAX_RESPONSE_LINE_BYTES} bytes on {op}"
                )
            newline_at = buffer.find(b"\n")
            if newline_at >= 0:
                return bytes(buffer[: newline_at + 1]).decode("utf-8", errors="replace")

    def call(self, op: str, payload: RpcPayload, timeout: float | None = None) -> RpcResult:
        """Send one request, return the result dict.

        `timeout` (default: the worker's configured timeout) bounds waiting
        for the response line: expiry raises RecoverableWorkerError so the
        supervisor's restart path engages (`stop()` kills the hung worker).
        The response is accumulated byte-wise on a non-blocking fd until a
        full `\n`-terminated line arrives or the deadline passes (issue
        001): `select` only guarantees *some* bytes are readable, so a
        worker dribbling a partial line can never wedge this call past the
        deadline. Lines past MAX_RESPONSE_LINE_BYTES fail the same way.

        A timed-out call's late response does not poison the next call
        (issue 137): reads loop within this call's deadline, discarding
        well-formed lines from strictly older requests, until the line
        matching this request arrives. A future or unparseable id is not
        a late line but a protocol break and stays Fatal, as does calling
        a worker that was never started.

        `payload` is `RpcPayload` (`dict[str, JsonValue]`, issue 035):
        JSON-shaped by contract. Callers holding `dict[str, object]`
        bridge with an explicit `cast` at the `_call_with_restart` seam
        (supervisor) — `dict` invariance means no implicit conversion —
        and `list[str]`/`dict[str, float]` literals need the same
        (embed/texts, end-to-end metrics).

        A malformed request (non-str `op`, non-dict `payload`) fails fast
        here as FatalWorkerError (issue 007 supervisor side): retrying the
        same bytes cannot succeed, so it must never surface as a raw
        pydantic error outside the branch-on-class taxonomy or burn
        restart budget. The op vocabulary itself stays unchecked — unknown
        ops are the worker's UNKNOWN_OP fatal, so version-skewed callers
        still get a wire answer instead of a local refusal.
        """
        if not isinstance(op, str):
            raise FatalWorkerError(
                f"worker {self._module} refusing malformed request: "
                f"op must be str, got {type(op).__name__}"
            )
        if not isinstance(payload, dict):
            raise FatalWorkerError(
                f"worker {self._module} refusing malformed request: "
                f"payload must be a dict, got {type(payload).__name__}"
            )
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise FatalWorkerError(f"worker {self._module} is not running")
        effective_timeout = self._timeout if timeout is None else timeout
        _require_finite_positive_timeout(self._module, op, effective_timeout)
        with self._call_lock:
            self._counter += 1
            request = WorkerRequest(id=f"req-{self._counter:06d}", op=op, payload=payload)
            try:
                proc.stdin.write(encode_request(request))
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RecoverableWorkerError(f"worker {self._module} pipe broken") from exc
            deadline = time.monotonic() + effective_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RecoverableWorkerError(
                        f"worker {self._module} timed out after {effective_timeout}s on {op}"
                    )
                line = self._read_response_line(proc, op, remaining)
                try:
                    response = decode_response(line)
                except ValueError as exc:
                    # Narrow on purpose: the line is already str, so the only
                    # failure is schema validation (pydantic ValidationError
                    # subclasses ValueError). Anything else (MemoryError and
                    # friends) propagates raw instead of masquerading as a
                    # worker protocol error.
                    raise RecoverableWorkerError(
                        f"worker {self._module} sent a malformed response line: {exc}"
                    ) from exc
                if response.id == request.id:
                    break
                if _is_stale_response(response.id, request.id):
                    # Late full line from an earlier timed-out call: the
                    # worker is serial, so our fresh response still arrives
                    # after it — discard and keep waiting within the same
                    # deadline instead of converting slowness into Fatal.
                    continue
                raise FatalWorkerError(f"worker {self._module} id mismatch: {response.id}")
            if not response.ok:
                code = response.error.code if response.error else "UNKNOWN"
                message = response.error.message if response.error else "unknown error"
                retryable = response.error.retryable if response.error else False
                error: RecoverableWorkerError | FatalWorkerError = (
                    RecoverableWorkerError(f"{code}: {message}")
                    if retryable
                    else FatalWorkerError(f"{code}: {message}")
                )
                raise error
            return response.result

    def health(self) -> RpcResult:
        return self.call("health", {})


def call_cancel(worker: SubprocessWorker, timeout: float = WORKER_STOP_GRACE_SECONDS) -> bool:
    """Stable client helper for mid-render cancel (Track A wire point).

    Abandons the in-flight call (close stdin + `stop()` from the calling
    thread, join with timeout inside `stop()`) and returns whether a
    process was abandoned. Track A calls this from the main thread while
    the render thread blocks in `call()`, then charges one budget unit
    via `restart_with_budget` and restarts. Never sends a wire op.
    """
    return worker.cancel(timeout=timeout)


def restart_with_budget(
    worker: SubprocessWorker,
    budget_remaining: int,
    *,
    op: str = "",
    worker_name: str = "",
) -> int:
    """Free-function refresh primitive (Track A wire point).

    Same single-charge semantics as
    `SubprocessWorker.restart_with_budget`: one restart charges exactly
    one unit, hook failures never double-charge here. Returns the
    decremented budget for the caller to store.
    """
    return worker.restart_with_budget(budget_remaining, op=op, worker_name=worker_name)


def call_with_retry_once(
    worker: SubprocessWorker,
    op: str,
    payload: RpcPayload,
    timeout: float | None = None,
    restart_hook: Any | None = None,
) -> RpcResult:
    """Free-function finalize retry (Track B/C wire point).

    `transport_from_restarting_call`-compatible retry-once shape for the
    idempotent ACE/SFX finalize renders: one restart + one retry on
    `RecoverableWorkerError`, Fatal passthrough. Worker-side idempotency
    holds (output overwrite + atomic replace, no partials), so the retry
    never duplicates timeline state.
    """
    return worker.call_with_retry_once(op, payload, timeout=timeout, restart_hook=restart_hook)


def restart_and_resume_single_charge(
    worker: SubprocessWorker,
    budget_remaining: int,
    resume_hook: Any | None,
    *,
    op: str = "",
    worker_name: str = "",
) -> int:
    """Single-charge restart + resume (Track D double-charge fix).

    The outer restart and its resume hook share ONE budget unit: the hook
    runs inside the same charge, and a hook `RecoverableWorkerError`
    propagates without charging again (the caller treats it as the same
    attempt's failure). Returns `budget_remaining - 1`. Raises
    `FatalWorkerError` without touching the worker when the budget is
    exhausted. Track A calls this instead of charging restart and resume
    separately; alternatively pass a dedicated `resume_budget` — both
    shapes are documented, this helper is the single-charge one.
    """
    remaining = restart_with_budget(worker, budget_remaining, op=op, worker_name=worker_name)
    if resume_hook is not None:
        resume_hook()
    return remaining


def is_wedged(
    last_progress_monotonic: float | None,
    now_monotonic: float,
    *,
    threshold_seconds: float = WEDGED_DEFAULT_THRESHOLD_SECONDS,
) -> bool:
    """Supervisor watchdog predicate (Track A wire point).

    True when `now - last` exceeds `threshold_seconds`. `None` (no
    progress ever recorded) reads as wedged only when the caller passes an
    explicit start time — bare None is not wedged (a freshly started
    worker has no progress yet). Keeps the RPC non-blocking + 8MiB cap
    intact: this is pure clock math over worker-reported progress
    timestamps (see `video_common.report_block_progress`), never a read.
    """
    if last_progress_monotonic is None:
        return False
    if not isinstance(last_progress_monotonic, (int, float)):
        return False
    if not isinstance(now_monotonic, (int, float)):
        return False
    try:
        gap = float(now_monotonic) - float(last_progress_monotonic)
    except (TypeError, ValueError):
        return False
    return gap > float(threshold_seconds)


def decide_timeout_seconds(requested: float | None = None) -> float:
    """Pin the synchronous decide RPC budget to [60, 120]s (Track D).

    `None` (or a non-finite/out-of-range value) pins to the 60s floor —
    a wedged director must fail fast instead of holding the commit loop
    behind the 600s default. In-range values pass through untouched.
    """
    if requested is None:
        return DECIDE_TIMEOUT_MIN_SECONDS
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        return DECIDE_TIMEOUT_MIN_SECONDS
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return DECIDE_TIMEOUT_MIN_SECONDS
    if not math.isfinite(value):
        return DECIDE_TIMEOUT_MIN_SECONDS
    if value < DECIDE_TIMEOUT_MIN_SECONDS:
        return DECIDE_TIMEOUT_MIN_SECONDS
    if value > DECIDE_TIMEOUT_MAX_SECONDS:
        return DECIDE_TIMEOUT_MAX_SECONDS
    return value


_CUDA1_LEASE = threading.Lock()
"""Serializes cuda:1 users (enhance / prefetch / prewarm, Track D lease).

The llama sidecar (Qwen3.5 GGUF, ~5 GiB on cuda:1) shares the 2060 with
background model-pass prewarm and director prefetch probes. All three
must hold this lease while touching cuda:1 — Track A wires the
acquire/idle checks around each, so they never co-reside and OOM the
6 GiB card.
"""


def cuda1_idle_fn() -> bool:
    """True when the cuda:1 lease is free (Track A wire point).

    Covers enhance + prefetch + prewarm: Track A checks this before
    submitting any of the three, and serializes them through
    `hold_cuda1_lease`. Non-blocking (a locked lease reads busy, never
    waits).
    """
    return not _CUDA1_LEASE.locked()


def hold_cuda1_lease(blocking: bool = False) -> Any:
    """Context manager holding the cuda:1 lease (Track A wire point).

    `blocking=False` (default) raises `RuntimeError` when the lease is
    held — the caller skips or defers instead of queueing behind a GPU
    render. `blocking=True` waits (finalize-time drain, never the commit
    loop).
    """
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _holder() -> Any:
        acquired = _CUDA1_LEASE.acquire(blocking=blocking)
        if not acquired:
            raise RuntimeError("cuda:1 lease busy — defer enhance/prefetch/prewarm")
        try:
            yield
        finally:
            _CUDA1_LEASE.release()

    return _holder()


def safe_stop_worker(worker: SubprocessWorker) -> bool:
    """Best-effort stop for `start_workers` unwind (Track A wire point).

    Never raises (OSError reads as already stopped). Returns whether a
    process was reaped. Track A calls this for every started worker when
    a later start fails, so stop errors never mask the original error.
    """
    return worker.safe_stop()


def stop_all_with_budget(
    workers: list[SubprocessWorker], timeout_per_worker: float = WORKER_STOP_GRACE_SECONDS
) -> dict[str, bool]:
    """Join every worker with a per-worker budget (Track D stop path).

    Stops in order, never raises: each entry maps `worker._module` to
    whether a process was reaped. `timeout_per_worker` is accepted for
    signature symmetry (the grace lives in `stop()`); passing a finite
    positive value documents the join budget Track A enforces around the
    whole shutdown.
    """
    _require_finite_positive_timeout("stop_all", "shutdown", timeout_per_worker)
    outcomes: dict[str, bool] = {}
    for worker in workers:
        try:
            outcomes[str(getattr(worker, "_module", "worker"))] = worker.safe_stop()
        except OSError:
            outcomes[str(getattr(worker, "_module", "worker"))] = False
    return outcomes


def worker_restart_metric(
    *, worker: str, op: str, attempt: int, budget: int, reason: str
) -> dict[str, JsonValue]:
    """`worker_restart`-style metric payload (Track D hook).

    Returns the code (data only — the caller emits via its own
    `_log_metric`, so rpc never touches supervisor logging). Covers
    cancel/refresh/retry paths uniformly, including the llama sidecar
    heals (caller passes `worker="llama-sidecar"`).
    """
    return {
        "event": "worker_restart",
        "worker": worker,
        "op": op,
        "attempt": attempt,
        "budget": budget,
        "reason": reason,
    }

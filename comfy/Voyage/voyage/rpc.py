"""Typed JSONL RPC protocol (DESIGN §§45-46, task group D).

Transport: supervisor launches a worker subprocess; requests go to the
worker's stdin, responses come from stdout, diagnostics from stderr.
Stdout is reserved for RPC — never log there from a worker.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from voyage.errors import FatalWorkerError, RecoverableWorkerError
from voyage.logrotate import rotate_log
from voyage.models import WorkerErrorDetail, WorkerRequest, WorkerResponse

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
)

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


def success(request_id: str, result: dict[str, Any]) -> WorkerResponse:
    return WorkerResponse(id=request_id, ok=True, result=result)


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
    longlive video worker's resident pipeline depends on this.
    """

    def __init__(
        self,
        module: str,
        workdir: Path,
        log_path: Path,
        init_op: str | None = "init",
        init_payload: dict[str, Any] | None = None,
        timeout: float = DEFAULT_RPC_TIMEOUT_SECONDS,
    ) -> None:
        self._module = module
        self._workdir = workdir
        self._log_path = log_path
        self._init_op = init_op
        self._init_payload = dict(init_payload) if init_payload else {}
        self._timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        # Supervisor-side log handle, opened per start() (issue 058): kept
        # so stop() can close it instead of leaking one fd per restart.
        self._log_file: TextIO | None = None
        self._counter = 0
        # Serializes concurrent callers (the supervisor's parallel director
        # prefetch shares the director worker with the commit path — the
        # JSONL stream cannot interleave two requests). Single-threaded
        # behavior is unchanged.
        self._call_lock = threading.Lock()

    def start(self) -> None:
        rotate_log(self._log_path)
        self._close_log_file()
        log_file = self._log_path.open("a", encoding="utf-8")
        self._log_file = log_file
        self._proc = subprocess.Popen(
            [sys.executable, "-m", self._module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file,
            text=True,
            cwd=str(self._workdir),
        )
        if self._init_op is not None:
            self.call(self._init_op, dict(self._init_payload))

    def _close_log_file(self) -> None:
        """Close the supervisor-side log handle, if any (best-effort)."""
        handle, self._log_file = self._log_file, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            self._close_log_file()
            return
        if proc.stdin:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # Peer already dead (e.g. SIGKILL); still reap below.
        try:
            proc.wait(timeout=WORKER_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            # Reap after the kill (issue 058): without this wait the child
            # stays a zombie — wait() returns once SIGKILL lands, so this
            # cannot block beyond a short grace.
            try:
                proc.wait(timeout=WORKER_STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        finally:
            self._close_log_file()

    def restart(self) -> None:
        """Stop (reaping zombies) and start a fresh worker process."""
        self.stop()
        self.start()

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
        assert stdout is not None  # checked by the caller
        # Test doubles pass a raw fd int as stdout; production passes a
        # TextIO. Both select and os.read accept either form.
        raw_stdout: Any = stdout
        if hasattr(raw_stdout, "fileno"):
            raw_fd = int(raw_stdout.fileno())
            readable: Any = raw_stdout
        else:
            raw_fd = int(raw_stdout)
            readable = raw_fd
        try:
            os.set_blocking(raw_fd, False)
        except (OSError, ValueError):
            # Best-effort only: select() already gated this fd readable, so
            # a single os.read still returns promptly even if the flag
            # change fails (odd fds in tests) — and the deadline bounds us
            # regardless.
            pass
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

    def call(
        self, op: str, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """Send one request, return the result dict.

        `timeout` (default: the worker's configured timeout) bounds waiting
        for the response line: expiry raises RecoverableWorkerError so the
        supervisor's restart path engages (`stop()` kills the hung worker).
        The response is accumulated byte-wise on a non-blocking fd until a
        full `\\n`-terminated line arrives or the deadline passes (issue
        001): `select` only guarantees *some* bytes are readable, so a
        worker dribbling a partial line can never wedge this call past the
        deadline. Lines past MAX_RESPONSE_LINE_BYTES fail the same way.
        """
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise FatalWorkerError(f"worker {self._module} is not running")
        effective_timeout = self._timeout if timeout is None else timeout
        with self._call_lock:
            self._counter += 1
            request = WorkerRequest(id=f"req-{self._counter:06d}", op=op, payload=payload)
            try:
                proc.stdin.write(encode_request(request))
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RecoverableWorkerError(f"worker {self._module} pipe broken") from exc
            line = self._read_response_line(proc, op, effective_timeout)
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
        if response.id != request.id:
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

    def health(self) -> dict[str, Any]:
        return self.call("health", {})

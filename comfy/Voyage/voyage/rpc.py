"""Typed JSONL RPC protocol (DESIGN §§45-46, task group D).

Transport: supervisor launches a worker subprocess; requests go to the
worker's stdin, responses come from stdout, diagnostics from stderr.
Stdout is reserved for RPC — never log there from a worker.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from voyage.errors import FatalWorkerError, RecoverableWorkerError
from voyage.models import WorkerRequest, WorkerResponse

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
    from voyage.models import WorkerErrorDetail

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
    ) -> None:
        self._module = module
        self._workdir = workdir
        self._log_path = log_path
        self._init_op = init_op
        self._init_payload = dict(init_payload) if init_payload else {}
        self._proc: subprocess.Popen[str] | None = None
        self._counter = 0

    def start(self) -> None:
        log_file = self._log_path.open("a", encoding="utf-8")
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

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # Peer already dead (e.g. SIGKILL); still reap below.
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

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

    def call(self, op: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise FatalWorkerError(f"worker {self._module} is not running")
        self._counter += 1
        request = WorkerRequest(id=f"req-{self._counter:06d}", op=op, payload=payload)
        try:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(encode_request(request))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RecoverableWorkerError(f"worker {self._module} pipe broken") from exc
        line = self._proc.stdout.readline()
        if not line:
            raise RecoverableWorkerError(f"worker {self._module} closed stdout")
        response = decode_response(line)
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

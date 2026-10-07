"""Shared worker main loop: read JSONL requests, dispatch, write JSONL.

Every worker process (`audio`, `video`, `director`, `audio_acestep`, and the
GPU video workers owned by other tracks) runs this loop — exactly one copy
of the framing, stdout quarantine, and error taxonomy (DESIGN §§45-46 for
the transport, §48 for the error classes). Handlers stay pure
`dict-in/dict-out` so the JSON-serializable boundary is visible in their
signatures; all fallible per-op checks live in `checked_request` /
`validate_*` and run before any model work.

Error taxonomy over the wire (issue 007): deterministic failures carry
`retryable=False` so the supervisor maps them straight to `FatalWorkerError`
instead of burning restart budget; a handler raising
`RecoverableWorkerError` keeps `retryable=True` by explicit contract (issue
201); unexpected failures keep the default `retryable=True` so the
supervisor restarts once. Upstream model code prints
to stdout, so each handler runs under `redirect_stdout(sys.stderr)` — stdout
is reserved for RPC framing (see `voyage/rpc.py`).
"""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from collections.abc import Callable
from typing import Any

from voyage.errors import RecoverableWorkerError, VoyageError
from voyage.rpc import decode_request, encode_response, failure, success

Handler = Callable[[dict[str, Any]], dict[str, Any]]
"""One op handler: JSON-serializable payload in, JSON-serializable result out."""

ERROR_MALFORMED = "MALFORMED"
"""Torn JSONL line with a salvageable id: retrying the same bytes cannot succeed."""

ERROR_UNKNOWN_OP = "UNKNOWN_OP"
"""No handler registered for the requested op: a supervisor/worker version skew."""

ERROR_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
"""Handler is a stub in this image (e.g. GPU op on a fake worker)."""

ERROR_INVALID_PAYLOAD = "INVALID_PAYLOAD"
"""Deterministic payload/plumbing failure past `checked_request` (bad numbers, keys)."""

ERROR_WORKER = "WORKER_ERROR"
"""Unexpected failure: retryable so the supervisor's restart path engages."""

CANCEL_OP = "cancel"
"""Cooperative cancel op name (Track D liveness).

Workers are serial: a `generate_blocks` in flight cannot serve this, so
mid-render interrupt stays client-side abandon-and-restart
(`rpc.call_cancel`). An idle worker answers `handle_cancel` below so the
wire name is stable and Track A can probe cancel support without a render.
"""


def handle_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    """Idle cancel handler: nothing in flight, nothing to abandon.

    Returns `cancelled=False` (no render was abandoned) with the reason so
    callers distinguish idle-noop from a client-side abandon (which never
    returns a wire response — the process is gone). Registered by
    `video_common.standard_serve_map` when the caller passes no explicit
    cancel handler.
    """
    del payload
    return {"cancelled": False, "reason": "idle-or-noop"}


def serve(handlers: dict[str, Handler]) -> None:
    stdin = sys.stdin
    stdout = sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = decode_request(line)
        except Exception as decode_exc:
            # Malformed line (issue 007): never stall the supervisor for a
            # full timeout. When an id survives inside otherwise-bad JSON,
            # answer MALFORMED (fatal — retrying the same bytes cannot
            # succeed); a non-JSON line carries no id to answer, so log it
            # and wait for the next well-formed request.
            print(f"malformed request line: {decode_exc}", file=sys.stderr)
            salvaged_id = _salvaged_request_id(line)
            if salvaged_id is not None:
                stdout.write(
                    encode_response(
                        failure(
                            salvaged_id,
                            ERROR_MALFORMED,
                            f"malformed request: {decode_exc}",
                            retryable=False,
                        )
                    )
                )
                stdout.flush()
            continue
        handler = handlers.get(request.op)
        if handler is None:
            stdout.write(
                encode_response(
                    failure(
                        request.id, ERROR_UNKNOWN_OP, f"unknown op: {request.op}", retryable=False
                    )
                )
            )
            stdout.flush()
            continue
        try:
            # Upstream model code prints to stdout; quarantine it to stderr
            # so the JSONL framing on stdout survives. Responses are written
            # through the captured `stdout` handle below, unaffected.
            with contextlib.redirect_stdout(sys.stderr):
                result = handler(request.payload)
        except NotImplementedError as exc:
            stdout.write(
                encode_response(
                    failure(request.id, ERROR_NOT_IMPLEMENTED, str(exc), retryable=False)
                )
            )
        except RecoverableWorkerError as exc:
            # Explicitly recoverable by handler contract (issue 201): the
            # worker asked for a restart, so the wire preserves
            # retryable=True and the supervisor restarts instead of going
            # Fatal. Must precede the generic VoyageError arm below (the
            # recoverable class subclasses it).
            stdout.write(
                encode_response(failure(request.id, type(exc).__name__, str(exc), retryable=True))
            )
        except VoyageError as exc:
            # Preserve the error class over the wire (issue 007): the code
            # is the class name and retryable=False, so the supervisor maps
            # deterministic failures (bad geometry, bad config, version
            # skew) straight to Fatal instead of burning restart budget.
            stdout.write(
                encode_response(failure(request.id, type(exc).__name__, str(exc), retryable=False))
            )
        except (ValueError, KeyError, TypeError) as exc:
            # Deterministic payload/plumbing failure past checked_request
            # (bad numbers, missing keys): fail fast, never retry.
            traceback.print_exc(file=sys.stderr)
            stdout.write(
                encode_response(
                    failure(
                        request.id,
                        ERROR_INVALID_PAYLOAD,
                        f"{type(exc).__name__}: {exc}",
                        retryable=False,
                    )
                )
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            stdout.write(encode_response(failure(request.id, ERROR_WORKER, str(exc))))
        else:
            stdout.write(encode_response(success(request.id, result)))
        stdout.flush()


def _salvaged_request_id(line: str) -> str | None:
    """Best-effort request id from a line that failed schema validation."""
    try:
        raw: Any = json.loads(line)
    except ValueError:
        return None
    if isinstance(raw, dict) and isinstance(raw.get("id"), str):
        return str(raw["id"])
    return None


def checked_request(payload: dict[str, Any], **required: type[Any]) -> None:
    """Fail fast on missing or mistyped payload fields (issue 007).

    The op boundary contract for every worker handler: presence plus type
    are enforced here, before any filesystem side effect or model work, so
    a bad payload surfaces as INVALID_PAYLOAD (fatal) instead of burning a
    GPU load and failing deep inside the worker. Plain ints satisfy a float
    requirement (JSON has no int/float distinction worth dying over); bools
    never satisfy int.
    """
    for key, expected in required.items():
        if key not in payload:
            raise KeyError(f"missing payload field: {key}")
        value = payload[key]
        if isinstance(value, bool) and expected is not bool:
            raise TypeError(
                f"payload field {key!r} must be {expected.__name__}, got {type(value).__name__}"
            )
        if expected is float and isinstance(value, int):
            continue
        if not isinstance(value, expected):
            raise TypeError(
                f"payload field {key!r} must be {expected.__name__}, got {type(value).__name__}"
            )


def validate_benchmark_counts(warmup: int, measured: int) -> None:
    """Reject benchmark counts that would measure nothing (issue 060).

    Every worker's `handle_benchmark` divides by `len(walls)` and peaks,
    so `measured <= 0` (or a negative `warmup`, which makes `range(...)`
    empty the same way) is a `ZeroDivisionError`. Call this first —
    before any session/stack check — so bad counts fail fast on CPU
    without loading GPU state.
    """
    if warmup < 0 or measured <= 0:
        raise ValueError(
            "benchmark needs warmup >= 0 and measured >= 1 "
            f"(got warmup={warmup}, measured={measured})"
        )

"""Shared worker main loop: read JSONL requests, dispatch, write JSONL."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any

from voyage.rpc import decode_request, encode_response, failure, success

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def serve(handlers: dict[str, Handler]) -> None:
    stdin = sys.stdin
    stdout = sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            decode_request(line)
        except Exception:
            continue  # Malformed line: stdout framing is length-delimited JSONL,
            # so skip and wait for the next well-formed request.
        request = decode_request(line)
        handler = handlers.get(request.op)
        if handler is None:
            stdout.write(
                encode_response(
                    failure(request.id, "UNKNOWN_OP", f"unknown op: {request.op}", retryable=False)
                )
            )
            stdout.flush()
            continue
        try:
            result = handler(request.payload)
        except NotImplementedError as exc:
            stdout.write(
                encode_response(failure(request.id, "NOT_IMPLEMENTED", str(exc), retryable=False))
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            stdout.write(encode_response(failure(request.id, "WORKER_ERROR", str(exc))))
        else:
            stdout.write(encode_response(success(request.id, result)))
        stdout.flush()


def checked_request(payload: dict[str, Any], **required: type) -> None:
    for key, _type in required.items():
        if key not in payload:
            raise KeyError(f"missing payload field: {key}")

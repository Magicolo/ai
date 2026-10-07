"""Issue 286: SFX extract drains ffmpeg stderr (no pipe deadlock, bounded reap).

`_extract_frames` streamed stdout frame-by-frame while `stderr=PIPE`
sat undrained — past ~64 KiB the child blocks on stderr while the
parent blocks on stdout, with no timeout anywhere. A daemon drain
thread now keeps the pipe empty and the trailing `communicate()` carries
a timeout (kill + re-reap on expiry).

All stdlib (slim gates image); the flood shape mirrors the issue repro.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from voyage.workers.sfx_mmaudio import _drain_in_background

_FLOOD_WRITER = (
    "import sys;"
    "sys.stdout.buffer.write(b'F' * 300);"
    "sys.stdout.buffer.flush();"
    "sys.stderr.buffer.write(b'e' * 2000000)"
)
"""300 stdout bytes then 2 MB stderr: a parent demanding more stdout than
offered blocks while the child blocks on stderr (the issue shape)."""


def _flooded_extract() -> dict[str, Any]:
    """Run the deadlock shape through the production drain helper."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _FLOOD_WRITER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    chunks: list[bytes] = []
    drainer = _drain_in_background(proc.stderr, chunks)
    assert isinstance(drainer, threading.Thread) and drainer.daemon is True
    demanded = proc.stdout.read(1000000)
    proc.wait(timeout=30)
    _, tail = proc.communicate(timeout=30)
    drainer.join(timeout=30)
    return {"stdout": demanded, "stderr": b"".join(chunks) + (tail or b"")}


def test_drain_survives_flooded_stderr() -> None:
    """2 MB of stderr beside a short stdout completes without deadlock."""
    outcome = _run_bounded(_flooded_extract, timeout_seconds=90.0)
    assert outcome["stdout"] == b"F" * 300
    assert outcome["stderr"] == b"e" * 2000000


def _run_bounded(work: Any, timeout_seconds: float) -> Any:
    """Run `work` to completion, failing (not hanging) past the timeout."""
    slot: list[Any] = []
    errors: list[BaseException] = []

    def _target() -> None:
        try:
            slot.append(work())
        except BaseException as exc:  # noqa: BLE001 — re-raised below on the main thread
            errors.append(exc)

    runner = threading.Thread(target=_target, name="issue-286-bounded-probe", daemon=True)
    runner.start()
    runner.join(timeout=timeout_seconds)
    assert not runner.is_alive(), "stderr drain deadlocked past its bound"
    if errors:
        raise errors[0]
    assert slot, "drain probe returned nothing"
    return slot[0]


def test_extract_fails_loud_on_missing_input(tmp_path: Path) -> None:
    """The rewired `_extract_frames` still fails loud (drain path exercised)."""
    pytest.importorskip("torch", reason="extract needs torch (absent from slim image)")
    from voyage.workers.sfx_mmaudio import _extract_frames

    with pytest.raises(RuntimeError, match="sfx frame extract"):
        _extract_frames(str(tmp_path / "absent.mp4"), 0.0, 1.0)

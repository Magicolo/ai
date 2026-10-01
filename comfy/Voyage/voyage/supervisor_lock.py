"""Run-lock helpers for the supervisor (DESIGN §73).

Split from `voyage.supervisor` (issue 081): the single-writer run
lock's best-effort pid read as an importable pure function with no
supervisor state. `voyage.supervisor` re-exports the name below and
keeps `Supervisor._read_lock_holder` as a one-line delegating wrapper
so existing callers keep working; new code imports from here directly.

- `read_lock_holder`: pid recorded by the lock holder, or 'unknown'
  (staleness-honest per issue 004 — a recorded pid for a dead process
  is residue, EPERM still names the pid).
"""

from __future__ import annotations

import os
from pathlib import Path


def read_lock_holder(lock_path: Path) -> str:
    """Pid recorded by the lock holder, or 'unknown' (best-effort).

    Staleness-honest (issue 004): the lock dies with its holder, so a
    recorded pid for a dead process is residue from a previous run —
    the live holder simply has not written its pid yet
    (write-after-acquire). Report 'unknown' rather than naming a dead
    process; EPERM (alive but unsignalable) still names the pid.
    """
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if not text:
        return "unknown"
    try:
        pid = int(text, 10)
    except ValueError:
        return "unknown"
    if pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "unknown"
    except PermissionError:
        return text
    except OSError:
        return "unknown"
    return text

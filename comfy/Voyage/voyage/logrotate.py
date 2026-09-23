"""Daily (time-based) log rotation with retention (Phase 6 slice D).

`metrics.jsonl` and the three worker logs are append-only event streams;
without rotation a multi-day run grows them without bound (DESIGN §60).
The live path never changes (`metrics.jsonl` stays the reader contract —
scoreboard, `inspect metrics`, tests); when the existing file was last
written on a previous calendar day it is renamed to
`<stem>-YYYY-MM-DD<suffix>` before the new write lands. Rotated siblings
older than `keep_days` are pruned so disk use stays bounded.

Note: the `# noqa: UP017` marks below are deliberate — the video worker
image is py3.10, where `datetime.UTC` does not exist yet (same precedent
as `persistence.py`).
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

DEFAULT_KEEP_DAYS = 30

_ROTATED_SUFFIX = re.compile(r"^(?P<stem>.+)-(?P<day>\d{4}-\d{2}-\d{2})$")


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()  # noqa: UP017


def _rotated_name(path: Path, day: datetime.date) -> Path:
    return path.with_name(f"{path.stem}-{day.isoformat()}{path.suffix}")


def rotate_log(path: Path, keep_days: int = DEFAULT_KEEP_DAYS) -> Path | None:
    """Roll `path` to a dated sibling when it holds a previous day's writes.

    Returns the rotated sibling, or None when no rotation was needed
    (missing file, or already current). Never raises on best-effort
    housekeeping: a failed rotate/prune leaves the live file appendable.
    """
    try:
        if not path.exists():
            return None
        written = datetime.datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=datetime.timezone.utc,  # noqa: UP017
        ).date()
        today = _today()
        if written >= today:
            return None
        rotated = _rotated_name(path, written)
        if rotated.exists():
            # Clock skew / double rotate: keep both, never overwrite.
            rotated = path.with_name(
                f"{path.stem}-{written.isoformat()}-{today.isoformat()}{path.suffix}"
            )
        path.rename(rotated)
        _prune_siblings(path, keep_days)
        return rotated
    except OSError:
        return None


def append_line(path: Path, line: str, keep_days: int = DEFAULT_KEEP_DAYS) -> None:
    """Append one line, rotating to a fresh daily file when day changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(path, keep_days)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _prune_siblings(path: Path, keep_days: int) -> None:
    """Delete dated siblings of `path` older than `keep_days` (best-effort)."""
    today = _today()
    for sibling in path.parent.glob(f"{path.stem}-*{path.suffix}"):
        match = _ROTATED_SUFFIX.match(sibling.stem)
        if match is None or match.group("stem") != path.stem:
            continue
        try:
            day = datetime.date.fromisoformat(match.group("day"))
        except ValueError:
            continue
        if (today - day).days > keep_days and sibling.is_file():
            try:
                sibling.unlink()
            except OSError:
                continue

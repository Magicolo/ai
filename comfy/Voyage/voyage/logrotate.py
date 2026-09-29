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

from voyage import paths

DEFAULT_KEEP_DAYS = 30

GAUGE_CADENCE_INTERVAL_SEGMENTS = 1
"""Default gauge cadence: sample on every segment (issue 032).

Matches the supervisor's `RESOURCE_GAUGE_INTERVAL_SEGMENTS` default;
raise the interval to pay the 3-health-RPC fan-out every K segments
instead of every commit.
"""

METRICS_FILENAME = "metrics.jsonl"
"""Live metrics filename; rotated siblings are `metrics-YYYY-MM-DD.jsonl`."""

_ROTATED_SUFFIX = re.compile(r"^(?P<stem>.+)-(?P<day>\d{4}-\d{2}-\d{2})$")


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()  # noqa: UP017


def should_sample_gauges(
    segment_number: int, interval_segments: int = GAUGE_CADENCE_INTERVAL_SEGMENTS
) -> bool:
    """Cadence gate for per-commit gauge sampling (issue 032).

    Pure helper so the policy is unit-testable: sample when
    `segment_number` lands on the interval grid. Non-positive
    intervals clamp to 1 (sample every segment), mirroring the
    supervisor's `max(1, RESOURCE_GAUGE_INTERVAL_SEGMENTS)`.

    Supervisor hook (not wired here — `supervisor.py` is out of scope
    for this change): replace the inline
    `if int(segment_id) % interval != 0: return` grid check in
    `Supervisor._sample_gauges` with
    `if not should_sample_gauges(int(segment_id), interval): return`
    so the cadence lives in one tested place.
    """
    interval = max(1, interval_segments)
    return segment_number % interval == 0


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


def iter_metric_files(run_dir: Path) -> list[Path]:
    """Live `metrics.jsonl` plus rotated siblings, oldest-first (issue 049).

    Rotation keeps history complete but splits it across dated siblings;
    every reader that needs full history (scoreboard stages, status
    last-commit, soak/benchmark averages) must iterate this list instead
    of opening the live file directly. The live file sorts last so its
    events win on duplicate segment ids. Never raises: a missing/unreadable
    logs dir yields whatever subset exists (possibly empty).
    """
    logs_dir = run_dir / paths.LOGS_DIRNAME
    live = logs_dir / METRICS_FILENAME
    try:
        siblings = sorted(
            sibling
            for sibling in logs_dir.glob(f"{live.stem}-*{live.suffix}")
            if sibling.is_file()
            and (match := _ROTATED_SUFFIX.match(sibling.stem)) is not None
            and match.group("stem") == live.stem
        )
    except OSError:
        siblings = []
    files = list(siblings)
    try:
        if live.is_file():
            files.append(live)
    except OSError:
        pass
    return files


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

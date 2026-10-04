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

import contextlib
import datetime
import json
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import fsync_dir as fsync_dir

DEFAULT_KEEP_DAYS = 30

GAUGE_CADENCE_INTERVAL_SEGMENTS = 1
"""Default gauge cadence: sample on every segment (issue 032).

Matches the supervisor's `RESOURCE_GAUGE_INTERVAL_SEGMENTS` default;
raise the interval to pay the 3-health-RPC fan-out every K segments
instead of every commit.
"""

METRICS_FILENAME = "metrics.jsonl"
"""Live metrics filename; rotated siblings are `metrics-YYYY-MM-DD.jsonl`."""

WORKER_LOG_FILENAMES = ("video-worker.log", "audio-worker.log", "director-worker.log")
"""Worker stderr logs held open for the whole worker lifetime (issue 056).

`rotate_log` renames the path, which orphans an open writer on the
renamed inode — so these three logs must rotate copytruncate-style
(`rotate_open_log`) while workers run, never via `rotate_log` mid-run
(`SubprocessWorker.start` keeps the start-time rename; the handle is
fresh there).
"""

MAX_WORKER_LOG_BYTES = 10 * 1024 * 1024
"""Size trigger for open-handle worker-log rotation (issue 056).

10 MiB mirrors the Kubernetes `containerLogMaxSize` default: a healthy
infinite run that never restarts a worker still rolls each worker log
after ~10 MiB instead of growing without bound. Time-based rotation
(previous-day mtime) applies regardless of size, same as `rotate_log`.
"""

MAX_METRICS_BYTES = 10 * 1024 * 1024
"""Size trigger for `rotate_log` on metrics/live logs (issue 057).

Same 10 MiB default as the worker logs: a single-day flood (verbose
loop, hot gauges) rolls the live file even without a day boundary, so
rotation bounds disk use by size *or* time. `None` at call time reads
this constant, so tests may shrink the trigger via monkeypatch. The
copytruncate path (`rotate_open_log`, issue 056) keeps its own
`MAX_WORKER_LOG_BYTES` trigger — this one covers rename-based
`rotate_log` only.
"""

METRICS_SCHEMA_VERSION = 1
"""Version stamp for new metric lines (issue 058).

Readers accept unversioned legacy lines (schema missing → version 0
assumed); writers stamp every new line via `format_metric_line`.
"""

MAX_METRIC_LINE_BYTES = 16 * 1024
"""Cap for one serialized metric line (issue 058).

One oversized entry (e.g. a huge stage dict) must not clog the
pipeline — `format_metric_line` truncates string fields to fit and
marks the line `truncated: true`.
"""

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


def rotate_log(
    path: Path,
    keep_days: int = DEFAULT_KEEP_DAYS,
    max_bytes: int | None = None,
) -> Path | None:
    """Roll `path` to a dated sibling on day change *or* size overflow (057).

    Returns the rotated sibling, or None when no rotation was needed
    (missing file, or current day and within `max_bytes`). `max_bytes=None`
    reads the current `MAX_METRICS_BYTES` at call time so tests may shrink
    the trigger via monkeypatch (same seam as `rotate_open_log`). Never
    raises on best-effort housekeeping: a failed rotate/prune leaves the
    live file appendable.

    Durability (101): the rename is followed by `fsync_dir` on the parent
    so the directory entry survives power loss (in-tree contract in
    `voyage/atomic.py`); failures stay best-effort here — the live file
    remains appendable.
    """
    if max_bytes is None:
        max_bytes = MAX_METRICS_BYTES
    try:
        if not path.exists():
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        written = datetime.datetime.fromtimestamp(
            stat.st_mtime,
            tz=datetime.timezone.utc,  # noqa: UP017
        ).date()
        today = _today()
        if written >= today and stat.st_size <= max_bytes:
            return None
        rotated = _rotated_name(path, written)
        if rotated.exists():
            # Clock skew / double rotate: keep both, never overwrite.
            rotated = path.with_name(
                f"{path.stem}-{written.isoformat()}-{today.isoformat()}{path.suffix}"
            )
        path.rename(rotated)
        with contextlib.suppress(OSError):
            fsync_dir(path.parent)
        _prune_siblings(path, keep_days)
        return rotated
    except OSError:
        return None


def append_line(
    path: Path,
    line: str,
    keep_days: int = DEFAULT_KEEP_DAYS,
    max_bytes: int | None = None,
) -> None:
    """Append one line, rotating on day change or size overflow (057/101).

    Metrics are advisory (unlike takes), but the tail is still synced —
    file `flush` + `os.fsync` + `fsync_dir` — so a crash cannot silently
    eat buffered events the forensics surface (`status`/`scoreboard`/`soak`)
    depends on. Rotation itself stays best-effort (`rotate_log` never
    raises); a failed append/ sync raises — a failing disk should fail
    loudly, not lose history silently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(path, keep_days, max_bytes)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(path.parent)


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
    """Delete dated siblings of `path` older than `keep_days` (best-effort).

    Durability (101): when at least one sibling is unlinked, the parent
    directory is `fsync_dir`-synced so the deletions survive power loss.
    Never raises — a failed prune or sync leaves the files for the next
    rotation pass.
    """
    today = _today()
    pruned = 0
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
            pruned += 1
    if pruned:
        with contextlib.suppress(OSError):
            fsync_dir(path.parent)


def format_metric_line(run_id: str, event: dict[str, object]) -> str:
    """Serialize one metric event with the schema baseline (issue 058).

    Every line carries `ts` (epoch float, compat), `ts_iso` (ISO-8601 UTC),
    `run_id` (correlation id), and `schema` (METRICS_SCHEMA_VERSION). Base
    fields win over same-named event keys so callers cannot spoof the
    correlation/version stamps. Lines longer than `MAX_METRIC_LINE_BYTES`
    have their longest string fields halved until they fit and gain
    `truncated: true` — one oversized entry never clogs the pipeline.
    """
    now = time.time()
    iso = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).isoformat()  # noqa: UP017
    merged: dict[str, object] = {**event, "ts": now, "ts_iso": iso, "run_id": run_id}
    merged["schema"] = METRICS_SCHEMA_VERSION
    line = json.dumps(merged)
    if len(line.encode("utf-8")) <= MAX_METRIC_LINE_BYTES:
        return line
    shortened: dict[str, object] = dict(event)
    while True:
        longest_key: str | None = None
        longest_len = 0
        for key, value in shortened.items():
            if isinstance(value, str) and len(value) > longest_len:
                longest_key = key
                longest_len = len(value)
        if longest_key is None or longest_len == 0:
            break
        shortened[longest_key] = str(shortened[longest_key])[: longest_len // 2]
        candidate: dict[str, object] = {
            **shortened,
            "ts": now,
            "ts_iso": iso,
            "run_id": run_id,
            "schema": METRICS_SCHEMA_VERSION,
            "truncated": True,
        }
        line = json.dumps(candidate)
        if len(line.encode("utf-8")) <= MAX_METRIC_LINE_BYTES:
            return line
    minimal: dict[str, object] = {
        "event": str(event.get("event", "unknown")),
        "ts": now,
        "ts_iso": iso,
        "run_id": run_id,
        "schema": METRICS_SCHEMA_VERSION,
        "truncated": True,
    }
    return json.dumps(minimal)


def parse_metric_lines(
    lines: Iterable[str], run_id: str | None = None
) -> tuple[list[dict[str, object]], int]:
    """Parse metric lines with loud torn-line accounting (issue 058).

    Returns `(events, torn_count)`: `torn_count` is the number of lines
    that were non-empty but not valid JSON objects (never silently
    dropped — callers surface the count). When `run_id` is given, only
    events carrying that correlation id are returned; mismatched ids are
    skipped without counting as torn. Blank lines are ignored.
    """
    events: list[dict[str, object]] = []
    torn = 0
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            parsed: Any = json.loads(stripped)
        except ValueError:
            torn += 1
            continue
        if not isinstance(parsed, dict):
            torn += 1
            continue
        if run_id is not None and parsed.get("run_id") != run_id:
            continue
        events.append(parsed)
    return (events, torn)


def _unique_rotated(path: Path, day: datetime.date, today: datetime.date) -> Path:
    """Dated sibling that never overwrites an existing archive (issue 056)."""
    rotated = _rotated_name(path, day)
    if not rotated.exists():
        return rotated
    # Clock skew / double rotate / repeated same-day size rolls: keep
    # both, never overwrite (mirrors `rotate_log`).
    candidate = path.with_name(f"{path.stem}-{day.isoformat()}-{today.isoformat()}{path.suffix}")
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        numbered = path.with_name(
            f"{path.stem}-{day.isoformat()}-{today.isoformat()}-{counter}{path.suffix}"
        )
        if not numbered.exists():
            return numbered
        counter += 1


def rotate_open_log(
    path: Path,
    max_bytes: int | None = None,
    keep_days: int = DEFAULT_KEEP_DAYS,
) -> Path | None:
    """Copytruncate-rotate a log that a live worker holds open (issue 056).

    Rename-based `rotate_log` is unsafe mid-run: the worker's stderr
    handle keeps writing to the renamed inode, so the live path stays
    empty while the archive grows without bound. This copies the
    content to a dated sibling, then truncates the live file in place
    (same inode), so open `O_APPEND` writers continue into the live
    file at offset 0 with no reopen handshake.

    Rotates when the mtime day is stale (same policy as `rotate_log`)
    or when the size exceeds `max_bytes` (`None` reads the current
    `MAX_WORKER_LOG_BYTES` at call time so tests may shrink the
    trigger via monkeypatch). Returns the archived sibling,
    or None when no rotation was needed. Never raises: a failed
    rotate leaves the live file appendable.
    """
    if max_bytes is None:
        max_bytes = MAX_WORKER_LOG_BYTES
    try:
        if not path.is_file():
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        try:
            written = datetime.datetime.fromtimestamp(
                stat.st_mtime,
                tz=datetime.timezone.utc,  # noqa: UP017
            ).date()
        except (OSError, ValueError, OverflowError):
            return None
        today = _today()
        if written >= today and stat.st_size <= max_bytes:
            return None
        rotated = _unique_rotated(path, written, today)
        try:
            data = path.read_bytes()
        except OSError:
            return None
        try:
            rotated.write_bytes(data)
        except OSError:
            return None
        try:
            with path.open("r+b") as handle:
                handle.truncate(0)
        except OSError:
            return None
        _prune_siblings(path, keep_days)
        return rotated
    except OSError:
        return None


def rotate_worker_logs(
    logs_dir: Path,
    max_bytes: int | None = None,
    keep_days: int = DEFAULT_KEEP_DAYS,
) -> list[Path]:
    """Copytruncate-rotate the three worker logs in `logs_dir` (issue 056).

    Per-committed-segment cadence hook for the supervisor: cheap
    (`stat` × 3 when quiet), best-effort, never raises — a missing
    logs dir or an unreadable file yields whatever subset rotated.
    `max_bytes=None` reads the current `MAX_WORKER_LOG_BYTES` at call
    time (same monkeypatch seam as `rotate_open_log`).
    """
    if max_bytes is None:
        max_bytes = MAX_WORKER_LOG_BYTES
    rotated: list[Path] = []
    try:
        if not logs_dir.is_dir():
            return rotated
    except OSError:
        return rotated
    for name in WORKER_LOG_FILENAMES:
        sibling = rotate_open_log(logs_dir / name, max_bytes, keep_days)
        if sibling is not None:
            rotated.append(sibling)
    return rotated


def read_all_metric_events(run_dir: Path) -> list[dict[str, object]]:
    """All metric events across live + rotated siblings, oldest-first (049).

    Rotation splits history across dated siblings; readers needing full
    history must use this instead of opening the live file directly. Torn
    lines are skipped; a missing/unreadable logs dir yields whatever
    subset exists. (Moved from the deleted `cli_status` module: the
    `status` verb is gone but rotation-tolerant reading stays.)
    """
    events: list[dict[str, object]] = []
    for events_path in iter_metric_files(run_dir):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def last_commit_stages(run_dir: Path) -> tuple[str, dict[str, object]] | None:
    """Newest segment_committed event (id + stages) across rotated logs.

    Scans live + dated siblings newest-first via `iter_metric_files`:
    after a daily rotation the live file alone would silently drop recent
    history. Returns None when no commit is found. (Moved from the
    deleted `cli_status` module with `read_all_metric_events`.)
    """
    for events_path in reversed(iter_metric_files(run_dir)):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "segment_committed":
                stages = event.get("stages")
                segment_id = event.get("segment_id")
                if isinstance(stages, dict) and isinstance(segment_id, str):
                    return segment_id, stages
    return None

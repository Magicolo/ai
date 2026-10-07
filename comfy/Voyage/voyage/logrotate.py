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
from collections.abc import Iterable, Mapping, Sequence
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
"""Live metrics filename; rotated siblings are `metrics-YYYY-MM-DD.jsonl`.

Collision archives also exist on disk (`metrics-<written>-<today>[-N].jsonl`
from the old emission, `metrics-<day>-<N>.jsonl` from the current one):
every reader/pruner accepts all shapes via `_rotated_content_day`
(issue 216), so single-date behavior is unchanged.
"""

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
"""Single-date rotated sibling shape (original emission, still produced)."""

_ROTATED_MULTI_SUFFIX = re.compile(
    r"^(?P<base>.+?)(?P<dates>(?:-\d{4}-\d{2}-\d{2})+)(?P<counter>-\d+)?$"
)
"""Compound rotated sibling shapes (issue 216).

Covers the double-rotate emission (`<stem>-<written>-<today>[-N]`, from
`rotate_log`'s collision path and the old `_unique_rotated`) plus the
counter-only emission (`<stem>-<day>-<N>`, the current `_unique_rotated`).
The lazy `base` keeps the shortest stem so
`metrics-2026-01-01-2026-10-07-2` parses as base `metrics` with two dates,
not base `metrics-2026-01-01`.
"""


def _rotated_content_day(stem: str, base: str) -> datetime.date | None:
    """Content day of a rotated sibling stem, or None when not a sibling.

    `base` is the live stem (`metrics`, `video-worker`, ...). Returns the
    FIRST date group — the day the content was written — so retention prunes
    by content age. Every date group must be calendar-valid (a month-13
    lookalike stays invisible, like before).
    """
    match = _ROTATED_MULTI_SUFFIX.match(stem)
    if match is None or match.group("base") != base:
        return None
    days = re.findall(r"\d{4}-\d{2}-\d{2}", match.group("dates"))
    if not days:
        return None
    try:
        parsed = [datetime.date.fromisoformat(day) for day in days]
    except ValueError:
        return None
    return parsed[0]


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
            # `_unique_rotated` also adds the missing counter loop (a third
            # same-day collision used to overwrite the compound file).
            rotated = _unique_rotated(path, written, today)
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
            if sibling.is_file() and _rotated_content_day(sibling.stem, live.stem) is not None
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
        day = _rotated_content_day(sibling.stem, path.stem)
        if day is None:
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

    Schema contract (issue 224): new writes carry `schema: 1` via
    `format_metric_line`; lines without a `schema` key read as v0 legacy
    (pre-224 supervisor / SFX / finalize writers) and are accepted
    unchanged — tolerance here is the legacy fallback, never a second
    contract.
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
    """Dated sibling that never overwrites an existing archive (issue 056).

    Collision form is counter-only (`<stem>-<day>-<N>`, issue 216): the old
    compound `<written>-<today>` names stay readable (the matcher covers both
    shapes), but new archives never mint them. `today` is kept so the
    `rotate_log` collision call site stays unchanged.
    """
    del today
    rotated = _rotated_name(path, day)
    if not rotated.exists():
        return rotated
    # Clock skew / double rotate / repeated same-day size rolls: keep
    # both, never overwrite (mirrors `rotate_log`).
    counter = 2
    while True:
        numbered = path.with_name(f"{path.stem}-{day.isoformat()}-{counter}{path.suffix}")
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

    Schema-tolerant like `parse_metric_lines` (issue 224): v1 lines and
    v0 legacy lines mix freely in one stream.
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


STALE_RUNNING_SECONDS = 1800.0
"""RUNNING staleness threshold in seconds (Track E observability, DESIGN §59).

Why 30 minutes: a healthy commit advances `state.json` every few minutes
(video render + validate + commit); a RUNNING state whose mtime is older
than this never advanced because the process died (SIGKILL/OOM) without
resting at FAILED. Read-only heuristic — `is_stale_running` never mutates
state, it only flags for the operator / reconcile path.
"""

HEARTBEAT_EVENT = "video_heartbeat"
"""Metric event name for intra-stage liveness (Track A, DESIGN §59).

Long GPU stages (video render, prompt enhance, model pass) run minutes
silent; Track A emits one `video_heartbeat` per stage tick via
`emit_heartbeat` so `metrics.jsonl` shows progress even when the console
is quiet/non-TTY. Console twin is `console.should_heartbeat` + `heartbeat`.
"""

GENERATION_TIMING_KEYS: tuple[str, ...] = (
    "director",
    "prompt_enhance",
    "video",
    "audio",
    "validate",
    "commit",
    "motion_sense",
)
"""Canonical generation `stage_seconds` keys (Track E timing vocab, DESIGN §59).

`prompt_enhance` is the enhancer split out of the old monolithic `video`
wall (Track A fills it from the enhancer summary); `motion_sense` is the
post-commit cheap-tier sense (Track A fills it from `sense_motion`
seconds). Older runs carry only the five legacy keys — readers must treat
missing keys as unknown, never zero.
"""

FINALIZE_TIMING_KEYS: tuple[str, ...] = (
    "triage",
    "upscale",
    "interpolate",
    "mastering",
    "music takes",
    "sfx bed",
    "mix audio",
    "publish video",
    "drain",
    "concat",
    "seam",
    "morph",
)
"""Canonical finalize `final_stages` display keys (Track E timing vocab).

`drain`/`concat`/`seam`/`morph` are the durable-path legs (same words the
model-pass ledger uses without the `_s` suffix); `mastering` is the
SonicMaster tail. Legacy/tmpdir runs show fewer rows — same missing-key
rule as generation.
"""

MODEL_PASS_TIMING_KEYS: tuple[str, ...] = (
    "upscale_poll_s",
    "interp_poll_s",
    "drain_s",
    "concat_s",
    "seam_s",
    "morph_s",
    "mastering_poll_s",
    "upscale_chunks_done",
    "interp_chunks_done",
    "upscale_frames_done",
    "interp_frames_done",
    "chunks_drained",
    "seams_done",
    "morphs_done",
    "mastering_chunks_done",
    "mastering_frames_done",
)
"""Canonical durable model-pass ledger keys (Track E timing vocab).

Mirrors `augment_finalize._ensure_model_pass_timings` zero-init list so
the table, the metric, and the initializer can never drift apart (the
`TIMING_KEYS` test pins the sets). `_s` suffix = seconds, `_done`/`_frames`
suffixes = counts — never mix them into the display table without mapping
through `media._model_pass_stage_rows`.
"""

TIMING_KEYS: frozenset[str] = frozenset(
    (*GENERATION_TIMING_KEYS, *FINALIZE_TIMING_KEYS, *MODEL_PASS_TIMING_KEYS)
)
"""Unified timing vocabulary (Track E, DESIGN §59).

One set so generation, finalize, and model-pass miners share key names:
`mastering_*` + `drain`/`concat`/`seam`/`morph` sort here, never as
one-off literals in a new writer. Display code maps `_s` ledger keys to
the short display names via `media._model_pass_stage_rows`; metric code
emits the ledger keys verbatim.
"""


def emit_heartbeat(
    run_id: str,
    segment_id: str,
    stage: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one `video_heartbeat` event dict for Track A (DESIGN §59).

    Pure builder — no I/O, no stamping. Track A fills `stage` (e.g.
    `"video"`, `"prompt enhance"`, `"upscale"`) and any `extra` fields
    (elapsed, frames done/total), then logs via `Supervisor._log_metric`
    (which stamps `run_id`/`ts`/`ts_iso`/`schema` through
    `format_metric_line`) or directly via
    `append_line(path, format_metric_line(run_id, emit_heartbeat(...)))`.
    `run_id` rides in the dict so the direct path carries correlation
    without a second lookup; `_log_metric` overwrites it with the same
    value, so both paths agree.
    """
    event: dict[str, object] = {
        "event": HEARTBEAT_EVENT,
        "run_id": run_id,
        "segment_id": segment_id,
        "stage": stage,
    }
    if extra:
        event.update(extra)
    return event


def tail_worker_logs(logs_dir: Path, *, lines: int = 20) -> dict[str, list[str]]:
    """Last `lines` lines per worker log, best-effort (Track A pump, DESIGN §59).

    Pure read helper for the worker-log-tail pump: Track A calls this on
    the commit thread, then forwards the tails to
    `console.pump_worker_log_tail` for verbose/heartbeat display. Never
    raises — a missing/unreadable log yields an empty list for that
    worker, never a failed commit. `lines` clamps to >= 0.
    """
    wanted = max(0, lines)
    tails: dict[str, list[str]] = {}
    try:
        is_dir = logs_dir.is_dir()
    except OSError:
        is_dir = False
    if not is_dir:
        return {name: [] for name in WORKER_LOG_FILENAMES}
    for name in WORKER_LOG_FILENAMES:
        try:
            content = (logs_dir / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            tails[name] = []
            continue
        split = content.splitlines()
        tails[name] = split[-wanted:] if wanted and len(split) > wanted else split
    return tails


def is_stale_running(
    status: str,
    mtime: float,
    now: float,
    *,
    threshold_seconds: float = STALE_RUNNING_SECONDS,
) -> bool:
    """Whether a RUNNING state is stale (Track A/C reconcile, DESIGN §59).

    Pure predicate: True only when `status == "RUNNING"` and `now - mtime`
    exceeds `threshold_seconds`. Future mtimes (clock skew) and non-RUNNING
    statuses read as fresh — staleness is evidence of a dead writer, never
    a guess about a live one. Read-only: callers decide whether to heal
    (via `heal_event`) or leave it for the operator.
    """
    if status != "RUNNING":
        return False
    if not (mtime <= now):
        return False
    return (now - mtime) > threshold_seconds


def reconcile_event(
    deleted: Sequence[str],
    adopted: Sequence[str],
    counts: Mapping[str, int],
) -> dict[str, object]:
    """Build one run-reconcile event dict for Track A/C (DESIGN §59).

    Pure builder: `deleted` = removed uncommitted segment ids, `adopted` =
    checksum-adopted DONE ids, `counts` = reconciled counters (e.g.
    `{"segments": n}`). Track A/C logs it via `_log_metric` next to the
    console `format_reconcile_output` line so the metric stream and the
    screen agree. Empty inputs are valid (a no-op reconcile still records
    that the scan ran).
    """
    return {
        "event": "run_reconciled",
        "deleted": list(deleted),
        "adopted": list(adopted),
        "counts": dict(counts),
    }


def heal_event(segment_id: str, action: str, detail: str = "") -> dict[str, object]:
    """Build one run-heal event dict for Track A/C (DESIGN §59).

    Pure builder: `action` names the heal (`"adopted"`, `"reclaimed"`,
    `"stale_running_healed"`), `detail` carries the human reason. Logged
    via `_log_metric` at the heal site — never synthesized by readers.
    """
    event: dict[str, object] = {
        "event": "run_healed",
        "segment_id": segment_id,
        "action": action,
    }
    if detail:
        event["detail"] = detail
    return event


def format_finalize_metric(run_id: str, payload: Mapping[str, object]) -> str:
    """Serialize one `finalize_completed` line with the schema baseline (Track C).

    Wrapper over `format_metric_line` so the finalize path gets the same
    `run_id`/`ts_iso`/`schema`/16 KiB-cap contract as every commit-path
    event (issue 224). `payload` carries the finalize fields (segments,
    geometry, timings); `event` is forced to `"finalize_completed"` so a
    caller-supplied key can never spoof it. Track C routes the
    `media.finalize_completed` write through here (media.py itself is
    untouched by Track E — this is the seam it calls).
    """
    merged: dict[str, object] = dict(payload)
    merged["event"] = "finalize_completed"
    return format_metric_line(run_id, merged)


def build_stage_seconds(
    *,
    director: float = 0.0,
    prompt_enhance: float = 0.0,
    video: float = 0.0,
    audio: float = 0.0,
    validate: float = 0.0,
    commit: float = 0.0,
    motion_sense: float = 0.0,
    enhance_ms: float | None = None,
    sense_ms: float | None = None,
) -> dict[str, float]:
    """Assemble one generation `stage_seconds` dict (Track A fills, DESIGN §59).

    Seconds by contract (same units the console `timing_table` prints).
    `enhance_ms`/`sense_ms` are millisecond overrides for the
    `prompt_enhance`/`motion_sense` split: when given, they win over the
    same-named seconds value (divided by 1000) so Track A can forward
    worker-measured milliseconds without converting at the call site.
    All values round to milliseconds; missing stages read as 0.0 (unknown
    stages are omitted by the caller, never zero-filled here beyond the
    explicit args).
    """
    if enhance_ms is not None:
        prompt_enhance = float(enhance_ms) / 1000.0
    if sense_ms is not None:
        motion_sense = float(sense_ms) / 1000.0
    return {
        "director": round(float(director), 3),
        "prompt_enhance": round(float(prompt_enhance), 3),
        "video": round(float(video), 3),
        "audio": round(float(audio), 3),
        "validate": round(float(validate), 3),
        "commit": round(float(commit), 3),
        "motion_sense": round(float(motion_sense), 3),
    }


def prewarm_heartbeat_event(
    segment_id: str | None,
    *,
    upscale_frames: int,
    interp_frames: int,
    skip_reason: str,
    segments_seen: int,
    upscale_seconds: float = 0.0,
    interp_seconds: float = 0.0,
) -> dict[str, object]:
    """Build one always-on pre-warm heartbeat event (Track A, DESIGN §59).

    Unlike the legacy `prewarm_progress` (fired only on new ledgered
    frames), this fires every commit — even when both frame counts are
    zero — so `metrics.jsonl` proves the background pass ran (or why it
    held back via `skip_reason`) instead of leaving idle passes invisible.
    `segments_seen` is the sweep's segment count (0 when the driver never
    swept). Logged via `_log_metric` at the post-commit report site.
    """
    return {
        "event": "prewarm_heartbeat",
        "segment_id": segment_id,
        "upscale_frames": int(upscale_frames),
        "interp_frames": int(interp_frames),
        "skip_reason": str(skip_reason),
        "segments_seen": int(segments_seen),
        "upscale_seconds": round(float(upscale_seconds), 3),
        "interp_seconds": round(float(interp_seconds), 3),
    }


def gauges_skipped_event(segment_id: str, reason: str) -> dict[str, object]:
    """Build one `gauges_skipped{reason}` event (Track A, DESIGN §59).

    Pure builder for the skip-busy path: when `_sample_gauges` skips the
    director probe (prefetch in flight) or the whole cadence grid misses
    via `should_sample_gauges`, Track A logs this with `reason` (e.g.
    `"prefetch_in_flight"`, `"cadence_miss"`) instead of staying silent,
    so gauge gaps are attributable. `reason` is a stable token, never a
    free sentence.
    """
    return {
        "event": "gauges_skipped",
        "segment_id": segment_id,
        "reason": str(reason),
    }


def health_alert_event(
    alerts: Sequence[str],
    *,
    segment_id: str | None = None,
    detail: str = "",
) -> dict[str, object]:
    """Build one `health_alert` metric event (Track A/C, DESIGN §64).

    Pure builder over `doctor.health_alerts` output: one alert string per
    list member (already `WARN:`/`CRIT:`-prefixed by the doctor). Logged
    via `_log_metric` wherever Track A/C evaluates health (gauges tail,
    preflight) so the metric stream and the console `warn` agree. Empty
    `alerts` is valid (records a clean bill, never skipped silently).
    """
    event: dict[str, object] = {
        "event": "health_alert",
        "alerts": list(alerts),
    }
    if segment_id is not None:
        event["segment_id"] = segment_id
    if detail:
        event["detail"] = detail
    return event


def worker_log_rotation_event(
    rotated: Sequence[Path | str],
    *,
    segment_id: str | None = None,
) -> dict[str, object]:
    """Build one worker-log rotation metric event (Track E, DESIGN §60).

    Pure builder: `rotated` holds the archived siblings
    `rotate_worker_logs` returned (empty = nothing rolled). Logged via
    `_log_metric` at the per-commit rotation tick so log rolls are
    visible in the metric stream, not just on disk. Names serialize as
    strings; the live paths never appear here.
    """
    event: dict[str, object] = {
        "event": "worker_log_rotated",
        "rotated": [str(sibling) for sibling in rotated],
        "count": len(list(rotated)),
    }
    if segment_id is not None:
        event["segment_id"] = segment_id
    return event


def prompt_enhanced_segment_id(event: Mapping[str, object]) -> str | None:
    """Segment id of a `prompt_enhanced` event, old or new key (Track E).

    New writers emit `segment_id` (canonical); the supervisor still emits
    `segment` (pre-rename). Readers must accept both — this helper is the
    single place that does, so Track A can rename the writer without
    breaking history scans. Returns None when neither key carries a
    string.
    """
    candidate = event.get("segment_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    legacy = event.get("segment")
    if isinstance(legacy, str) and legacy:
        return legacy
    return None


def commit_av_drift() -> None:
    """Null A/V drift for video-only commits (Track E, DESIGN §56).

    Commits are video-only (all backends deferred — no audio exists at
    commit time), so there is no alignment to measure: readers must treat
    the commit-time `av_drift_seconds` as null (unknown), never as a real
    zero-second alignment. The legacy writer still stamps `0.0` for shape
    stability — `normalize_av_drift` maps that sentinel to None. The real
    alignment lives at finalize time (`media.check_av_alignment`); this
    helper exists so Track C routes the finalize value instead of
    re-measuring at commit.
    """
    return None


def normalize_av_drift(value: object) -> float | None:
    """Map a commit-time drift cell to real-or-null (Track E).

    `None` stays None (new null writers); the legacy `0.0` sentinel reads
    as None (video-only, nothing aligned); any other finite number passes
    through (finalize-time real alignment). Non-numeric cells read as
    None instead of raising — a hand-edited metric must degrade, never
    fail validation.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number == 0.0:
            return None
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    return None


def count_torn_metric_lines(run_dir: Path) -> int:
    """Torn metric lines across live + rotated siblings (Track E/C, DESIGN §60).

    Loud accounting twin of `read_all_metric_events` (which skips torn
    silently): sums `parse_metric_lines` torn counts per file so the
    scoreboard `--json` and the `validate_metrics` hook can surface log
    health. Never raises — a missing/unreadable logs dir reads as zero.
    """
    total = 0
    for events_path in iter_metric_files(run_dir):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        _events, torn = parse_metric_lines(lines)
        total += torn
    return total


def validate_metrics(run_dir: Path, *, run_id: str | None = None) -> tuple[list[str], list[str]]:
    """Read-only metrics.jsonl contract check for Track C (DESIGN §§59-60).

    Returns `(errors, warnings)`: errors fail the run (missing live file,
    `run_id` mismatch, no `segment_committed` per DONE segment, frames
    disagreement between the metric and the segment manifest); warnings
    flag legacy v0 lines (missing `schema` — pre-224 writers) without
    failing. Presence = the live `metrics.jsonl` must exist when any DONE
    segment does. Schema = every parsed line for this run should carry
    `schema == METRICS_SCHEMA_VERSION`. Run correlation = every line's
    `run_id` must equal the expected id (explicit arg or the run manifest
    name). One-commit-per-DONE = each DONE id needs exactly one
    `segment_committed` (adopted replays that re-emit are tolerated as
    >= 1, never zero). Frames agreement = the committed `frames` must
    equal the manifest `metrics.frames`/`video.frames` when both are
    known. Never raises — unreadable files degrade to errors, never
    exceptions.
    """
    from voyage import paths as voyage_paths

    errors: list[str] = []
    warnings: list[str] = []
    logs_dir = run_dir / voyage_paths.LOGS_DIRNAME
    live = logs_dir / METRICS_FILENAME
    segments_root = run_dir / voyage_paths.SEGMENTS_DIRNAME
    try:
        done_ids = sorted(
            entry.name
            for entry in segments_root.iterdir()
            if entry.is_dir() and (entry / voyage_paths.DONE_MARKER).is_file()
        )
    except OSError as exc:
        return ([f"segments dir unreadable: {exc}"], [])
    if done_ids and not live.is_file():
        errors.append(f"missing {METRICS_FILENAME} for {len(done_ids)} DONE segments")
        return (errors, warnings)
    if not live.is_file():
        return (errors, warnings)
    try:
        lines = live.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return ([f"{METRICS_FILENAME} unreadable: {exc}"], warnings)
    events, _torn = parse_metric_lines(lines)
    expected_run: str | None = run_id
    if expected_run is None:
        try:
            from voyage.persistence import read_manifest

            manifest = read_manifest(run_dir)
            name = manifest.get("name")
            expected_run = str(name) if isinstance(name, str) and name else None
        except Exception:  # noqa: BLE001 - manifest absence is a state error elsewhere
            expected_run = None
    for event in events:
        if not isinstance(event, dict):
            continue
        if expected_run is not None and event.get("run_id") != expected_run:
            errors.append(
                f"run_id mismatch: expected {expected_run!r}, "
                f"found {event.get('run_id')!r} on {event.get('event')!r}"
            )
            break
        if "schema" not in event:
            warnings.append(f"legacy v0 line for event {event.get('event')!r} (no schema)")
            break
        schema = event.get("schema")
        if schema != METRICS_SCHEMA_VERSION:
            errors.append(f"schema mismatch: expected {METRICS_SCHEMA_VERSION}, got {schema!r}")
            break
    by_segment: dict[str, list[dict[str, object]]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "segment_committed":
            continue
        segment_id = event.get("segment_id")
        if isinstance(segment_id, str):
            by_segment.setdefault(segment_id, []).append(event)
    for segment_id in done_ids:
        commits = by_segment.get(segment_id, [])
        if not commits:
            errors.append(f"{segment_id} DONE but no segment_committed event")
            continue
        try:
            from voyage.segment_manifest import load_segment_manifest

            manifest_section = load_segment_manifest(voyage_paths.segment_dir(run_dir, segment_id))
            metrics = manifest_section.get("metrics")
            manifest_frames: int | None = None
            if isinstance(metrics, dict):
                raw = metrics.get("frames")
                if isinstance(raw, int) and not isinstance(raw, bool):
                    manifest_frames = raw
                else:
                    video = metrics.get("video")
                    if isinstance(video, dict):
                        raw_video = video.get("frames")
                        if isinstance(raw_video, int) and not isinstance(raw_video, bool):
                            manifest_frames = raw_video
        except Exception:  # noqa: BLE001 - torn manifests are validate_run territory
            manifest_frames = None
        if manifest_frames is None:
            continue
        for commit in commits:
            raw_frames = commit.get("frames")
            if isinstance(raw_frames, bool) or not isinstance(raw_frames, int):
                errors.append(f"{segment_id} segment_committed frames not an int")
                break
            if raw_frames != manifest_frames:
                errors.append(
                    f"{segment_id} frames mismatch: metric {raw_frames} "
                    f"!= manifest {manifest_frames}"
                )
                break
    return (errors, warnings)

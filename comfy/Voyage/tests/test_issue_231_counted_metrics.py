"""Issue 231: counted metric reader with loud torn accounting.

`read_all_metric_events_counted` returns (events, torn) matching the
silent variant's events exactly; scoreboard rows surface torn lines in
their errors cells and the status summary routes through the counted
reader.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.logrotate import (
    read_all_metric_events,
    read_all_metric_events_counted,
)
from voyage.scoreboard import run_status_summary, scoreboard_rows


def _init_run(run_dir: Path) -> None:
    initialize_run_directory(run_dir, run_id="issue-231")


def _write_committed_segment(run_dir: Path, segment_id: str) -> None:
    from voyage.segment_manifest import write_segment_manifest

    segment = run_dir / paths.SEGMENTS_DIRNAME / segment_id
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_text("", encoding="utf-8")
    write_segment_manifest(
        segment,
        {
            "metrics": {"video": {"frames": 48}},
            "transition": {
                "destination": {"canonical_name": "probe harbor"},
                "phase": "HOLD",
            },
            "prompt_plan": {},
            "audio_state": {"take_ids": []},
            "world_state": {},
            "checksums": {},
        },
    )


def test_counted_matches_silent_events_and_counts_torn(tmp_path: Path) -> None:
    """Counted events equal silent events; torn counts non-object lines (231)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    logs_dir = run_dir / paths.LOGS_DIRNAME
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event": "a"}\n')
        handle.write("not-json\n")
        handle.write("[1, 2]\n")
        handle.write('{"event": "b"}\n')
    silent = read_all_metric_events(run_dir)
    events, torn = read_all_metric_events_counted(run_dir)
    assert [event["event"] for event in events if event["event"] in ("a", "b")] == ["a", "b"]
    assert [event for event in silent if event.get("event") in ("a", "b")] == [
        event for event in events if event.get("event") in ("a", "b")
    ]
    assert torn == 2


def test_counted_empty_run_is_zero(tmp_path: Path) -> None:
    """Missing logs read as ([], 0), never raise (231)."""
    events, torn = read_all_metric_events_counted(tmp_path / "absent-run")
    assert events == []
    assert torn == 0


def test_scoreboard_rows_surface_torn_in_errors(tmp_path: Path) -> None:
    """Torn lines land in every row's errors cell (231)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _write_committed_segment(run_dir, "000000")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event": "segment_committed"\n')
    rows = scoreboard_rows(run_dir)
    assert rows
    assert any("torn metric lines: 1" in str(error) for row in rows for error in row["errors"])  # type: ignore[union-attr]


def test_status_summary_torn_matches_counted(tmp_path: Path) -> None:
    """Status torn equals the counted reader (231)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _write_committed_segment(run_dir, "000000")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("torn-tail\n")
    _, torn = read_all_metric_events_counted(run_dir)
    summary = run_status_summary(run_dir)
    assert summary["torn"] == torn == 1


def test_silent_variant_stays_silent_but_documented(tmp_path: Path) -> None:
    """Silent reader still skips torn without a count (documented convenience)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    logs_dir = run_dir / paths.LOGS_DIRNAME
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("torn\n")
        handle.write(json.dumps({"event": "kept"}) + "\n")
    events = read_all_metric_events(run_dir)
    assert any(event.get("event") == "kept" for event in events)
    assert "counted" in (read_all_metric_events.__doc__ or "")

"""Rank-2 ledger/rotation/metrics hardening (issues 054/057/058/101-remainders).

CPU-only: fake/silent workers, tmp_path runs, no GPU/network/host paths.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# 054 — SFX ledger two-worker race + fsync_dir twin (101 leg)
# ---------------------------------------------------------------------------


def test_054_append_sfx_window_syncs_directory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ledger append persists the namespace entry (101 twin of append_take)."""
    from voyage import sfx_finalize as finalize
    from voyage.sfx_finalize import SfxWindow, append_sfx_window

    calls: list[Path] = []
    monkeypatch.setattr(
        finalize, "fsync_dir", lambda directory: calls.append(Path(directory)), raising=False
    )
    ledger = tmp_path / "audio" / "sfx" / "sfx.jsonl"
    append_sfx_window(
        ledger, SfxWindow("w0000", 0.0, 8.0, "rain", 7), "audio/sfx/w0000.wav", "small_44k"
    )
    assert calls == [ledger.parent]


def test_054_validate_is_order_insensitive(tmp_path: Path) -> None:
    """Out-of-order + duplicate ledger lines still validate clean (already landed)."""
    from voyage.sfx_finalize import validate_sfx_ledger

    sfx_dir = tmp_path / "audio" / "sfx"
    sfx_dir.mkdir(parents=True)
    (sfx_dir / "w0000.wav").write_bytes(b"RIFF" + b"\0" * 32)
    (sfx_dir / "w0001.wav").write_bytes(b"RIFF" + b"\0" * 32)
    lines = [
        {"window_id": "w0001", "start": 7.0, "duration": 5.0, "path": "audio/sfx/w0001.wav"},
        {"window_id": "w0000", "start": 0.0, "duration": 8.0, "path": "audio/sfx/w0000.wav"},
        {"window_id": "w0000", "start": 0.0, "duration": 8.0, "path": "audio/sfx/w0000.wav"},
    ]
    (sfx_dir / "sfx.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    assert validate_sfx_ledger(tmp_path, 12.0) == []


def test_054_two_workers_append_in_window_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slow-first worker appends on completion: both lines land whole, plan order not required.

    Durability-first contract (SFX resume fix): each window's ledger line is
    appended as that window completes, so a kill mid-batch keeps completed
    lines instead of orphaning them. Completion order under threads is
    therefore plan-order-independent — consumers (existing-by-window_id,
    last-wins dedupe, order-insensitive validate) never depend on line order.
    This test pins the race-safety half: both lines land whole and parse.
    """
    from voyage import sfx_finalize as finalize
    from voyage.sfx_finalize import load_sfx_ledger, render_sfx_bed

    monkeypatch.setattr(finalize, "augment_devices", lambda **kwargs: ("cuda:0", "cuda:1"))

    class _SleepyWorker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def start(self) -> None:
            return None

        def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
            import wave

            del op
            # w0000 sleeps so w0001 finishes first when run under threads.
            if str(payload["window_id"]) == "w0000":
                time.sleep(0.3)
            out = Path(str(payload["output_path"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            duration = float(payload["duration_seconds"])
            frames = max(1, int(48000 * duration))
            with wave.open(str(out), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(b"\0" * frames * 4)
            return {"duration_seconds": duration}

        def stop(self) -> None:
            return None

    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _SleepyWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    bed = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        [(0.0, 12.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        0,
        48000,
        2,
        2,
    )
    assert bed.exists()
    records = load_sfx_ledger(run_dir / "audio" / "sfx" / "sfx.jsonl")
    assert sorted(record["window_id"] for record in records) == ["w0000", "w0001"]


def test_054_concurrent_appends_all_parse(tmp_path: Path) -> None:
    """Two threads hammering the ledger leave only whole JSON lines."""
    import threading

    from voyage.sfx_finalize import SfxWindow, append_sfx_window, load_sfx_ledger

    ledger = tmp_path / "audio" / "sfx" / "sfx.jsonl"

    def _hammer(base: int) -> None:
        for index in range(20):
            append_sfx_window(
                ledger,
                SfxWindow(f"w{base + index:04d}", float(index), 8.0, "rain", index),
                f"audio/sfx/w{base + index:04d}.wav",
                "small_44k",
            )

    threads = [
        threading.Thread(target=_hammer, args=(0,)),
        threading.Thread(target=_hammer, args=(100,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_sfx_ledger(ledger)) == 40


# ---------------------------------------------------------------------------
# 057 + 101-metrics — rotation size trigger, fsync, dir-sync
# ---------------------------------------------------------------------------


def test_057_append_line_flushes_and_syncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Metrics append is durable: file fsync + directory sync (101 leg)."""
    from voyage import logrotate as rotate

    fsynced: list[int] = []
    dir_calls: list[Path] = []
    real_fsync = os.fsync

    def _recording_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _recording_fsync)
    monkeypatch.setattr(
        rotate, "fsync_dir", lambda directory: dir_calls.append(Path(directory)), raising=False
    )
    log = tmp_path / "metrics.jsonl"
    rotate.append_line(log, '{"event": "one"}')
    assert log.read_text(encoding="utf-8") == '{"event": "one"}\n'
    assert fsynced, "append_line must fsync the file"
    assert dir_calls == [log.parent]


def test_057_size_trigger_rotates_same_day(tmp_path: Path) -> None:
    """A same-day log past max_bytes rolls even without a day boundary."""
    from voyage import logrotate as rotate

    log = tmp_path / "metrics.jsonl"
    log.write_text("x" * 100, encoding="utf-8")
    rotated = rotate.rotate_log(log, max_bytes=10)
    assert rotated is not None and rotated.exists()
    assert log.exists() is False or log.stat().st_size == 0 or not log.exists()


def test_057_rotate_syncs_directory_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rename-based rotation persists the directory entry."""
    from voyage import logrotate as rotate

    dir_calls: list[Path] = []
    monkeypatch.setattr(
        rotate, "fsync_dir", lambda directory: dir_calls.append(Path(directory)), raising=False
    )
    log = tmp_path / "metrics.jsonl"
    log.write_text('{"event": "old"}\n', encoding="utf-8")
    aged = time.time() - 2 * 86400
    os.utime(log, (aged, aged))
    rotated = rotate.rotate_log(log)
    assert rotated is not None
    assert rotate.fsync_dir is not None  # seam exists
    assert dir_calls, "rotate_log must fsync_dir after rename"


def test_057_prune_syncs_directory_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pruning old siblings persists the directory entry."""
    import datetime

    from voyage import logrotate as rotate

    dir_calls: list[Path] = []
    monkeypatch.setattr(
        rotate, "fsync_dir", lambda directory: dir_calls.append(Path(directory)), raising=False
    )
    today = datetime.datetime.now(datetime.timezone.utc).date()
    old = tmp_path / f"metrics-{(today - datetime.timedelta(days=40)).isoformat()}.jsonl"
    old.write_text("", encoding="utf-8")
    rotate._prune_siblings(tmp_path / "metrics.jsonl", keep_days=30)
    assert not old.exists()
    assert dir_calls, "_prune_siblings must fsync_dir after unlink"


# ---------------------------------------------------------------------------
# 101-concepts — jsonl fsync_dir + missing-index-key detector
# ---------------------------------------------------------------------------


def test_101_concepts_append_syncs_directory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concept jsonl append persists the namespace entry like the .npy side."""
    from voyage import concepts as concepts_module
    from voyage.concepts import ConceptStore

    dir_calls: list[Path] = []
    real = concepts_module.fsync_dir

    def _recording_fsync_dir(directory: Path) -> None:
        dir_calls.append(Path(directory))
        real(directory)

    monkeypatch.setattr(
        concepts_module,
        "fsync_dir",
        _recording_fsync_dir,
    )
    store = ConceptStore(tmp_path / "novelty")
    store.append("a calm reef", accepted=True, vector=[1.0, 0.0])
    assert (tmp_path / "novelty" / "concepts.jsonl").exists()
    assert len(dir_calls) >= 2, "vector + jsonl sides must each sync the directory"


def test_101_validate_concepts_reports_missing_index_key(tmp_path: Path) -> None:
    """Crash between jsonl append and index write fails loud (no silent drift)."""
    import json

    from voyage.concepts import ConceptStore, validate_concepts

    store = ConceptStore(tmp_path / "novelty")
    store.append("a calm reef", accepted=True, vector=[1.0, 0.0])
    assert validate_concepts(tmp_path / "novelty") == []
    index_path = tmp_path / "novelty" / "concept_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["concept-000000"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    errors = validate_concepts(tmp_path / "novelty")
    assert any("concept-000000" in error for error in errors)


# ---------------------------------------------------------------------------
# 058 — metrics schema baseline (version + run_id filter + torn accounting)
# ---------------------------------------------------------------------------


def test_058_metric_line_carries_schema_version_and_iso_ts() -> None:
    """New lines are self-describing: schema + ISO ts alongside epoch compat."""
    from voyage.logrotate import METRICS_SCHEMA_VERSION, format_metric_line

    assert isinstance(METRICS_SCHEMA_VERSION, int)
    event = json.loads(format_metric_line("run-1", {"event": "segment_committed"}))
    assert event["run_id"] == "run-1"
    assert event["schema"] == METRICS_SCHEMA_VERSION
    assert isinstance(event["ts"], float)
    assert isinstance(event["ts_iso"], str) and "T" in event["ts_iso"]


def test_058_metric_line_caps_size_with_truncated_flag() -> None:
    """One oversized entry cannot clog the pipeline: truncate + flag, stay valid."""
    from voyage.logrotate import MAX_METRIC_LINE_BYTES, format_metric_line

    line = format_metric_line("run-1", {"event": "segment_committed", "blob": "x" * 100000})
    assert len(line.encode("utf-8")) <= MAX_METRIC_LINE_BYTES
    assert json.loads(line)["truncated"] is True


def test_058_parse_filters_run_id_and_counts_torn() -> None:
    """Readers filter by correlation id and account torn lines loudly."""
    from voyage.logrotate import format_metric_line, parse_metric_lines

    lines = [
        format_metric_line("run-1", {"event": "a"}),
        format_metric_line("run-2", {"event": "b"}),
        "not-json {{{",
        "",
    ]
    events, torn = parse_metric_lines(lines, run_id="run-1")
    assert [event["event"] for event in events] == ["a"]
    assert torn == 1
    all_events, all_torn = parse_metric_lines(lines)
    assert len(all_events) == 2 and all_torn == 1

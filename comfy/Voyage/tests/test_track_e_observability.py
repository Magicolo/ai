"""Track E observability helpers: contracts for Tracks A/C (DESIGN §§59-60, 64).

CPU-only, fake data, no GPU/network. Covers the ten Track E items with
pure-builder + file-shape tests so Tracks A/C can call the helpers
without reading implementation bodies. See AGENTS.md §11/§12 for scope.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.console import (
    HEARTBEAT_AFTER_SECONDS,
    VoyageConsole,
    note_prewarm_heartbeat,
    note_worker_rotation,
    prompt_enhance_bar,
    prompt_enhance_stage,
    pump_worker_log_tail,
    should_heartbeat,
    warn_health_alerts,
)
from voyage.doctor import (
    gauges_to_facts,
    health_alert_event,
    health_alerts,
    report_health_alerts,
)
from voyage.logrotate import (
    FINALIZE_TIMING_KEYS,
    GENERATION_TIMING_KEYS,
    MODEL_PASS_TIMING_KEYS,
    TIMING_KEYS,
    build_stage_seconds,
    commit_av_drift,
    count_torn_metric_lines,
    emit_heartbeat,
    format_finalize_metric,
    format_metric_line,
    gauges_skipped_event,
    heal_event,
    is_stale_running,
    normalize_av_drift,
    parse_metric_lines,
    prewarm_heartbeat_event,
    prompt_enhanced_segment_id,
    reconcile_event,
    tail_worker_logs,
    validate_metrics,
    worker_log_rotation_event,
)
from voyage.logrotate import (
    health_alert_event as log_health_alert_event,
)
from voyage.motion_sense import (
    ENHANCE_STAGE_KEY,
    SENSE_STAGE_KEY,
    MotionReading,
    motion_stage_fragment,
)
from voyage.scoreboard import (
    adopted_segment_ids,
    format_scoreboard_table,
    run_status_summary,
    scoreboard_rows,
    torn_metric_lines,
    validate_scoreboard,
)
from voyage.scoreboard import (
    main as scoreboard_main,
)


def _write_metrics_line(logs_dir: Path, event: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def test_emit_heartbeat_shape() -> None:
    event = emit_heartbeat("run-1", "000007", "video", {"elapsed": 12.5})
    assert event["event"] == "video_heartbeat"
    assert event["run_id"] == "run-1"
    assert event["segment_id"] == "000007"
    assert event["stage"] == "video"
    assert event["elapsed"] == 12.5
    line = format_metric_line("run-1", event)
    parsed = json.loads(line)
    assert parsed["event"] == "video_heartbeat"
    assert parsed["run_id"] == "run-1"
    assert isinstance(parsed["ts_iso"], str)


def test_tail_worker_logs_reads_last_lines(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "video-worker.log").write_text("a\nb\nc\n", encoding="utf-8")
    tails = tail_worker_logs(logs_dir, lines=2)
    assert tails["video-worker.log"] == ["b", "c"]
    assert tails["audio-worker.log"] == []
    assert tails["director-worker.log"] == []


def test_tail_worker_logs_missing_dir_is_empty(tmp_path: Path) -> None:
    tails = tail_worker_logs(tmp_path / "no-logs")
    assert all(lines == [] for lines in tails.values())


def test_is_stale_running_flags_old_running() -> None:
    now = 1_000_000.0
    assert is_stale_running("RUNNING", now - 3600.0, now) is True
    assert is_stale_running("RUNNING", now - 10.0, now) is False
    assert is_stale_running("COMPLETE", now - 99999.0, now) is False
    assert is_stale_running("RUNNING", now + 60.0, now) is False


def test_reconcile_and_heal_events() -> None:
    reconcile = reconcile_event(["000003"], ["000002"], {"segments": 2})
    assert reconcile["event"] == "run_reconciled"
    assert reconcile["deleted"] == ["000003"]
    assert reconcile["adopted"] == ["000002"]
    healed = heal_event("000002", "adopted", "checksum ok")
    assert healed["event"] == "run_healed"
    assert healed["segment_id"] == "000002"
    assert healed["action"] == "adopted"


def test_format_finalize_metric_forces_event_and_schema() -> None:
    line = format_finalize_metric("run-9", {"segments": 2, "event": "spoof"})
    parsed = json.loads(line)
    assert parsed["event"] == "finalize_completed"
    assert parsed["run_id"] == "run-9"
    assert parsed["schema"] == 1
    assert isinstance(parsed["ts_iso"], str)


def test_timing_keys_unify_vocab() -> None:
    assert "prompt_enhance" in GENERATION_TIMING_KEYS
    assert "motion_sense" in GENERATION_TIMING_KEYS
    assert "mastering" in FINALIZE_TIMING_KEYS
    for key in ("drain", "concat", "seam", "morph"):
        assert key in FINALIZE_TIMING_KEYS
    for key in (
        "mastering_poll_s",
        "drain_s",
        "concat_s",
        "seam_s",
        "morph_s",
    ):
        assert key in MODEL_PASS_TIMING_KEYS
        assert key in TIMING_KEYS
    assert "prompt_enhance" in TIMING_KEYS
    assert "motion_sense" in TIMING_KEYS


def test_build_stage_seconds_enhance_sense_split() -> None:
    built = build_stage_seconds(director=1.0, video=2.0, enhance_ms=1500.0, sense_ms=250.0)
    assert built["director"] == 1.0
    assert built["video"] == 2.0
    assert built["prompt_enhance"] == 1.5
    assert built["motion_sense"] == 0.25
    assert set(built) == set(GENERATION_TIMING_KEYS)


def test_generation_timing_table_mirrors_finalize() -> None:
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    console.generation_timing_table({"video": 2.0, "prompt_enhance": 1.0})
    out = stream.getvalue()
    assert "generation" in out
    assert "video 2.0s" in out
    assert "slowest" in out


def test_should_heartbeat_threshold() -> None:
    assert should_heartbeat(HEARTBEAT_AFTER_SECONDS + 1.0) is True
    assert should_heartbeat(0.5) is False


def test_heartbeat_bypasses_quiet() -> None:
    stream = io.StringIO()
    console = VoyageConsole(stream=stream, quiet=True)
    console.heartbeat("video 45.0s")
    assert "heartbeat" in stream.getvalue()


def test_prompt_enhance_stage_and_bar_helpers() -> None:
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    with prompt_enhance_stage(console, "ltx25"):
        pass
    assert "prompt enhance" in stream.getvalue()
    with prompt_enhance_bar(console, 3) as tracker:
        assert tracker is not None
        tracker.update()
        tracker.update()
    assert "prompt enhance prompts" in stream.getvalue()
    with prompt_enhance_stage(None, "ltx25"):
        pass
    with prompt_enhance_bar(None, 2) as tracker_none:
        assert tracker_none is None


def test_pump_worker_log_tail_verbose_gating() -> None:
    tails = {"video-worker.log": ["last line here"]}
    quiet_stream = io.StringIO()
    quiet_console = VoyageConsole(stream=quiet_stream)
    pump_worker_log_tail(quiet_console, tails, verbose_only=True)
    assert quiet_stream.getvalue() == ""
    verbose_stream = io.StringIO()
    verbose_console = VoyageConsole(verbose=True, stream=verbose_stream)
    pump_worker_log_tail(verbose_console, tails, verbose_only=True)
    assert "last line here" in verbose_stream.getvalue()
    pump_worker_log_tail(None, tails)
    pump_worker_log_tail(object(), tails)


def test_note_worker_rotation_verbose_only() -> None:
    silent = io.StringIO()
    plain = VoyageConsole(stream=silent)
    note_worker_rotation(plain, ["video-worker-2026-01-01.log"])
    assert silent.getvalue() == ""
    loud = io.StringIO()
    verbose = VoyageConsole(verbose=True, stream=loud)
    note_worker_rotation(verbose, ["video-worker-2026-01-01.log"])
    assert "rotated" in loud.getvalue()
    note_worker_rotation(verbose, [])
    note_worker_rotation(None, ["x"])


def test_note_prewarm_heartbeat_always_reports() -> None:
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    note_prewarm_heartbeat(
        console, upscale_frames=0, interp_frames=0, skip_reason="low vram", segments_seen=3
    )
    assert "held back" in stream.getvalue()
    stream2 = io.StringIO()
    note_prewarm_heartbeat(
        VoyageConsole(stream=stream2),
        upscale_frames=4,
        interp_frames=2,
        skip_reason="",
        segments_seen=1,
    )
    assert "+4f upscale" in stream2.getvalue()
    note_prewarm_heartbeat(None, upscale_frames=1, interp_frames=1, skip_reason="", segments_seen=1)


def test_prewarm_heartbeat_event_always_carries_fields() -> None:
    event = prewarm_heartbeat_event(
        "000001", upscale_frames=0, interp_frames=0, skip_reason="director busy", segments_seen=0
    )
    assert event["event"] == "prewarm_heartbeat"
    assert event["upscale_frames"] == 0
    assert event["skip_reason"] == "director busy"
    assert event["segments_seen"] == 0


def test_gauges_skipped_event() -> None:
    event = gauges_skipped_event("000002", "prefetch_in_flight")
    assert event["event"] == "gauges_skipped"
    assert event["reason"] == "prefetch_in_flight"


def test_doctor_health_alert_builders() -> None:
    facts = {
        "disk_by_mount": {"root": {"free_gib": 0.1, "used_fraction": 0.99}},
        "gpu_details": [],
        "models": {"checks": {}},
    }
    alerts = health_alerts(facts, min_free_gib=5.0)
    assert alerts
    event = health_alert_event(alerts, segment_id="000003")
    assert event["event"] == "health_alert"
    assert event["segment_id"] == "000003"
    twin = log_health_alert_event(alerts)
    assert twin["event"] == "health_alert"
    adapted = gauges_to_facts({"disk_free_gib": 10.0, "video_vram_free_gib": 1.0})
    assert "disk_by_mount" in adapted
    stream = io.StringIO()
    report_health_alerts(VoyageConsole(stream=stream), alerts)
    assert stream.getvalue() != ""
    warn_health_alerts(VoyageConsole(stream=io.StringIO()), [])
    report_health_alerts(object(), alerts)


def test_motion_stage_fragment() -> None:
    reading = MotionReading(energy=0.1, seconds=0.234)
    fragment = motion_stage_fragment(reading)
    assert fragment == {SENSE_STAGE_KEY: 0.234}
    assert ENHANCE_STAGE_KEY == "prompt_enhance"


def test_worker_log_rotation_event() -> None:
    event = worker_log_rotation_event(["a.log", "b.log"], segment_id="000004")
    assert event["event"] == "worker_log_rotated"
    assert event["count"] == 2
    assert event["segment_id"] == "000004"


def test_commit_av_drift_is_null() -> None:
    commit_av_drift()
    assert normalize_av_drift(None) is None
    assert normalize_av_drift(0.0) is None
    assert normalize_av_drift(0.42) == 0.42
    assert normalize_av_drift("bad") is None
    assert normalize_av_drift(True) is None


def test_prompt_enhanced_segment_id_back_compat() -> None:
    assert prompt_enhanced_segment_id({"segment_id": "000009"}) == "000009"
    assert prompt_enhanced_segment_id({"segment": "000008"}) == "000008"
    assert prompt_enhanced_segment_id({}) is None


def _write_committed_segment(run_dir: Path, segment_id: str) -> None:
    from voyage.segment_manifest import write_segment_manifest

    segment = run_dir / paths.SEGMENTS_DIRNAME / segment_id
    segment.mkdir(parents=True, exist_ok=True)
    (segment / paths.DONE_MARKER).write_text("", encoding="utf-8")
    write_segment_manifest(
        segment,
        {
            "metrics": {"video": {"frames": 48}},
            "transition": {"destination": {"canonical_name": "probe"}, "phase": "HOLD"},
            "prompt_plan": {},
            "audio_state": {"take_ids": []},
            "world_state": {},
            "checksums": {},
        },
    )


def test_scoreboard_adopted_and_null_deltas(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-e")
    _write_committed_segment(run_dir, "000000")
    _write_committed_segment(run_dir, "000001")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    _write_metrics_line(
        logs_dir, {"event": "segment_adopted", "segment_id": "000001", "reason": "orphan"}
    )
    _write_metrics_line(
        logs_dir, {"event": "segment_committed", "segment_id": "000000", "stages": {}}
    )
    assert adopted_segment_ids(run_dir) == {"000001"}
    rows = scoreboard_rows(run_dir)
    by_id = {row["segment_id"]: row for row in rows}
    assert by_id["000001"]["adopted"] is True
    assert by_id["000000"]["adopted"] is False
    assert by_id["000001"]["stages"] == {}


def test_scoreboard_null_delta_for_new_key(tmp_path: Path) -> None:
    from voyage.segment_manifest import write_segment_manifest

    run_dir = tmp_path / "run2"
    (run_dir / "segments").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.jsonl").write_text("", encoding="utf-8")
    for segment_id, keys in (("000000", ["motion_energy"]), ("000001", ["motion_energy"])):
        segment = run_dir / "segments" / segment_id
        segment.mkdir(parents=True, exist_ok=True)
        (segment / "DONE").write_text("ok", encoding="utf-8")
        metrics_block: dict[str, Any] = {"video": {"frames": 48}}
        if segment_id == "000000":
            metrics_block["visual"] = {"metrics": {"motion_energy": 0.2}}
        else:
            metrics_block["visual"] = {"metrics": {"motion_energy": 0.3, "visual_complexity": 0.4}}
        write_segment_manifest(
            segment,
            {
                "metrics": metrics_block,
                "transition": {"destination": {"canonical_name": "x"}, "phase": "HOLD"},
                "prompt_plan": {},
                "audio_state": {"take_ids": []},
                "world_state": {},
                "checksums": {},
            },
        )
        del keys
    rows = scoreboard_rows(run_dir)
    assert len(rows) == 2
    deltas = rows[1]["deltas"]
    assert isinstance(deltas, dict)
    assert deltas["visual_complexity"] is None


def test_scoreboard_torn_surfaced_and_validate_hook(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-e-torn")
    _write_committed_segment(run_dir, "000000")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event": "segment_committed"\n')
    assert torn_metric_lines(run_dir) == 1
    assert count_torn_metric_lines(run_dir) == 1
    assert validate_scoreboard(run_dir) != []
    summary = run_status_summary(run_dir)
    assert summary["torn"] == 1
    assert summary["partial"] == []


def test_scoreboard_json_includes_torn_and_status(tmp_path: Path, capsys: Any) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-e-json")
    _write_committed_segment(run_dir, "000000")
    assert scoreboard_main(["--run", str(run_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "torn" in payload
    assert "status" in payload
    assert payload["status"]["partial"] == []


def test_format_scoreboard_table_shows_adopted_and_status() -> None:
    table = format_scoreboard_table(
        [
            {
                "segment_id": "000000",
                "frames": 48,
                "video_exists": True,
                "destination": "x",
                "phase": "HOLD",
                "stages": {},
                "errors": [],
                "adopted": True,
            }
        ],
        [],
        {"status": "RUNNING", "stale_running": False, "torn": 0},
    )
    assert "adopted" in table
    assert "status: RUNNING" in table


def test_format_reconcile_output_surfaces_partial() -> None:
    from voyage.scoreboard import format_reconcile_output as fmt

    line = fmt(["000003"], ["000002"], ["000004"])
    assert "000003" in line
    assert "000002" in line
    assert "000004" in line
    assert fmt([], [], []) == "reconcile: deleted [none] adopted [none] partial [none]"


def test_validate_metrics_helper(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-e-validate")
    _write_committed_segment(run_dir, "000000")
    logs_dir = run_dir / paths.LOGS_DIRNAME
    with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            format_metric_line(
                "track-e-validate",
                {"event": "segment_committed", "segment_id": "000000", "frames": 48},
            )
            + "\n"
        )
    errors, warnings = validate_metrics(run_dir)
    assert errors == []
    assert isinstance(warnings, list)
    live = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    with live.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "legacy", "run_id": "track-e-validate"}\n')
    _errors2, warnings2 = validate_metrics(run_dir)
    assert any("v0" in warning for warning in warnings2)


def test_parse_metric_lines_counts_torn() -> None:
    events, torn = parse_metric_lines(['{"event": "ok"}', "torn{"])
    assert len(events) == 1
    assert torn == 1


def test_run_status_summary_read_only(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-e-status")
    before = sorted(str(path) for path in run_dir.rglob("*"))
    summary = run_status_summary(run_dir)
    assert summary["status"] in ("CREATED", "unknown")
    assert summary["committed"] == 0
    assert sorted(str(path) for path in run_dir.rglob("*")) == before
    stale = is_stale_running("RUNNING", time.time() - 99999.0, time.time())
    assert stale is True

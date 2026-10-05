"""Presented-frames coverage gate (redundant-finalize fix).

`_final_covers_timeline` compared ffprobe presented frames against
`state.timeline_frames` (source frames) — never equal under interp m=2 +
1.5x slow-mo, so the generate 'nothing to do' gate could never fire and
every revisit re-finalized. Coverage is now recorded in the manifest at
finalize success and the gate compares presented-against-presented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_record_final_coverage_round_trip(tmp_path: Path) -> None:
    from voyage.persistence import read_effective_config, record_final_coverage

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="coverage", seed=7)
    record_final_coverage(run_dir, presented_frames=476, segments=2)
    assert _manifest(run_dir)["final_coverage"] == {"segments": 2, "presented_frames": 476}
    # Extra manifest keys must stay tolerated by the config reader.
    assert read_effective_config(run_dir).name == "coverage"


def test_build_manifest_carries_no_coverage() -> None:
    """`configure` rebuilds the manifest fresh → coverage wipes for free."""
    from voyage.config import preset_config
    from voyage.persistence import build_manifest

    manifest = build_manifest(preset_config("fresh", "pastel neon line-art, peaceful", 7))
    assert "final_coverage" not in manifest


def _seeded_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "covered"
    initialize_run_directory(run_dir, run_id="covered", seed=7)
    (run_dir / "final.mp4").write_bytes(b"fake-final")
    return run_dir


def test_gate_missing_final_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.persistence import record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
    record_final_coverage(run_dir, presented_frames=48, segments=1)
    (run_dir / "final.mp4").unlink()
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=1) is False


def test_gate_without_coverage_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops

    run_dir = _seeded_run(tmp_path, monkeypatch)
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: 48)
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=1) is False


def test_gate_matching_coverage_is_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.persistence import record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
    record_final_coverage(run_dir, presented_frames=48, segments=1)
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: 48)
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=1) is True


def test_gate_stale_segments_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.persistence import record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
    record_final_coverage(run_dir, presented_frames=48, segments=1)
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: 48)
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=2) is False


def test_gate_stale_frames_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.persistence import record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
    record_final_coverage(run_dir, presented_frames=48, segments=1)
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: 96)
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=1) is False


def test_gate_probe_failure_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.persistence import record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
    record_final_coverage(run_dir, presented_frames=48, segments=1)
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: None)
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=1) is False


def _commit_two(run_dir: Path) -> None:
    from voyage.persistence import read_effective_config
    from voyage.supervisor import Supervisor

    supervisor = Supervisor(run_dir, read_effective_config(run_dir))
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _last_finalize_event(run_dir: Path) -> dict:
    lines = (run_dir / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if '"finalize_completed"' in line]
    assert events, "expected a finalize_completed event"
    return events[-1]


def test_finalize_run_records_invoker(tmp_path: Path) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="invoker", seed=7)
    _commit_two(run_dir)
    out = run_dir / "final.mp4"
    finalize_run(run_dir, out, invoker="generate")
    assert _last_finalize_event(run_dir)["invoker"] == "generate"


def test_finalize_run_invoker_defaults_to_none(tmp_path: Path) -> None:
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="invoker", seed=7)
    _commit_two(run_dir)
    finalize_run(run_dir, run_dir / "final.mp4")
    assert "invoker" in _last_finalize_event(run_dir)

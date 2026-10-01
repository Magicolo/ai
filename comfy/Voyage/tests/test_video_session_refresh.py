"""Proactive video-session refresh between segments (issue 198).

The ltx25 ComfyUI session OOMs on continuation blocks when it stays
resident across segments, while a fresh process rebuilt from the recovery
tape renders the identical block fine. `_refresh_video_session` automates
the proven manual-resume path in-process (restart + tape resume) so the
supervisor process — and its prefetch future — survives. Gated to the
backends that need it; everyone else keeps the resident session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.supervisor import VIDEO_BACKENDS_NEEDING_FRESH_SESSION, Supervisor


def _metric_events(run_dir: Path, event: str) -> list[dict[str, Any]]:
    metrics = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics.exists():
        return []
    events = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("event") == event:
                events.append(record)
    return events


def test_fresh_session_gate_lists_only_ltx25() -> None:
    """ltx25 needs it (198 OOM); ltx23/ltxv/fake stay resident (no reload tax)."""
    assert "ltx25" in VIDEO_BACKENDS_NEEDING_FRESH_SESSION
    assert "ltx23" not in VIDEO_BACKENDS_NEEDING_FRESH_SESSION
    assert "ltxv" not in VIDEO_BACKENDS_NEEDING_FRESH_SESSION
    assert "fake" not in VIDEO_BACKENDS_NEEDING_FRESH_SESSION


def test_should_refresh_only_when_more_remain(tmp_path: Path) -> None:
    """Finished batches never pay a pointless reload (no workers needed)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert config.video.backend == "fake"
    supervisor = Supervisor(run_dir, config)
    assert supervisor._should_refresh_video_session(None, 2) is False
    assert supervisor._should_refresh_video_session(3, 3) is False
    assert supervisor._should_refresh_video_session(3, 2) is False  # gate first


def test_should_refresh_true_path_for_ltx25(tmp_path: Path) -> None:
    """ltx25 refreshes mid-batch, never after the final commit."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, video_backend="ltx25")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    assert config.video.backend == "ltx25"
    supervisor = Supervisor(run_dir, config)
    assert supervisor._should_refresh_video_session(None, 0) is True
    assert supervisor._should_refresh_video_session(3, 2) is True
    assert supervisor._should_refresh_video_session(3, 3) is False


def test_refresh_replaces_video_process_and_commits_after(tmp_path: Path) -> None:
    """Restart + resume leaves a live worker; the next commit renders."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"
        before = supervisor._video.pid
        assert before is not None
        supervisor._refresh_video_session("000001")
        after = supervisor._video.pid
        assert after is not None
        assert after != before
        assert supervisor.commit_one_segment() == "000001"
    finally:
        supervisor.stop_workers()
    refreshed = _metric_events(run_dir, "video_session_refreshed")
    assert len(refreshed) == 1
    assert refreshed[0]["segment_id"] == "000001"

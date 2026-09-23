"""Scoreboard view: per-segment rows with metric deltas (slice 3)."""

from __future__ import annotations

from pathlib import Path

from voyage import paths
from voyage.config import default_config_toml, load_config
from voyage.persistence import build_manifest, initial_state, write_manifest, write_state
from voyage.scoreboard import METRIC_KEYS, scoreboard_rows
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, *, inspector: bool, run_id: str = "score") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    toml = default_config_toml(run_id, "pastel neon line-art, peaceful", 7)
    if inspector:
        toml = toml.replace("visual_inspector = false", "visual_inspector = true")
    (run_dir / paths.CONFIG_FILENAME).write_text(toml, encoding="utf-8")
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def _commit(run_dir: Path, count: int) -> None:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.run_segments(count) == [f"{n:06d}" for n in range(count)]
    finally:
        supervisor.stop_workers()


def test_scoreboard_two_segments_with_deltas(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=True)
    # Three commits: the piggyback inspector always covers the PREVIOUS
    # segment, so seg0/seg1 carry visual metrics and seg2 does not yet.
    _commit(run_dir, 3)
    rows = scoreboard_rows(run_dir)
    assert [row["segment_id"] for row in rows] == ["000000", "000001", "000002"]
    assert all(row["done"] for row in rows)
    assert all(row["frames"] == 48 for row in rows)
    assert set(rows[0]["stages"]) == {"inspect", "director", "video", "audio", "validate", "commit"}
    assert set(rows[0]["metrics"]) == set(METRIC_KEYS)
    assert rows[0]["deltas"] == dict.fromkeys(METRIC_KEYS, 0.0)
    expected = {
        key: round(rows[1]["metrics"][key] - rows[0]["metrics"][key], 3) for key in METRIC_KEYS
    }
    assert rows[1]["deltas"] == expected
    assert rows[0]["destination"]
    assert rows[0]["phase"]
    assert rows[0]["take_ids"]
    assert rows[0]["video_path"].endswith("000000/video.mp4")
    assert rows[0]["audio_path"].endswith("000000/audio.wav")


def test_scoreboard_missing_visual(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=False)
    _commit(run_dir, 1)
    (rows,) = scoreboard_rows(run_dir)
    assert rows["metrics"] is None
    assert rows["deltas"] is None
    assert rows["frames"] == 48


def test_scoreboard_skips_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir, inspector=True)
    _commit(run_dir, 2)
    (run_dir / "segments" / "000002").mkdir(parents=True)
    rows = scoreboard_rows(run_dir)
    assert [row["segment_id"] for row in rows] == ["000000", "000001"]

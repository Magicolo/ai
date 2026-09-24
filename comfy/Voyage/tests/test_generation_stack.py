"""Generation-stack tests: config defaults, drift cadence, prefetch, blend.

Covers the rhythm-cut music-video stack on fake (CPU-only) backends:
new AudioConfig/VoyageConfig fields, toml roundtrip, override plumbing,
drift cadence holds, parallel director prefetch hits, and the
finalize-only overlap blend (duration-exact, previews untouched).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from voyage import paths
from voyage.config import (
    AudioConfig,
    VoyageConfig,
    apply_draft_overrides,
    default_config_toml,
    load_config,
)
from voyage.media import build_final_audio, finalize_run
from voyage.media import probe as media_probe
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor


def _write_toml(tmp_path: Path) -> Path:
    path = tmp_path / "voyage.toml"
    text = default_config_toml("stack", "pastel neon line-art, peaceful", 7)
    path.write_text(text, encoding="utf-8")
    return path


def test_audio_config_rhythm_defaults() -> None:
    audio = AudioConfig()
    assert audio.beats_per_segment == 4
    assert audio.final_overlap_fraction == pytest.approx(0.10)
    assert audio.final_overlap_cap_seconds == pytest.approx(0.5)


def test_audio_config_rhythm_validators() -> None:
    with pytest.raises(ValidationError):
        AudioConfig(beats_per_segment=0)
    with pytest.raises(ValidationError):
        AudioConfig(final_overlap_fraction=0.6)
    with pytest.raises(ValidationError):
        AudioConfig(final_overlap_fraction=-0.1)
    with pytest.raises(ValidationError):
        AudioConfig(final_overlap_cap_seconds=-1.0)


def test_voyage_config_drift_default_and_validator() -> None:
    assert VoyageConfig().drift_every_n_segments == 1
    with pytest.raises(ValidationError):
        VoyageConfig(drift_every_n_segments=0)


def test_toml_roundtrip_carries_new_fields(tmp_path: Path) -> None:
    config, _ = load_config(_write_toml(tmp_path))
    assert config.audio.beats_per_segment == 4
    assert config.audio.final_overlap_fraction == pytest.approx(0.10)
    assert config.audio.final_overlap_cap_seconds == pytest.approx(0.5)
    assert config.voyage.drift_every_n_segments == 1


def test_overrides_plumb_beats_and_drift(tmp_path: Path) -> None:
    config, _ = load_config(_write_toml(tmp_path))
    out = apply_draft_overrides(config, beats_per_segment=8, drift_every_n_segments=3)
    assert out.audio.beats_per_segment == 8
    assert out.voyage.drift_every_n_segments == 3
    assert config.audio.beats_per_segment == 4  # pure: source untouched
    with pytest.raises(ValidationError):
        apply_draft_overrides(config, beats_per_segment=0)


def _init_run(run_dir: Path, style: str = "pastel neon line-art, peaceful") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml("stack", style, 7), encoding="utf-8"
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def _metric_events(run_dir: Path, event: str) -> list[dict[str, object]]:
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


def test_drift_cadence_holds_non_drift_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config = apply_draft_overrides(config, drift_every_n_segments=2)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"  # 0 % 2: drift (LLM path)
        assert supervisor.commit_one_segment() == "000001"  # 1 % 2: hold
    finally:
        supervisor.stop_workers()
    holds = _metric_events(run_dir, "drift_hold")
    assert len(holds) == 1
    assert holds[0]["segment_id"] == "000001"
    state = read_state(run_dir)
    assert state.committed_segments == 2


def test_director_prefetch_hits_next_commit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    # Segment 0 has no prefetch (nothing ran before it); segment 1 should
    # consume the proposal prefetched during segment 0's render window.
    misses = _metric_events(run_dir, "director_prefetch_miss")
    hits = _metric_events(run_dir, "director_prefetch_hit")
    assert [event["segment_id"] for event in misses] == ["000000"]
    assert [event["segment_id"] for event in hits] == ["000001"]


def _commit_two(run_dir: Path) -> None:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_finalize_blend_is_timeline_exact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    out = tmp_path / "final.mp4"
    assert finalize_run(run_dir, out).exists()
    duration = float(media_probe(out).get("format", {}).get("duration", 0.0))
    # 2 fake segments x 48f @24fps = 4.0s; overlap blend must not shorten it.
    assert duration == pytest.approx(4.0, abs=0.15)


def test_finalize_overlap_zero_keeps_legacy_splice(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    out = tmp_path / "final-legacy.mp4"
    assert finalize_run(run_dir, out, overlap_fraction=0.0).exists()
    duration = float(media_probe(out).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(4.0, abs=0.15)


def test_build_final_audio_falls_back_without_takes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    # No takes ledger on this path only when audio/ is removed: blend must
    # degrade to the hard-splice concat, not raise.
    import shutil

    shutil.rmtree(run_dir / "audio", ignore_errors=True)
    usable = [paths.segment_dir(run_dir, "000000"), paths.segment_dir(run_dir, "000001")]
    dest = build_final_audio(run_dir, usable, tmp_path, 24, 48000, 2)
    assert dest.exists() and dest.stat().st_size > 0


def test_generate_defaults_to_qwen_director_with_offline_fallback(tmp_path: Path) -> None:
    """`generate` without --director resolves qwen; missing weights (this
    CPU-only image) degrade to the deterministic fallback via the
    offline-first load — the run still validates and finalizes."""
    from voyage.cli import main

    run_dir = tmp_path / "run"
    code = main(
        [
            "generate",
            "--backend",
            "fake",
            "--duration",
            "2s",
            "--style",
            "pastel neon line-art, peaceful",
            "--output",
            str(run_dir),
            "--run-id",
            "gen-qwen-default",
            "--seed",
            "11",
        ]
    )
    assert code == 0
    assert (run_dir / "final.mp4").exists()
    from voyage.config import load_config as _load

    config, _ = _load(run_dir / paths.CONFIG_FILENAME)
    assert config.director.backend == "deterministic"  # file default untouched
    state = read_state(run_dir)
    assert state.committed_segments == 1

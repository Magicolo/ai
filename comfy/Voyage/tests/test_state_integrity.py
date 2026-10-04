"""State-integrity tests: checksums, orphans, numbering, atomicity (DESIGN §§56, 70).

All run against fake backends (real media, no GPU) in-container.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.audio.planner import AudioTake, append_take, load_takes
from voyage.cli import main, validate_run
from voyage.concepts import ConceptStore
from voyage.errors import MediaError
from voyage.media import (
    FinalizeOptions,
    av_drift_seconds,
    check_av_alignment,
    finalize_run,
    validate_audio,
    validate_video,
)
from voyage.persistence import read_effective_config, read_state
from voyage.rpc import SubprocessWorker
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "integrity") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _commit(run_dir: Path, count: int) -> list[str]:
    config, _ = read_effective_config(run_dir)
    return Supervisor(run_dir, config).run_segments(count)


def _rewrite_metrics(segment: Path, **overrides: object) -> None:
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    manifest = load_segment_manifest(segment)
    metrics = dict(manifest.get("metrics", {}))
    metrics.update(overrides)
    write_segment_manifest(segment, {**manifest, "metrics": metrics})


def _rewrite_checksums(segment: Path) -> None:
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest
    from voyage.supervisor import sha256_file

    manifest = load_segment_manifest(segment)
    checksums: dict[str, str] = {
        "video.mp4": sha256_file(segment / "video.mp4"),
        "audio.wav": sha256_file(segment / "audio.wav"),
    }
    tape = segment / "recovery.pt"
    if tape.is_file():
        checksums["recovery.pt"] = sha256_file(tape)
    write_segment_manifest(segment, {**manifest, "checksums": checksums})


def test_clean_run_validates(tmp_path: Path) -> None:
    """A genuine fake-backend commit must pass every new integrity check."""
    run_dir = tmp_path / "run"
    _init_run(run_dir, "clean")
    assert _commit(run_dir, 1) == ["000000"]
    assert validate_run(run_dir) == []
    assert main(["validate", "--run", str(run_dir)]) == 0


def test_validate_detects_checksum_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    video = run_dir / "segments" / "000000" / "video.mp4"
    with video.open("ab") as handle:
        handle.write(b"\x00")
    errors = validate_run(run_dir)
    assert any("checksum" in error and "video.mp4" in error for error in errors)
    assert main(["validate", "--run", str(run_dir)]) == 1


def test_validate_detects_audio_checksum_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    audio = run_dir / "segments" / "000000" / "audio.wav"
    with audio.open("ab") as handle:
        handle.write(b"\x00")
    errors = validate_run(run_dir)
    assert any("checksum" in error and "audio.wav" in error for error in errors)


def test_validate_detects_in_segment_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    (run_dir / "segments" / "000000" / "world_state.json.ab12.partial").write_bytes(b"")
    errors = validate_run(run_dir)
    assert any("orphan" in error for error in errors)


def test_validate_detects_done_partial_remnant(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    (run_dir / "segments" / "000000" / "DONE.partial").write_bytes(b"")
    errors = validate_run(run_dir)
    assert any("DONE.partial" in error for error in errors)


def test_validate_detects_segment_gap(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 2)
    second = run_dir / "segments" / "000001"
    staged = run_dir / "segments" / "000002"
    second.rename(staged)
    errors = validate_run(run_dir)
    assert any("000001" in error and ("gap" in error or "numbering" in error) for error in errors)


def test_validate_detects_nonpositive_frames(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    _rewrite_metrics(run_dir / "segments" / "000000", frames=0)
    errors = validate_run(run_dir)
    assert any("frames" in error for error in errors)


def test_validate_detects_missing_recovery_tape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    _rewrite_metrics(run_dir / "segments" / "000000", recovery_tape="/nonexistent/recovery.pt")
    errors = validate_run(run_dir)
    assert any("recovery" in error for error in errors)


def test_validate_detects_impossible_duration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    segment = run_dir / "segments" / "000000"
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    manifest = load_segment_manifest(segment)
    metrics = dict(manifest.get("metrics", {}))
    video_block = dict(metrics.get("video", {}))
    video_block["duration"] = 0.0
    metrics["video"] = video_block
    write_segment_manifest(segment, {**manifest, "metrics": metrics})
    errors = validate_run(run_dir)
    assert any("duration" in error for error in errors)


def test_no_partial_remnants_after_commit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 2)
    leftovers = list((run_dir / "segments").rglob("*.partial"))
    assert leftovers == []


def test_finalize_rejects_tampered_segment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    video = run_dir / "segments" / "000000" / "video.mp4"
    with video.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(MediaError, match="checksum"):
        finalize_run(run_dir, tmp_path / "final.mp4")


def _synth_wav(dest: Path, duration: float) -> None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_finalize_rejects_av_misalignment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    segment = run_dir / "segments" / "000000"
    _synth_wav(segment / "audio.wav", 10.0)
    _rewrite_checksums(segment)
    with pytest.raises(MediaError, match="[Aa]lignment"):
        finalize_run(run_dir, tmp_path / "final.mp4")


@pytest.mark.slow
def test_finalize_skip_bad_finalizes_rest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 2)
    bad_video = run_dir / "segments" / "000001" / "video.mp4"
    with bad_video.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(MediaError, match="checksum"):
        finalize_run(run_dir, tmp_path / "strict.mp4")
    out = finalize_run(run_dir, tmp_path / "lenient.mp4", skip_bad=True)
    assert out.exists()


def test_concept_store_vector_roundtrip(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    record, _ = store.propose("glowing neon lattice", vector=[1.0, 0.0, 0.0], segment=0)
    assert record.embedding_index == 0
    reopened = ConceptStore(tmp_path / "novelty")
    accepted, score = reopened.check_novel("glowing neon lattice", vector=[1.0, 0.0, 0.0])
    assert not accepted
    assert score == pytest.approx(1.0)
    index = json.loads((tmp_path / "novelty" / "concept_index.json").read_text(encoding="utf-8"))
    assert index[record.id] == 0


def test_ledger_append_roundtrip(tmp_path: Path) -> None:
    ledger = tmp_path / "takes.jsonl"
    take = AudioTake(
        take_id="take_0000",
        path=str(tmp_path / "take.wav"),
        caption="quiet drone",
        seed=7,
        covers_from=0.0,
        duration=45.0,
        segment_index=0,
    )
    append_take(ledger, take)
    loaded = load_takes(ledger)
    assert len(loaded) == 1
    assert loaded[0].take_id == "take_0000"
    assert loaded[0].caption == "quiet drone"


# --- 088 fold: tests/test_av_alignment_consumer.py (6 tests) ---
# """Issue 003 consumer tests: A/V drift enforced outside the finalizer.
#
# Covers the media helper (`av_drift_seconds` / `check_av_alignment`), the
# `validate` drift check over stored metrics durations, and the commit-path
# wiring (supervisor step 4 calls `check_av_alignment` on probed durations).
# Fake backends / real ffmpeg for the commit tests; pure math otherwise.
# """
# NOTE: source `_commit` is byte-identical to this file's `_commit` — reused,
# not duplicated.


def test_av_drift_seconds_is_absolute_difference() -> None:
    assert av_drift_seconds(2.0, 10.0) == pytest.approx(8.0)
    assert av_drift_seconds(10.0, 2.0) == pytest.approx(8.0)
    assert av_drift_seconds(1.5, 1.5) == pytest.approx(0.0)


def test_check_av_alignment_returns_drift_within_budget() -> None:
    drift = check_av_alignment(2.0, 2.5, "000000")
    assert drift == pytest.approx(0.5)


def test_check_av_alignment_rejects_beyond_budget() -> None:
    with pytest.raises(MediaError, match="[Aa]lignment"):
        check_av_alignment(2.0, 2.7, "000000")


def test_validate_passes_aligned_segment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="alignment")
    assert _commit(run_dir, 1) == ["000000"]
    assert validate_run(run_dir) == []


def test_validate_rejects_drifted_stored_durations(tmp_path: Path) -> None:
    """Rewritten-audio segment must fail validate, not just finalize."""
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="alignment")
    _commit(run_dir, 1)
    segment = run_dir / "segments" / "000000"
    manifest = load_segment_manifest(segment)
    metrics = dict(manifest["metrics"])
    video_block = dict(metrics.get("video", {}))
    audio_block = dict(metrics.get("audio", {}))
    video_block["duration"] = 2.0
    audio_block["duration"] = 10.0
    metrics["video"] = video_block
    metrics["audio"] = audio_block
    write_segment_manifest(segment, {**manifest, "metrics": metrics})
    errors = validate_run(run_dir)
    assert any("drift" in error and "000000" in error for error in errors)


def test_commit_rejects_av_drifted_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Commit-side wiring: drifted probed audio durations fail the commit."""
    import voyage.supervisor as supervisor_module

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="alignment")
    # The supervisor calls its own module-global `validate_audio` (bound by
    # `from voyage.media import ...`, which mypy strict does not treat as an
    # explicit re-export), so the drift wrapper must patch the supervisor
    # namespace while delegating to the defining module's original.

    def _drifted(path: Path, sample_rate: int, channels: int) -> dict[str, object]:
        info = validate_audio(path, sample_rate, channels)
        return {**info, "duration": float(info["duration"]) + 10.0}

    monkeypatch.setattr(supervisor_module, "validate_audio", _drifted)
    with pytest.raises(MediaError, match="alignment drift"):
        _commit(run_dir, 1)


# --- 088 fold: tests/test_integration.py (6 tests) ---
# """Integration tests: workers over RPC + full segment commit + validate.
#
# Fake backends render real media with ffmpeg, so validation, checksums,
# DONE markers, state advance, and finalizer concat are all exercised.
# """
# NOTE: source run initializer (fixed run_id "itest", seed 7) differs from
# this file's parameterized initializer — kept inline per the quintet
# prefix precedent (assertions byte-identical).


def test_workers_answer_health(tmp_path: Path) -> None:
    worker = SubprocessWorker("voyage.workers.video", tmp_path, tmp_path / "v.log")
    worker.start()
    try:
        assert worker.health()["status"] == "READY"
    finally:
        worker.stop()


def test_director_worker_decides(tmp_path: Path) -> None:
    worker = SubprocessWorker("voyage.workers.director", tmp_path, tmp_path / "d.log")
    worker.start()
    try:
        result = worker.call(
            "decide",
            {
                "decision_index": 0,
                "current_concept": "a",
                "destination_concept": "b",
                "phase": "ESTABLISH",
                "style": "s",
                # Explicit opt-out: the worker default is qwen since
                # 2026-09-29, so the deterministic path must say so.
                "backend": "deterministic",
            },
        )
        destination = cast(dict[str, Any], result["destination"])
        assert destination["canonical_name"] == "b"
        assert result["fallback"] is True
    finally:
        worker.stop()


def test_commit_one_segment_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="itest", seed=7)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        segment_id = supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert segment_id == "000000"
    segment = paths.segment_dir(run_dir, segment_id)
    for name in ("video.mp4", "audio.wav", "DONE", "manifest.json"):
        assert (segment / name).exists(), name
    for name in (
        "transition.json",
        "prompt_plan.json",
        "audio_state.json",
        "world_state.json",
        "metrics.json",
        "sha256.json",
    ):
        assert not (segment / name).exists(), name
    state = read_state(run_dir)
    assert state.committed_segments == 1
    assert state.next_segment_number == 1
    assert state.timeline_frames == config.video.segment_frames


def _committed_run(tmp_path: Path, segment_count: int) -> Path:
    """Commit `segment_count` fake segments; caller owns no workers after return."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="itest", seed=7)
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        for _ in range(segment_count):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    return run_dir


@pytest.mark.slow
def test_finalize_end_to_end_after_commit(tmp_path: Path) -> None:
    """Commit → finalize → valid presentation MP4 (issue 038 contract path)."""
    run_dir = _committed_run(tmp_path, 2)
    output_path = tmp_path / "final.mp4"
    assert finalize_run(run_dir, output_path) == output_path
    assert output_path.exists()
    info = validate_video(output_path, 1216, 704, 24)
    assert info["duration"] > 0


@pytest.mark.slow
def test_finalize_options_explicit_joint_style(tmp_path: Path) -> None:
    """The `FinalizeOptions` path (issue 045) finalizes identically."""
    run_dir = _committed_run(tmp_path, 2)
    for joint_style in ("blend", "hard-splice"):
        output_path = tmp_path / f"final-{joint_style}.mp4"
        options = FinalizeOptions(joint_style=joint_style)
        assert finalize_run(run_dir, output_path, options=options) == output_path
        info = validate_video(output_path, 1216, 704, 24)
        assert info["duration"] > 0


def test_finalize_options_rejects_unknown_joint_style() -> None:
    with pytest.raises(ValueError, match="joint_style"):
        FinalizeOptions(joint_style="crossfade-everything")  # type: ignore[arg-type]

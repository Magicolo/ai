"""State-integrity tests: checksums, orphans, numbering, atomicity (DESIGN §§56, 70).

All run against fake backends (real media, no GPU) in-container.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from voyage import paths
from voyage.audio.planner import AudioTake, append_take, load_takes
from voyage.cli import main, validate_run
from voyage.concepts import ConceptStore
from voyage.config import default_config_toml, load_config
from voyage.errors import MediaError
from voyage.media import finalize_run
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "integrity") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml(run_id, "pastel neon line-art, peaceful", 11),
        encoding="utf-8",
    )
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    from voyage.persistence import build_manifest, initial_state, write_manifest, write_state

    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


def _commit(run_dir: Path, count: int) -> list[str]:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    return Supervisor(run_dir, config).run_segments(count)


def _rewrite_metrics(segment: Path, **overrides: object) -> None:
    metrics_path = segment / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(overrides)
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_checksums(segment: Path) -> None:
    from voyage.supervisor import sha256_file

    (segment / "sha256.json").write_text(
        json.dumps(
            {
                "video.mp4": sha256_file(segment / "video.mp4"),
                "audio.wav": sha256_file(segment / "audio.wav"),
            }
        ),
        encoding="utf-8",
    )


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
    metrics_path = segment / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["video"]["duration"] = 0.0
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
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

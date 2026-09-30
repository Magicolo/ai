"""Consumer-side path + orphan coverage (issues 016-C, 058-C).

016-C: stored take/tape paths resolve run-relative (existing as-is,
otherwise `run_dir`-joined), so a relocated run keeps validating and
serving takes. 058-C: the validate orphan scan covers `*.tmp.npy` /
`*.tmp*` in segments and the novelty dir — torn vector temps included.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.audio.planner import AudioTake
from voyage.cli import validate_run
from voyage.config import load_config
from voyage.errors import MediaError
from voyage.paths import resolve_stored_path
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "relocatable") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _commit(run_dir: Path, count: int) -> list[str]:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    return Supervisor(run_dir, config).run_segments(count)


def _rewrite_metrics(segment: Path, **overrides: object) -> None:
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    manifest = load_segment_manifest(segment)
    metrics = dict(manifest.get("metrics", {}))
    metrics.update(overrides)
    write_segment_manifest(segment, {**manifest, "metrics": metrics})


def _make_take(take_id: str = "take_0000", path: str = "") -> AudioTake:
    return AudioTake(
        take_id=take_id,
        path=path,
        caption="quiet drone",
        seed=7,
        covers_from=0.0,
        duration=45.0,
        segment_index=0,
    )


def test_resolve_stored_path_prefers_existing_absolute(tmp_path: Path) -> None:
    """Outside-the-run absolutes raise (issue 015) — existence no longer trusts."""
    target = tmp_path / "take.wav"
    target.write_bytes(b"data")
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(tmp_path / "run", str(target))


def test_resolve_stored_path_joins_missing_relative(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert resolve_stored_path(run_dir, "audio/take_0000.wav") == run_dir / "audio/take_0000.wav"


def test_resolve_stored_path_keeps_missing_absolute(tmp_path: Path) -> None:
    """Missing outside-the-run absolutes raise (issue 015)."""
    missing = tmp_path / "gone" / "take.wav"
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(tmp_path / "run", str(missing))


def test_resolve_stored_path_reanchors_legacy_absolute(tmp_path: Path) -> None:
    """Pre-fix absolute entries heal after a move via layout re-anchoring."""
    run_dir = tmp_path / "run"
    target = run_dir / "audio" / "take_0000.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")
    stale = tmp_path / "old-location" / "audio" / "take_0000.wav"
    assert resolve_stored_path(run_dir, str(stale)) == target


def test_take_resolved_path_survives_relocation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True)
    take_file = audio_dir / "take_0000.wav"
    take_file.write_bytes(b"data")
    take = _make_take(path="audio/take_0000.wav")
    assert take.resolved_path(run_dir) == take_file
    moved = tmp_path / "moved"
    shutil.move(str(run_dir), str(moved))
    assert take.resolved_path(moved) == moved / "audio/take_0000.wav"
    assert take.resolved_path(moved).exists()


def test_validate_tape_resolves_run_relative(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    segment = run_dir / "segments" / "000000"
    _rewrite_metrics(segment, recovery_tape="recovery.pt")
    assert any("recovery" in error for error in validate_run(run_dir))
    (run_dir / "recovery.pt").write_bytes(b"tape")
    assert validate_run(run_dir) == []


def test_relocated_run_validates_with_relative_refs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    segment = run_dir / "segments" / "000000"
    (run_dir / "recovery.pt").write_bytes(b"tape")
    _rewrite_metrics(segment, recovery_tape="recovery.pt")
    assert validate_run(run_dir) == []
    moved = tmp_path / "moved"
    shutil.move(str(run_dir), str(moved))
    assert validate_run(moved) == []


def test_validate_flags_segment_tmp_orphan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    orphan = run_dir / "segments" / "000000" / "concept_vectors.npy.12345.tmp.npy"
    orphan.write_bytes(b"torn")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "tmp.npy" in error for error in errors)


def test_validate_flags_novelty_tmp_orphan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    novelty_dir = run_dir / "novelty"
    novelty_dir.mkdir(exist_ok=True)
    orphan = novelty_dir / "concept_vectors.npy.12345.tmp.npy"
    orphan.write_bytes(b"torn")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "tmp.npy" in error for error in errors)


def test_validate_clean_run_has_no_transient_orphans(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    assert validate_run(run_dir) == []

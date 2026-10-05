"""Orphan scan covers audio/ + run root (issue 098).

Why this module exists: `validate_run` orphan-scanned only `segments/` +
`novelty/` while `atomic_write_*` stages `*.partial` in every
`destination.parent` — including `audio/` takes and the run root
(`state.json`). A crashed take or state advance left residue the
canonical "is this run clean?" tool never reported. The root pass is
scoped to top-level `*.partial` only so the recursive `segments/` +
`novelty/` + `audio/` hits are not double-counted; `logs/` rotation
siblings (`metrics-YYYY-MM-DD.jsonl`) never match the globs.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage.cli_validate import validate_run


def test_validate_flags_audio_partial(tmp_path: Path) -> None:
    """A crashed take staging file under audio/ must be reported."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    orphan = audio_dir / "take_0000.wav.partial"
    orphan.write_text("torn", encoding="utf-8")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "take_0000" in error for error in errors)


def test_validate_flags_root_partial(tmp_path: Path) -> None:
    """A crashed state advance (`state.json.*.partial`) must be reported."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    orphan = run_dir / "state.json.abc123.partial"
    orphan.write_text("torn", encoding="utf-8")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "state.json" in error for error in errors)


def test_validate_ignores_logs_rotation_sibling(tmp_path: Path) -> None:
    """Dated `logs/` rotation siblings are legitimate, never orphans."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    sibling = run_dir / "logs" / "metrics-2026-01-01.jsonl"
    sibling.write_text("{}\n", encoding="utf-8")
    errors = validate_run(run_dir)
    assert validate_run(run_dir) == [] or not any("orphan" in error for error in errors)
    assert errors == []


def test_validate_flags_augment_partial(tmp_path: Path) -> None:
    """A crashed chunk encode under `augment/<hash>/` must be reported."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    plan_dir = run_dir / "augment" / ("a" * 16)
    plan_dir.mkdir(parents=True)
    orphan = plan_dir / "chunk_00.partial.mp4"
    orphan.write_bytes(b"torn")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "chunk_00" in error for error in errors)


def test_validate_flags_final_staging_tmpdir(tmp_path: Path) -> None:
    """A stranded `voyage-final-*` staging dir (killed finalize) must be reported."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    stranded = run_dir / "voyage-final-abc123"
    stranded.mkdir(parents=True)
    (stranded / "model_intermediate.mp4").write_bytes(b"torn")
    errors = validate_run(run_dir)
    assert any("orphan" in error and "voyage-final-abc123" in error for error in errors)


def test_validate_flags_sidecar_ledger_without_output(tmp_path: Path) -> None:
    """A ledgered interp chunk with no PNG dir is an inconsistency notice (read-only)."""
    import json

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    plan_dir = run_dir / "augment" / ("b" * 16)
    plan_dir.mkdir(parents=True)
    record = {
        "chunk_index": 0,
        "start_frame": 0,
        "source_frames": 4,
        "expected_frames": 7,
        "upscale_factor": 2,
        "multiplier": 2,
        "crf": 15,
        "preset": "veryfast",
        "source_key": "abc",
        "weights_key": "wkey",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 48,
        "stage": "interpolated",
        "path": "augment/plan/interpolated_00",
    }
    (plan_dir / "chunks.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors = validate_run(run_dir)
    assert any(("b" * 16) in error and "interpolated" in error for error in errors)


def test_validate_clean_when_sidecar_output_matches_ledger(tmp_path: Path) -> None:
    """A fully ledgered + present chunk dir stays silent (no false positive)."""
    import json

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    plan_dir = run_dir / "augment" / ("c" * 16)
    png_dir = plan_dir / "interpolated_00"
    png_dir.mkdir(parents=True)
    for position in range(3):
        (png_dir / f"frame_{position:06d}.png").write_bytes(b"fake-png")
    record = {
        "chunk_index": 0,
        "start_frame": 0,
        "source_frames": 2,
        "expected_frames": 3,
        "upscale_factor": 2,
        "multiplier": 2,
        "crf": 15,
        "preset": "veryfast",
        "source_key": "abc",
        "weights_key": "wkey",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 48,
        "stage": "interpolated",
        "path": "augment/plan/interpolated_00",
    }
    (plan_dir / "chunks.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors = validate_run(run_dir)
    assert not any(("c" * 16) in error for error in errors)

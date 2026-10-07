"""Wiring tests for joints/sfx stage-skip gates (DESIGN §56 stage skipping).

Why this file exists: the track helpers (`augment_work_complete`,
`sfx_bed_complete`, stamps/validators) are covered by their own track
suites, but the orchestrator wiring — persistence markers, the
`_resume_completed_pass` fail-open paths, and the `spawn_workers`
plumbing — is only exercised here. Every test is CPU-only and
ffmpeg-free (fail-open paths return before any probe/spawn).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from voyage.augment_finalize import _resume_completed_pass, run_durable_model_pass
from voyage.persistence import (
    create_run_dir,
    read_augment_coverage,
    read_sfx_coverage,
    record_augment_coverage,
    record_sfx_coverage,
)
from voyage.sfx_finalize import (
    _render_single_track,
    render_sfx_bed,
    stamp_sfx_coverage,
    validate_sfx_coverage,
)

from tests.conftest import DEFAULT_RUN_ID, DEFAULT_STYLE, initialize_run_directory


def _resume_kwargs(run_dir: Path, work_dir: Path) -> dict:
    """Keyword arguments for `_resume_completed_pass` (positional segments vary)."""
    return {
        "weights": None,
        "weights_key": "weights",
        "out_width": 64,
        "out_height": 32,
        "source_fps": 24.0,
        "source_fps_key": 24,
        "upscale_factor": 2,
        "multiplier": 4,
        "chunk_frames": 32,
        "crf": 30,
        "preset": "slow",
        "device": "cpu",
        "work_dir": work_dir,
        "interp_backend": "rife",
        "drain_fn": None,
        "concat_fn": None,
        "assemble_fn": None,
        "joint_interp_fn": None,
        "timings": None,
        "progress": None,
    }


def test_augment_coverage_round_trip(tmp_path: Path) -> None:
    """Recorded augment marker reads back identical (persistence wiring)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    marker = {
        "fingerprint": "abc123",
        "segments": 2,
        "joints": 1,
        "jointed_timeline_sha": "def456",
        "interp_backend": "rife",
        "weights_key": "weights",
    }
    record_augment_coverage(run_dir, marker)
    assert read_augment_coverage(run_dir) == marker


def test_sfx_coverage_round_trip(tmp_path: Path) -> None:
    """Recorded sfx marker reads back identical (persistence wiring)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    marker = {
        "timeline_ms": 8000,
        "conditioning_source": "proxy",
        "dual_pan": False,
        "bed_digest": "bed",
        "ledger_digest": "ledger",
        "model_size": "large",
    }
    record_sfx_coverage(run_dir, marker)
    assert read_sfx_coverage(run_dir) == marker


def test_coverage_readers_fail_open_without_manifest(tmp_path: Path) -> None:
    """Readers return None (never raise) when no manifest exists."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert read_augment_coverage(run_dir) is None
    assert read_sfx_coverage(run_dir) is None


def test_coverage_readers_fail_open_on_non_dict(tmp_path: Path) -> None:
    """Readers return None when the stored marker is not a dict."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["augment_coverage"] = ["not", "a", "dict"]
    raw["sfx_coverage"] = 42
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    assert read_augment_coverage(run_dir) is None
    assert read_sfx_coverage(run_dir) is None


def test_resume_returns_none_without_manifest(tmp_path: Path) -> None:
    """No marker anywhere means no skip (fail-open None, no sources needed)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    assert (
        _resume_completed_pass(run_dir, [], **_resume_kwargs(run_dir, work_dir)) is None  # type: ignore[arg-type]
    )


def test_resume_returns_none_on_non_divisible_geometry(tmp_path: Path) -> None:
    """Geometry bail-out precedes every source/marker read (pure check)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    kwargs = _resume_kwargs(run_dir, work_dir)
    kwargs["out_width"] = 65
    assert _resume_completed_pass(run_dir, [], **kwargs) is None  # type: ignore[arg-type]


def test_resume_raises_on_missing_source(tmp_path: Path) -> None:
    """A usable segment with no pollable source fails loud (drain parity)."""
    from voyage.media import MediaError

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(MediaError, match="no pollable source"):
        _resume_completed_pass(
            run_dir,
            [tmp_path / "000000"],
            **_resume_kwargs(run_dir, work_dir),  # type: ignore[arg-type]
        )


def test_sfx_stamp_validate_round_trip() -> None:
    """Stamped sfx marker validates (schema wiring, no files needed)."""
    marker = stamp_sfx_coverage(
        run_dir=None,
        timeline_seconds=8.0,
        conditioning_source="proxy",
        dual_pan=False,
        bed_digest="bed",
        model_size="large",
        ledger_digest="ledger",
    )
    assert validate_sfx_coverage(marker) == marker
    assert marker["timeline_ms"] == 8000


def test_run_durable_model_pass_accepts_should_stop() -> None:
    """Cancel flag threads into the durable pass (cancellable joints work)."""
    params = inspect.signature(run_durable_model_pass).parameters
    assert "should_stop" in params
    assert params["should_stop"].default is None


def test_sfx_render_paths_accept_spawn_workers() -> None:
    """Spawn-free joins stay available (pre-check-True never spawns workers)."""
    for function in (render_sfx_bed, _render_single_track):
        params = inspect.signature(function).parameters
        assert "spawn_workers" in params
        assert params["spawn_workers"].default is True


def test_create_run_dir_keeps_manifest_minimal(tmp_path: Path) -> None:
    """Coverage keys stay out of fresh manifests (configure wipes them anyway)."""
    from voyage.config import preset_config

    run_dir = tmp_path / "run"
    create_run_dir(run_dir, preset_config(DEFAULT_RUN_ID, DEFAULT_STYLE, 11))
    raw = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "augment_coverage" not in raw
    assert "sfx_coverage" not in raw

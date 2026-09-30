"""Issue 003 consumer tests: A/V drift enforced outside the finalizer.

Covers the media helper (`av_drift_seconds` / `check_av_alignment`), the
`validate` drift check over stored metrics durations, and the commit-path
wiring (supervisor step 4 calls `check_av_alignment` on probed durations).
Fake backends / real ffmpeg for the commit tests; pure math otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.cli import validate_run
from voyage.config import load_config
from voyage.errors import MediaError
from voyage.media import av_drift_seconds, check_av_alignment, validate_audio
from voyage.supervisor import Supervisor


def _commit(run_dir: Path, count: int) -> list[str]:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    return Supervisor(run_dir, config).run_segments(count)


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
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="alignment")
    _commit(run_dir, 1)
    metrics_path = run_dir / "segments" / "000000" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["video"]["duration"] = 2.0
    payload["audio"]["duration"] = 10.0
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
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

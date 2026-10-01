"""Commit-time take-joint compensation (boba seg21: A/V drift 0.959s).

Joining abutting slices with a crossfade absorbs `fade` seconds, so the
assembled preview runs short of the video whenever a take boundary falls
inside the segment window — the 0.6 s A/V gate then fails the commit,
and retries reproduce the same take geometry forever. The slice walk
must extend the tail slice by the absorption (mirroring
`build_final_audio`'s issue-095 compensation), clamped to the take
file, and pass the fade explicitly so assembly consumes exactly what
was added. The render stub writes real sine (no GPU); slicing, assembly
and probing are all genuine ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.audio.planner import AudioTake, append_take
from voyage.config import ProjectConfig, load_config
from voyage.models import DirectorDestination, EvolutionDecision
from voyage.supervisor import Supervisor

_CAPTION = "slow ambient electronic composition"
_VIDEO_TIME = 86.083333
_DURATION = 121 / 24  # 5.0417 s: fresh-session 121-frame LTXV segment


def _sine_wav(path: Path, seconds: float) -> Path:
    """Render a mono sine WAV of (near-)exact `seconds` (fast, small)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=8000:duration={seconds}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "8000",
            "-ac",
            "1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return path


def _probe_seconds(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    return float(proc.stdout.strip())


def _stub_render(supervisor: Supervisor, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the GPU round-trip: render the requested take as sine."""

    def _render(
        self: Supervisor,
        segment_id: str,
        payload: dict[str, object],
        recovery_path: str | None,
    ) -> dict[str, object]:
        output = Path(str(payload["output_path"]))
        duration = payload["duration_seconds"]
        assert isinstance(duration, (int, float))
        _sine_wav(output, float(duration))
        return {"take_path": str(output)}

    monkeypatch.setattr(Supervisor, "_with_audio_gpu", _render)


def _run_with_old_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Supervisor, ProjectConfig, Path, dict[str, Any] | None]:
    """Ledger holds one take ending mid-window; the planner chains the next."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="joint")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    _sine_wav(run_dir / "audio" / "take_a.wav", 8.0)
    append_take(
        run_dir / "audio" / "takes.jsonl",
        AudioTake(
            take_id="take_0000",
            path="audio/take_a.wav",
            caption=_CAPTION,
            seed=0,
            covers_from=80.0,
            duration=8.0,
            segment_index=20,
        ),
    )
    segment = run_dir / "segments" / "000021"
    segment.mkdir(parents=True, exist_ok=True)
    supervisor = Supervisor(run_dir, config)
    _stub_render(supervisor, monkeypatch)
    return supervisor, config, segment, None


def _decision() -> EvolutionDecision:
    return EvolutionDecision(
        decision_index=21,
        destination=DirectorDestination(canonical_name="joint valley"),
    )


def test_straddling_boundary_assembles_exact_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boba seg21 geometry: 1.917 s sliver + joint must not shorten the preview."""
    supervisor, config, segment, _ = _run_with_old_take(tmp_path, monkeypatch)
    plan, _, _, _ = supervisor._ensure_audio_coverage(
        config, 21, "000021", segment, _VIDEO_TIME, _DURATION, _decision(), None
    )
    # Two serving takes prove the window really straddled the boundary
    # (a single-slice run would pass vacuously).
    assert plan.take_ids == ["take_0000", "take_0001"]
    assert abs(_probe_seconds(segment / "audio.wav") - _DURATION) <= 0.06


def test_single_slice_stays_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No boundary in the window: stream-copy path is untouched by compensation."""
    supervisor, config, segment, _ = _run_with_old_take(tmp_path, monkeypatch)
    supervisor._ensure_audio_coverage(config, 21, "000021", segment, 60.0, 4.0, _decision(), None)
    assert abs(_probe_seconds(segment / "audio.wav") - 4.0) <= 0.06

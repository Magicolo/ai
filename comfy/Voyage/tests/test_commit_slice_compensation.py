"""Video-only commit cover: all backends are deferred, `audio.wav` obsolete.

`Supervisor._cover_audio` never writes `audio.wav` at commit — ACE music
renders at finalize from the stored director decisions. For streaming
backends it derives the conditioning tail (`video_tail.mp4`) so the next
segment chains instead of going fresh; `fake` renders statelessly and
needs no tail. The render stub writes real testsrc video (no GPU); tail
derivation and frame counting are genuine ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage.fake_backends import FakeVideoBackend
from voyage.models import DirectorDestination, EvolutionDecision
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor

_TESTSRC = FakeVideoBackend()


def _render_testsrc_segment_video(path: Path, *, frames: int, fps: int) -> Path:
    """Real ffmpeg `testsrc` segment video (fake-backend-first, real media)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _TESTSRC.generate_segment(
        path, prompt="tail probe", seed=7, width=64, height=64, fps=fps, frames=frames
    )
    return path


def _count_frames(path: Path) -> int:
    """Frame count via ffprobe (decodes — exact, no container estimate)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ffprobe failed for {path}: {proc.stderr[-500:]}"
    return int(proc.stdout.strip())


def _decision() -> EvolutionDecision:
    return EvolutionDecision(
        decision_index=21,
        destination=DirectorDestination(canonical_name="joint valley"),
    )


def test_streaming_cover_derives_tail_without_audio_wav(tmp_path: Path) -> None:
    """Streaming cover: tail derived for chaining, never `audio.wav`."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="joint", video_backend="ltxv")
    config = read_effective_config(run_dir)
    segment = run_dir / "segments" / "000021"
    _render_testsrc_segment_video(segment / "video.mp4", frames=121, fps=24)
    supervisor = Supervisor(run_dir, config)
    stage_seconds: dict[str, float] = {}
    covered = supervisor._cover_audio(
        config, 21, "000021", segment, 0.0, 121 / 24, _decision(), None, stage_seconds
    )
    assert covered.take_action == "deferred"
    assert covered.audio_ahead == 0.0
    assert covered.audio_plan.take_ids == []
    assert not (segment / "audio.wav").exists()
    assert stage_seconds["audio"] >= 0.0
    tail = segment / "video_tail.mp4"
    assert tail.is_file()
    assert _count_frames(tail) == 25


def test_fake_cover_skips_tail_without_audio_wav(tmp_path: Path) -> None:
    """Fake cover: stateless, so no tail derive and never `audio.wav`."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="joint")
    config = read_effective_config(run_dir)
    assert config.video.backend == "fake"
    segment = run_dir / "segments" / "000021"
    segment.mkdir(parents=True, exist_ok=True)
    supervisor = Supervisor(run_dir, config)
    stage_seconds: dict[str, float] = {}
    covered = supervisor._cover_audio(
        config, 21, "000021", segment, 60.0, 4.0, _decision(), None, stage_seconds
    )
    assert covered.take_action == "deferred"
    assert covered.audio_ahead == 0.0
    assert covered.audio_plan.take_ids == []
    assert not (segment / "audio.wav").exists()
    assert not (segment / "video_tail.mp4").exists()
    assert stage_seconds["audio"] >= 0.0

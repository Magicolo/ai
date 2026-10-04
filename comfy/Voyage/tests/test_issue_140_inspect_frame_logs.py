"""Issue 140: the visual-inspector frame PNG must not orphan segment dirs.

`_inspect_frame_view` extracts one middle frame for the VLM read; the frame
is a transient view, not a committed artifact, so it lives under
`logs/inspect/<segment>.png` — segment dirs stay exactly the checksummed
artifacts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from voyage.persistence import read_effective_config


class _StubDirector:
    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert op == "inspect"
        assert Path(str(payload["frame_path"])).exists()
        return {"inspected": True, "scene_summary": "a neon dune"}


def _testsrc_video(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    return dest


def test_inspect_frame_view_writes_logs_not_segment_dir(tmp_path: Path) -> None:
    """The VLM view lands in logs/inspect/; the segment dir stays clean."""
    from tests.conftest import initialize_run_directory
    from voyage import paths
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="frameview", seed=7)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor._director = _StubDirector()  # type: ignore[assignment]
    prev_dir = paths.segment_dir(run_dir, "000000")
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_video = _testsrc_video(prev_dir / "video.mp4")

    scene = supervisor._inspect_frame_view(prev_dir, prev_video)

    assert scene == "a neon dune"
    assert not (prev_dir / "inspect_frame.png").exists()
    assert (run_dir / "logs" / "inspect" / "000000.png").exists()


def test_inspect_frame_view_failure_stays_silent(tmp_path: Path) -> None:
    """A missing video still returns '' without littering either directory."""

    from tests.conftest import initialize_run_directory
    from voyage import paths
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="frameview-fail", seed=7)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor._director = _StubDirector()  # type: ignore[assignment]
    prev_dir = paths.segment_dir(run_dir, "000001")
    prev_dir.mkdir(parents=True, exist_ok=True)

    assert supervisor._inspect_frame_view(prev_dir, prev_dir / "video.mp4") == ""
    assert not (prev_dir / "inspect_frame.png").exists()
    assert not (run_dir / "logs" / "inspect" / "000001.png").exists()

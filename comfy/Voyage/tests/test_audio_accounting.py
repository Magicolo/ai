"""Take/file duration accounting (issues 094, 095).

094: ACE renders are not sample-exact — the ledger must be clamped to the
probed file at commit (`take_short_seconds` metric), not the request.
095: take-joint windows must tile their nominal range exactly (the window
crossfade absorbs fade*(joints) of unique content otherwise, and the
final `-shortest` mux then trims video frames). All media is real ffmpeg
output (sine), no GPU, no mocks except where noted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from voyage.audio.planner import AudioTake, append_take
from voyage.errors import MediaError
from voyage.media import (
    assemble_segment_audio,
    build_final_audio,
    probed_take_seconds,
)
from voyage.persistence import read_effective_config


def _sine_wav(path: Path, seconds: float) -> Path:
    """Render a mono sine WAV of (near-)exact `seconds`."""
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
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    return float(proc.stdout.strip())


def test_probed_take_seconds_matches_file(tmp_path: Path) -> None:
    wav = _sine_wav(tmp_path / "take.wav", 2.0)
    assert probed_take_seconds(wav) == pytest.approx(2.0, abs=0.05)


def test_probed_take_seconds_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(MediaError):
        probed_take_seconds(tmp_path / "nope.wav")


def _write_segment(segment: Path, frames: int) -> None:
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "metrics.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")


def _write_takes(run_dir: Path, takes: list[AudioTake]) -> None:
    ledger = run_dir / "audio" / "takes.jsonl"
    for take in takes:
        append_take(ledger, take)


def _two_joint_run(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Two 2 s segments (timeline 4.0) with a take joint at 3.0 inside
    the second window ([1.9, 4.0]): slices 1.1 s + 1.0 s, fade 0.2."""
    run_dir = tmp_path / "run"
    segments = [run_dir / "segments" / "000000", run_dir / "segments" / "000001"]
    for segment in segments:
        _write_segment(segment, 48)
    take_a = _sine_wav(run_dir / "audio" / "take_0000.wav", 3.0)
    take_b = _sine_wav(run_dir / "audio" / "take_0001.wav", 5.0)
    _write_takes(
        run_dir,
        [
            AudioTake(
                take_id="take_0000",
                path="audio/take_0000.wav",
                caption="cap",
                seed=1,
                covers_from=0.0,
                duration=3.0,
                segment_index=0,
            ),
            AudioTake(
                take_id="take_0001",
                path="audio/take_0001.wav",
                caption="cap",
                seed=2,
                covers_from=3.0,
                duration=5.0,
                segment_index=0,
            ),
        ],
    )
    assert take_a.exists() and take_b.exists()
    return run_dir, segments


def test_joint_window_tiles_nominal_range(tmp_path: Path) -> None:
    """Issue 095: a take joint inside a window must not shorten the mix."""
    run_dir, segments = _two_joint_run(tmp_path)
    out = build_final_audio(
        run_dir,
        segments,
        tmp_path / "tmp",
        24,
        8000,
        1,
        overlap_fraction=0.10,
        overlap_cap_seconds=0.5,
    )
    assert _probe_seconds(out) == pytest.approx(4.0, abs=0.05)


def test_jointless_window_still_exact(tmp_path: Path) -> None:
    """Single-take control: no joints, no compensation, still exact."""
    run_dir = tmp_path / "run"
    segments = [run_dir / "segments" / "000000", run_dir / "segments" / "000001"]
    for segment in segments:
        _write_segment(segment, 48)
    _sine_wav(run_dir / "audio" / "take_0000.wav", 8.0)
    _write_takes(
        run_dir,
        [
            AudioTake(
                take_id="take_0000",
                path="audio/take_0000.wav",
                caption="cap",
                seed=1,
                covers_from=0.0,
                duration=8.0,
                segment_index=0,
            ),
        ],
    )
    out = build_final_audio(
        run_dir,
        segments,
        tmp_path / "tmp",
        24,
        8000,
        1,
        overlap_fraction=0.10,
        overlap_cap_seconds=0.5,
    )
    assert _probe_seconds(out) == pytest.approx(4.0, abs=0.05)


def test_assemble_joint_fade_override_is_exact(tmp_path: Path) -> None:
    """`joint_fade` makes assembly consume exactly what was compensated."""
    first = _sine_wav(tmp_path / "a.wav", 1.1)
    second = _sine_wav(tmp_path / "b.wav", 1.0)
    dest = tmp_path / "joint.wav"
    assemble_segment_audio([first, second], dest, 0.2, joint_fade=0.2)
    assert _probe_seconds(dest) == pytest.approx(1.1 + 1.0 - 0.2, abs=0.05)


def test_commit_clamps_short_take_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue 094: a short render clamps the ledger + records the metric."""
    import voyage.supervisor as supervisor_module
    from tests.conftest import initialize_run_directory
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="clamp")
    (run_dir / "audio").mkdir(exist_ok=True)

    short = _sine_wav(tmp_path / "short.wav", 1.0)

    def _probed_short(path: Path) -> float:
        return _probe_seconds(short)

    monkeypatch.setattr(supervisor_module, "probed_take_seconds", _probed_short)
    config, _digest = read_effective_config(run_dir)
    committed = Supervisor(run_dir, config).run_segments(1)
    assert committed == ["000000"]
    ledger = [
        json.loads(line)
        for line in (run_dir / "audio" / "takes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ledger and ledger[0]["duration"] == pytest.approx(1.0, abs=0.05)

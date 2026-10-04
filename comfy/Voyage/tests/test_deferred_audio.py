"""Deferred ACE music: commit stub + finalize-time take rendering (DESIGN §140).

Why: ltxv/causvid commit no ACE takes (full move to finalize); `usable`
segments carry timeline-exact silent stubs while `ensure_deferred_takes`
replays director decisions at finalize. ltx25/ltx23 joint path untouched.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from voyage.audio_finalize import (
    deferred_tail_frames,
    derive_conditioning_tail,
    ensure_deferred_takes,
    is_deferred_backend,
    write_deferred_stub_audio,
)
from voyage.config import ProjectConfig
from voyage.errors import MediaError
from voyage.fake_backends import FakeVideoBackend
from voyage.models import EvolutionDecision
from voyage.supervisor import Supervisor


def _write_sine_wav(path: Path, duration_seconds: float) -> None:
    """Stdlib sine stub: 48 kHz stereo s16le, exact duration."""
    rate, channels, freq = 48000, 2, 440.0
    frames = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for index in range(frames):
            sample = int(12000.0 * math.sin(2.0 * math.pi * freq * index / rate))
            wav.writeframes(struct.pack("<hh", sample, sample))


def _stub_render(payload: dict[str, Any], output_path: Path) -> None:
    """Render seam: stdlib sine at the payload duration (no GPU)."""
    duration = payload["duration_seconds"]
    assert isinstance(duration, (int, float))
    _write_sine_wav(output_path, float(duration))


def _segment_with_decision(
    run_dir: Path, index: int, frames: int, caption: str, energy: float
) -> Path:
    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    manifest = {
        "format": 1,
        "transition": {
            "decision": {
                "decision_index": index,
                "audio": {"music_caption": caption, "energy": energy},
            }
        },
        "metrics": {"frames": frames},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    return segment


def test_is_deferred_backend_only_ltxv_causvid() -> None:
    """Every streaming CUDA backend defers; only fake commits inline."""
    assert is_deferred_backend("ltxv")
    assert is_deferred_backend("causvid")
    assert is_deferred_backend("ltx25")
    assert is_deferred_backend("ltx23")
    assert not is_deferred_backend("fake")


def test_ensure_deferred_takes_forwards_sample_rate_and_channels(tmp_path: Path) -> None:
    """Takes render at the run's audio rate/channels, not a hardcoded pair.

    The commit stub uses the configured rate/channels, so finalize takes
    must match — a hardcoded 48k/2 would split-brain any run configured
    otherwise. Defaults stay 48k stereo when not given.
    """
    from voyage.audio_finalize import ensure_deferred_takes

    run_dir = tmp_path / "run"
    segment = _segment_with_decision(run_dir, 0, 96, "hollow winds", 0.4)
    seen: list[dict[str, Any]] = []

    def _capture(payload: dict[str, Any], output_path: Path) -> None:
        seen.append(dict(payload))
        _stub_render(payload, output_path)

    ensure_deferred_takes(
        run_dir=run_dir,
        usable=[segment],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_capture,
        sample_rate=44100,
        channels=1,
    )
    assert seen, "expected at least one rendered take"
    assert seen[0]["sample_rate"] == 44100
    assert seen[0]["channels"] == 1

    run_dir2 = tmp_path / "run2"
    segment2 = _segment_with_decision(run_dir2, 0, 96, "hollow winds", 0.4)
    seen2: list[dict[str, Any]] = []

    def _capture2(payload: dict[str, Any], output_path: Path) -> None:
        seen2.append(dict(payload))
        _stub_render(payload, output_path)

    ensure_deferred_takes(
        run_dir=run_dir2,
        usable=[segment2],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_capture2,
    )
    assert seen2[0]["sample_rate"] == 48000
    assert seen2[0]["channels"] == 2


def test_write_deferred_stub_audio_is_timeline_exact(tmp_path: Path) -> None:
    """Stub is silent 48k stereo s16le matching the segment duration."""
    from voyage.media_audio import probed_take_seconds

    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    out = write_deferred_stub_audio(segment, 4.0, 48000, 2)
    assert out == segment / "audio.wav"
    assert abs(probed_take_seconds(out) - 4.0) < 0.05


def test_cover_audio_deferred_branch_skips_take_path(tmp_path: Path) -> None:
    """Deferred branch writes the stub and never calls the take renderer."""
    from voyage.supervisor import Supervisor

    segment = tmp_path / "segments" / "000000"
    _render_testsrc_segment_video(segment / "video.mp4", frames=121, fps=24)

    @contextmanager
    def _stage(_kind: str, _label: str) -> Any:
        yield

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("take path must not run for deferred backends")

    fake_self = SimpleNamespace(_stage=_stage, _ensure_audio_coverage=_forbidden)
    config = SimpleNamespace(
        video=SimpleNamespace(backend="ltxv"),
        audio=SimpleNamespace(
            backend="acestep",
            music_style="ambient electronic",
            sample_rate=48000,
            channels=2,
        ),
    )
    covered = Supervisor._cover_audio(
        cast("Supervisor", fake_self),
        cast("ProjectConfig", config),
        0,
        "000000",
        segment,
        0.0,
        4.0,
        cast("EvolutionDecision", SimpleNamespace()),
        None,
        {},
    )
    assert covered.take_action == "deferred"
    assert covered.audio_plan.take_ids == []
    assert (segment / "audio.wav").exists()
    assert (segment / "video_tail.mp4").exists()
    assert _count_frames(segment / "video_tail.mp4") == 25


def test_ensure_deferred_takes_renders_and_ledgers(tmp_path: Path) -> None:
    """Replay renders takes covering both segments and appends the ledger."""
    seg0 = _segment_with_decision(tmp_path, 0, 121, "dark drone", 0.4)
    seg1 = _segment_with_decision(tmp_path, 1, 96, "dark drone evolving", 0.5)
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0, seg1],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    assert len(takes) >= 1
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert ledger.exists()
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == len(takes)
    for take in takes:
        assert (tmp_path / take["path"]).exists()
    coverage = max(t["covers_from"] + t["duration"] for t in takes)
    assert coverage >= 121 / 24.0 + 96 / 24.0 - 1e-6


def test_ensure_deferred_takes_fail_loud_on_render_error(tmp_path: Path) -> None:
    """A failed take render raises and appends nothing to the ledger."""
    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)

    def _boom(payload: dict[str, object], output_path: Path) -> None:
        raise RuntimeError("synthetic ACE failure")

    with pytest.raises(MediaError):
        ensure_deferred_takes(
            run_dir=tmp_path,
            usable=[seg0],
            source_fps=24.0,
            run_seed=0,
            render_take_fn=_boom,
        )
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert not ledger.exists() or ledger.read_text().strip() == ""


def test_ensure_deferred_takes_cover_stretched_timeline(tmp_path: Path) -> None:
    """Slow-mo stretch scales take coverage (1.5x: 4 s content -> 6 s music)."""
    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        stretch=1.5,
    )
    coverage = max(t["covers_from"] + t["duration"] for t in takes)
    assert coverage >= (96 / 24.0) * 1.5 - 1e-6


def _video_segment(
    run_dir: Path,
    index: int,
    frames: int = 96,
    fps: int = 24,
    caption: str = "dark drone",
    energy: float = 0.4,
) -> Path:
    """Committed-shape segment: DONE + testsrc video + stub audio + manifest."""
    import hashlib
    import subprocess

    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    duration = frames / fps
    video = segment / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={fps}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    write_deferred_stub_audio(segment, duration, 48000, 2)

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "format": 1,
        "transition": {
            "decision": {
                "decision_index": index,
                "audio": {"music_caption": caption, "energy": energy},
            }
        },
        "metrics": {"frames": frames},
        "checksums": {
            "video.mp4": _sha(video),
            "audio.wav": _sha(segment / "audio.wav"),
        },
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    (segment / "DONE").write_bytes(b"")
    return segment


def test_build_final_audio_deferred_fails_loud_on_stub(tmp_path: Path) -> None:
    """Deferred mix with no rendered takes raises instead of shipping silence."""
    from voyage.media_audio import build_final_audio

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    with pytest.raises(MediaError):
        build_final_audio(
            run_dir,
            [seg0],
            tmp_path / "tmp",
            24,
            48000,
            2,
            deferred=True,
        )


def test_build_final_audio_deferred_blends_rendered_takes(tmp_path: Path) -> None:
    """Deferred mix after `ensure_deferred_takes` blends real music."""
    from voyage.media_audio import build_final_audio, probed_take_seconds

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    seg1 = _video_segment(run_dir, 1, caption="dark drone evolving", energy=0.5)
    ensure_deferred_takes(
        run_dir=run_dir,
        usable=[seg0, seg1],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    dest = build_final_audio(
        run_dir,
        [seg0, seg1],
        tmp_path / "tmp",
        24,
        48000,
        2,
        deferred=True,
    )
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 8.0) < 0.3


def test_build_final_audio_stretch_uses_stretched_timeline(tmp_path: Path) -> None:
    """`stretch=1.5` mixes the 4 s segment into a 6 s slow-mo timeline."""
    from voyage.media_audio import build_final_audio, probed_take_seconds

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    ensure_deferred_takes(
        run_dir=run_dir,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        stretch=1.5,
    )
    dest = build_final_audio(
        run_dir,
        [seg0],
        tmp_path / "tmp",
        24,
        48000,
        2,
        deferred=True,
        stretch=1.5,
    )
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 6.0) < 0.6


def test_spawn_ace_render_fn_factory_exists() -> None:
    """Production ACE renderer factory is importable (spawning needs a GPU)."""
    from voyage import audio_finalize

    assert callable(audio_finalize.spawn_ace_render_fn)


def test_finalize_run_deferred_renders_takes_before_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`finalize_run(deferred_audio=True)` renders takes, then mixes/muxes."""
    import json as json_module

    from voyage import audio_finalize
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _video_segment(run_dir, 0)
    _video_segment(run_dir, 1, caption="dark drone evolving", energy=0.5)
    monkeypatch.setattr(
        audio_finalize,
        "spawn_ace_render_fn",
        lambda run_dir, models_dir, device="cuda:0": (_stub_render, lambda: None),
    )
    out = run_dir / "final.mp4"
    finalize_run(
        run_dir,
        out,
        width=320,
        height=240,
        fps=24,
        min_fps=0,
        min_width=0,
        min_height=0,
        use_model_pass=False,
        interp_multiplier=1,
        deferred_audio=True,
        seed=0,
    )
    assert out.exists()
    ledger = run_dir / "audio" / "takes.jsonl"
    assert ledger.exists()
    takes = [json_module.loads(line) for line in ledger.read_text().strip().splitlines()]
    assert len(takes) >= 1


def test_ensure_deferred_for_finalize_noops_on_complete_ledger(tmp_path: Path) -> None:
    """Re-finalize with a complete ledger spawns no worker and renders nothing."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    assert not audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
    )

    # `models_dir=None` would raise inside the spawn factory — returning
    # False proves the pending check fired first and no worker was spawned.
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        device="cuda:0",
    )
    assert rendered is False


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


def test_derive_conditioning_tail_writes_25f_tail(tmp_path: Path) -> None:
    """Commit-time tail derive: last 25 frames of the segment video.

    Why: the deferred branch skips the commit-time audio swap whose
    rebuild derived `video_tail.mp4`; without this derive the next
    segment's resident tail path is missing and the worker goes fresh
    (121f) instead of continuing (96f).
    """
    segment = tmp_path / "segments" / "000000"
    _render_testsrc_segment_video(segment / "video.mp4", frames=121, fps=24)
    derived = derive_conditioning_tail(segment, deferred_tail_frames("ltxv"))
    assert derived is True
    tail = segment / "video_tail.mp4"
    assert tail.is_file()
    assert _count_frames(tail) == 25


def test_derive_conditioning_tail_noops_when_present(tmp_path: Path) -> None:
    """An existing tail is adopted untouched (resume no-op)."""
    segment = tmp_path / "segments" / "000000"
    _render_testsrc_segment_video(segment / "video.mp4", frames=121, fps=24)
    assert derive_conditioning_tail(segment, deferred_tail_frames("ltxv")) is True
    before = (segment / "video_tail.mp4").stat().st_mtime_ns
    assert derive_conditioning_tail(segment, deferred_tail_frames("ltxv")) is False
    assert (segment / "video_tail.mp4").stat().st_mtime_ns == before


def test_derive_conditioning_tail_fails_loud_without_video(tmp_path: Path) -> None:
    """No segment video means fail loud, never a silent missing tail."""
    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    with pytest.raises(MediaError):
        derive_conditioning_tail(segment, deferred_tail_frames("ltxv"))


def test_deferred_tail_frames_match_worker_resume_counts() -> None:
    """Tail lengths pin each worker's resume derive (short tails adopt wrong)."""
    assert deferred_tail_frames("ltxv") == 25
    assert deferred_tail_frames("ltx25") == 25
    assert deferred_tail_frames("ltx23") == 25
    assert deferred_tail_frames("causvid") == 25
    assert deferred_tail_frames("causvid", overlap_frames=7) == 25
    assert deferred_tail_frames("causvid", overlap_frames=8) == 29
    with pytest.raises(MediaError):
        deferred_tail_frames("fake")

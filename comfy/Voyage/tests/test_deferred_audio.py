"""Always-deferred finalize: ledger-only music rendering (DESIGN §140).

Why: every backend commits video only — no per-segment `audio.wav`
previews. `ensure_deferred_takes` replays director decisions at finalize
(ACE-Step on cuda:0, or the fake sine worker via `audio_backend`
routing); `build_final_audio` blends the takes ledger and fails loud
without it.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from voyage.audio_finalize import (
    deferred_tail_frames,
    derive_conditioning_tail,
    ensure_deferred_takes,
)
from voyage.errors import MediaError
from voyage.fake_backends import FakeVideoBackend


def _write_sine_wav(path: Path, duration_seconds: float) -> None:
    """Stdlib sine stub: 48 kHz stereo s16le, exact duration.

    Batches the whole buffer into ONE `writeframes` call: the old
    per-sample loop paid a syscall per frame (~2 s per rendered second,
    ~60 s for this module alone).
    """
    rate, channels, freq = 48000, 2, 440.0
    frames = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = array.array(
        "h",
        (int(12000.0 * math.sin(2.0 * math.pi * freq * index / rate)) for index in range(frames)),
    )
    stereo = array.array("h", (sample for value in mono for sample in (value, value)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(stereo.tobytes())


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


def test_ensure_deferred_takes_forwards_sample_rate_and_channels(tmp_path: Path) -> None:
    """Takes render at the run's audio rate/channels, not a hardcoded pair.

    The finalize takes must match the run's configured rate/channels —
    a hardcoded 48k/2 would split-brain any run configured otherwise.
    Defaults stay 48k stereo when not given.
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


def test_deferred_render_pending_tracks_ledger_coverage(tmp_path: Path) -> None:
    """Dry walk is True fresh, False once the ledger covers the timeline."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    assert audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
    )
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


def _video_segment(
    run_dir: Path,
    index: int,
    frames: int = 96,
    fps: int = 24,
    caption: str = "dark drone",
    energy: float = 0.4,
) -> Path:
    """Committed-shape video-only segment: DONE + testsrc video + manifest."""
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
        "checksums": {"video.mp4": _sha(video)},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    (segment / "DONE").write_bytes(b"")
    return segment


def test_build_final_audio_fails_loud_without_takes(tmp_path: Path) -> None:
    """Ledger-only mix with no rendered takes raises instead of shipping silence."""
    from voyage.media_audio import build_final_audio

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    with pytest.raises(MediaError, match="no rendered takes"):
        build_final_audio(
            run_dir,
            [seg0],
            tmp_path / "tmp",
            24,
            48000,
            2,
        )


def test_build_final_audio_blends_rendered_takes(tmp_path: Path) -> None:
    """Mix after `ensure_deferred_takes` blends real music."""
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
        stretch=1.5,
    )
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 6.0) < 0.6


def test_spawn_ace_render_fn_factory_exists() -> None:
    """Production ACE renderer factory is importable (spawning needs a GPU)."""
    from voyage import audio_finalize

    assert callable(audio_finalize.spawn_ace_render_fn)


def test_spawn_fake_render_fn_factory_exists() -> None:
    """Offline fake renderer factory is importable (spawning needs only ffmpeg)."""
    from voyage import audio_finalize

    assert callable(audio_finalize.spawn_fake_render_fn)


def test_spawn_fake_render_fn_renders_sine(tmp_path: Path) -> None:
    """The fake factory renders real sine audio through a live worker."""
    from voyage import audio_finalize

    render, shutdown = audio_finalize.spawn_fake_render_fn(tmp_path)
    try:
        out = tmp_path / "take.wav"
        render(
            {
                "segment_id": "000000",
                "style": "dark drone",
                "energy": 0.4,
                "seed": 0,
                "duration_seconds": 1.0,
                "sample_rate": 48000,
                "channels": 2,
            },
            out,
        )
        assert out.exists() and out.stat().st_size > 0
    finally:
        shutdown()


def test_ensure_routing_fake_uses_fake_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`audio_backend="fake"` renders via the fake factory, never ACE."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    calls: list[str] = []

    def _recording_fake(run_dir: Path) -> tuple[object, object]:
        calls.append("fake")
        return (_stub_render, lambda: None)

    monkeypatch.setattr(audio_finalize, "spawn_fake_render_fn", _recording_fake)

    def _forbidden(run_dir: Path, models_dir: Path | str | None, device: str = "cuda:0") -> object:
        raise AssertionError("ACE factory must not run for audio_backend='fake'")

    monkeypatch.setattr(audio_finalize, "spawn_ace_render_fn", _forbidden)
    # `models_dir=None` would raise inside the ACE factory — rendering
    # proves the fake route (which needs no weights) was taken.
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        audio_backend="fake",
    )
    assert rendered is True
    assert calls == ["fake"]
    assert (tmp_path / "audio" / "takes.jsonl").exists()


def test_ensure_routing_acestep_uses_ace_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`audio_backend="acestep"` (default) renders via the ACE factory."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    seen: list[object] = []

    def _fake_spawn_fake(run_dir: Path) -> object:
        raise AssertionError("fake factory must not run for audio_backend='acestep'")

    def _recording_ace(
        run_dir: Path, models_dir: Path | str | None, device: str = "cuda:0"
    ) -> tuple[object, object]:
        seen.append(models_dir)
        return (_stub_render, lambda: None)

    monkeypatch.setattr(audio_finalize, "spawn_fake_render_fn", _fake_spawn_fake)
    monkeypatch.setattr(audio_finalize, "spawn_ace_render_fn", _recording_ace)
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir="/models",
    )
    assert rendered is True
    assert seen == ["/models"]


def test_ensure_routing_unknown_backend_fails_loud(tmp_path: Path) -> None:
    """An unknown audio backend raises instead of guessing a renderer."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    with pytest.raises(MediaError, match="unknown audio backend"):
        audio_finalize.ensure_deferred_for_finalize(
            run_dir=tmp_path,
            usable=[seg0],
            source_fps=24.0,
            stretch=1.0,
            run_seed=0,
            models_dir="/models",
            audio_backend="orchestra",
        )


def test_ensure_fake_end_to_end_needs_no_models(tmp_path: Path) -> None:
    """Fake routing renders takes through a live worker with no models_dir."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        audio_backend="fake",
    )
    assert rendered is True
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert ledger.exists() and ledger.read_text().strip() != ""


def test_finalize_run_always_renders_takes_before_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`finalize_run` renders takes, then mixes/muxes (no opt-in flag)."""
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
        upscale=1,
        interpolate=1,
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

    Why: the commit path holds no audio worker whose rebuild derived
    `video_tail.mp4`; without this derive the next segment's resident
    tail path is missing and the worker goes fresh (121f) instead of
    continuing (96f).
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
    """Tail lengths pin each streaming worker's resume derive."""
    assert deferred_tail_frames("ltxv") == 25
    assert deferred_tail_frames("ltx25") == 25
    assert deferred_tail_frames("ltx23") == 25
    assert deferred_tail_frames("causvid") == 25
    assert deferred_tail_frames("causvid", overlap_frames=7) == 25
    assert deferred_tail_frames("causvid", overlap_frames=8) == 29
    with pytest.raises(MediaError):
        deferred_tail_frames("fake")

"""SonicMaster mastering runner: routing, resample, duration gate, cache (Track B).

Covers the Track B choke contract without GPU/torch: the fixed
prompt/sampler constants, the chunk-window math, the
dubbed->mixed / music-only-or-silent->final_audio routing rule, the
44.1 kHz resample round-trip through `master_fn`, the 0.6 s
duration gate, and the cache invariant (slots hold pre-master bytes
only; the knob versions the fingerprints). Real ffmpeg throughout
(slim image ships it); synthetic stdlib sine fixtures only.
"""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import pytest

import voyage.mastering as mastering
from voyage.errors import MediaError
from voyage.final_mix_cache import (
    bed_fingerprint,
    load_bed_cache,
    load_music_cache,
    music_fingerprint,
    store_bed_cache,
    store_music_cache,
)
from voyage.mastering import (
    MASTERING_CHUNK_SECONDS,
    MASTERING_GUIDANCE,
    MASTERING_OVERLAP_SECONDS,
    MASTERING_PROMPT,
    MASTERING_SAMPLE_RATE,
    MASTERING_SAMPLER,
    MASTERING_SEED,
    MASTERING_STEPS,
    master_chunk_windows,
    master_fn,
    maybe_master_ship_audio,
    select_master_source,
)


def _write_sine_wav(
    path: Path,
    seconds: float,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    frequency: float = 440.0,
) -> Path:
    """Stdlib sine WAV, exact duration, one writeframes call (test-stub rule)."""
    import array

    frames = int(round(seconds * sample_rate))
    samples: array.array[int] = array.array("h")
    for index in range(frames):
        value = int(round(12000.0 * math.sin(2.0 * math.pi * frequency * index / sample_rate)))
        samples.extend([value] * channels)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def _write_silent_wav(
    path: Path, seconds: float, *, sample_rate: int = 48000, channels: int = 2
) -> Path:
    """Stdlib silence WAV, exact duration, one writeframes call."""
    frames = int(round(seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00" * frames * channels * 2)
    return path


def _probe_wav(path: Path) -> tuple[float, int, int]:
    """Probed (duration, sample_rate, channels) for a WAV file."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
    return (frames / rate if rate > 0 else 0.0, rate, channels)


def _render_clip_with_audio(video: Path, audio: Path, *, seconds: float = 2.0) -> Path:
    """Tiny real mp4 carrier (testsrc video + WAV audio, CPU-only)."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x64:rate=8:duration={seconds}",
            "-i",
            str(audio),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"clip render failed: {proc.stderr[-500:]}"
    return video


def test_constants_pin_fixed_prompt_and_sampler() -> None:
    assert MASTERING_PROMPT == "Master this track for me, please!"
    assert MASTERING_SAMPLE_RATE == 44100
    assert MASTERING_CHUNK_SECONDS == 30.0
    assert MASTERING_OVERLAP_SECONDS == 10.0
    assert MASTERING_SAMPLER == "euler"
    assert MASTERING_STEPS == 10
    assert MASTERING_GUIDANCE == 1.0
    assert MASTERING_SEED == 0


def test_chunk_windows_short_is_single() -> None:
    assert master_chunk_windows(3.0) == [(0.0, 3.0)]
    assert master_chunk_windows(30.0) == [(0.0, 30.0)]
    exact = master_chunk_windows(70.0)
    assert exact == [(0.0, 30.0), (20.0, 30.0), (40.0, 30.0)]


def test_chunk_windows_stride_and_tail_clamp() -> None:
    windows = master_chunk_windows(75.0)
    assert windows[0] == (0.0, 30.0)
    assert windows[1] == (20.0, 30.0)
    assert windows[2] == (40.0, 30.0)
    assert windows[3][0] == 60.0
    assert windows[3][0] + windows[3][1] == pytest.approx(75.0)
    for first, second in zip(windows, windows[1:], strict=False):
        assert second[0] == pytest.approx(first[0] + 20.0)
        assert first[0] + first[1] - second[0] == pytest.approx(10.0)


def test_chunk_windows_reject_bad_geometry() -> None:
    with pytest.raises(ValueError):
        master_chunk_windows(0.0)
    with pytest.raises(ValueError):
        master_chunk_windows(10.0, chunk_seconds=0.0)
    with pytest.raises(ValueError):
        master_chunk_windows(10.0, chunk_seconds=5.0, overlap_seconds=5.0)


def test_select_master_source_routes_all_three_paths(tmp_path: Path) -> None:
    mixed = _write_sine_wav(tmp_path / "mixed.wav", 1.0, frequency=440.0)
    music = _write_sine_wav(tmp_path / "music.wav", 1.0, frequency=660.0)
    silent = _write_silent_wav(tmp_path / "silent.wav", 1.0)
    assert select_master_source(mixed, music) == mixed
    assert select_master_source(None, music) == music
    assert select_master_source(None, silent) == silent


def test_master_fn_resample_round_trip(tmp_path: Path) -> None:
    source = _write_sine_wav(tmp_path / "in.wav", 3.0)
    dest = tmp_path / "out.wav"
    timings: dict[str, float] = {}
    result = master_fn(source, dest, 48000, 2, timings=timings, progress=None)
    assert result == dest
    duration, rate, channels = _probe_wav(dest)
    assert rate == 48000
    assert channels == 2
    assert abs(duration - 3.0) <= 0.6
    assert timings["master_s"] >= 0.0
    assert timings["master_resample_in_s"] >= 0.0
    assert timings["master_render_s"] >= 0.0
    assert timings["master_join_s"] >= 0.0
    assert timings["master_resample_out_s"] >= 0.0
    leftovers = list(tmp_path.glob("master_chunk_*")) + list(tmp_path.glob("voyage-master-*"))
    assert leftovers == []


def test_master_fn_threads_fixed_chunk_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, object]] = []
    original = mastering._render_mastered_chunk

    def _recording(
        chunk_in: Path,
        chunk_out: Path,
        *,
        prompt: str = mastering.MASTERING_PROMPT,
        sampler: str = mastering.MASTERING_SAMPLER,
        steps: int = mastering.MASTERING_STEPS,
        guidance: float = mastering.MASTERING_GUIDANCE,
        seed: int = mastering.MASTERING_SEED,
    ) -> Path:
        seen.append(
            {
                "prompt": prompt,
                "sampler": sampler,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            }
        )
        return original(chunk_in, chunk_out)

    monkeypatch.setattr(mastering, "_render_mastered_chunk", _recording)
    monkeypatch.setattr(mastering, "MASTERING_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(mastering, "MASTERING_OVERLAP_SECONDS", 1.0)
    source = _write_sine_wav(tmp_path / "in.wav", 5.0)
    master_fn(source, tmp_path / "out.wav", 48000, 2)
    expected = len(master_chunk_windows(5.0, 2.0, 1.0))
    assert len(seen) == expected
    assert seen
    for record in seen:
        assert record["prompt"] == "Master this track for me, please!"
        assert record["sampler"] == "euler"
        assert record["steps"] == 10
        assert record["guidance"] == 1.0
        assert record["seed"] == 0


def test_master_fn_duration_gate_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _truncating(chunk_in: Path, chunk_out: Path, **kwargs: object) -> Path:
        del chunk_in, kwargs
        _write_silent_wav(chunk_out, 0.5, sample_rate=44100, channels=2)
        return chunk_out

    monkeypatch.setattr(mastering, "_render_mastered_chunk", _truncating)
    source = _write_sine_wav(tmp_path / "in.wav", 3.0)
    with pytest.raises(MediaError, match="drifts from input"):
        master_fn(source, tmp_path / "out.wav", 48000, 2)


def test_master_fn_rejects_bad_shape(tmp_path: Path) -> None:
    source = _write_sine_wav(tmp_path / "in.wav", 1.0)
    with pytest.raises(ValueError, match="channels"):
        master_fn(source, tmp_path / "out.wav", 48000, 3)
    with pytest.raises(ValueError, match="sample_rate"):
        master_fn(source, tmp_path / "out.wav", 0, 2)
    with pytest.raises(MediaError):
        master_fn(tmp_path / "missing.wav", tmp_path / "out.wav", 48000, 2)


def test_maybe_master_ship_disabled_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _write_sine_wav(tmp_path / "music.wav", 2.0)
    ship = _render_clip_with_audio(tmp_path / "ship.mp4", audio)
    calls: list[Path] = []
    monkeypatch.setattr(mastering, "MASTERING_ENABLED", False)

    def _spy(
        input_wav: Path,
        output_wav: Path,
        sample_rate: int,
        channels: int,
        timings: dict[str, float] | None = None,
        progress: object = None,
    ) -> Path:
        del output_wav, sample_rate, channels, timings, progress
        calls.append(input_wav)
        raise AssertionError("master_fn must not run while disabled")

    monkeypatch.setattr(mastering, "master_fn", _spy)
    assert maybe_master_ship_audio(ship, tmp_path, 48000, 2) == ship
    assert calls == []


def test_maybe_master_ship_enabled_masters_video_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _write_sine_wav(tmp_path / "music.wav", 2.0)
    ship = _render_clip_with_audio(tmp_path / "ship.mp4", audio)
    monkeypatch.setattr(mastering, "MASTERING_ENABLED", True)
    out = maybe_master_ship_audio(ship, tmp_path, 48000, 2)
    assert out != ship
    assert out.is_file() and out.stat().st_size > 0
    assert out.name == "final_mastered.mp4"


def test_maybe_master_ship_explicit_enabled_overrides_knob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _write_sine_wav(tmp_path / "music.wav", 2.0)
    ship = _render_clip_with_audio(tmp_path / "ship.mp4", audio)
    monkeypatch.setattr(mastering, "MASTERING_ENABLED", False)
    out = maybe_master_ship_audio(ship, tmp_path, 48000, 2, enabled=True)
    assert out != ship
    assert out.is_file() and out.stat().st_size > 0
    assert out.name == "final_mastered.mp4"


def test_maybe_master_ship_explicit_disabled_overrides_knob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _write_sine_wav(tmp_path / "music.wav", 2.0)
    ship = _render_clip_with_audio(tmp_path / "ship.mp4", audio)
    monkeypatch.setattr(mastering, "MASTERING_ENABLED", True)
    assert maybe_master_ship_audio(ship, tmp_path, 48000, 2, enabled=False) == ship


def test_fingerprints_ignore_the_mastering_knob(tmp_path: Path) -> None:
    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    (segment / "DONE").write_text("", encoding="utf-8")
    (segment / "video.mp4").write_bytes(b"\x00" * 64)
    usable = [segment]
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "takes.jsonl").write_text("", encoding="utf-8")
    plain = music_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.10,
        overlap_cap_seconds=6.0,
        audio_stretch=1.0,
        audio_fps=24.0,
    )
    knobbed = music_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.10,
        overlap_cap_seconds=6.0,
        audio_stretch=1.0,
        audio_fps=24.0,
        mastering_enabled=True,
    )
    assert plain == knobbed
    plain_bed = bed_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        backend="fake",
        model_size="small",
        seed=11,
        caption_override=None,
        music_digest=plain,
        dual_pan=False,
    )
    knobbed_bed = bed_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        backend="fake",
        model_size="small",
        seed=11,
        caption_override=None,
        music_digest=plain,
        dual_pan=False,
        mastering_enabled=True,
    )
    assert plain_bed == knobbed_bed


def test_cache_slots_never_hold_mastered_bytes(tmp_path: Path) -> None:
    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    (segment / "DONE").write_text("", encoding="utf-8")
    (segment / "video.mp4").write_bytes(b"\x00" * 64)
    usable = [segment]
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "takes.jsonl").write_text("", encoding="utf-8")
    digest = music_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.10,
        overlap_cap_seconds=6.0,
        audio_stretch=1.0,
        audio_fps=24.0,
    )
    pre_master = _write_sine_wav(tmp_path / "pre.wav", 2.0, frequency=440.0)
    store_music_cache(tmp_path, digest, pre_master)
    assert load_music_cache(tmp_path, digest) is not None
    mastered = _write_sine_wav(tmp_path / "mastered.wav", 2.0, frequency=880.0)
    slot = tmp_path / "audio" / "final_music_cache.wav"
    assert slot.read_bytes() == pre_master.read_bytes()
    assert slot.read_bytes() != mastered.read_bytes()
    stem_dir = tmp_path / "audio" / "sfx"
    stem_dir.mkdir(parents=True, exist_ok=True)
    (stem_dir / "sfx.jsonl").write_text("{}\n", encoding="utf-8")
    bed_digest = bed_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        backend="fake",
        model_size="small",
        seed=11,
        caption_override=None,
        music_digest=digest,
        dual_pan=False,
    )
    store_bed_cache(tmp_path, bed_digest, pre_master, source_seconds=2.0)
    loaded = load_bed_cache(tmp_path, bed_digest)
    assert loaded is not None
    assert loaded[0].read_bytes() == pre_master.read_bytes()
    assert loaded[0].read_bytes() != mastered.read_bytes()

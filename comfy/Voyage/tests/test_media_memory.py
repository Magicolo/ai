"""Rank-1 media/RAM perf issues 043/044/045 (CPU-only, synthetic media).

043: final publish must stream staged MP4s to the destination (atomic_copy)
    instead of `staged.read_bytes()` into the heap.
044: frame sampling must stream the rawvideo decode (Popen + incremental
    reads, early stop) with one owner per frame, and the estimate-drift
    fallback must be capped instead of full-decoding the clip.
045: SFX conditioning must use a single ffmpeg pass (pure argv math +
    window/padding/byte-cap helpers are CPU-testable; torch/GPU execution
    itself is a needs-GPU-box residual).

Real ffmpeg throughout (slim image ships it); synthetic testsrc clips only.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import initialize_run_directory
from voyage.persistence import read_effective_config


def _make_clip(dest: Path, size: str = "320x240", rate: int = 8, duration: float = 4.0) -> Path:
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
            f"testsrc=size={size}:rate={rate}:duration={duration}",
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


def _reference_full_decode(clip: Path, width: int, height: int) -> list[np.ndarray]:  # type: ignore[type-arg]
    """The pre-044 algorithm: capture the whole rawvideo stdout, then split."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"scale={width}:{height}",
            "-vsync",
            "0",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    raw = proc.stdout
    stride = width * height * 3
    total = len(raw) // stride
    assert total > 0
    return [
        np.frombuffer(raw[index * stride : (index + 1) * stride], dtype=np.uint8).reshape(
            height, width, 3
        )
        for index in range(total)
    ]


# ---------------------------------------------------------------------------
# 043: atomic_copy + publish path never holds the whole MP4 in RAM
# ---------------------------------------------------------------------------


def test_atomic_copy_is_byte_identical(tmp_path: Path) -> None:
    from voyage.atomic import atomic_copy

    source = tmp_path / "source.mp4"
    source.write_bytes(os.urandom(5 * 1024 * 1024))
    dest = tmp_path / "nested" / "final.mp4"
    assert atomic_copy(source, dest) == dest
    assert dest.read_bytes() == source.read_bytes()
    assert list(tmp_path.rglob("*.partial")) == []


def test_atomic_copy_empty_file_round_trips(tmp_path: Path) -> None:
    from voyage.atomic import atomic_copy

    source = tmp_path / "empty.mp4"
    source.write_bytes(b"")
    dest = tmp_path / "copy.mp4"
    atomic_copy(source, dest)
    assert dest.read_bytes() == b""


def test_atomic_copy_cleans_partial_on_missing_source(tmp_path: Path) -> None:
    from voyage.atomic import atomic_copy

    with pytest.raises((OSError, FileNotFoundError)):
        atomic_copy(tmp_path / "nope.mp4", tmp_path / "dest.mp4")
    assert list(tmp_path.rglob("*.partial")) == []


def _commit_two_segments(run_dir: Path) -> None:
    from voyage.supervisor import Supervisor

    initialize_run_directory(run_dir, run_id="mem043", style="pastel neon line-art, peaceful")
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _guard_mp4_read_bytes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Fail any `Path.read_bytes()` on an MP4 (the 043 publish defect)."""
    calls: list[Path] = []
    real_read_bytes = Path.read_bytes

    def _guard(self: Path) -> bytes:
        if self.suffix == ".mp4":
            raise AssertionError(f"publish path read {self} into RAM (issue 043)")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guard)
    return calls


def test_finalize_fastpath_publishes_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage.media import finalize_run, probe, validate_video

    run_dir = tmp_path / "run"
    _commit_two_segments(run_dir)
    _guard_mp4_read_bytes(monkeypatch)
    out = tmp_path / "final-copy.mp4"
    assert finalize_run(run_dir, out, min_fps=0, min_width=0, min_height=0).exists()
    probed = validate_video(out, 768, 432, 24)
    assert probed["fps"] == pytest.approx(24.0, abs=0.5)
    duration = float(probe(out).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(4.0, abs=0.15)


@pytest.mark.slow
def test_finalize_reencode_path_publishes_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default floors force the re-encode tail — same streaming publish."""
    from voyage.media import finalize_run, validate_video

    run_dir = tmp_path / "run"
    _commit_two_segments(run_dir)
    _guard_mp4_read_bytes(monkeypatch)
    out = tmp_path / "final-lift.mp4"
    assert finalize_run(run_dir, out).exists()
    validate_video(out, 1216, 704, 24)


# ---------------------------------------------------------------------------
# 044: streamed decode, one owner per frame, capped drift fallback
# ---------------------------------------------------------------------------


def test_sample_frames_matches_reference_decode(tmp_path: Path) -> None:
    """Identical sampled frames vs the old capture-everything algorithm."""
    from voyage.vision.metrics import sample_frames, select_frame_indices

    clip = _make_clip(tmp_path / "clip.mp4")
    selected = sample_frames(clip, count=3, width=160)
    reference = _reference_full_decode(clip, 160, 120)
    picks = select_frame_indices(len(reference), 3)
    assert len(selected) == 3
    for got, pick in zip(selected, picks, strict=True):
        np.testing.assert_array_equal(got, reference[pick])


def test_sample_frames_never_uses_capture_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock-gate: the decode path must not `subprocess.run` video bytes."""
    import subprocess as subprocess_module

    from voyage.vision.metrics import sample_frames

    clip = _make_clip(tmp_path / "clip.mp4")
    real_run = subprocess_module.run

    def _forbidden(*args: object, **kwargs: object) -> object:
        argv = args[0] if args else []
        if isinstance(argv, list) and "rawvideo" in argv:
            raise AssertionError("sample_frames captured video bytes via subprocess.run")
        return real_run(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(subprocess_module, "run", _forbidden)
    try:
        frames = sample_frames(clip, count=3, width=160)
    finally:
        monkeypatch.setattr(subprocess_module, "run", real_run)
    assert len(frames) == 3
    assert all(frame.shape == (120, 160, 3) for frame in frames)


def test_sample_frames_single_frame_never_uses_capture_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess as subprocess_module

    from voyage.vision.metrics import sample_frames

    clip = _make_clip(tmp_path / "clip.mp4")
    real_run = subprocess_module.run

    def _forbidden(*args: object, **kwargs: object) -> object:
        argv = args[0] if args else []
        if isinstance(argv, list) and "rawvideo" in argv:
            raise AssertionError("sample_frames captured video bytes via subprocess.run")
        return real_run(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(subprocess_module, "run", _forbidden)
    (frame,) = sample_frames(clip, count=1, width=160)
    assert frame.shape == (120, 160, 3)


def test_sample_frames_drifted_estimate_still_serves_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wildly wrong probe estimate must not full-decode the clip."""
    import voyage.vision.metrics as metrics_module
    from voyage.vision.metrics import sample_frames

    clip = _make_clip(tmp_path / "clip.mp4")
    monkeypatch.setattr(metrics_module, "estimate_frame_total", lambda _s, _f: 100000)
    frames = sample_frames(clip, count=3, width=160)
    assert len(frames) == 3
    assert all(frame.shape == (120, 160, 3) for frame in frames)


def test_sample_frames_missing_estimate_serves_identical_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.vision.metrics as metrics_module
    from voyage.vision.metrics import sample_frames

    clip = _make_clip(tmp_path / "clip.mp4")
    expected = sample_frames(clip, count=3, width=160)
    monkeypatch.setattr(metrics_module, "estimate_frame_total", lambda _s, _f: None)
    fallback = sample_frames(clip, count=3, width=160)
    assert len(fallback) == 3
    for got, want in zip(fallback, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_sample_frames_bounded_rss(tmp_path: Path) -> None:
    """Resource-tracked smoke: sampling a small clip grows peak RSS modestly."""
    import resource

    from voyage.vision.metrics import sample_frames

    clip = _make_clip(tmp_path / "clip.mp4", duration=8.0)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    frames = sample_frames(clip, count=3, width=160)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert len(frames) == 3
    # Generous margin (kilobytes on Linux): the streamed path holds ~3
    # scaled frames (~170 KiB), never the whole rawvideo stdout.
    assert (after - before) / 1024 < 120


def test_fallback_decode_cap_exists() -> None:
    """The drift fallback carries an explicit decode-length ceiling."""
    import voyage.vision.metrics as metrics_module

    cap = metrics_module.FALLBACK_MAX_FRAMES
    assert isinstance(cap, int) and cap >= 512
    # Real segments are < 200 frames; the cap only binds pathological clips.
    assert cap <= 8192


# ---------------------------------------------------------------------------
# 045: single-pass conditioning — pure helpers (CPU-testable)
# ---------------------------------------------------------------------------


def test_window_frame_counts_are_exact() -> None:
    from voyage.audio.mmaudio_sfx import window_frame_counts

    assert window_frame_counts(8.0) == (64, 200)
    assert window_frame_counts(1.0) == (8, 25)
    with pytest.raises(ValueError, match="duration_seconds"):
        window_frame_counts(0.0)
    with pytest.raises(ValueError, match="duration_seconds"):
        window_frame_counts(61.0)


def test_effective_stacked_bytes_and_cap() -> None:
    from voyage.audio.mmaudio_sfx import (
        MAX_STACKED_BYTES,
        check_stacked_bytes,
        effective_stacked_bytes,
    )

    eight = effective_stacked_bytes(8.0)
    # 200 sync float32 @224² + 64 clip float32 @384².
    assert eight == 200 * 3 * 224 * 224 * 4 + 64 * 3 * 384 * 384 * 4
    assert check_stacked_bytes(8.0) == eight
    assert effective_stacked_bytes(60.0) <= MAX_STACKED_BYTES
    assert check_stacked_bytes(60.0) == effective_stacked_bytes(60.0)
    # Seconds beyond the window ceiling reject on duration first.
    with pytest.raises(ValueError, match="duration_seconds"):
        check_stacked_bytes(61.0)


def test_stacked_byte_cap_rejects_absurd_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The byte cap binds geometry, not just seconds (what MAX_WINDOW misses)."""
    import voyage.audio.mmaudio_sfx as sfx_module
    from voyage.audio.mmaudio_sfx import check_stacked_bytes

    monkeypatch.setattr(sfx_module, "SYNC_SIZE", 4096)
    with pytest.raises(ValueError, match="stacked bytes"):
        check_stacked_bytes(8.0)


def test_tile_to_length_repeats_whole_batches() -> None:
    from voyage.audio.mmaudio_sfx import tile_to_length

    assert tile_to_length([1, 2, 3], 7) == [1, 2, 3, 1, 2, 3, 1]
    assert tile_to_length([1, 2, 3], 3) == [1, 2, 3]
    assert tile_to_length([1, 2, 3], 0) == []
    assert tile_to_length([], 0) == []
    with pytest.raises(ValueError, match="empty"):
        tile_to_length([], 16)
    with pytest.raises(ValueError, match="length"):
        tile_to_length([1], -1)


def test_pad_to_sync_floor_tiles_short_tails() -> None:
    from voyage.audio.mmaudio_sfx import MIN_SYNC_FRAMES, pad_to_sync_floor

    clip = [f"c{i}" for i in range(4)]
    sync = [f"s{i}" for i in range(12)]
    padded_clip, padded_sync, effective = pad_to_sync_floor(clip, sync, 0.5)
    assert len(padded_sync) == MIN_SYNC_FRAMES == 16
    assert len(padded_clip) == 5  # duration-implied clip count at 0.64 s
    assert effective == pytest.approx(16 / 25.0)
    # Whole-batch tiling preserves temporal order.
    assert padded_sync[:12] == sync
    assert padded_sync[12:] == sync[:4]


def test_pad_to_sync_floor_passes_through_healthy_windows() -> None:
    from voyage.audio.mmaudio_sfx import pad_to_sync_floor

    clip = [f"c{i}" for i in range(64)]
    sync = [f"s{i}" for i in range(200)]
    out_clip, out_sync, effective = pad_to_sync_floor(clip, sync, 8.0)
    assert out_clip == clip and out_sync == sync and effective == 8.0


def test_pad_to_sync_floor_rejects_unservable_windows() -> None:
    from voyage.audio.mmaudio_sfx import pad_to_sync_floor

    with pytest.raises(ValueError, match="unservable"):
        pad_to_sync_floor(["c0"], [], 0.5)
    with pytest.raises(ValueError, match="unservable"):
        pad_to_sync_floor([], [], 0.5)


def test_single_pass_argv_is_one_ffmpeg_at_top_geometry() -> None:
    from voyage.workers.sfx_mmaudio import sfx_single_pass_argv

    argv = sfx_single_pass_argv("vid.mp4", 1.5, 8.0, 200)
    assert argv[0] == "ffmpeg"
    assert argv.count("ffmpeg") == 1
    assert "-vf" in argv
    vf = argv[argv.index("-vf") + 1]
    assert "fps=25" in vf and "scale=384:384" in vf
    assert argv[argv.index("-frames:v") + 1] == "200"
    assert argv[argv.index("-ss") + 1] == "1.500000"


def test_derive_clip_indices_is_evenly_spaced() -> None:
    from voyage.workers.sfx_mmaudio import derive_clip_indices

    assert derive_clip_indices(200, 64)[0] == 0
    assert derive_clip_indices(200, 64)[-1] == 199
    assert len(derive_clip_indices(200, 64)) == 64
    assert derive_clip_indices(200, 64) == sorted(derive_clip_indices(200, 64))
    assert derive_clip_indices(100, 1) == [50]
    with pytest.raises(ValueError, match="frame_total"):
        derive_clip_indices(0, 4)
    with pytest.raises(ValueError, match="wanted_clip"):
        derive_clip_indices(200, 0)


@pytest.mark.skipif(
    os.environ.get("VOYAGE_REQUIRE_TORCH_TESTS") != "1",
    reason="needs torch (GPU box); pure helpers above cover the CPU gates",
)
def test_single_pass_extract_shapes_on_testsrc(tmp_path: Path) -> None:
    """Integration (needs-GPU-box residual): one spawn, exact shapes."""
    pytest.importorskip("torch")
    from voyage.workers.sfx_mmaudio import _extract_frames

    clip = _make_clip(tmp_path / "sfx.mp4", size="768x512", rate=24, duration=2.0)
    clip_frames, sync_frames, resolved = _extract_frames(str(clip), 0.0, 2.0)
    assert len(clip_frames) == 16 and len(sync_frames) == 50
    assert resolved == pytest.approx(2.0, abs=0.05)

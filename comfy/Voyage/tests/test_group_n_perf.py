"""Issue group N (perf/media pipeline): probe/decode/sweep dedupe (CPU-only).

Pins the group-N fixes with stubbed subprocess/PIL/sha seams — no
ffmpeg, no torch, no GPU. Real-filesystem `tmp_path` runs with fake
bytes; decode stubs stand in for ffmpeg.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 70,
    checksum: str = "abc123",
) -> Path:
    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": frames},
        "checksums": {"video.mp4": checksum},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


class _StubCompleted:
    """Minimal `subprocess.CompletedProcess` shape (returncode + streams)."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _StubSubprocess:
    """Raising-by-default subprocess double; tests arm `run` per case."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.handler: Any = None

    def run(self, argv: Any, **kwargs: Any) -> _StubCompleted:
        self.calls.append([str(part) for part in argv])
        assert self.handler is not None, "stub subprocess has no handler"
        return self.handler(argv, kwargs)


def _arm_video_common(monkeypatch: pytest.MonkeyPatch, stub: _StubSubprocess) -> None:
    import voyage.workers.video_common as video_common

    monkeypatch.setattr(video_common, "subprocess", stub)
    video_common._FRAME_COUNT_CACHE.clear()


def test_count_video_frames_known_frames_skips_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest frames return without spawning anything (issue 249)."""
    import voyage.workers.video_common as video_common

    stub = _StubSubprocess()
    _arm_video_common(monkeypatch, stub)
    assert video_common.count_video_frames(tmp_path / "video.mp4", known_frames=96) == 96
    assert stub.calls == []
    with pytest.raises(ValueError):
        video_common.count_video_frames(tmp_path / "video.mp4", known_frames=0)
    with pytest.raises(TypeError):
        video_common.count_video_frames(tmp_path / "video.mp4", known_frames=True)  # type: ignore[arg-type]


def test_count_video_frames_header_first_then_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Header `nb_frames` wins; repeats hit the identity cache (issue 249)."""
    import voyage.workers.video_common as video_common

    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake-video")
    stub = _StubSubprocess()

    def _handler(argv: Any, kwargs: Any) -> _StubCompleted:
        flat = " ".join(str(part) for part in argv)
        assert "-count_frames" not in flat
        return _StubCompleted(0, "232\n")

    stub.handler = _handler
    _arm_video_common(monkeypatch, stub)
    assert video_common.count_video_frames(source) == 232
    assert video_common.count_video_frames(source) == 232
    assert len(stub.calls) == 1


def test_count_video_frames_decode_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty header falls back to the decode path (issue 249)."""
    import voyage.workers.video_common as video_common

    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake-video")
    stub = _StubSubprocess()
    seen: list[str] = []

    def _handler(argv: Any, kwargs: Any) -> _StubCompleted:
        flat = " ".join(str(part) for part in argv)
        seen.append(flat)
        if "-count_frames" in flat:
            return _StubCompleted(0, "121\n")
        return _StubCompleted(0, "N/A\n")

    stub.handler = _handler
    _arm_video_common(monkeypatch, stub)
    assert video_common.count_video_frames(source) == 121
    assert any("-count_frames" in entry for entry in seen)


def test_tail_start_frame_threads_total_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known totals skip the probe; shorts still fail loud (issue 249)."""
    import voyage.workers.video_common as video_common

    stub = _StubSubprocess()
    _arm_video_common(monkeypatch, stub)
    assert video_common.tail_start_frame(tmp_path / "v.mp4", 25, total_frames=100) == 75
    assert stub.calls == []
    with pytest.raises(ValueError):
        video_common.tail_start_frame(tmp_path / "v.mp4", 25, total_frames=24)


def test_presented_frames_known_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known counts skip the probe; header results cache (issue 249)."""
    import voyage.media as media_module

    media_module._PRESENTED_FRAMES_CACHE.clear()
    assert media_module.presented_frames(tmp_path / "final.mp4", known_frames=428) == 428
    assert media_module.presented_frames(tmp_path / "final.mp4", known_frames=0) is None
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake-video")
    calls: list[list[str]] = []

    def _fake_run_capture(argv: list[str]) -> _StubCompleted:
        calls.append(argv)
        return _StubCompleted(0, "428\n")

    monkeypatch.setattr(media_module, "run_capture", _fake_run_capture)
    assert media_module.presented_frames(video) == 428
    assert media_module.presented_frames(video) == 428
    assert len(calls) == 1
    assert all("-count_frames" not in " ".join(argv) for argv in calls)


def test_morph_probe_frames_header_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trim probes try the header before decoding (issue 249)."""
    import voyage.augment_morph as morph

    morph._MORPH_FRAME_COUNT_CACHE.clear()
    video = tmp_path / "trim.mp4"
    video.write_bytes(b"fake-video")
    stub = _StubSubprocess()

    def _handler(argv: Any, kwargs: Any) -> _StubCompleted:
        flat = " ".join(str(part) for part in argv)
        assert "-count_frames" not in flat
        return _StubCompleted(0, "13\n")

    stub.handler = _handler
    monkeypatch.setattr(morph, "subprocess", stub)
    assert morph._probe_frames(video) == 13
    assert morph._probe_frames(video) == 13
    assert len(stub.calls) == 1


def test_audio_duration_cached_per_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged audio files probe once no matter how many stages ask (250)."""
    import voyage.media_audio as media_audio

    media_audio._AUDIO_DURATION_CACHE.clear()
    take = tmp_path / "take.wav"
    take.write_bytes(b"fake-audio")
    probes: list[Path] = []
    real_probe = media_audio.probe

    def _counting(path: Path) -> dict[str, Any]:
        probes.append(path)
        return {"format": {"duration": 4.0}}

    monkeypatch.setattr(media_audio, "probe", _counting)
    assert media_audio._audio_duration_seconds(take) == 4.0
    assert media_audio.probed_take_seconds(take) == 4.0
    assert media_audio._audio_duration_seconds(take) == 4.0
    assert len(probes) == 1
    take.write_bytes(b"fake-audio-changed")
    assert media_audio._audio_duration_seconds(take) == 4.0
    assert len(probes) == 2
    assert real_probe is not None


def test_weights_key_memoizes_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeat calls with unchanged weights hash nothing (issue 250)."""
    import voyage.augment_finalize as finalize_module

    finalize_module._WEIGHTS_KEY_CACHE.clear()
    interp = tmp_path / "rife.safetensors"
    esrgan = tmp_path / "esrgan.pth"
    interp.write_bytes(b"interp-weights")
    esrgan.write_bytes(b"esrgan-weights")

    class _Weights:
        def interp_leg(self, backend: str) -> Path:
            return interp

        @property
        def realesrgan(self) -> Path:
            return esrgan

    hashes: list[str] = []
    import voyage.hashing as hashing

    real_sha = hashing.sha256_file

    def _counting(path: Path) -> str:
        hashes.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(finalize_module, "sha256_file", _counting)
    first = finalize_module.weights_key_for(_Weights(), "rife")
    second = finalize_module.weights_key_for(_Weights(), "rife")
    assert first == second and "|" in first
    assert len(hashes) == 2
    esrgan.write_bytes(b"esrgan-weights-v2")
    third = finalize_module.weights_key_for(_Weights(), "rife")
    assert third != first
    assert len(hashes) == 4


def test_joint_video_sha_cached_per_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Joint videos hash once per process (issues 250/252)."""
    import voyage.augment_joints as joints

    joints._JOINT_SHA_CACHE.clear()
    video = tmp_path / "joint.mp4"
    video.write_bytes(b"joint-bytes")
    hashes: list[str] = []
    import voyage.hashing as hashing

    real_sha = hashing.sha256_file

    def _counting(path: Path) -> str:
        hashes.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(joints, "sha256_file", _counting)
    first = joints.joint_video_sha(video)
    assert joints.joint_video_sha(video) == first
    assert len(hashes) == 1
    video.write_bytes(b"joint-bytes-changed")
    assert joints.joint_video_sha(video) != first
    assert len(hashes) == 2


def _shared_poll_kwargs(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "weights_path": Path("/models/realesrgan/realesr-animevideov3.pth"),
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "chunk_frames": 32,
    }
    params.update(overrides)
    return params


def _content_decode_factory(calls: list[tuple[int, int]]) -> Any:
    def _decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
        calls.append((start, count))
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for position in range(count):
            frame = dest_dir / f"frame_{position:06d}.png"
            frame.write_bytes(f"frame-{start + position:06d}".encode())
            written.append(frame)
        return written

    return _decode


def _passthrough_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
    return list(frame_paths)


def test_shared_decode_runs_once_and_slices_windows(tmp_path: Path) -> None:
    """One full decode feeds every chunk with identical bytes (issue 251)."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _make_segment(tmp_path, frames=70)
    calls: list[tuple[int, int]] = []
    result = upscale_poll_once(
        tmp_path,
        **_shared_poll_kwargs(
            decode_fn=_content_decode_factory(calls),
            upscale_fn=_passthrough_upscale,
            shared_segment_decode=True,
        ),
    )
    assert result.chunks_done == 3
    assert calls == [(0, 70)]
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    assert not (plan_dir / "shared_decode.partial").exists()
    first = (plan_dir / "upscaled_00" / "frame_000000.png").read_bytes()
    assert first == b"frame-000000"
    second = (plan_dir / "upscaled_01" / "frame_000000.png").read_bytes()
    assert second == b"frame-000031"
    tail = (plan_dir / "upscaled_02" / "frame_000007.png").read_bytes()
    assert tail == b"frame-000069"
    assert len(list((plan_dir / "upscaled_02").glob("frame_*.png"))) == 8


def test_default_poll_still_decodes_per_chunk(tmp_path: Path) -> None:
    """Opt-out default keeps the legacy per-chunk decode shape (issue 251)."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _make_segment(tmp_path, frames=70)
    calls: list[tuple[int, int]] = []
    result = upscale_poll_once(
        tmp_path,
        **_shared_poll_kwargs(
            decode_fn=_content_decode_factory(calls),
            upscale_fn=_passthrough_upscale,
        ),
    )
    assert result.chunks_done == 3
    assert calls == [(0, 32), (31, 32), (62, 8)]


def test_shared_decode_single_missing_chunk_stays_per_chunk(tmp_path: Path) -> None:
    """A lone missing chunk never pays a full decode (issue 251)."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _make_segment(tmp_path, frames=70)
    calls: list[tuple[int, int]] = []
    decode = _content_decode_factory(calls)
    first = upscale_poll_once(
        tmp_path,
        **_shared_poll_kwargs(decode_fn=decode, upscale_fn=_passthrough_upscale, chunk_ids=[0, 1]),
    )
    assert first.chunks_done == 2
    calls.clear()
    second = upscale_poll_once(
        tmp_path,
        **_shared_poll_kwargs(
            decode_fn=decode, upscale_fn=_passthrough_upscale, shared_segment_decode=True
        ),
    )
    assert second.chunks_done == 1
    assert calls == [(62, 8)]


def test_slice_shared_frames_rejects_short_decodes(tmp_path: Path) -> None:
    """Slicing past the shared decode fails loud (issue 251)."""
    from voyage.augment_upscale_poller import _slice_shared_frames
    from voyage.errors import MediaError

    frames = [tmp_path / f"frame_{index:06d}.png" for index in range(4)]
    with pytest.raises(MediaError):
        _slice_shared_frames(frames, 2, 4, tmp_path / "partial", "000000")


def _fake_pil(monkeypatch: pytest.MonkeyPatch, opens: list[str]) -> None:
    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")

    class _FakeImage:
        def __init__(self, path: Path) -> None:
            opens.append(str(path))
            self.size = (64, 32)

        def __enter__(self) -> _FakeImage:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    image_module.open = _FakeImage  # type: ignore[attr-defined]
    pil_module.Image = _FakeImage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)


def test_png_size_cache_opens_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeat size gates reuse cached headers (issue 252)."""
    import voyage.augment as augment

    augment._PNG_SIZE_CACHE.clear()
    chunk_dir = tmp_path / "upscaled_00"
    chunk_dir.mkdir()
    for index in range(3):
        (chunk_dir / f"frame_{index:06d}.png").write_bytes(b"fake-png")
    opens: list[str] = []
    _fake_pil(monkeypatch, opens)
    assert augment.chunk_frames_match_size(chunk_dir, (64, 32)) is True
    assert augment.chunk_frames_match_size(chunk_dir, (64, 32)) is True
    assert len(opens) == 3
    assert augment.chunk_frames_match_size(chunk_dir, (64, 33)) is False


def test_gc_skips_unchanged_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeat GC with nothing changed collects nothing and re-stats nothing (252)."""
    import voyage.augment_drain as drain

    drain._LAST_GC_FINGERPRINT = None
    _make_segment(tmp_path, frames=8)
    orphan = tmp_path / "augment" / ("ab" * 8)
    orphan.mkdir(parents=True, exist_ok=True)
    old = 1_700_000_000.0
    os.utime(orphan, (old, old))
    recursive_stats: list[Path] = []
    real_newest = drain._newest_modification_time

    def _counting(target: Path) -> float | None:
        recursive_stats.append(target)
        return real_newest(target)

    monkeypatch.setattr(drain, "_newest_modification_time", _counting)
    params: dict[str, Any] = {
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "upscale_factor": 2,
        "crf": 15,
        "preset": "veryfast",
        "grace_days": 7.0,
        "now_seconds": old + 8 * 86400.0,
    }
    assert drain.prune_orphan_plan_dirs(tmp_path, **params) == 1
    assert not orphan.exists()
    assert len(recursive_stats) == 1
    assert drain.prune_orphan_plan_dirs(tmp_path, **params) == 0
    assert len(recursive_stats) == 1


def test_gc_reruns_when_tree_changes(tmp_path: Path) -> None:
    """A new orphan after a skip re-arms the GC (issue 252)."""
    import voyage.augment_drain as drain

    drain._LAST_GC_FINGERPRINT = None
    _make_segment(tmp_path, frames=8)
    params: dict[str, Any] = {
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "upscale_factor": 2,
        "crf": 15,
        "preset": "veryfast",
        "grace_days": 7.0,
        "now_seconds": 1_700_000_000.0 + 8 * 86400.0,
    }
    assert drain.prune_orphan_plan_dirs(tmp_path, **params) == 0
    assert drain.prune_orphan_plan_dirs(tmp_path, **params) == 0
    orphan = tmp_path / "augment" / ("cd" * 8)
    orphan.mkdir(parents=True, exist_ok=True)
    old = 1_700_000_000.0
    os.utime(orphan, (old, old))
    assert drain.prune_orphan_plan_dirs(tmp_path, **params) == 1


def test_list_plan_dirs_shared_enumeration(tmp_path: Path) -> None:
    """One helper lists plan dirs for sweep/heal/GC (issue 252)."""
    from voyage.augment_sidecar import list_plan_dirs

    assert list_plan_dirs(tmp_path) == []
    root = tmp_path / "augment"
    root.mkdir()
    (root / "deadbeefdeadbeef").mkdir()
    (root / "note.txt").write_text("not a dir", encoding="utf-8")
    assert [path.name for path in list_plan_dirs(tmp_path)] == ["deadbeefdeadbeef"]

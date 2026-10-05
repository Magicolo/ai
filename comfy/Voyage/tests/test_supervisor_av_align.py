"""Supervisor commit-side gates: video continuity, worker-report clamps, log rotation.

Video-only since the all-deferred audio cleanup (fake backends, real
ffmpeg media, no GPU):

- Video continuity: sequential commits advance numbering and the frame
  timeline from per-segment video frames (the old 003 A/V-drift commit
  gate now lives at finalize-time over the takes ledger).
- 006: absurd worker-reported frame counts (negative, zero, bool,
  non-int, over ceiling) and foreign tape paths fail fast; the
  orphan-adoption path enforces the frame ceiling too.
- 056: worker logs rotate mid-run (per committed segment) so the
  infinite path stays bounded.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.atomic import JsonValue
from voyage.errors import MediaError
from voyage.hashing import sha256_file
from voyage.persistence import read_effective_config
from voyage.rpc import RpcPayload, RpcResult
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path, run_id: str = "av-align") -> None:
    initialize_run_directory(run_dir, run_id=run_id)


def _commit(run_dir: Path, count: int) -> list[str]:
    config = read_effective_config(run_dir)
    return Supervisor(run_dir, config).run_segments(count)


def _started_supervisor(run_dir: Path) -> Supervisor:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    return supervisor


def _lie_about_video(supervisor: Supervisor, video_block: dict[str, object]) -> Any:
    """Wrap the video worker to report a crafted `video` block."""
    original = supervisor._video.call

    def _lie(op: str, payload: RpcPayload, timeout: float | None = None) -> RpcResult:
        result = original(op, payload, timeout=timeout)
        if op == "generate_blocks":
            result = dict(result)
            result["video"] = cast(JsonValue, video_block)
        return result

    return _lie


def _build_orphan_from_committed(run_dir: Path, next_number: int) -> Path:
    """Copy segment 000000's video/manifest into the next segment dir.

    Video-only since the all-deferred audio cleanup: only `video.mp4`
    is copied with a matching checksum entry (plus `recovery.pt` when
    the source carries one). A legacy `audio.wav` is copied along when
    the source still has one (current joint supervisor) so the orphan
    reaches the gate under test instead of failing on a missing preview;
    new deferred commits carry no audio and the orphan carries none.

    Returns the orphan segment dir (with DONE, matching checksums).
    The caller tampers afterwards (absurd frames) and rewrites the
    manifest checksums so checksums still pass — the adoption gate
    under test is the only thing that may refuse.
    """
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    src = run_dir / paths.SEGMENTS_DIRNAME / "000000"
    segment_id = paths.format_segment_id(next_number)
    dst = run_dir / paths.SEGMENTS_DIRNAME / segment_id
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "video.mp4", dst / "video.mp4")
    if (src / "audio.wav").exists():
        shutil.copy(src / "audio.wav", dst / "audio.wav")
    manifest = load_segment_manifest(src)
    world_state = dict(manifest.get("world_state", {}))
    world_state["segment_id"] = segment_id
    manifest["world_state"] = world_state
    checksums: dict[str, str] = {"video.mp4": sha256_file(dst / "video.mp4")}
    if (dst / "audio.wav").exists():
        checksums["audio.wav"] = sha256_file(dst / "audio.wav")
    if (src / "recovery.pt").is_file():
        shutil.copy(src / "recovery.pt", dst / "recovery.pt")
        checksums["recovery.pt"] = sha256_file(dst / "recovery.pt")
    manifest["checksums"] = checksums
    write_segment_manifest(dst, manifest)
    (dst / paths.DONE_MARKER).write_bytes(b"")
    return dst


def test_commit_continues_video_timeline(tmp_path: Path) -> None:
    """Video-only continuation: two commits advance numbering + timeline.

    Replaces the old A/V-drift commit gate (003, now finalize-time):
    each segment contributes its video frames, `video.mp4` exists per
    segment, and the timeline is the frame sum — no audio duration
    involved.
    """
    from voyage.persistence import read_state

    run_dir = tmp_path / "run"
    _init_run(run_dir)
    assert _commit(run_dir, 2) == ["000000", "000001"]
    state = read_state(run_dir)
    assert state.committed_segments == 2
    assert state.next_segment_number == 2
    frames_total = 0
    for segment_id in ("000000", "000001"):
        segment = run_dir / paths.SEGMENTS_DIRNAME / segment_id
        assert (segment / "video.mp4").exists()
        assert (segment / paths.DONE_MARKER).exists()
        from voyage.segment_manifest import load_segment_manifest

        metrics = load_segment_manifest(segment).get("metrics", {})
        assert isinstance(metrics, dict)
        segment_frames = int(metrics.get("frames", 0))
        assert segment_frames > 0
        frames_total += segment_frames
    assert state.timeline_frames == frames_total


def test_worker_reported_negative_frames_rejected(tmp_path: Path) -> None:
    """`frames=-5` fails fast instead of corrupting the timeline (006)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        supervisor._video.call = _lie_about_video(supervisor, {"frames": -5})  # type: ignore[method-assign]
        with pytest.raises(MediaError, match="implausible"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_worker_reported_bool_frames_rejected(tmp_path: Path) -> None:
    """`frames=True` (bool is an int subclass) fails fast (006)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        supervisor._video.call = _lie_about_video(supervisor, {"frames": True})  # type: ignore[method-assign]
        with pytest.raises(MediaError, match="implausible"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_adopt_rejects_absurd_frames(tmp_path: Path) -> None:
    """DONE orphan with `frames=10**9` must not adopt silently (006)."""
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest

    run_dir = tmp_path / "run"
    _init_run(run_dir)
    assert _commit(run_dir, 1) == ["000000"]
    orphan = _build_orphan_from_committed(run_dir, 1)
    manifest = load_segment_manifest(orphan)
    metrics = dict(manifest.get("metrics", {}))
    metrics["frames"] = 10**9
    write_segment_manifest(orphan, {**manifest, "metrics": metrics})
    supervisor = _started_supervisor(run_dir)
    try:
        with pytest.raises(MediaError, match="implausible"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_foreign_tape_directory_rejected(tmp_path: Path) -> None:
    """A tape pointing at a directory fails fast (006)."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    supervisor = _started_supervisor(run_dir)
    try:
        tape = str(run_dir / paths.SEGMENTS_DIRNAME)
        supervisor._video.call = _lie_about_video(  # type: ignore[method-assign]
            supervisor, {"frames": 48, "recovery_path": tape}
        )
        with pytest.raises(MediaError, match="not a regular file"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_rotate_open_log_truncates_in_place(tmp_path: Path) -> None:
    """Open-handle rotation archives content and truncates live (056)."""
    from voyage.logrotate import rotate_open_log

    log = tmp_path / "video-worker.log"
    handle = log.open("a", encoding="utf-8")
    try:
        handle.write("old worker output\n")
        handle.flush()
        rotated = rotate_open_log(log, max_bytes=1)
        assert rotated is not None
        assert rotated.read_text(encoding="utf-8") == "old worker output\n"
        assert log.read_text(encoding="utf-8") == ""
        handle.write("new worker output\n")
        handle.flush()
        assert log.read_text(encoding="utf-8") == "new worker output\n"
    finally:
        handle.close()


def test_worker_logs_rotate_mid_run_keeps_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-bloated worker logs shrink after commits; dir stays bounded (056)."""
    import voyage.logrotate as logrotate_module

    run_dir = tmp_path / "run"
    _init_run(run_dir)
    monkeypatch.setattr(logrotate_module, "MAX_WORKER_LOG_BYTES", 1024)
    logs_dir = run_dir / paths.LOGS_DIRNAME
    for name in ("video-worker.log", "audio-worker.log", "director-worker.log"):
        (logs_dir / name).write_text("x" * 4096, encoding="utf-8")
    assert _commit(run_dir, 2) == ["000000", "000001"]
    for name in ("video-worker.log", "audio-worker.log", "director-worker.log"):
        assert (logs_dir / name).stat().st_size < 4096
        # The pre-bloat must have been archived mid-run, not left growing
        # live: at least one dated sibling per worker log.
        assert list(logs_dir.glob(f"{Path(name).stem}-*{Path(name).suffix}")) != []

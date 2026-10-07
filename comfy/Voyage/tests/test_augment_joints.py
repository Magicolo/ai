"""Fix-then-uniform-pass joints: source-res morph fix, uniform model pass (DESIGN §140).

Order of operations is seam fix -> upscale -> interpolate: the fix stage
renders morph 2+2 bridges at source resolution only (no anchor upscale),
the 4 bridge frames become a joint source video that flows through the
same upscale + interp pollers as committed segments (standard ChunkKey
ledger shapes in the joint's own plan dir), and the drain assembles
[trimA, joint, trimB] with source-derived trims. Real ffmpeg on synthetic
testsrc clips; interp is a stub (no torch at module scope).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voyage import augment_joints
from voyage.augment_finalize import _poll_to_completion, run_durable_model_pass
from voyage.augment_interp_poller import InterpPollResult
from voyage.augment_upscale_poller import UpscalePollResult


def _ffmpeg(*argv: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *argv], check=True)


def _frame_count(mp4: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    counts = [int(line) for line in out.stdout.splitlines() if line.strip()]
    assert counts, f"no frames probed in {mp4}"
    assert all(count == counts[0] for count in counts), f"probe disagrees: {counts!r}"
    return counts[0]


def _make_clip(path: Path, frames: int, fps: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x64:rate={fps}:duration={frames / fps}",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(path),
    )
    assert _frame_count(path) == frames
    return path


# 64x64 solid-gray RGB PNG (zlib-built, ffmpeg-validated): the stub interp
# must return *decodable* bytes because the fix stage runs real ffmpeg over
# the bridge PNGs (garbage bytes would fail the joint encode).
def _gray_png_64() -> bytes:
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        framed = struct.pack(">I", len(data)) + tag + data
        return framed + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * 64 for _ in range(64))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


_FAKE_PNG_BYTES = _gray_png_64()


def _stub_interp(png_a: Path, png_b: Path, weights: object, moment: float, device: str) -> bytes:
    del weights, moment, device
    assert png_a.exists() and png_b.exists()
    return _FAKE_PNG_BYTES


def _make_segment(
    run_dir: Path,
    segment_id: str,
    *,
    frames: int = 8,
    checksum: str = "ck",
    real_clip: bool = False,
) -> Path:
    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    video = segment_dir / "video.mp4"
    if real_clip:
        _make_clip(video, frames)
    else:
        video.write_bytes(b"fake-video")
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


def test_joint_fix_key_is_pair_and_recipe_sensitive() -> None:
    key = augment_joints.joint_fix_key
    base = key(a_sha="a" * 64, b_sha="b" * 64, width=1216, height=704, fps_key=24)
    assert key(a_sha="b" * 64, b_sha="a" * 64, width=1216, height=704, fps_key=24) != base
    assert key(a_sha="a" * 64, b_sha="b" * 64, width=640, height=704, fps_key=24) != base
    assert key(a_sha="a" * 64, b_sha="b" * 64, width=1216, height=704, fps_key=25) != base
    assert key(a_sha="a" * 64, b_sha="b" * 64, width=1216, height=704, fps_key=24) == base
    assert base.startswith(augment_joints.JOINT_FIX_KEY_PREFIX + "|")


def test_segment_keep_ranges() -> None:
    keep = augment_joints.segment_keep
    # Multiplier 1: A keeps through model frame a-3, B drops its first 2.
    assert keep(0, 2, 12, 1) == (0, 10)
    assert keep(1, 2, 12, 1) == (2, 12)
    # Multiplier 4: model indices scale, same source semantics.
    assert keep(0, 2, 12, 4) == (0, (12 - 3) * 4 + 1)
    assert keep(1, 2, 12, 4) == (8, (12 - 1) * 4 + 1)
    # Middle segments lose both ends; a lone segment passes whole.
    assert keep(1, 3, 96, 4) == (8, (96 - 3) * 4 + 1)
    assert keep(0, 1, 96, 4) == (0, (96 - 1) * 4 + 1)
    # Too short to anchor both sides fails loud, never an empty trim.
    from voyage.errors import MediaError

    with pytest.raises(MediaError):
        keep(1, 3, 4, 4)


def test_render_joint_source_writes_four_frame_video_and_ledger(tmp_path: Path) -> None:
    video_a = _make_clip(tmp_path / "a.mp4", 12)
    video_b = _make_clip(tmp_path / "b.mp4", 12)
    joint_dir = tmp_path / "joint_000000_000001"
    calls = {"interp": 0}

    def _counting_interp(
        png_a: Path, png_b: Path, weights: object, moment: float, device: str
    ) -> bytes:
        calls["interp"] += 1
        return _stub_interp(png_a, png_b, weights, moment, device)

    unit = augment_joints.render_joint_source(
        joint_dir=joint_dir,
        video_a=video_a,
        a_frames=12,
        video_b=video_b,
        b_frames=12,
        source_key="fix-key",
        source_fps=24,
        crf=18,
        preset="veryfast",
        left_id="000000",
        right_id="000001",
        interp_fn=_counting_interp,
        weights=None,
        device="cpu",
    )
    assert unit.left_id == "000000"
    assert unit.right_id == "000001"
    assert unit.left_frames == 12
    assert unit.right_frames == 12
    assert _frame_count(unit.joint_video) == 4
    record = json.loads((joint_dir / "record.json").read_text(encoding="utf-8"))
    assert record["source_key"] == "fix-key"
    assert record["joint_sha"] == unit.joint_key
    assert record["left_id"] == "000000"
    # Ledger hit: a second render with the same key never re-interps.
    again = augment_joints.render_joint_source(
        joint_dir=joint_dir,
        video_a=video_a,
        a_frames=12,
        video_b=video_b,
        b_frames=12,
        source_key="fix-key",
        source_fps=24,
        crf=18,
        preset="veryfast",
        left_id="000000",
        right_id="000001",
        interp_fn=_counting_interp,
        weights=None,
        device="cpu",
    )
    assert again.joint_key == unit.joint_key
    assert calls["interp"] == 4


def test_render_joint_source_heals_key_clash(tmp_path: Path) -> None:
    video_a = _make_clip(tmp_path / "a.mp4", 12)
    video_b = _make_clip(tmp_path / "b.mp4", 12)
    joint_dir = tmp_path / "joint_000000_000001"
    kwargs: dict[str, Any] = {
        "joint_dir": joint_dir,
        "video_a": video_a,
        "a_frames": 12,
        "video_b": video_b,
        "b_frames": 12,
        "source_fps": 24,
        "crf": 18,
        "preset": "veryfast",
        "left_id": "000000",
        "right_id": "000001",
        "interp_fn": _stub_interp,
        "weights": None,
        "device": "cpu",
    }
    first = augment_joints.render_joint_source(source_key="key-one", **kwargs)
    second = augment_joints.render_joint_source(source_key="key-two", **kwargs)
    assert second.joint_video == first.joint_video
    record = json.loads((joint_dir / "record.json").read_text(encoding="utf-8"))
    assert record["source_key"] == "key-two"


def test_ensure_joint_units_orders_and_rejects_geometry_mix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _make_segment(run_dir, "000000", frames=12, checksum="aa", real_clip=True)
    _make_segment(run_dir, "000001", frames=12, checksum="bb", real_clip=True)
    from voyage.augment_upscale_poller import committed_segment_sources

    sources, _skipped = committed_segment_sources(run_dir)
    ordered = sorted(sources, key=lambda source: source.segment_id)
    units = augment_joints.ensure_joint_units(
        run_dir,
        ordered,
        source_fps=24,
        crf=18,
        preset="veryfast",
        interp_fn=_stub_interp,
        weights=None,
        device="cpu",
    )
    assert len(units) == 1
    (unit,) = units
    assert unit.left_id == "000000"
    assert unit.right_id == "000001"
    assert _frame_count(unit.joint_video) == 4
    as_source = unit.as_source()
    assert as_source.total_frames == 4
    assert as_source.video_path == unit.joint_video
    # A lone source yields no joints (no probing, no rendering).
    assert (
        augment_joints.ensure_joint_units(
            run_dir,
            ordered[:1],
            source_fps=24,
            crf=18,
            preset="veryfast",
            interp_fn=_stub_interp,
            weights=None,
            device="cpu",
        )
        == []
    )
    # Mixed source geometry fails loud, never a silently stretched joint.
    wide = _make_clip(tmp_path / "wide.mp4", 12)
    _ffmpeg(
        "-i",
        str(wide),
        "-vf",
        "scale=128:64",
        "-frames:v",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(tmp_path / "wide_scaled.mp4"),
    )
    from voyage.augment_upscale_poller import SegmentSource

    mixed = [
        ordered[0],
        SegmentSource(
            segment_id="000002",
            segment_dir=tmp_path,
            video_path=tmp_path / "wide_scaled.mp4",
            source_key="cc",
            total_frames=12,
        ),
    ]
    with pytest.raises(Exception, match="geometry"):
        augment_joints.ensure_joint_units(
            run_dir,
            mixed,
            source_fps=24,
            crf=18,
            preset="veryfast",
            interp_fn=_stub_interp,
            weights=None,
            device="cpu",
        )


def test_assemble_joint_timeline_trims_and_whole_joints(tmp_path: Path) -> None:
    """End-to-end assembly math on real clips: 10 + 4 + 10 = 24 at m=1."""
    seg_a = _make_clip(tmp_path / "seg_a.mp4", 12)
    seg_b = _make_clip(tmp_path / "seg_b.mp4", 12)
    joint = _make_clip(tmp_path / "joint.mp4", 4)
    final = augment_joints.assemble_joint_timeline(
        [seg_a, seg_b],
        [joint],
        source_counts=[12, 12],
        multiplier=1,
        joint_root=tmp_path / "timeline",
        fps=24,
        crf=18,
        preset="veryfast",
        pix_fmt="yuv420p",
    )
    assert _frame_count(final) == 10 + 4 + 10
    # A lone segment passes through untouched (no joints, no re-encode).
    passthrough = augment_joints.assemble_joint_timeline(
        [seg_a],
        [],
        source_counts=[12],
        multiplier=1,
        joint_root=tmp_path / "timeline_lone",
        fps=24,
        crf=18,
        preset="veryfast",
        pix_fmt="yuv420p",
    )
    assert passthrough == seg_a


def _make_weights(work: Path) -> Any:
    work.mkdir(parents=True, exist_ok=True)
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    rife = work / "rife.safetensors"
    rife.write_bytes(b"rife-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    from voyage.augment import AugmentWeights

    return AugmentWeights(film=film, rife=rife, realesrgan=esrgan)


def _stub_joint_unit(joint_dir: Path, joint_video: Path) -> Any:
    return augment_joints.JointUnit(
        joint_dir=joint_dir,
        joint_video=joint_video,
        joint_key="joint-key",
        left_id="000000",
        right_id="000001",
        left_frames=8,
        right_frames=8,
    )


def test_poll_routes_joints_through_both_legs_after_segments(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Joints poll through the same legs as segments (sources= override)."""
    import voyage.augment_finalize as finalize_module

    run_dir = tmp_path / "run"
    _make_segment(run_dir, "000000", frames=8, checksum="ck0")
    _make_segment(run_dir, "000001", frames=8, checksum="ck1")
    joint_video = _make_clip(tmp_path / "joint.mp4", 4)
    joint_dir = tmp_path / "joint_000000_000001"
    joint_dir.mkdir(parents=True, exist_ok=True)
    (joint_dir / "joint.mp4").write_bytes(joint_video.read_bytes())
    monkeypatch.setattr(
        finalize_module,
        "ensure_joint_units",
        lambda *args, **kwargs: [_stub_joint_unit(joint_dir, joint_dir / "joint.mp4")],
    )
    calls: list[tuple[str, Any, Any]] = []

    def _rec_upscale(run_dir_arg: Path, **kwargs: Any) -> UpscalePollResult:
        calls.append(("up", kwargs.get("segment_ids"), kwargs.get("sources")))
        return UpscalePollResult(1, 0, 0, 0, 0)

    def _rec_interp(run_dir_arg: Path, **kwargs: Any) -> InterpPollResult:
        calls.append(("ip", kwargs.get("segment_ids"), kwargs.get("sources")))
        return InterpPollResult(1, 0, 0, 0, 0, 0)

    _poll_to_completion(
        run_dir,
        weights=_make_weights(tmp_path / "weights"),
        weights_key="k|k",
        out_width=64,
        out_height=64,
        source_fps=24.0,
        upscale_factor=1,
        multiplier=2,
        chunk_frames=32,
        device="cpu",
        crf=18,
        preset="veryfast",
        upscale_poll_fn=_rec_upscale,
        interp_poll_fn=_rec_interp,
    )
    legs = [leg for leg, _ids, _sources in calls]
    assert legs == ["up", "ip", "up", "ip", "up", "ip"]
    # Segment polls scan the run (no sources override); joint polls route
    # the fix-stage units through the same legs.
    assert calls[0][2] is None
    assert calls[1][2] is None
    joint_up = calls[4]
    joint_ip = calls[5]
    assert joint_up[1] == [joint_dir.name]
    assert joint_ip[1] == [joint_dir.name]
    assert [source.segment_id for source in joint_up[2]] == [joint_dir.name]
    assert [source.segment_id for source in joint_ip[2]] == [joint_dir.name]
    assert joint_up[2][0].total_frames == 4


def test_poll_runs_joint_upscale_in_interp_only_mode(tmp_path: Path, monkeypatch: Any) -> None:
    """Interp-only polls still run the joint upscale (new units, no prior leg)."""
    import voyage.augment_finalize as finalize_module

    run_dir = tmp_path / "run"
    _make_segment(run_dir, "000000", frames=8, checksum="ck0")
    _make_segment(run_dir, "000001", frames=8, checksum="ck1")
    joint_video = _make_clip(tmp_path / "joint.mp4", 4)
    joint_dir = tmp_path / "joint_000000_000001"
    joint_dir.mkdir(parents=True, exist_ok=True)
    (joint_dir / "joint.mp4").write_bytes(joint_video.read_bytes())
    monkeypatch.setattr(
        finalize_module,
        "ensure_joint_units",
        lambda *args, **kwargs: [_stub_joint_unit(joint_dir, joint_dir / "joint.mp4")],
    )
    legs: list[str] = []

    def _rec_upscale(run_dir_arg: Path, **kwargs: Any) -> UpscalePollResult:
        sources = kwargs.get("sources") or []
        legs.extend(f"up:{source.segment_id}" for source in sources)
        return UpscalePollResult(1, 0, 0, 0, 0)

    def _rec_interp(run_dir_arg: Path, **kwargs: Any) -> InterpPollResult:
        return InterpPollResult(1, 0, 0, 0, 0, 0)

    _poll_to_completion(
        run_dir,
        weights=_make_weights(tmp_path / "weights"),
        weights_key="k|k",
        out_width=64,
        out_height=64,
        source_fps=24.0,
        upscale_factor=1,
        multiplier=2,
        chunk_frames=32,
        device="cpu",
        crf=18,
        preset="veryfast",
        upscale_poll_fn=_rec_upscale,
        interp_poll_fn=_rec_interp,
        include_upscale=False,
        include_interp=True,
    )
    assert legs == [f"up:{joint_dir.name}"]


def test_run_durable_assembles_trim_joint_trim_order(tmp_path: Path, monkeypatch: Any) -> None:
    """Drain consumes segments + joint plan dirs, assembles [trimA, joint, trimB]."""
    import voyage.augment_finalize as finalize_module
    from voyage.augment_finalize import plan_dir_for_segment

    run_dir = tmp_path / "run"
    first = _make_segment(run_dir, "000000", frames=8, checksum="ck0")
    second = _make_segment(run_dir, "000001", frames=8, checksum="ck1")
    joint_video = _make_clip(tmp_path / "joint.mp4", 4)
    joint_dir = tmp_path / "joint_000000_000001"
    joint_dir.mkdir(parents=True, exist_ok=True)
    (joint_dir / "joint.mp4").write_bytes(joint_video.read_bytes())
    monkeypatch.setattr(
        finalize_module,
        "ensure_joint_units",
        lambda *args, **kwargs: [_stub_joint_unit(joint_dir, joint_dir / "joint.mp4")],
    )
    weights = _make_weights(tmp_path / "weights")
    work = tmp_path / "work"
    work.mkdir()
    drained: list[Path] = []
    assembled: dict[str, Any] = {}
    seen_keys: list[str] = []

    def stub_poll(run_dir_arg: Path, **kwargs: Any) -> Any:
        if "weights_key" in kwargs:
            seen_keys.append(kwargs["weights_key"])
        if "multiplier" in kwargs:
            return SimpleNamespace(chunks_done=0, chunks_waiting=0)
        return SimpleNamespace(chunks_done=0)

    def stub_drain(plan_dir: Path, **kwargs: Any) -> Any:
        drained.append(plan_dir)
        intermediate = plan_dir / "model_intermediate.mp4"
        intermediate.parent.mkdir(parents=True, exist_ok=True)
        intermediate.write_bytes(b"fake-intermediate")
        return SimpleNamespace(intermediate_mp4=intermediate, chunks_drained=1)

    def stub_assemble(
        segment_mains: list[Path],
        joint_mains: list[Path],
        **kwargs: Any,
    ) -> Path:
        assembled["segments"] = list(segment_mains)
        assembled["joints"] = list(joint_mains)
        assembled["counts"] = kwargs.get("source_counts")
        dest = work / "jointed.mp4"
        dest.write_bytes(b"fake-final")
        return dest

    final, final_fps = run_durable_model_pass(
        run_dir,
        [first, second],
        out_width=64,
        out_height=64,
        source_fps=24.0,
        weights=weights,
        multiplier=1,
        chunk_frames=4,
        device="cpu",
        work_dir=work,
        upscale_poll_fn=stub_poll,
        interp_poll_fn=stub_poll,
        drain_fn=stub_drain,
        assemble_fn=stub_assemble,
    )
    assert final_fps == 24
    assert final.read_bytes() == b"fake-final"
    assert len(drained) == 3  # two segments + the joint plan dir
    assert assembled["counts"] == [8, 8]
    assert len(assembled["segments"]) == 2
    assert len(assembled["joints"]) == 1
    joint_plan = plan_dir_for_segment(
        run_dir,
        source_key="joint-key",
        weights_key=seen_keys[0],
        out_width=64,
        out_height=64,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    assert joint_plan in drained


def test_ensure_joint_units_stops_between_units_and_resumes(tmp_path: Path) -> None:
    """A stop returns finished units; the next pass resumes the rest."""
    run_dir = tmp_path / "run"
    for index in range(4):
        _make_segment(run_dir, f"{index:06d}", frames=12, checksum=f"ck{index}", real_clip=True)
    from voyage.augment_upscale_poller import committed_segment_sources

    sources, _skipped = committed_segment_sources(run_dir)
    ordered = sorted(sources, key=lambda source: source.segment_id)
    checks = {"count": 0}

    def _stop_after_two_checks() -> bool:
        checks["count"] += 1
        return checks["count"] >= 3

    kwargs: dict[str, Any] = {
        "source_fps": 24,
        "crf": 18,
        "preset": "veryfast",
        "interp_fn": _stub_interp,
        "weights": None,
        "device": "cpu",
    }
    partial = augment_joints.ensure_joint_units(
        run_dir, ordered, should_stop=_stop_after_two_checks, **kwargs
    )
    assert [unit.joint_dir.name for unit in partial] == [
        "joint_000000_000001",
        "joint_000001_000002",
    ]
    assert all(unit.joint_video.is_file() for unit in partial)
    full = augment_joints.ensure_joint_units(run_dir, ordered, **kwargs)
    assert [unit.joint_dir.name for unit in full] == [
        "joint_000000_000001",
        "joint_000001_000002",
        "joint_000002_000003",
    ]
    assert [unit.joint_video for unit in full[:2]] == [unit.joint_video for unit in partial]
    assert all(unit.joint_video.is_file() for unit in full)


class _FakeJointTracker:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.total: int | None = None
        self.extras: list[str] = []

    def update(self, advance: int = 1) -> None:
        self.updates.append(advance)

    def set_total(self, total: int) -> None:
        self.total = total

    def set_extra(self, extra: str) -> None:
        self.extras.append(extra)


class _FakeJointBarCM:
    def __init__(self, tracker: _FakeJointTracker) -> None:
        self.tracker = tracker

    def __enter__(self) -> _FakeJointTracker:
        return self.tracker

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeJointProgress:
    def __init__(self) -> None:
        self.bars: list[tuple[str, int | None, _FakeJointTracker]] = []
        self.verbose = False

    def bar(self, label: str, total: int | None = None) -> _FakeJointBarCM:
        tracker = _FakeJointTracker()
        self.bars.append((label, total, tracker))
        return _FakeJointBarCM(tracker)


def _settling_joint_stubs() -> tuple[Any, Any, list[tuple[str, str]]]:
    """Poll stubs that render once per scope, then settle (ledger-hit)."""
    from voyage.augment_interp_poller import InterpPollResult
    from voyage.augment_upscale_poller import UpscalePollResult

    calls: list[tuple[str, str]] = []
    seen: set[Any] = set()

    def _scope(kwargs: Any) -> tuple[str, tuple[str, ...]]:
        ids = tuple(kwargs.get("segment_ids") or ())
        if kwargs.get("sources") is not None:
            return ("joint", ids)
        return ("segment", ids)

    def _upscale_stub(run_dir: Path, **kwargs: Any) -> Any:
        scope = ("up", _scope(kwargs))
        calls.append(("up", scope[1][0]))
        if scope in seen:
            return UpscalePollResult(
                segments_seen=1,
                segments_skipped=0,
                chunks_done=0,
                chunks_skipped=1,
                partials_pruned=0,
                frames_done=0,
                frames_skipped=4,
            )
        seen.add(scope)
        unit_id = scope[1][1][0] if scope[1][1] else "unknown"
        kwargs["on_chunk"](unit_id, 0, 1)
        kwargs["on_chunk_frames"](unit_id, 4)
        return UpscalePollResult(
            segments_seen=1,
            segments_skipped=0,
            chunks_done=1,
            chunks_skipped=0,
            partials_pruned=0,
            frames_done=4,
            frames_skipped=0,
        )

    def _interp_stub(run_dir: Path, **kwargs: Any) -> Any:
        scope = ("ip", _scope(kwargs))
        calls.append(("ip", scope[1][0]))
        if scope in seen:
            return InterpPollResult(
                segments_seen=1,
                segments_skipped=0,
                chunks_done=0,
                chunks_skipped=1,
                chunks_waiting=0,
                partials_pruned=0,
                frames_done=0,
                frames_skipped=4,
            )
        seen.add(scope)
        unit_id = scope[1][1][0] if scope[1][1] else "unknown"
        kwargs["on_chunk"](unit_id, 0, 1)
        kwargs["on_pair_frames"](unit_id, 4.0)
        return InterpPollResult(
            segments_seen=1,
            segments_skipped=0,
            chunks_done=1,
            chunks_skipped=0,
            chunks_waiting=0,
            partials_pruned=0,
            frames_done=4,
            frames_skipped=0,
        )

    return _upscale_stub, _interp_stub, calls


def test_poll_joint_bar_combined_with_live_status(tmp_path: Path) -> None:
    """The finalize joint sweep shows one bar with leg status, ending full."""
    from voyage.augment_finalize import _poll_to_completion

    run_dir = tmp_path / "run"
    _make_segment(run_dir, "000000", frames=12, checksum="aa", real_clip=True)
    _make_segment(run_dir, "000001", frames=12, checksum="bb", real_clip=True)
    weights = SimpleNamespace(
        realesrgan=tmp_path / "esrgan.pth",
        film=tmp_path / "film.safetensors",
        rife=tmp_path / "rife.safetensors",
    )
    (tmp_path / "esrgan.pth").write_bytes(b"e" * 32)
    (tmp_path / "film.safetensors").write_bytes(b"f" * 32)
    (tmp_path / "rife.safetensors").write_bytes(b"i" * 32)
    upscale_stub, interp_stub, calls = _settling_joint_stubs()
    progress = _FakeJointProgress()
    _poll_to_completion(
        run_dir,
        weights=weights,
        weights_key="weights-abc",
        out_width=64,
        out_height=64,
        source_fps=24.0,
        upscale_factor=1,
        multiplier=1,
        chunk_frames=8,
        device="cpu",
        crf=18,
        preset="veryfast",
        upscale_poll_fn=upscale_stub,
        interp_poll_fn=interp_stub,
        timings={},
        progress=progress,  # type: ignore[arg-type]
        joint_interp_fn=_stub_interp,
    )
    joint_bars = [bar for bar in progress.bars if bar[0] == "joint frames"]
    # One bar per poll pass (the settle pass re-opens instant-full, same
    # shape as the per-segment bars); the rendering pass carries the work.
    assert len(joint_bars) == 2
    _label, total, tracker = joint_bars[0]
    assert total == 8
    assert sum(tracker.updates) == 8
    assert any(extra.startswith("upscaling joint_") for extra in tracker.extras)
    assert any(extra.startswith("interpolating joint_") for extra in tracker.extras)
    assert any(extra.endswith("(1/1)") for extra in tracker.extras)
    _settle_label, _settle_total, settle_tracker = joint_bars[1]
    # Settle pass: everything ledger-hit, so the remainder bump fills the
    # bar with no live status (mirrors skipped segments).
    assert sum(settle_tracker.updates) == 8
    assert settle_tracker.extras == []
    assert ("up", "joint") in calls
    assert ("ip", "joint") in calls

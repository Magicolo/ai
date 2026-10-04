"""Morph-cut 2+2 joint assembly for ltx25/ltx23 finalize (DESIGN §140).

Replaces the mids-insert seam (freeze-motion bridge) with a count-preserving
morph-cut: per segment joint, A[-2:]+B[:2] are replaced by 4 FILM bridge
frames morphed between anchors A[-3] and B[+2]. Frame count is unchanged,
so committed audio timelines stay untouched.

Uses real ffmpeg on synthetic testsrc clips (ffmpeg ships in every voyage
image); FILM is a stub (no torch at module scope — same rule as the seam
tests). Ephemeral-harness numbers: morph 2+2 gaps ~3, through-bridge mean
steps at within-motion level, fullres transfer verified.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from voyage import augment_morph
from voyage.augment_finalize import run_durable_model_pass


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
    # MPEG-TS probes print per-program lines; they must agree.
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
# must return *decodable* bytes because the assembly tests run real ffmpeg
# over the bridge PNGs (garbage bytes would fail the bridge encode).
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


def test_morph_joint_key_is_pair_and_recipe_sensitive() -> None:
    key = augment_morph.morph_joint_key
    base = key(a_sha="a" * 64, b_sha="b" * 64, width=1216, height=704, fps_key=24)
    assert key(a_sha="b" * 64, b_sha="a" * 64, width=1216, height=704, fps_key=24) != base
    assert key(a_sha="a" * 64, b_sha="b" * 64, width=640, height=704, fps_key=24) != base
    assert key(a_sha="a" * 64, b_sha="b" * 64, width=1216, height=704, fps_key=24) == base
    assert base.startswith("morph2x2|")


def test_morph_backend_gate() -> None:
    assert frozenset({"ltx25", "ltx23"}) == augment_morph.MORPH_BACKENDS
    assert augment_morph.morph_enabled_for_backend("ltx25")
    assert augment_morph.morph_enabled_for_backend("ltx23")
    assert not augment_morph.morph_enabled_for_backend("ltxv")
    assert not augment_morph.morph_enabled_for_backend("causvid")
    assert not augment_morph.morph_enabled_for_backend("fake")


def test_morph_backend_for_run_reads_voyage_toml(tmp_path: Path) -> None:
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="morph", style="probe", seed=11, video_backend="ltx25")
    assert augment_morph.morph_backend_for_run(run_dir) == "ltx25"
    initialize_run_directory(run_dir, run_id="morph", style="probe", seed=11, video_backend="fake")
    assert augment_morph.morph_backend_for_run(run_dir) is None
    assert augment_morph.morph_backend_for_run(tmp_path / "missing") is None


def test_morph_anchor_and_trim_counts_preserve_total() -> None:
    assert augment_morph.morph_anchors(a_count=96, b_count=96) == (93, 2)
    assert augment_morph.morph_trims(a_count=96, b_count=96) == ((0, 94), (2, 96))
    kept, bridged = augment_morph.morph_counts([96, 96, 96])
    assert kept + bridged == 96 * 3
    assert bridged == 2 * 4


def test_render_morph_once_writes_four_bridges_and_ledger(tmp_path: Path) -> None:
    joint_dir = tmp_path / "morph_000000_000001"
    anchor_a = tmp_path / "a.png"
    anchor_b = tmp_path / "b.png"
    anchor_a.write_bytes(b"fake-png-a")
    anchor_b.write_bytes(b"fake-png-b")
    key = augment_morph.morph_joint_key(
        a_sha="a" * 64, b_sha="b" * 64, width=64, height=64, fps_key=6
    )
    result = augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=key,
        interp_fn=_stub_interp,
        weights=None,
        device="cpu",
    )
    assert [p.name for p in result.frames] == [f"bridge_{i:02d}.png" for i in range(4)]
    record = json.loads((joint_dir / "record.json").read_text(encoding="utf-8"))
    assert record["source_key"] == key
    assert record["bridge_count"] == 4
    again = augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=key,
        interp_fn=_stub_interp,
        weights=None,
        device="cpu",
    )
    assert again.frames == result.frames
    with pytest.raises(ValueError, match="ledger has"):
        augment_morph.render_morph_once(
            joint_dir=joint_dir,
            a_anchor=anchor_a,
            b_anchor=anchor_b,
            source_key=key + "x",
            interp_fn=_stub_interp,
            weights=None,
            device="cpu",
        )


def test_assemble_morphed_timeline_real_ffmpeg(tmp_path: Path) -> None:
    seg_a = _make_clip(tmp_path / "a.mp4", 12)
    seg_b = _make_clip(tmp_path / "b.mp4", 12)
    out = augment_morph.assemble_morphed_timeline(
        [seg_a, seg_b],
        joint_root=tmp_path / "joints",
        fps=24,
        crf=18,
        preset="veryfast",
        pix_fmt="yuv420p",
        interp_fn=_stub_interp,
        weights=None,
        device="cpu",
    )
    assert _frame_count(out) == 24
    # One ledgered joint dir per adjacent pair; sibling work files (trims/,
    # pieces list, assembled timeline) mirror the drain's plan-dir layout.
    joints = sorted(
        path
        for path in (tmp_path / "joints").iterdir()
        if path.is_dir() and path.name.startswith("morph_")
    )
    assert len(joints) == 1
    record = json.loads((joints[0] / "record.json").read_text(encoding="utf-8"))
    assert record["bridge_count"] == 4


def test_single_segment_needs_no_joints(tmp_path: Path) -> None:
    seg = _make_clip(tmp_path / "only.mp4", 12)

    def _explode(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("no joints expected")

    out = augment_morph.assemble_morphed_timeline(
        [seg],
        joint_root=tmp_path / "joints",
        fps=24,
        crf=18,
        preset="veryfast",
        pix_fmt="yuv420p",
        interp_fn=_explode,
        weights=None,
        device="cpu",
    )
    assert _frame_count(out) == 12
    assert list((tmp_path / "joints").iterdir()) == []


def test_three_segments_middle_trim_drops_both_ends(tmp_path: Path) -> None:
    segs = [_make_clip(tmp_path / f"s{i}.mp4", 12) for i in range(3)]
    out = augment_morph.assemble_morphed_timeline(
        segs,
        joint_root=tmp_path / "joints",
        fps=24,
        crf=18,
        preset="veryfast",
        pix_fmt="yuv420p",
        interp_fn=_stub_interp,
        weights=None,
        device="cpu",
    )
    # Trims [0,10) + [2,10) + [2,12) = 28 kept + 2x4 bridge = 36 = 3x12.
    assert _frame_count(out) == 36
    joints = sorted(
        path
        for path in (tmp_path / "joints").iterdir()
        if path.is_dir() and path.name.startswith("morph_")
    )
    assert len(joints) == 2


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 8,
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


def test_durable_morph_replaces_seams_with_trim_bridge_trim_order(
    tmp_path: Path,
) -> None:
    first = _make_segment(tmp_path, "000000", frames=12, checksum="abc123")
    second = _make_segment(tmp_path, "000001", frames=12, checksum="def456")
    work = tmp_path / "work"
    work.mkdir()
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    weights = SimpleNamespace(film=film, realesrgan=esrgan)
    calls: dict[str, list[Any]] = {"concat": []}

    def stub_poll(run_dir: Path, **kwargs: Any) -> Any:
        if "multiplier" in kwargs:
            return SimpleNamespace(chunks_done=0, chunks_waiting=0)
        return SimpleNamespace(chunks_done=0)

    def stub_drain(plan_dir: Path, **kwargs: Any) -> Any:
        intermediate = plan_dir / "model_intermediate.mp4"
        _make_clip(intermediate, 12)
        return SimpleNamespace(intermediate_mp4=intermediate, chunks_drained=1)

    def stub_concat(chunks: list[Path], dest: Path) -> Path:
        calls["concat"].append(list(chunks))
        dest.write_bytes(b"".join(chunk.read_bytes() for chunk in chunks))
        return dest

    def _seam(*args: object, **kwargs: object) -> object:
        raise AssertionError("mids-insert seams must not run for ltx morph")

    with patch("voyage.augment_finalize.weights_key_for", return_value="wkey"):
        final, final_fps = run_durable_model_pass(
            tmp_path,
            [first, second],
            out_width=64,
            out_height=64,
            source_fps=24.0,
            weights=weights,
            multiplier=1,
            chunk_frames=4,
            device="cuda:1",
            work_dir=work,
            upscale_poll_fn=stub_poll,
            interp_poll_fn=stub_poll,
            drain_fn=stub_drain,
            concat_fn=stub_concat,
            seam_interp_fn=_seam,
            morph_joints=True,
            morph_interp_fn=_stub_interp,
        )
    assert final_fps == 24
    assert _frame_count(final) == 24
    (concat_order,) = calls["concat"]
    names = [Path(p).name for p in concat_order]
    assert names[0].endswith("trim"), names
    assert names[1].startswith("bridge"), names
    assert names[2].endswith("trim"), names
    assert len(names) == 3

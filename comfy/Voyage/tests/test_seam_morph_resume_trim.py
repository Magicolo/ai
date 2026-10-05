"""Morph trim content keys (DESIGN §140 seamless continuation).

Why this module exists: `assemble_morphed_timeline` reused `seg_<i>_trim`
on exists+non-empty alone, so a re-rendered segment (same index, new
pixels) or a retuned recipe silently served the stale trim — the timeline
kept the old pictures while joints morphed the new ones. Trims now carry
a content-addressed sidecar (`morph_trim_key` beside the extensionless TS,
same atomic discipline as the joint `record.json`); these tests pin that
same-index/different-content re-renders while same-content resumes skip.
Real ffmpeg on synthetic clips (ffmpeg ships in every voyage image).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voyage import augment_morph


def _run_ffmpeg(*argv: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *argv], check=True)


def _make_clip(path: Path, frames: int, source: str = "testsrc", fps: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"{source}=size=64x64:rate={fps}:duration={frames / fps}",
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
    return path


def _gray_bridge_bytes() -> bytes:
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        framed = struct.pack(">I", len(data)) + tag + data
        return framed + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * 64 for _ in range(64))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _stub_bridge(
    anchor_a: Path, anchor_b: Path, weights: object, moment: float, device: str
) -> bytes:
    del anchor_a, anchor_b, weights, moment, device
    return _gray_bridge_bytes()


def _assemble(clips: list[Path], joint_root: Path) -> Path:
    return augment_morph.assemble_morphed_timeline(
        clips,
        joint_root=joint_root,
        fps=24,
        crf=18,
        preset="veryfast",
        pix_fmt="yuv420p",
        interp_fn=_stub_bridge,
        weights=None,
        device="cpu",
    )


def test_trim_keys_differ_by_content_and_recipe() -> None:
    """Trim keys fork on content, range, geometry, and every recipe knob."""
    base = {
        "video_sha": "a" * 64,
        "start_frame": 0,
        "end_frame": 10,
        "width": 64,
        "height": 64,
        "fps_key": 24,
        "crf": 18,
        "preset": "veryfast",
        "pixel_format": "yuv420p",
    }
    reference = augment_morph.morph_trim_key(**base)
    assert augment_morph.morph_trim_key(**{**base, "video_sha": "b" * 64}) != reference
    assert augment_morph.morph_trim_key(**{**base, "end_frame": 9}) != reference
    assert augment_morph.morph_trim_key(**{**base, "width": 128}) != reference
    assert augment_morph.morph_trim_key(**{**base, "fps_key": 32}) != reference
    assert augment_morph.morph_trim_key(**{**base, "crf": 15}) != reference
    assert augment_morph.morph_trim_key(**{**base, "preset": "fast"}) != reference
    assert augment_morph.morph_trim_key(**{**base, "pixel_format": "yuv444p"}) != reference
    assert augment_morph.morph_trim_key(**base) == reference


def test_same_content_resume_skips_trim_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same clips assembled twice: the second pass reuses keyed trims (no re-encode)."""
    import voyage.augment_morph as morph_module

    first_clip = _make_clip(tmp_path / "first.mp4", 12)
    second_clip = _make_clip(tmp_path / "second.mp4", 12, source="smptebars")
    joint_root = tmp_path / "joints"
    _assemble([first_clip, second_clip], joint_root)
    trim_record = joint_root / "trims" / "seg_000000_trim.json"
    assert trim_record.is_file()
    first_key = trim_record.read_text(encoding="utf-8")

    calls: list[str] = []
    real_trim = morph_module._trim_keep

    def _counting_trim(*args: object, **kwargs: object) -> Path:
        calls.append("trim")
        return real_trim(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(morph_module, "_trim_keep", _counting_trim)  # type: ignore[attr-defined]
    _assemble([first_clip, second_clip], joint_root)
    assert calls == []
    assert trim_record.read_text(encoding="utf-8") == first_key


def test_rerendered_segment_refreshes_its_trim(tmp_path: Path) -> None:
    """Same index with new pixels: the trim re-renders and the record forks."""
    first_clip = _make_clip(tmp_path / "first.mp4", 12)
    second_clip = _make_clip(tmp_path / "second.mp4", 12, source="smptebars")
    joint_root = tmp_path / "joints"
    _assemble([first_clip, second_clip], joint_root)
    trim_record = joint_root / "trims" / "seg_000000_trim.json"
    old_record = trim_record.read_text(encoding="utf-8")
    old_trim_bytes = (joint_root / "trims" / "seg_000000_trim").read_bytes()
    # Re-render the first segment in place: same path and frame count, new pixels.
    _make_clip(tmp_path / "first.mp4", 12, source="smptebars")
    _assemble([first_clip, second_clip], joint_root)
    new_record = trim_record.read_text(encoding="utf-8")
    assert new_record != old_record
    refreshed_trim = (joint_root / "trims" / "seg_000000_trim").read_bytes()
    assert refreshed_trim != old_trim_bytes

"""Source-res joint fix + uniform model-pass joints (DESIGN §140).

Order of operations for augmentations: seam fix -> upscale -> interpolate.
The fix stage renders morph 2+2 bridges at SOURCE resolution only — no
anchor upscale, no extra interpolation. The 4 bridge frames become a joint
source video that flows through the SAME upscale + interp pollers as
committed segments (standard ChunkKey ledger shapes in the joint's own
plan dir, keyed on the joint video sha), and the drain assembles
``[trimA, joint, trimB]`` with source-derived trims.

Continuity (multiplier ``m``, ``A``/``B`` source counts ``a``/``b``):
``A`` keeps model frames ``[0, (a-3)*m+1)`` — ending exactly at upscaled
``A[-3]``; the joint drains whole (``3m+1`` frames from its 4 bridges,
starting at bridge moment 1/5 just after ``A[-3]`` and ending at moment
4/5 just before ``B[+2]``); ``B`` keeps ``[2m, end)`` — starting exactly
at upscaled ``B[+2]``. No duplicated or dropped anchor frames.

Replaces the mids-insert seam path (post-model interpolated endpoints):
joints hold for every backend (no backend gate — the fix is model work,
not a backend feature).

Stdlib only at module scope (supervisor §12 GPU ban): pixels enter
through the caller-supplied ``interp_fn`` (tests) or the morph default's
function-local lazy torch import (production); ffmpeg drives extracts
and encodes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage import augment_morph as morph
from voyage.atomic import fsync_dir
from voyage.errors import MediaError
from voyage.hashing import sha256_file

JOINT_SOURCES_DIRNAME = "joint_sources"
"""Run-relative dir holding one subdir per adjacent segment pair."""

JOINT_DIR_TEMPLATE = "joint_{left:06d}_{right:06d}"
"""Joint subdir per adjacent usable position (content-keyed record inside)."""

JOINT_VIDEO_FILENAME = "joint.mp4"
"""Four source-res bridge frames (the joint unit the pollers consume)."""

JOINT_RECORD_FILENAME = "record.json"
"""Fix-stage ledger beside the joint video (fix-keyed resume, clash heals)."""

JOINT_TS_TEMPLATE = "joint_{index:06d}_ts"
"""Extensionless TS piece per joint in the assembly (mirrors trim pieces)."""

JOINT_FIX_KEY_PREFIX = "jointfix2x2"
"""Fix-key namespace (fresh: no pre-fix ledgers exist under this prefix)."""

JOINT_TS_KEY_PREFIX = "jointts"
"""Joint-TS piece key namespace (source content + recipe, fork-proof)."""

InterpFn = Callable[[Path, Path, Any, float, str], bytes]
"""(anchor_a_png, anchor_b_png, weights, moment, device) -> PNG bytes."""


@dataclass(frozen=True)
class JointUnit:
    """One rendered fix-stage joint between two adjacent usable segments."""

    joint_dir: Path
    joint_video: Path
    joint_key: str
    left_id: str
    right_id: str
    left_frames: int
    right_frames: int

    def as_source(self) -> Any:
        """The joint as a poller unit (SegmentSource shape, 4 source frames)."""
        from voyage.augment_upscale_poller import SegmentSource

        return SegmentSource(
            segment_id=self.joint_dir.name,
            segment_dir=self.joint_dir,
            video_path=self.joint_video,
            source_key=self.joint_key,
            total_frames=morph.MORPH_BRIDGE,
        )


def joint_fix_key(*, a_sha: str, b_sha: str, width: int, height: int, fps_key: int) -> str:
    """Content-addressed fix key (pair order + source geometry sensitive).

    Order matters (``A|B != B|A``); a re-rendered side or a settings
    change forks the key and forces a fresh fix instead of reusing a
    stale joint video.
    """
    return f"{JOINT_FIX_KEY_PREFIX}|{a_sha}|{b_sha}|{width}x{height}@{fps_key}"


def joint_sources_root(run_dir: Path) -> Path:
    """Run-relative root holding the per-pair joint dirs."""
    from voyage.augment_sidecar import AUGMENT_DIRNAME

    return run_dir / AUGMENT_DIRNAME / JOINT_SOURCES_DIRNAME


def existing_joint_videos(run_dir: Path) -> list[Path]:
    """Joint source videos already on disk (prune liveness without rendering)."""
    root = joint_sources_root(run_dir)
    if not root.is_dir():
        return []
    found = [
        video
        for child in sorted(root.iterdir())
        if child.is_dir() and (video := child / JOINT_VIDEO_FILENAME).is_file()
    ]
    return found


def render_joint_source(
    *,
    joint_dir: Path,
    video_a: Path,
    a_frames: int,
    video_b: Path,
    b_frames: int,
    source_key: str,
    source_fps: int,
    crf: int,
    preset: str,
    left_id: str,
    right_id: str,
    interp_fn: InterpFn | None = None,
    weights: Any = None,
    device: str = "cpu",
    interp_backend: str = "rife",
) -> JointUnit:
    """Render one joint's 4-frame source video (fresh work) or resume the ledger hit.

    Anchors are ``A[-3]``/``B[+2]`` in SOURCE coordinates (``morph_anchors``,
    fail loud on short sides); bridges render at source resolution via
    ``render_morph_once`` into ``joint_dir/bridges/`` (its own record lives
    there, namespaced away from the joint record) and encode to
    ``joint.mp4`` at the source fps with the run's chunk recipe. An
    exact-key record plus a non-empty 4-frame video is finished work and
    is never re-rendered; a key clash, short/empty output, or torn ledger
    re-renders (last-wins).
    """
    if not isinstance(joint_dir, Path):
        raise TypeError(f"joint_dir must be a Path (got {type(joint_dir).__name__})")
    joint_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = joint_dir / JOINT_RECORD_FILENAME
    joint_video = joint_dir / JOINT_VIDEO_FILENAME
    if ledger_path.is_file():
        try:
            record = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        if not isinstance(record, dict):
            record = {}
        if record.get("source_key") == source_key and _video_has_frames(joint_video, 4):
            return JointUnit(
                joint_dir=joint_dir,
                joint_video=joint_video,
                joint_key=record.get("joint_sha", ""),
                left_id=record.get("left_id", ""),
                right_id=record.get("right_id", ""),
                left_frames=int(record.get("left_frames", 0)),
                right_frames=int(record.get("right_frames", 0)),
            )
    anchor_a_index, anchor_b_index = morph.morph_anchors(a_frames, b_frames)
    anchor_a_png = joint_dir / "anchor_a.png"
    anchor_b_png = joint_dir / "anchor_b.png"
    morph._extract_frame_png(video_a, anchor_a_index, anchor_a_png)
    morph._extract_frame_png(video_b, anchor_b_index, anchor_b_png)
    bridges = morph.render_morph_once(
        joint_dir=joint_dir / "bridges",
        a_anchor=anchor_a_png,
        b_anchor=anchor_b_png,
        source_key=f"bridges|{source_key}",
        interp_fn=interp_fn,
        weights=weights,
        device=device,
        interp_backend=interp_backend,
    )
    _encode_joint_video(bridges.frames, joint_video, source_fps, crf, preset)
    joint_sha = sha256_file(joint_video)
    _write_joint_record(
        ledger_path,
        source_key,
        joint_sha,
        left_id,
        right_id,
        a_frames,
        b_frames,
    )
    return JointUnit(
        joint_dir=joint_dir,
        joint_video=joint_video,
        joint_key=joint_sha,
        left_id=left_id,
        right_id=right_id,
        left_frames=a_frames,
        right_frames=b_frames,
    )


def ensure_joint_units(
    run_dir: Path,
    ordered_sources: list[Any],
    *,
    source_fps: int,
    crf: int,
    preset: str,
    interp_fn: InterpFn | None = None,
    weights: Any = None,
    device: str = "cpu",
    interp_backend: str = "rife",
) -> list[JointUnit]:
    """Fix every adjacent pair (ledger-hit when already rendered), in order.

    ``ordered_sources`` are poller SegmentSources in presentation order
    (segment_id order); geometry probes once per source video. A lone
    source yields no joints. Raises on short sides (``morph_anchors``)
    — a segment too short to anchor is a fail-loud data error, never a
    silent skip.
    """
    units: list[JointUnit] = []
    root = joint_sources_root(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    dimensions: dict[str, tuple[int, int]] = {}
    checksums: dict[str, str] = {}

    def _dimensions(video: Path) -> tuple[int, int]:
        key = str(video)
        if key not in dimensions:
            dimensions[key] = morph._probe_size(video)
        return dimensions[key]

    def _checksum(video: Path) -> str:
        key = str(video)
        if key not in checksums:
            checksums[key] = sha256_file(video)
        return checksums[key]

    for position in range(len(ordered_sources) - 1):
        first = ordered_sources[position]
        second = ordered_sources[position + 1]
        width, height = _dimensions(first.video_path)
        other_width, other_height = _dimensions(second.video_path)
        if (width, height) != (other_width, other_height):
            raise MediaError(
                f"joint {first.segment_id}|{second.segment_id} mixes source geometry "
                f"{width}x{height} vs {other_width}x{other_height} "
                "(segments in one run must share geometry)"
            )
        fix_key = joint_fix_key(
            a_sha=_checksum(first.video_path),
            b_sha=_checksum(second.video_path),
            width=width,
            height=height,
            fps_key=source_fps,
        )
        joint_dir = root / JOINT_DIR_TEMPLATE.format(left=position, right=position + 1)
        unit = render_joint_source(
            joint_dir=joint_dir,
            video_a=first.video_path,
            a_frames=first.total_frames,
            video_b=second.video_path,
            b_frames=second.total_frames,
            source_key=fix_key,
            source_fps=source_fps,
            crf=crf,
            preset=preset,
            left_id=first.segment_id,
            right_id=second.segment_id,
            interp_fn=interp_fn,
            weights=weights,
            device=device,
            interp_backend=interp_backend,
        )
        units.append(unit)
    return units


def segment_keep(
    index: int, segment_count: int, source_frames: int, multiplier: int
) -> tuple[int, int]:
    """Model-frame keep range for one drained segment intermediate.

    Interior cuts land on upscaled anchors: the tail cut ``(a-3)*m+1``
    keeps through ``A[-3]`` (the joint's first bridge moment follows
    it); the head cut ``2m`` starts at ``B[+2]`` (the joint's last
    bridge moment precedes it). First/last segments keep their outer
    edge whole. Fail loud on an empty keep (segments too short to
    joint — real segments are 96+ frames).
    """
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or isinstance(segment_count, bool)
        or not isinstance(segment_count, int)
        or isinstance(source_frames, bool)
        or not isinstance(source_frames, int)
        or isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
    ):
        raise TypeError("segment_keep needs int index/segment_count/source_frames/multiplier")
    if segment_count < 1 or source_frames < 1 or multiplier < 1:
        raise ValueError("segment_keep needs segment_count/source_frames/multiplier >= 1")
    if not 0 <= index < segment_count:
        raise ValueError(f"segment_keep index {index} outside [0, {segment_count})")
    start = 2 * multiplier if index > 0 else 0
    if index < segment_count - 1:
        end = (source_frames - 3) * multiplier + 1
    else:
        end = (source_frames - 1) * multiplier + 1
    if end - start <= 0:
        raise MediaError(
            f"joint trim of a {source_frames}f segment keeps no model frames "
            f"[{start}, {end}) (segment too short to joint at multiplier {multiplier})"
        )
    return (start, end)


def assemble_joint_timeline(
    segment_intermediates: list[Path],
    joint_intermediates: list[Path],
    *,
    source_counts: list[int],
    multiplier: int,
    joint_root: Path,
    fps: int,
    crf: int,
    preset: str,
    pix_fmt: str,
    concat_fn: morph.ConcatFn | None = None,
) -> Path:
    """Assemble ``[trimA, joint, trimB, ...]`` into one timeline (TS pieces).

    Segment trims come from ``segment_keep`` ranges re-encoded via
    ``_trim_keep`` (content-keyed trim ledger, same as the morph path);
    joint intermediates drain whole — decoded to PNGs and re-encoded to
    TS under a ``jointts|`` key (source content + recipe, fork-proof).
    Pieces join via ``concat_fn`` (default: stream-copy concat). A lone
    segment passes through untouched (no joints, no re-encode).
    """
    if not isinstance(segment_intermediates, list) or not segment_intermediates:
        raise ValueError(
            f"joint assembly needs at least one segment (got {segment_intermediates!r})"
        )
    if len(joint_intermediates) != len(segment_intermediates) - 1:
        raise ValueError(
            f"joint assembly needs one joint per boundary "
            f"(got {len(segment_intermediates)} segments, {len(joint_intermediates)} joints)"
        )
    if len(source_counts) != len(segment_intermediates):
        raise ValueError("source_counts must cover every segment intermediate")
    rate = _require_rate(fps)
    joint_root.mkdir(parents=True, exist_ok=True)
    join = concat_fn if concat_fn is not None else morph._default_concat
    if len(segment_intermediates) == 1:
        return segment_intermediates[0]
    trim_dir = joint_root / "trims"
    trim_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []
    checksum_cache: dict[str, str] = {}

    def _checksum(video: Path) -> str:
        key = str(video)
        if key not in checksum_cache:
            checksum_cache[key] = sha256_file(video)
        return checksum_cache[key]

    for index, (video, count) in enumerate(zip(segment_intermediates, source_counts, strict=True)):
        start, end = segment_keep(index, len(segment_intermediates), count, multiplier)
        width, height = morph._probe_size(video)
        trim = trim_dir / f"seg_{index:06d}_trim"
        trim_key = morph.morph_trim_key(
            video_sha=_checksum(video),
            start_frame=start,
            end_frame=end,
            width=width,
            height=height,
            fps_key=rate,
            crf=crf,
            preset=preset,
            pixel_format=pix_fmt,
        )
        trim_record = morph._trim_record_path(trim)
        stored = morph._read_trim_record(trim_record)
        if not (trim.exists() and trim.stat().st_size > 0 and stored.get("source_key") == trim_key):
            trim_record.unlink(missing_ok=True)
            morph._trim_keep(video, start, end, trim, rate, crf, preset, pix_fmt)
            morph._write_trim_record(trim_record, trim_key, end - start)
        pieces.append(trim)
        if index < len(joint_intermediates):
            pieces.append(
                _joint_ts_piece(
                    joint_intermediates[index],
                    trim_dir / JOINT_TS_TEMPLATE.format(index=index),
                    rate,
                    crf,
                    preset,
                    pix_fmt,
                    checksum_cache,
                )
            )
    final = joint_root / "jointed_timeline.mp4"
    join(pieces, final)
    expected = sum(
        end - start
        for start, end in (
            segment_keep(i, len(segment_intermediates), c, multiplier)
            for i, c in enumerate(source_counts)
        )
    ) + sum(morph._probe_frames(joint) for joint in joint_intermediates)
    if morph._probe_frames(final) != expected:
        raise MediaError(
            f"jointed timeline holds {morph._probe_frames(final)} frames "
            f"(expected {expected}; inputs changed mid-assemble?)"
        )
    fsync_dir(joint_root)
    return final


def _joint_ts_piece(
    joint_mp4: Path,
    dest: Path,
    rate: int,
    crf: int,
    preset: str,
    pix_fmt: str,
    checksum_cache: dict[str, str],
) -> Path:
    """Re-encode one drained joint intermediate to TS (ledgered, fork-proof)."""
    from voyage.augment import ffmpeg_decode_chunk

    key = str(joint_mp4)
    if key not in checksum_cache:
        checksum_cache[key] = sha256_file(joint_mp4)
    joint_key = f"{JOINT_TS_KEY_PREFIX}|{checksum_cache[key]}|{rate}|crf{crf}|{preset}|{pix_fmt}"
    record_path = morph._trim_record_path(dest)
    stored = morph._read_trim_record(record_path)
    if dest.exists() and dest.stat().st_size > 0 and stored.get("source_key") == joint_key:
        return dest
    record_path.unlink(missing_ok=True)
    if dest.exists():
        dest.unlink(missing_ok=True)
    frame_count = morph._probe_frames(joint_mp4)
    work_dir = dest.parent / f"{dest.name}.pngs"
    if work_dir.exists():
        import shutil

        shutil.rmtree(work_dir)
    frames = ffmpeg_decode_chunk(joint_mp4, work_dir, 0, frame_count, fps=None)
    morph._encode_ts(str(work_dir / "frame_%06d.png"), dest, rate, crf, preset, pix_fmt)
    if not frames:
        raise MediaError(f"joint TS decode produced no frames for {joint_mp4}")
    morph._write_trim_record(record_path, joint_key, frame_count)
    return dest


def _encode_joint_video(
    bridge_pngs: tuple[Path, ...] | list[Path],
    dest: Path,
    source_fps: int,
    crf: int,
    preset: str,
) -> Path:
    """Encode the 4 source-res bridges to the joint source video (chunk recipe)."""
    from voyage.augment import ffmpeg_encode_chunk

    frames = list(bridge_pngs)
    if len(frames) != morph.MORPH_BRIDGE:
        raise MediaError(f"joint fix holds {len(frames)} bridges (expected {morph.MORPH_BRIDGE})")
    pattern = frames[0].parent / "bridge_%02d.png"
    if not all(path.parent == frames[0].parent for path in frames):
        raise MediaError("joint bridges span directories (render went somewhere unexpected)")
    return ffmpeg_encode_chunk(pattern, dest, source_fps, crf=crf, preset=preset)


def _video_has_frames(video: Path, count: int) -> bool:
    """True when `video` exists, is non-empty, and probes to `count` frames."""
    if not video.is_file():
        return False
    try:
        if video.stat().st_size == 0:
            return False
        return morph._probe_frames(video) == count
    except (OSError, MediaError):
        return False


def _write_joint_record(
    record_path: Path,
    source_key: str,
    joint_sha: str,
    left_id: str,
    right_id: str,
    left_frames: int,
    right_frames: int,
) -> None:
    """Atomically write one joint fix record (partial + replace + fsync)."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    partial = record_path.with_name(f"{record_path.name}.partial")
    partial.write_text(
        json.dumps(
            {
                "source_key": source_key,
                "joint_sha": joint_sha,
                "joint_frames": morph.MORPH_BRIDGE,
                "left_id": left_id,
                "right_id": right_id,
                "left_frames": left_frames,
                "right_frames": right_frames,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    os.replace(partial, record_path)
    fsync_dir(record_path.parent)


def _require_rate(fps: int | float) -> int:
    """Positive integer fps (bool rejected — `True` is an int that means 1)."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TypeError(f"fps must be a number (got {type(fps).__name__})")
    import math

    rate = int(round(fps))
    if not math.isfinite(float(fps)) or rate <= 0:
        raise ValueError(f"fps must be finite and > 0 (got {fps!r})")
    return rate

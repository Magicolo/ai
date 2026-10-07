"""Morph-cut 2+2 joint assembly for the no-model-pass native path (DESIGN §140).

Per segment joint, ``A[-2:]+B[:2]`` are replaced by 4 bridge frames
morphed between anchors ``A[-3]`` and ``B[+2]`` of the passed mp4s.
Frame count is unchanged, so committed audio timelines stay untouched.
The durable model pass fixes joints at source resolution instead (see
`voyage.augment_joints`) and shares this module's trim/TS/concat
helpers — but never calls the assembly below.

Ephemeral-harness numbers behind the recipe: morph 2+2 gaps ~3 meanabs,
through-bridge mean steps at within-motion level, fullres transfer verified;
feathered masks and wider carries showed no further gain.

Stdlib only at module scope (supervisor §12 GPU ban): torch enters through
the caller-supplied `interp_fn` (tests) or the default FILM render's
function-local lazy import (production). ffmpeg drives
anchor extraction, trims, and encodes.

Joint pieces are extensionless MPEG-TS (explicit `-f mpegts` on encode):
unlike MP4, TS byte-concatenates into a decodable stream, so the injected
`concat_fn` seam in tests (byte-join) decodes the exact frame total while
production's concat-demuxer stream copy accepts the same pieces. Trim
filenames carry no extension by contract (`*_trim`, `bridge_*`).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage.atomic import fsync_dir
from voyage.errors import MediaError, StateError

MORPH_BACKENDS = frozenset({"ltx25", "ltx23"})
"""Backends whose finalize joints get morph-cuts (user decision: ltx pair only)."""

MORPH_KEY_VERSION = "morph2x2-v2"
"""Joint key version: v2 anchors bridges on source-boundary frames (v1
used post-model intermediate anchors). Bumping the key orphans v1
bridges — the record clash auto-heals by re-rendering, never migrating."""

MORPH_ANCHOR_A = 3
"""`A[-3]`: last frozen-context frame feeding the bridge (eyeball winner 2+2)."""

MORPH_ANCHOR_B = 2
"""`B[+2]`: first fresh frame feeding the bridge."""

MORPH_DROP_A = 2
"""Frames dropped from A's tail per joint (`A[-2:]`)."""

MORPH_DROP_B = 2
"""Frames dropped from B's head per joint (`B[:2]`)."""

MORPH_BRIDGE = 4
"""FILM bridge frames per joint (moments 1/5..4/5 between the anchors)."""

MORPH_RECORD_FILENAME = "record.json"
"""Ledger beside the bridge PNGs (source-keyed resume, clash auto-heals)."""

MORPH_JOINT_TEMPLATE = "morph_{left:06d}_{right:06d}"
"""Joint dir per adjacent segment position (content-keyed record inside)."""

MORPH_TRIM_TEMPLATE = "seg_{index:06d}_trim"
"""Per-segment trim piece (extensionless TS; middle segments lose both ends)."""

MORPH_BRIDGE_FILENAME = "bridge"
"""Bridge piece beside the trims (extensionless TS, 4 frames)."""

MORPHED_FILENAME = "morphed_timeline.mp4"
"""Assembled timeline (concat of trim/bridge pieces in segment order)."""

MORPH_MOMENTS = tuple((position + 1) / (MORPH_BRIDGE + 1) for position in range(MORPH_BRIDGE))
"""Bridge moments 1/5..4/5 (direct FILM times, no bisection needed)."""

InterpFn = Callable[[Path, Path, Any, float, str], bytes]
"""(anchor_a_png, anchor_b_png, weights, moment, device) -> PNG bytes, one frame."""

ConcatFn = Callable[[list[Path], Path], Path]
"""(pieces_in_order, dest) -> dest; production uses stream-copy concat."""


@dataclass(frozen=True)
class MorphRender:
    """Four ledgered bridge PNGs for one joint."""

    frames: tuple[Path, ...]


def morph_enabled_for_backend(backend: str) -> bool:
    """True only for the ltx pair (all other backends keep plain concat)."""
    return backend in MORPH_BACKENDS


def morph_backend_for_run(run_dir: Path) -> str | None:
    """Backend for morph gating, read from the run manifest (None = plain path).

    Any missing/unreadable manifest reads as None — finalize must never
    break runs that predate the morph, it just concats them as before.
    """
    try:
        from voyage.persistence import read_effective_config
    except ImportError:
        return None
    try:
        config = read_effective_config(run_dir)
    except StateError:
        return None
    backend = config.video.backend
    if not morph_enabled_for_backend(backend):
        return None
    return backend


def morph_joint_key(*, a_sha: str, b_sha: str, width: int, height: int, fps_key: int) -> str:
    """Content-addressed joint key (pair order + recipe sensitive, fork-proof).

    Order matters (`A|B != B|A`); geometry/fps fork the key so a re-finalize
    at new settings never reuses a stale bridge. The version prefix
    orphans pre-source-anchor (v1) bridges.
    """
    return f"{MORPH_KEY_VERSION}|{a_sha}|{b_sha}|{width}x{height}@{fps_key}"


def morph_anchors(a_count: int, b_count: int) -> tuple[int, int]:
    """Anchor indices (`A[-3]`, `B[+2]`) for a joint (fail loud on short sides)."""
    if a_count < MORPH_ANCHOR_A or b_count <= MORPH_ANCHOR_B:
        raise ValueError(
            f"morph needs A >= {MORPH_ANCHOR_A} and B > {MORPH_ANCHOR_B} frames "
            f"(got {a_count}, {b_count})"
        )
    return (a_count - MORPH_ANCHOR_A, MORPH_ANCHOR_B)


def morph_trims(a_count: int, b_count: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Joint-local keep ranges: A keeps `[0, n-2)`, B keeps `[2, n)`."""
    if a_count <= MORPH_DROP_A or b_count <= MORPH_DROP_B:
        raise ValueError(
            f"morph needs more than {MORPH_DROP_A}/{MORPH_DROP_B} frames per side "
            f"(got {a_count}, {b_count})"
        )
    return ((0, a_count - MORPH_DROP_A), (MORPH_DROP_B, b_count))


def morph_counts(counts: list[int]) -> tuple[int, int]:
    """(kept, bridged) frame totals: every joint swaps 4 kept for 4 bridge."""
    if not counts:
        raise ValueError("morph needs at least one segment count")
    joints = len(counts) - 1
    bridged = MORPH_BRIDGE * joints
    return (sum(counts) - bridged, bridged)


def resolve_morph_device(
    explicit: str | None = None, *, visible: tuple[str, ...] | None = None
) -> str:
    """Device for the interp bridge render (explicit wins, else augment order, else CPU).

    Mirrors the finalize interp pinning (cuda:1 on the 2-GPU box, sharing
    the 2060 with the upscale leg and the llama sidecar) via
    `interp_pass_devices`; a box with no CUDA
    falls back to CPU (slow but working) instead of failing the finalize.
    """
    if explicit:
        return explicit
    from voyage.augment import interp_pass_devices

    devices = interp_pass_devices(devices=visible)
    candidate = devices[0] if devices else "cpu"
    if candidate.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return candidate
        except ImportError:
            pass
        return "cpu"
    return candidate


def _sha_file(path: Path) -> str:
    from voyage.hashing import sha256_file

    return sha256_file(path)


def _run_ffmpeg(argv: list[str], purpose: str) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaError(f"morph {purpose} failed: {proc.stderr[-2000:]}")


def _probe_frames(video: Path) -> int:
    proc = subprocess.run(
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
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"morph frame probe failed for {video}: {proc.stderr[-2000:]}")
    try:
        counts = [int(line) for line in proc.stdout.splitlines() if line.strip()]
    except ValueError:
        raise MediaError(f"morph frame probe unparseable for {video}: {proc.stdout!r}") from None
    if not counts:
        raise MediaError(f"morph found no frames in {video}")
    if any(count != counts[0] for count in counts):
        # MPEG-TS probes print per-program lines; they must agree.
        raise MediaError(f"morph frame probe disagrees for {video}: {proc.stdout!r}")
    if counts[0] <= 0:
        raise MediaError(f"morph found no frames in {video}")
    return counts[0]


def _probe_size(video: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"morph size probe failed for {video}: {proc.stderr[-2000:]}")
    try:
        width_text, height_text = proc.stdout.strip().split(",")
        return (int(width_text), int(height_text))
    except ValueError:
        raise MediaError(f"morph size probe unparseable for {video}: {proc.stdout!r}") from None


def _extract_frame_png(video: Path, index: int, dest: Path) -> Path:
    """Extract one frame by index as PNG (fail loud on empty output)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select='eq(n\\,{index})'",
            "-frames:v",
            "1",
            str(dest),
        ],
        f"anchor extract {video.name} frame {index}",
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"morph anchor extract produced empty {dest}")
    return dest


def _encode_ts(
    frames_pattern: str, dest: Path, fps: int, crf: int, preset: str, pix_fmt: str
) -> Path:
    """Encode a PNG sequence to extensionless MPEG-TS (byte-concatenable pieces)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frames_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            pix_fmt,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-f",
            "mpegts",
            str(dest),
        ],
        f"TS piece encode {dest.name}",
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"morph TS encode produced empty output {dest}")
    return dest


def _trim_keep(
    video: Path, start: int, end: int, dest: Path, fps: int, crf: int, preset: str, pix_fmt: str
) -> Path:
    """Frame-accurate keep of `[start, end)` re-encoded to extensionless TS."""
    count = end - start
    if count <= 0:
        raise ValueError(f"morph trim needs a positive keep (got [{start}, {end}))")
    if start == 0:
        argv: list[str] = ["-i", str(video), "-frames:v", str(count)]
    else:
        # select+setpts re-stamps the kept range onto clean zero-based
        # timestamps (integer-exact, no float seek): a bare select keeps
        # source timestamps and vsync pads the head back up. -frames:v caps
        # the tail (select alone is open-ended).
        argv = [
            "-i",
            str(video),
            "-vf",
            f"select='gte(n\\,{start})',setpts=N/FRAME_RATE/TB",
            "-frames:v",
            str(count),
        ]
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            *argv,
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            pix_fmt,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-an",
            "-f",
            "mpegts",
            str(dest),
        ],
        f"trim {video.name} [{start}, {end})",
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"morph trim produced empty output {dest}")
    return dest


def _write_concat_list(entries: list[Path], dest: Path) -> Path:
    """Concat-demuxer list (single-quote escaping mirrors the chunk helper)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        f"file '{str(entry).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
        for entry in entries
    )
    dest.write_text(lines, encoding="utf-8")
    return dest


def _default_concat(pieces: list[Path], dest: Path) -> Path:
    """Stream-copy concat of same-codec pieces (TS or mp4 alike)."""
    concat_list = _write_concat_list(pieces, dest.parent / "pieces.txt")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(dest),
        ],
        f"morphed concat {dest.name}",
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError(f"morphed concat produced empty output {dest}")
    return dest


def _default_morph_pngs(
    a_anchor: Path,
    b_anchor: Path,
    joint_dir: Path,
    *,
    weights_path: Path,
    device: str,
    interp_backend: str = "rife",
) -> list[Path]:
    """Render the 4 bridge PNGs via the resident interp leg (lazy torch import)."""
    from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames
    from voyage.workers.augment_worker import (
        interpolate_mids,
        interpolate_rife_mids,
        validate_interp_backend,
    )

    validate_interp_backend(interp_backend)
    frames = load_png_frames_as_tensors([a_anchor, b_anchor])
    if interp_backend == "rife":
        tensors = interpolate_rife_mids(frames, weights_path, moments=MORPH_MOMENTS, device=device)
    else:
        tensors = interpolate_mids(frames, weights_path, moments=MORPH_MOMENTS, device=device)
    written = write_tensors_as_png_frames(tensors, joint_dir / "tensors")
    renamed = []
    for position, tensor_png in enumerate(written):
        dest = joint_dir / f"bridge_{position:02d}.png"
        if dest.exists():
            dest.unlink()
        tensor_png.rename(dest)
        renamed.append(dest)
    with contextlib.suppress(OSError):
        (joint_dir / "tensors").rmdir()
    return renamed


MORPH_TRIM_RECORD_SUFFIX = ".json"
"""Trim sidecar suffix: `seg_000000_trim` + `.json` beside the extensionless TS.

Why a per-trim JSON (not a shared `chunks.jsonl`): trims are single-file
TS pieces, not chunk PNG dirs — forcing the `ChunkKey` ledger shape would
bury a one-file identity under chunk/window fields it never uses. A
lightweight JSON beside each trim mirrors the joint `record.json` pattern
(source-keyed, atomic partial+replace, fsynced) and keeps the trim dir
human-inspectable (`seg_000000_trim` + `seg_000000_trim.json`).
"""


def morph_trim_key(
    *,
    video_sha: str,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
    fps_key: int,
    crf: int,
    preset: str,
    pixel_format: str,
) -> str:
    """Content-addressed trim key (segment content + keep range + recipe, fork-proof).

    Same-index/different-content reuses (re-rendered segment, retuned
    recipe) must never serve a stale trim: the key binds the source video
    checksum, the exact `[start, end)` keep, the probed geometry, and the
    full encode recipe (fps, crf, preset, pixel format), so any of them
    changing forks the key and forces a fresh `_trim_keep`.
    """
    return (
        f"trim|{video_sha}|{start_frame}-{end_frame}|"
        f"{width}x{height}@{fps_key}|crf{crf}|{preset}|{pixel_format}"
    )


def _trim_record_path(trim_path: Path) -> Path:
    """Sidecar JSON path for one trim (same dir, trim name + `.json`)."""
    return trim_path.parent / (trim_path.name + MORPH_TRIM_RECORD_SUFFIX)


def _read_trim_record(record_path: Path) -> dict[str, object]:
    """Best-effort trim record read (torn/missing/non-dict → empty, heal below).

    Why lenient: a crash between trim write and record write leaves a
    valid trim with no record, and a crash mid-record-write leaves a torn
    JSON — both are healable by re-rendering, never fail-loud here.
    """
    try:
        parsed = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_trim_record(record_path: Path, source_key: str, trim_frames: int) -> None:
    """Atomically write one trim record (partial + replace + fsync, joint pattern)."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    partial = record_path.with_name(f"{record_path.name}.partial")
    partial.write_text(
        json.dumps({"source_key": source_key, "trim_frames": trim_frames}, indent=1),
        encoding="utf-8",
    )
    os.replace(partial, record_path)
    fsync_dir(record_path.parent)


def _drop_morph_outputs(joint_dir: Path) -> None:
    """Remove stale bridge PNGs plus the ledger so a heal re-renders clean.

    Why a helper: both heal paths (key clash, hit-but-short output) need
    the same drop — bridges for the wrong pair must never survive beside
    a fresh record, and anchor PNGs are deliberately kept (the caller
    re-extracts them only on a key change; on a same-key heal they are
    still valid inputs for the re-render).
    """
    for stale_bridge in sorted(joint_dir.glob("bridge_*.png")):
        with contextlib.suppress(OSError):
            if stale_bridge.is_file() or stale_bridge.is_symlink():
                stale_bridge.unlink()
    with contextlib.suppress(OSError):
        (joint_dir / MORPH_RECORD_FILENAME).unlink()


def render_morph_once(
    *,
    joint_dir: Path,
    a_anchor: Path,
    b_anchor: Path,
    source_key: str,
    interp_fn: InterpFn | None = None,
    weights: Any = None,
    device: str = "cpu",
    interp_backend: str = "rife",
) -> MorphRender:
    """Render one joint's 4 bridge PNGs (fresh work) or resume the ledger hit.

    Ledger-truth plus output-truth resume (DESIGN §§56-57, chunk-poller
    parity): an exact-key record plus its 4 complete bridge PNGs on disk
    is finished work and is never re-rendered. A ledger hit whose output
    is gone, short, or empty — and a key clash from a re-rendered side —
    drops the stale bridges plus the ledger and re-renders instead of
    failing loud (last-wins: the fresh record overwrites the stale one,
    so a wrong-pair bridge can never survive beside a new key). A torn
    or unreadable ledger heals the same way — the interrupted joint has
    no valid record and is simply redone. Never raises for a healable
    mismatch; only genuine render failures (wrong bridge count, empty
    output, missing interp weights) fail loud.
    """
    if not isinstance(joint_dir, Path):
        raise TypeError(f"joint_dir must be a Path (got {type(joint_dir).__name__})")
    joint_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = joint_dir / MORPH_RECORD_FILENAME
    expected = [joint_dir / f"bridge_{position:02d}.png" for position in range(MORPH_BRIDGE)]
    if ledger_path.is_file():
        try:
            record = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        if not isinstance(record, dict):
            record = {}
        if record.get("source_key") == source_key:
            found = [path for path in expected if path.is_file() and path.stat().st_size > 0]
            if len(found) == MORPH_BRIDGE and record.get("bridge_count") == MORPH_BRIDGE:
                return MorphRender(frames=tuple(expected))
        # Auto-heal: key clash, hit-but-short output, or torn ledger —
        # drop stale bridges plus the record and fall through to re-render.
        _drop_morph_outputs(joint_dir)
    if interp_fn is None:
        from voyage.workers import augment_worker

        backend = augment_worker.validate_interp_backend(interp_backend)
        if weights is None:
            raise ValueError("morph default render needs weights (interp leg)")
        written = _default_morph_pngs(
            a_anchor,
            b_anchor,
            joint_dir,
            weights_path=weights,
            device=device,
            interp_backend=backend,
        )
    else:
        written = []
        for position, moment in enumerate(MORPH_MOMENTS):
            data = interp_fn(a_anchor, b_anchor, weights, moment, device)
            dest = joint_dir / f"bridge_{position:02d}.png"
            dest.write_bytes(data)
            if dest.stat().st_size == 0:
                raise MediaError(f"morph bridge render produced empty {dest}")
            written.append(dest)
    if len(written) != MORPH_BRIDGE:
        raise MediaError(f"morph rendered {len(written)} bridges, expected {MORPH_BRIDGE}")
    record = {"source_key": source_key, "bridge_count": MORPH_BRIDGE}
    partial = ledger_path.with_name(f"{ledger_path.name}.partial")
    partial.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(partial, ledger_path)
    fsync_dir(joint_dir)
    return MorphRender(frames=tuple(written))


def assemble_morphed_timeline(
    segment_mp4s: list[Path],
    *,
    joint_root: Path,
    fps: int,
    crf: int,
    preset: str,
    pix_fmt: str,
    interp_fn: InterpFn | None = None,
    weights: Any = None,
    device: str = "cpu",
    concat_fn: ConcatFn | None = None,
    interp_backend: str = "rife",
) -> Path:
    """Assemble a count-preserving morphed timeline over segment mp4s.

    Per joint: anchors `A[-3]`/`B[+2]` of the passed mp4s morph into 4
    bridge frames (ledgered under `joint_root/morph_II_JJ/`); trims keep
    `A[:-2]`/`B[2:]` (middle segments lose both ends), each trim keyed on
    its source content plus keep range plus recipe (`morph_trim_key`
    beside the TS piece) so a re-rendered segment or retuned recipe
    never reuses a stale trim. Pieces join as `[trim_0, bridge_0,
    trim_1, ...]` via `concat_fn` (default: stream-copy concat). Frame
    total is unchanged, so audio needs no work. A lone segment concats
    through untouched (no joints, no interp calls).

    This is the no-model-pass native path (media.py): anchors come from
    the passed mp4s themselves. The durable model pass fixes joints at
    source resolution instead (see `voyage.augment_joints`) and never
    calls this with external anchors.
    """
    if not isinstance(segment_mp4s, list) or not segment_mp4s:
        raise ValueError(f"morph needs at least one segment mp4 (got {segment_mp4s!r})")
    from voyage.workers import augment_worker

    backend = augment_worker.validate_interp_backend(interp_backend)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(f"morph fps must be a positive number (got {fps!r})")
    rate = int(round(fps))
    joint_root.mkdir(parents=True, exist_ok=True)
    join = concat_fn if concat_fn is not None else _default_concat
    counts = [_probe_frames(video) for video in segment_mp4s]
    if len(segment_mp4s) == 1:
        # Lone segment: nothing to morph — pass through untouched so
        # joint_root stays empty (no joints, no interp calls, count kept).
        return segment_mp4s[0]
    trim_dir = joint_root / "trims"
    trim_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []
    checksum_by_path: dict[str, str] = {}
    dimensions_by_path: dict[str, tuple[int, int]] = {}

    def _cached_checksum(video_path: Path) -> str:
        cache_key = str(video_path)
        if cache_key not in checksum_by_path:
            checksum_by_path[cache_key] = _sha_file(video_path)
        return checksum_by_path[cache_key]

    def _cached_dimensions(video_path: Path) -> tuple[int, int]:
        cache_key = str(video_path)
        if cache_key not in dimensions_by_path:
            dimensions_by_path[cache_key] = _probe_size(video_path)
        return dimensions_by_path[cache_key]

    for index, (video, count) in enumerate(zip(segment_mp4s, counts, strict=True)):
        start = 0 if index == 0 else MORPH_DROP_B
        end = count if index == len(segment_mp4s) - 1 else count - MORPH_DROP_A
        if end - start <= 0:
            raise MediaError(f"morph trim of {video.name} keeps no frames ({count}f)")
        trim = trim_dir / (MORPH_TRIM_TEMPLATE.format(index=index))
        trim_width, trim_height = _cached_dimensions(video)
        trim_key = morph_trim_key(
            video_sha=_cached_checksum(video),
            start_frame=start,
            end_frame=end,
            width=trim_width,
            height=trim_height,
            fps_key=rate,
            crf=crf,
            preset=preset,
            pixel_format=pix_fmt,
        )
        trim_record = _trim_record_path(trim)
        stored_trim = _read_trim_record(trim_record)
        if not (
            trim.exists() and trim.stat().st_size > 0 and stored_trim.get("source_key") == trim_key
        ):
            # Stale or legacy trim (missing/torn/mismatched record) — drop
            # the record first so a crash during `_trim_keep` leaves no
            # record (next run heals by re-rendering), then re-encode and
            # publish the fresh key.
            with contextlib.suppress(OSError):
                trim_record.unlink()
            _trim_keep(video, start, end, trim, rate, crf, preset, pix_fmt)
            _write_trim_record(trim_record, trim_key, end - start)
        pieces.append(trim)
        if index < len(segment_mp4s) - 1:
            other = segment_mp4s[index + 1]
            other_count = counts[index + 1]
            anchor_a, anchor_b = morph_anchors(count, other_count)
            joint_dir = joint_root / MORPH_JOINT_TEMPLATE.format(left=index, right=index + 1)
            width, height = _cached_dimensions(video)
            key = morph_joint_key(
                a_sha=_cached_checksum(video),
                b_sha=_cached_checksum(other),
                width=width,
                height=height,
                fps_key=rate,
            )
            anchor_a_png = joint_dir / "anchor_a.png"
            anchor_b_png = joint_dir / "anchor_b.png"
            rendered = True
            ledger_path = joint_dir / MORPH_RECORD_FILENAME
            # The bridge pixels depend on the interp backend, so the key
            # carries it — switching backends re-renders instead of
            # reusing the other backend's bridges.
            joint_key = f"{backend}|{key}"
            if ledger_path.is_file():
                try:
                    record = json.loads(ledger_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    record = {}
                rendered = record.get("source_key") != joint_key
            if rendered:
                _extract_frame_png(video, anchor_a, anchor_a_png)
                _extract_frame_png(other, anchor_b, anchor_b_png)
            render_morph_once(
                joint_dir=joint_dir,
                a_anchor=anchor_a_png,
                b_anchor=anchor_b_png,
                source_key=joint_key,
                interp_fn=interp_fn,
                weights=weights,
                device=device,
                interp_backend=backend,
            )
            bridge = joint_dir / MORPH_BRIDGE_FILENAME
            if not (bridge.exists() and bridge.stat().st_size > 0):
                pattern = str(joint_dir / "bridge_%02d.png")
                found = sorted(joint_dir.glob("bridge_*.png"))
                if len(found) != MORPH_BRIDGE:
                    raise MediaError(f"morph joint {joint_dir.name} holds {len(found)} bridges")
                _encode_ts(pattern, bridge, rate, crf, preset, pix_fmt)
            pieces.append(bridge)
    final = joint_root / MORPHED_FILENAME
    join(pieces, final)
    kept, bridged = morph_counts(counts)
    if _probe_frames(final) != kept + bridged:
        raise MediaError(
            f"morphed timeline holds {_probe_frames(final)} frames "
            f"(expected {kept + bridged}; segments changed mid-assemble?)"
        )
    fsync_dir(joint_root)
    return final

"""Shared video-worker scaffolding.

Extracted from `video_ltxv.py` / `video_causvid.py` (plus a removed third
worker, issue 079), whose operational skeleton (mp4 writes, atomic JSON tape writes, benchmark
warmup+measured loops, `serve()` dispatch maps) was copy-pasted three ways
and had already diverged on tape keys and profile fields. Workers keep
their generate/denoise logic and session classes; everything here is
plumbing that must change once, not three times.

Deliberate non-share (documented, not migrated): the removed worker's
recovery tape was a torch `.pt` tensor bundle (not JSON) and its
segment mp4 write used `imageio.get_writer` inside `generate_blocks`
(issue 079 — both deleted with the backend).

On-demand tail derivation (run-file pruning, DESIGN §§5.3-5.4):
`generate_blocks` (ltxv + causvid) no longer persists `video_tail.mp4`
beside the segment video; the tape still records the would-be
`conditioning_tail_path`. At resume, `ensure_conditioning_tail` adopts an
existing tail as before or derives the last `DERIVED_TAIL_FRAMES` frames
from the sibling segment video via ffmpeg, so the supervisor's
resume/rebuild flow works unchanged while committed segments carry one
fewer file.

Top level is numpy + stdlib (+ `voyage.atomic`, itself stdlib-only)
only (both in every image); imageio stays
function-level — the slim gates image has none, so CPU tests stub
`sys.modules["imageio.v2"]`.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from voyage.atomic import fsync_dir
from voyage.workers.loop import Handler, checked_request, validate_benchmark_counts

BoundaryKind = Literal["fresh", "continue"]
"""Per-block continuity vocabulary (issue 045).

Replaces the bare `scene_cut: bool` thread with a named state: `fresh`
starts a new scene (cut prefix, tail ignored), `continue` extends the
rolling stream. The RPC wire keeps `scene_cuts: list[bool]` (compat);
`boundary_from_scene_cut` converts at the boundary.
"""


def boundary_from_scene_cut(scene_cut: bool) -> BoundaryKind:
    """Name a scene-cut flag (issue 045)."""
    return "fresh" if scene_cut else "continue"


def session_scratch_parent(
    payload: dict[str, Any], *, stored_scratch_dir: Any | None = None
) -> Path:
    """Parent dir for a worker session work_root (boba /tmp-quota incident).

    The supervisor passes absolute `scratch_dir` (`run_dir/tmp/`) in every
    production init payload; session `mkdtemp` dirs land there instead of
    the host /tmp tmpfs. Rebuild handlers pass their `_INIT_PARAMS`
    value as `stored_scratch_dir` (their payload only carries the tape
    path); legacy/test callers without either fall back to `CWD/tmp/` —
    worker processes always spawn with CWD=run_dir (`rpc.py`), so the
    fallback is still the run, never /tmp. A non-str value fails loud
    (issue 118 discipline); the dir is created here so `mkdtemp(dir=...)`
    never races a missing parent.
    """
    from voyage import paths as voyage_paths

    raw = stored_scratch_dir if stored_scratch_dir is not None else payload.get("scratch_dir")
    if raw is None:
        parent = Path.cwd() / voyage_paths.SCRATCH_DIRNAME
    else:
        if not isinstance(raw, str) or not raw.strip():
            raise TypeError(f"payload field 'scratch_dir' must be a non-empty str, got {raw!r}")
        parent = Path(raw)
    parent.mkdir(parents=True, exist_ok=True)
    return parent


T = TypeVar("T")
"""Element type for `_strict_element` (issue 118)."""


def _strict_element(value: Any, field: str, expected: type[T]) -> T:
    """One wire element with `checked_request` discipline (issue 118).

    Bare `str()`/`int()`/`bool()` never fail and invent values (`None →
    "None"`, `"false" → True`, `1.9 → 1`, `True → 1px`) — a mistyped seed
    then truncates instead of erroring, breaking the determinism story
    (§62). Mirror `checked_request`: exact type, bools never satisfy int.
    """
    if isinstance(value, bool) and expected is not bool:
        raise TypeError(
            f"payload field {field!r} must be {expected.__name__}, got {type(value).__name__}"
        )
    if not isinstance(value, expected):
        raise TypeError(
            f"payload field {field!r} must be {expected.__name__}, got {type(value).__name__}"
        )
    return value


def _strict_optional_int(value: Any, field: str) -> int | None:
    """Geometry element: int or None, never bool/str/float (issue 118)."""
    if value is None:
        return None
    return _strict_element(value, field, int)


@dataclass(frozen=True)
class GenerateBlocksRequest:
    """Validated `generate_blocks` payload (issue 045).

    The three video workers parsed prompts/seeds/scene_cuts inline with
    `assert isinstance` shape checks — triplicated, positional-arg prone,
    and (as asserts) strippable. All cross-process construction goes
    through `from_payload`: one validated struct, keyword-only use.
    Session `generate_blocks` bodies keep their signatures; handlers
    unpack this struct into keyword calls.
    """

    prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    scene_cuts: tuple[bool, ...]
    output_path: Path
    width: int | None
    height: int | None
    fps: int
    segment_id: str = "000000"
    prompt_plan_digest: str | None = None
    requested_frames: int | None = None
    profile_stages: bool = False

    @property
    def boundaries(self) -> tuple[BoundaryKind, ...]:
        """Per-block continuity states derived from `scene_cuts`."""
        return tuple(boundary_from_scene_cut(cut) for cut in self.scene_cuts)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        width_default: int | None,
        height_default: int | None,
    ) -> GenerateBlocksRequest:
        """Parse + validate a `generate_blocks` RPC payload.

        Handles the multi-block form (`prompts`/`seeds`/`scene_cuts`
        lists) and the single-block form (`prompt`/`seed`/`scene_cut`
        scalars); geometry falls back to the backend-native defaults.
        Raises KeyError (missing), TypeError (mistyped) or ValueError
        (empty/mismatched) — never asserts, so `-O` cannot strip it.
        """
        checked_request(payload, segment_id=str, output_path=str, fps=int)
        if "prompts" in payload or "seeds" in payload:
            checked_request(payload, prompts=list, seeds=list)
            raw_prompts = payload["prompts"]
            raw_seeds = payload["seeds"]
            if not isinstance(raw_prompts, list) or not isinstance(raw_seeds, list):
                raise TypeError("payload prompts/seeds must be lists")
            prompts = tuple(_strict_element(item, "prompts[]", str) for item in raw_prompts)
            seeds = tuple(_strict_element(item, "seeds[]", int) for item in raw_seeds)
            raw_cuts = payload.get("scene_cuts", [False] * len(prompts))
            if not isinstance(raw_cuts, list) or len(raw_cuts) != len(prompts):
                raise ValueError("payload scene_cuts must match prompts in length")
            scene_cuts = tuple(_strict_element(item, "scene_cuts[]", bool) for item in raw_cuts)
        else:
            checked_request(payload, prompt=str, seed=int)
            prompts = (str(payload["prompt"]),)
            seeds = (int(payload["seed"]),)
            raw_cut = payload.get("scene_cut", False)
            scene_cuts = (_strict_element(raw_cut, "scene_cut", bool),)
        if not prompts or not (len(prompts) == len(seeds) == len(scene_cuts)):
            raise ValueError("prompts/seeds/scene_cuts must be non-empty equal-length lists")
        width = _strict_optional_int(payload.get("width", width_default), "width")
        height = _strict_optional_int(payload.get("height", height_default), "height")
        raw_requested = payload.get("frames")
        if isinstance(raw_requested, bool):
            requested = None
        else:
            requested = int(raw_requested) if isinstance(raw_requested, int) else None
        raw_digest = payload.get("prompt_plan_hash")
        raw_profile = payload.get("profile_stages", False)
        profile_stages = _strict_element(raw_profile, "profile_stages", bool)
        return cls(
            prompts=prompts,
            seeds=seeds,
            scene_cuts=scene_cuts,
            output_path=Path(str(payload["output_path"])),
            width=width,
            height=height,
            fps=int(payload["fps"]),
            segment_id=str(payload["segment_id"]),
            prompt_plan_digest=str(raw_digest) if isinstance(raw_digest, str) else None,
            requested_frames=requested,
            profile_stages=profile_stages,
        )


TAIL_FILENAME = "video_tail.mp4"
"""Crash-recovery anchor beside the segment video (ltxv + causvid).

Since run-file pruning this is the derive-target name, not a
generate-time artifact: `generate_blocks` records the would-be path in
the tape, and `ensure_conditioning_tail` materializes it at resume.
"""

TAPE_FILENAME = "recovery.pt"
"""Recovery-tape filename (kept for supervisor discovery; JSON content)."""

DERIVED_TAIL_FRAMES = 25
"""Frames in an on-demand derived conditioning tail (run-file pruning).

Matches the ltxv conditioning tail exactly and covers the causvid
default re-encode window (9 frames at overlap 3) with room — resume
takes the newest window it needs from the derived file. Callers with a
larger window pass an explicit `tail_frames` instead.
"""

SEGMENT_VIDEO_FILENAME = "video.mp4"
"""Committed segment video beside the tape (supervisor layout, §§5.3-5.4)."""

TAIL_DERIVE_TIMEOUT_SECONDS = 120.0
"""Bound for each ffmpeg/ffprobe spawn in tail derivation (short trims)."""

MAX_RECOVERY_TAPE_BYTES = 1024**3
"""Byte ceiling for recovery tapes (issue 171).

Legitimate tapes are megabytes (tail latents bf16 `[1,8,C,H,W]` plus
embeds plus RNG state — DESIGN §22 logs ~7 MB), so any gigabyte-scale tape
is corrupt or hostile; `weights_only=True` stops code execution, not
allocation, and torch still materializes every tensor in the archive. The
cap sits two orders of magnitude above legitimate — no false-positive
surface. Enforced at the supervisor gates and (residually) at the worker
`torch.load` call sites.
"""


def check_recovery_tape_size(resolved: Path) -> Path:
    """Reject implausibly large tapes before `torch.load` (issue 171).

    One `stat` syscall against a multi-minute GPU rebuild: a multi-GB
    `.pt` — runaway write, a weights file renamed `.pt`, a hostile worker
    report — fails here with ValueError instead of OOMing the worker and
    burning the whole restart budget on a knowably oversized input.
    Returns `resolved` for chaining.
    """
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValueError(f"recovery tape unreadable: {resolved}: {exc}") from exc
    if size > MAX_RECOVERY_TAPE_BYTES:
        raise ValueError(
            f"implausible tape size {size} bytes (>{MAX_RECOVERY_TAPE_BYTES}) "
            f"at {resolved} — re-render from seed instead of resuming"
        )
    return resolved


def verify_conditioning_tail_sha(segment_dir: Path, tape: dict[str, Any]) -> None:
    """Recompute a taped conditioning-tail hash, if the tape carries one.

    Issue 123: ltxv/causvid persist `conditioning_tail_sha256` at commit
    but no resume path recomputes it, so a truncated tail (disk-full
    mid-write, partial copy, bit-rot) passes `Path.exists()` and becomes
    the conditioning anchor — the next segment conditions on garbage
    without crashing, the worst failure mode for a continuity system.
    A mismatch raises ValueError ("re-render from seed"). No-op when the
    tape carries no hash (run-file pruning records the would-be tail path
    with nothing persisted to hash) or when the tail file is absent (the
    derive path materializes it — absence is not corruption).
    """
    digest = tape.get("conditioning_tail_sha256")
    raw_path = tape.get("conditioning_tail_path")
    if not isinstance(digest, str) or not digest or not isinstance(raw_path, str):
        return
    tail_path = Path(raw_path)
    if not tail_path.is_absolute():
        tail_path = segment_dir / tail_path
    if not tail_path.is_file():
        return
    from voyage.hashing import sha256_file

    actual = sha256_file(tail_path)
    if actual != digest:
        raise ValueError(
            f"tail sha mismatch at {tail_path} (taped {digest[:16]}…, "
            f"actual {actual[:16]}…) — re-render from seed instead of resuming"
        )


EMBED_CACHE_CAPACITY = 8
"""Resident text-embed entries per worker session (issues 014, 030).

Every distinct prompt costs a CPU T5 encode (~25 s on ltxv); the cache
avoids re-encoding repeats within a session, and the
bound keeps drift-every-N runs from pinning VRAM monotonically. Entries
are stored CPU-side — callers move to the worker device on use — so an
entry costs host RAM (~4 MB), never resident VRAM.
"""


class EmbedCache:
    """Small LRU over opaque text-embed values (issues 014, 030).

    Torch-agnostic on purpose: values are opaque (nested tensor
    structures differ per backend), eviction is pure key order, and
    device placement stays with the caller (`move_to_cpu` /
    `move_to_device` below). `get` refreshes recency; `put` evicts the
    least-recently-used entry past capacity.
    """

    def __init__(self, capacity: int = EMBED_CACHE_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"embed cache capacity must be >= 1 (got {capacity})")
        self._capacity = capacity
        self._entries: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    @property
    def capacity(self) -> int:
        return self._capacity

    def get(self, key: str) -> Any | None:
        """Return the entry and refresh its recency, or None on a miss."""
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, value: Any) -> None:
        """Store `value` under `key`, evicting the LRU entry past capacity."""
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop all entries (session teardown)."""
        self._entries.clear()


def move_to_cpu(value: Any) -> Any:
    """Recursively move nested tensor structures to CPU (issue 030).

    Duck-typed (`.cpu()`) so this module stays torch-free and CPU tests
    can use doubles: mappings and lists/tuples recurse element-wise,
    objects with `.cpu()` move, everything else passes through.
    """
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [move_to_cpu(item) for item in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    if hasattr(value, "cpu"):
        return value.cpu()
    return value


def move_to_device(value: Any, device: Any) -> Any:
    """Recursively move nested tensor structures to `device` (issue 030).

    Mirror of :func:`move_to_cpu` for cache hits: mappings and
    lists/tuples recurse, objects with `.to(device)` move, everything
    else passes through. Same-device `.to()` is a no-op upstream, so
    hits cost no copy when the entry already sits on the device.
    """
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [move_to_device(item, device) for item in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    if hasattr(value, "to"):
        return value.to(device)
    return value


def cuda_device_index(device: str) -> int:
    """CUDA index behind a `"cuda[:N]"` session device string (issue 124).

    Single home for the `"cuda:N" → N` parse the video workers need to
    index device-scoped torch telemetry (`get_device_name`,
    `mem_get_info`, peak-memory stats) at the *session* device instead
    of device 0. Bare `"cuda"` means index 0 (torch's own default).
    Raises ValueError on non-CUDA strings — placement must fail fast,
    never silently report GPU 0's numbers for a `cuda:1` session.
    """
    prefix, _, index_text = device.partition(":")
    if prefix != "cuda":
        raise ValueError(f"expected a CUDA device like 'cuda:0' (got {device!r})")
    if not index_text.strip():
        return 0
    if not index_text.strip().isdigit():
        raise ValueError(f"expected a CUDA device like 'cuda:0' (got {device!r})")
    return int(index_text.strip())


def torch_device_arg(device_index: int) -> tuple[int, ...]:
    """Positional device arg for torch peak-memory calls (issue 124).

    Empty on device 0: torch defaults peak reads to the current device
    (0 on single-GPU boxes), and the zero-arg shape is what the worker
    test fakes pin — so the default path keeps it byte-for-byte, and
    only non-zero sessions route explicitly.
    """
    return () if device_index == 0 else (device_index,)


def write_tape_atomic(tape_path: Path, tape: dict[str, Any]) -> Path:
    """Atomically + durably write a JSON recovery tape (sorted keys, trailing newline).

    Writes to a `.tmp` sibling, flushes + fsyncs the file, renames, then
    fsyncs the directory (issue 122, DESIGN §31 — the `concepts`
    `_append_vector` contract). Rename-without-fsync is atomic against
    *process* crashes but can lose the latest tape on OS crash / power
    loss — exactly the tape a restart needs. A failed fsync raises before
    the rename, so the previous tape survives. Returns `tape_path`.
    """
    tape_text = json.dumps(tape, indent=2, sort_keys=True) + "\n"
    tape_tmp = tape_path.with_suffix(".tmp")
    with open(tape_tmp, "w", encoding="utf-8") as handle:
        handle.write(tape_text)
        handle.flush()
        os.fsync(handle.fileno())
    tape_tmp.replace(tape_path)
    fsync_dir(tape_path.parent)
    return tape_path


class TailEnsureOutcome(NamedTuple):
    """Where the usable conditioning tail lives, and whether it was derived."""

    path: Path
    derived: bool


def find_segment_video(segment_dir: Path) -> Path | None:
    """Locate the segment video a missing tail can be derived from.

    Prefers the supervisor-layout `video.mp4`; otherwise the newest
    `*.mp4` in the directory (mtime, then name) — never the tail file
    itself. None when nothing qualifies.
    """
    preferred = segment_dir / SEGMENT_VIDEO_FILENAME
    if preferred.is_file():
        return preferred
    candidates = [
        candidate
        for candidate in segment_dir.glob("*.mp4")
        if candidate.is_file() and candidate.name != TAIL_FILENAME
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name))


_FRAME_COUNT_CACHE: dict[str, tuple[int, int, int]] = {}
"""ffprobe frame counts keyed by path -> (mtime_ns, size, frames).

Issue 249: tail derives and freshness gates re-probed the same segment
videos on every pass. The (mtime_ns, size) identity keeps the cache
honest across re-renders — a re-rendered video stats differently and
re-probes. Per-process only (workers are short-lived subprocesses);
failures are never cached, so a transient ffprobe error retries next
call.
"""


def _frame_count_identity(source_video: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) for a path, None when it cannot be stated."""
    try:
        stat = source_video.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _ffprobe_header_frames(source_video: Path) -> int | None:
    """Container-header frame count (`nb_frames`), no decode.

    Returns None when ffprobe fails or the container omits the count
    (MPEG-TS `N/A`, some mkv) — the caller falls back to the decode
    path. Never raises: header reads are advisory, the decode is truth.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "csv=p=0",
                str(source_video),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=TAIL_DERIVE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        count = int(proc.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return count if count > 0 else None


def _ffprobe_decode_frames(source_video: Path) -> int:
    """Exact frame count via ffprobe decode (no container estimate).

    Raises ValueError when the source is not a regular file, RuntimeError
    when ffprobe fails or its count is unparseable (operational failure,
    not a validation one).
    """
    if not source_video.is_file():
        raise ValueError(f"segment video missing: {source_video}")
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
            str(source_video),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=TAIL_DERIVE_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe frame count failed for {source_video}: {proc.stderr.strip()[-500:]}"
        )
    try:
        return int(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe frame count unparseable for {source_video}: {proc.stdout.strip()!r}"
        ) from exc


def count_video_frames(source_video: Path, known_frames: int | None = None) -> int:
    """Exact frame count for `source_video` (issue 249).

    `known_frames` (e.g. manifest `metrics.frames`, already trusted at
    the pollers) skips both probes — validated positive-int, bools
    rejected. Otherwise the container-header `nb_frames` is tried
    first (no decode) with the decode path as fallback; results cache
    per (path, mtime_ns, size). Default None preserves the old
    decode-first behavior exactly.
    """
    if known_frames is not None:
        if isinstance(known_frames, bool) or not isinstance(known_frames, int):
            raise TypeError(f"known_frames must be an int (got {type(known_frames).__name__})")
        if known_frames < 1:
            raise ValueError(f"known_frames must be positive (got {known_frames})")
        return known_frames
    identity = _frame_count_identity(source_video)
    if identity is not None:
        cached = _FRAME_COUNT_CACHE.get(str(source_video))
        if cached is not None and (cached[0], cached[1]) == identity:
            return cached[2]
    header = _ffprobe_header_frames(source_video)
    if header is not None:
        if identity is not None:
            _FRAME_COUNT_CACHE[str(source_video)] = (identity[0], identity[1], header)
        return header
    total = _ffprobe_decode_frames(source_video)
    if identity is not None:
        _FRAME_COUNT_CACHE[str(source_video)] = (identity[0], identity[1], total)
    return total


def tail_start_frame(source_video: Path, tail_frames: int, total_frames: int | None = None) -> int:
    """First frame index of the last-`tail_frames` window (ffprobe-backed).

    `total_frames` threads an already-known count (manifest frames);
    None probes as before. Raises ValueError for non-positive counts
    or short sources.
    """
    if tail_frames < 1:
        raise ValueError(f"tail_frames must be positive (got {tail_frames})")
    total = count_video_frames(source_video, known_frames=total_frames)
    if total < tail_frames:
        raise ValueError(
            f"segment video {source_video} has {total} frames, "
            f"need {tail_frames} for the conditioning tail"
        )
    return total - tail_frames


def build_tail_trim_argv(
    source_video: Path, dest_tail: Path, start_frame: int, tail_frames: int
) -> list[str]:
    """ffmpeg argv trimming `tail_frames` frames from `start_frame` (pure).

    Frame-exact (not time-based): `-frames:v` caps the output so the
    derived tail length never depends on timestamps. Source timestamps
    are preserved verbatim — retiming via `setpts` drops a frame on this
    ffmpeg build (measured 24 instead of 25), and both tail consumers
    (ltxv conditioning, causvid re-encode) decode to frames, where only
    the count matters. Arg-lists only (never shell); single quotes are
    ffmpeg filter quoting, not shell.
    """
    if tail_frames < 1:
        raise ValueError(f"tail_frames must be positive (got {tail_frames})")
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative (got {start_frame})")
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_video),
        "-vf",
        f"select='gte(n,{start_frame})'",
        "-frames:v",
        str(tail_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-f",
        "mp4",
        str(dest_tail),
    ]


def derive_tail_from_segment_video(
    source_video: Path,
    dest_tail: Path,
    tail_frames: int = DERIVED_TAIL_FRAMES,
    total_frames: int | None = None,
) -> Path:
    """Trim the last `tail_frames` frames of `source_video` into `dest_tail`.

    `total_frames` threads an already-known count (manifest frames) so
    the derive costs one ffmpeg trim instead of probe + trim; None
    probes as before. Writes to a `.tmp` sibling then renames (a crash
    mid-derive leaves a flagged temp, never a torn tail). Returns
    `dest_tail`. Raises ValueError for bad counts/short sources,
    RuntimeError when ffmpeg fails or yields an empty file.
    """
    dest_tmp = dest_tail.with_suffix(".tmp")
    start = tail_start_frame(source_video, tail_frames, total_frames=total_frames)
    argv = build_tail_trim_argv(source_video, dest_tmp, start, tail_frames)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=TAIL_DERIVE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg tail derive timed out after {TAIL_DERIVE_TIMEOUT_SECONDS}s "
            f"({source_video} -> {dest_tail})"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg tail derive failed ({source_video} -> {dest_tail}): "
            f"{proc.stderr.strip()[-500:]}"
        )
    if not dest_tmp.is_file() or dest_tmp.stat().st_size == 0:
        dest_tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg tail derive yielded no output ({source_video} -> {dest_tail})")
    dest_tmp.replace(dest_tail)
    return dest_tail


def ensure_conditioning_tail(
    tail_path: Path, tail_frames: int = DERIVED_TAIL_FRAMES
) -> TailEnsureOutcome:
    """Return the usable conditioning tail, deriving it when missing.

    An existing tail file is adopted untouched (resume behaves exactly as
    before, including a stale sha — the tape hash stays advisory). A
    missing one is derived from the sibling segment video into `tail_path`
    itself, so later resumes hit it directly. Raises ValueError only when
    both the tail and any derivable video are absent.
    """
    if tail_path.is_file():
        return TailEnsureOutcome(tail_path, False)
    source = find_segment_video(tail_path.parent)
    if source is None:
        raise ValueError(
            f"conditioning tail missing: {tail_path} "
            f"(no segment video in {tail_path.parent} to derive it from)"
        )
    derive_tail_from_segment_video(source, tail_path, tail_frames=tail_frames)
    return TailEnsureOutcome(tail_path, True)


def clip_array_to_uint8(scaled_frames: NDArray[Any]) -> NDArray[np.uint8]:
    """Clip float frames in ~[0, 1] to uint8 (issue 065).

    VAE overshoot outside [0, 1] wraps modulo 256 on a bare `astype`
    (1.01 becomes near-black) — clip first, then convert.
    """
    return np.clip(scaled_frames * 255.0, 0, 255).astype("uint8")


def save_mp4(frames: list[NDArray[np.uint8]], output_path: Path, frames_per_second: int) -> None:
    """Write (H, W, C) uint8 frames as h264 mp4 (ltxv + causvid writer)."""
    if frames_per_second <= 0:
        raise ValueError(f"mp4 frame rate must be positive (got {frames_per_second})")
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    imageio.mimsave(str(output_path), frames, fps=frames_per_second, codec="libx264")


class BenchmarkHarnessOutcome(NamedTuple):
    """Measured-only results from :func:`run_benchmark_harness`."""

    wall_seconds: list[float]
    peak_gib: list[float]


def run_benchmark_harness(
    warmup_count: int,
    measured_count: int,
    temporary_prefix: str,
    probe: Callable[[Path, bool], None],
    clock: Callable[[], float] = time.monotonic,
    reset_peak_memory: Callable[[], None] | None = None,
    read_peak_gib: Callable[[], float] | None = None,
    staging_parent: Path | None = None,
) -> BenchmarkHarnessOutcome:
    """Run warmup + measured probes in a scratch dir, timing each.

    Owns the `TemporaryDirectory`, the warmup/measured loop, wall timing and
    VRAM-peak sampling; callers pass a `probe` that renders one artifact to
    the given path (`measured` is true for measured iterations — collect
    results then — and false for warmup) and format their own
    backend-specific response from the returned measured-only walls/peaks.
    Counts are validated first so bad values fail fast on CPU without
    touching GPU state. `staging_parent` routes the scratch under the run
    (`run_dir/tmp/`, never bare host /tmp — issue 287); None keeps the
    TMPDIR default (which the supervisor points at the run scratch in
    production, bare /tmp only in bench/test callers).
    """
    validate_benchmark_counts(warmup_count, measured_count)
    wall_seconds: list[float] = []
    peak_gib: list[float] = []
    with tempfile.TemporaryDirectory(prefix=temporary_prefix, dir=staging_parent) as tmp:
        for index in range(warmup_count + measured_count):
            if reset_peak_memory is not None:
                reset_peak_memory()
            started = clock()
            probe(Path(tmp) / f"b{index}.mp4", index >= warmup_count)
            elapsed = clock() - started
            peak = read_peak_gib() if read_peak_gib is not None else 0.0
            if index >= warmup_count:
                wall_seconds.append(elapsed)
                peak_gib.append(peak)
    return BenchmarkHarnessOutcome(wall_seconds=wall_seconds, peak_gib=peak_gib)


def standard_serve_map(
    backend_name: str,
    *,
    handle_init: Handler,
    handle_health: Handler,
    handle_generate_blocks: Handler,
    handle_benchmark: Handler,
    handle_evict_gpu: Handler,
    handle_rebuild: Handler,
    handle_resume: Handler,
    handle_cancel: Handler | None = None,
) -> dict[str, Handler]:
    """Build the standard nine-op worker `serve()` dispatch map.

    Every video worker serves the same ops; only the `checkpoint` id prefix
    varies, which `backend_name` supplies (e.g. `"ltxv"` →
    `"ltxv-<segment_id>"`). `handle_cancel` is optional for backward
    compatibility (existing callers pass seven handlers and get the nine-op
    map exactly — see `test_standard_serve_map_keys_and_checkpoint`); pass
    one or omit it to get the idle `loop.handle_cancel` tenth op (Track D
    cancel wire name, stable for Track A probing).
    """
    serve_map: dict[str, Handler] = {
        "init": handle_init,
        "health": handle_health,
        "generate_blocks": handle_generate_blocks,
        "benchmark": handle_benchmark,
        "evict_gpu": handle_evict_gpu,
        "rebuild": handle_rebuild,
        "checkpoint": lambda payload: {
            "checkpoint_id": f"{backend_name}-{payload.get('segment_id', 'none')}"
        },
        "resume": handle_resume,
        "shutdown": lambda _payload: {"stopped": True},
    }
    # Backward-compatible default: nine ops exactly (existing callers/tests
    # pin the key set). New workers opt into the Track D `cancel` wire name
    # via `standard_serve_map_with_cancel` below or by passing an explicit
    # `handle_cancel` here.
    if handle_cancel is not None:
        serve_map["cancel"] = handle_cancel
    return serve_map


def standard_serve_map_with_cancel(
    backend_name: str,
    *,
    handle_init: Handler,
    handle_health: Handler,
    handle_generate_blocks: Handler,
    handle_benchmark: Handler,
    handle_evict_gpu: Handler,
    handle_rebuild: Handler,
    handle_resume: Handler,
    handle_cancel: Handler | None = None,
) -> dict[str, Handler]:
    """Ten-op map with the Track D `cancel` wire name (Track A wire point).

    Same nine ops as `standard_serve_map` plus `cancel` (idle handler by
    default, explicit handler when passed). New workers prefer this; the
    nine-op builder stays frozen for backward compatibility.
    """
    from voyage.workers.loop import handle_cancel as _idle_cancel

    base = standard_serve_map(
        backend_name,
        handle_init=handle_init,
        handle_health=handle_health,
        handle_generate_blocks=handle_generate_blocks,
        handle_benchmark=handle_benchmark,
        handle_evict_gpu=handle_evict_gpu,
        handle_rebuild=handle_rebuild,
        handle_resume=handle_resume,
    )
    base["cancel"] = handle_cancel if handle_cancel is not None else _idle_cancel
    return base


def report_block_progress(
    block_index: int, total_blocks: int, *, backend: str = "", stream: Any = None
) -> None:
    """Worker-side progress line on stderr per diffusion chunk/block (Track D).

    One line per block (`voyage_progress backend=<b> block=i/n`) so the
    supervisor watchdog can tell a slow render from a wedged one without
    touching RPC (non-blocking + 8MiB cap intact). Never raises (progress
    must not fail a render); defaults to `sys.stderr` (stdout is RPC
    framing). Call once per block/rollout inside `generate_blocks`.
    """
    import contextlib as _contextlib
    import sys as _sys

    target = stream if stream is not None else _sys.stderr
    with _contextlib.suppress(OSError):
        print(
            f"voyage_progress backend={backend} block={block_index + 1}/{total_blocks}",
            file=target,
            flush=True,
        )


def touch_progress_file(progress_path: Path) -> Path:
    """Record now as the last-progress timestamp (Track D watchdog).

    Atomic write of `time.monotonic()` to `progress_path` (sibling of the
    segment video or under the session work_root). The supervisor polls
    the file mtime/content via `read_progress_timestamp` and gates on
    `rpc.is_wedged`. Never raises (best-effort); returns the path.
    """
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(f"{time.monotonic():.3f}\n", encoding="utf-8")
    except OSError:
        pass
    return progress_path


def read_progress_timestamp(progress_path: Path) -> float | None:
    """Read the last-progress monotonic timestamp, None when unknown."""
    try:
        text = progress_path.read_text(encoding="utf-8").strip().split()[0]
        value = float(text)
    except (OSError, ValueError, IndexError):
        return None
    if value != value or value == float("inf") or value == float("-inf"):
        return None
    return value


def prune_work_root(work_root: Path) -> None:
    """Prune session scratch contents on refresh (Track D).

    Deletes every child of `work_root` (files + dirs, best-effort) but
    keeps the root itself, so a refreshed session restarts from an empty
    ComfyUI output/input/temp without losing the session dir identity.
    Never raises (a prune failure must not fail the refresh — the next
    render overwrites anyway). Track A calls this on the session-refresh
    path (which charges one budget unit via `rpc.restart_with_budget`).
    """
    try:
        if not work_root.is_dir():
            return
        for child in work_root.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    import shutil as _shutil

                    _shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        pass


def validate_resume_trust(
    tape: dict[str, Any],
    *,
    expected_profile_hash: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_tail_frames: int | None = None,
    expected_model_revision: str | None = None,
) -> dict[str, Any]:
    """Compare tape profile/geometry/tail/model against the session (Track D).

    Mismatch raises `ValueError` (Fatal over the wire — never resume across
    numerics/geometry). Returns the validated tape for chaining. Every
    expectation is optional (None = skip that axis); callers pass what the
    session knows (profile hash + geometry + tail frames + model revision).
    The supervisor gate helper `resume_gate_for_supervisor` wraps this with
    the metric hook (caller emits `resume_trust_mismatch`).
    """
    if not isinstance(tape, dict):
        raise TypeError("recovery tape must be a JSON object")
    if expected_profile_hash is not None:
        actual = tape.get("profile_hash")
        if not isinstance(actual, str) or actual != expected_profile_hash:
            raise ValueError(
                f"recovery tape profile_hash mismatch (taped {actual!r} "
                f"vs session {expected_profile_hash!r}) — re-render from seed"
            )
    if expected_width is not None:
        actual_width = tape.get("width")
        if actual_width != expected_width:
            raise ValueError(
                f"recovery tape width mismatch (taped {actual_width!r} "
                f"vs session {expected_width!r}) — re-render from seed"
            )
    if expected_height is not None:
        actual_height = tape.get("height")
        if actual_height != expected_height:
            raise ValueError(
                f"recovery tape height mismatch (taped {actual_height!r} "
                f"vs session {expected_height!r}) — re-render from seed"
            )
    if expected_tail_frames is not None:
        actual_tail = tape.get("conditioning_tail_frames", tape.get("num_overlap_frames"))
        if actual_tail != expected_tail_frames:
            raise ValueError(
                f"recovery tape tail_frames mismatch (taped {actual_tail!r} "
                f"vs session {expected_tail_frames!r}) — re-render from seed"
            )
    if expected_model_revision is not None:
        actual_revision = tape.get("model_revision", tape.get("checkpoint_revision"))
        if not isinstance(actual_revision, str) or actual_revision != expected_model_revision:
            raise ValueError(
                f"recovery tape model_revision mismatch (taped {actual_revision!r} "
                f"vs session {expected_model_revision!r}) — re-render from seed"
            )
    return tape


def resume_gate_for_supervisor(
    tape: dict[str, Any],
    *,
    expected_profile_hash: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_tail_frames: int | None = None,
    expected_model_revision: str | None = None,
    worker_name: str = "video",
    segment_id: str = "",
) -> tuple[bool, dict[str, Any]]:
    """Supervisor gate helper for resume trust (Track A wire point).

    Returns `(trusted, metric)`: `trusted` False when
    `validate_resume_trust` raises (caller skips resume, renders fresh,
    and emits `metric` as `resume_trust_mismatch`); True with a
    `resume_trusted` metric otherwise. Never raises (the gate itself must
    not fail the commit — only the resume decision).
    """
    try:
        validate_resume_trust(
            tape,
            expected_profile_hash=expected_profile_hash,
            expected_width=expected_width,
            expected_height=expected_height,
            expected_tail_frames=expected_tail_frames,
            expected_model_revision=expected_model_revision,
        )
    except ValueError as exc:
        return False, {
            "event": "resume_trust_mismatch",
            "worker": worker_name,
            "segment_id": segment_id,
            "reason": str(exc),
        }
    return True, {
        "event": "resume_trusted",
        "worker": worker_name,
        "segment_id": segment_id,
    }


def checked_torch_load(path: Path, **kwargs: Any) -> Any:
    """`torch.load` gated on `check_recovery_tape_size` (Track D).

    One `stat` before the load: implausible sizes fail fast with
    ValueError instead of OOMing the worker and burning restart budget.
    All video-worker `torch.load` call sites must route through here.
    """
    import torch as _torch

    check_recovery_tape_size(path)
    return _torch.load(str(path), **kwargs)


def is_idempotent_op(op: str) -> bool:
    """True when retrying `op` cannot duplicate timeline state (Track D).

    The ACE/SFX finalize renders (`generate_audio` / `generate_sfx`) and
    the read-only probes (`health`, `benchmark`, `checkpoint`, `cancel`)
    overwrite their outputs atomically — a retry-once restart is safe.
    `generate_blocks` commits segment state and is NOT idempotent (the
    supervisor's full budget path owns it, never the finalize helper).
    """
    return op in ("generate_audio", "generate_sfx", "health", "benchmark", "checkpoint", "cancel")

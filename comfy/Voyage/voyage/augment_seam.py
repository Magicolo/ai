"""Seam interpolation between adjacent usable segments (DESIGN §§56-57, §140).

The durable finalize path drains one intermediate per usable segment and
concats them in order — without seams every segment joint is a hard cut:
the pair (A-last, B-first) never saw FILM. For each adjacent usable pair
this module renders the `multiplier - 1` mids between A's last
interpolated frame and B's first into a dedicated joint sidecar dir, so
the final concat interleaves [A, seam, B].

The joint dir reuses the segment plan shape (composite source key
`"<key-a>|<key-b>"`, single chunk index 0, `interpolated_00/` PNGs, one
ledger record): a settings change or a re-render of either side forks the
dir instead of reusing a stale seam, and `drain_interpolated_plan` drains
it unchanged. Endpoints are never dropped or duplicated — they stay in
their segments, the seam contributes mids only (so `multiplier=1` needs
no seam at all).

Stdlib only at module scope (supervisor §12 GPU ban): FILM enters
through the injected `interp_fn` seam or a function-local lazy import.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from voyage.augment_sidecar import (
    STAGE_INTERPOLATED,
    ChunkKey,
    append_chunk_record,
    chunk_cache_hit,
    chunk_output_complete,
    completed_stages,
    load_chunk_ledger,
    plan_dir_for_segment,
    prune_stale_partials,
)
from voyage.errors import MediaError

SEAM_CHUNK_INDEX = 0
"""Joint dirs hold exactly one chunk (the seam has no chunking)."""

SEAM_SOURCE_FRAMES = 2
"""A seam interpolates one pair: A's last frame + B's first frame."""

SeamInterpFn = Callable[[Path, Path, Path, int], list[Path]]
"""(before_png, after_png, dest_dir, multiplier) -> mids written (m-1 PNGs)."""


def seam_plan_dir(
    run_dir: Path,
    *,
    key_a: str,
    key_b: str,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    crf: int,
    preset: str,
) -> Path:
    """Joint plan dir for one adjacent segment pair (order-sensitive).

    The composite source key binds the seam to both sides' exact bytes:
    re-rendering either segment misses the old seam instead of reusing it.
    """
    if not key_a or not key_b:
        raise ValueError("seam needs both segment source keys (got empty)")
    return plan_dir_for_segment(
        run_dir,
        source_key=f"{key_a}|{key_b}",
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )


def _interp_pngs(plan_dir: Path, index: int) -> list[Path]:
    png_dir = plan_dir / f"interpolated_{index:02d}"
    if not png_dir.is_dir():
        return []
    return sorted(png_dir.glob("frame_*.png"))


def _last_interp_index(plan_dir: Path) -> int | None:
    records = load_chunk_ledger(plan_dir / "chunks.jsonl")
    indexes = [
        index for index, stages in completed_stages(records).items() if STAGE_INTERPOLATED in stages
    ]
    if not indexes:
        return None
    return max(indexes)


def seam_endpoints(plan_dir_a: Path, plan_dir_b: Path) -> tuple[Path, Path] | None:
    """Boundary pair for one seam, or None while either side is incomplete.

    Returns (A's last interpolated frame, B's first interpolated frame).
    Missing dirs/PNGs mean the interp poller has not finished that side
    yet — the caller waits, it never fails loud here.
    """
    last_index = _last_interp_index(plan_dir_a)
    if last_index is None:
        return None
    before_candidates = _interp_pngs(plan_dir_a, last_index)
    after_candidates = _interp_pngs(plan_dir_b, SEAM_CHUNK_INDEX)
    if not before_candidates or not after_candidates:
        return None
    return (before_candidates[-1], after_candidates[0])


def _default_seam_pngs(
    before_png: Path,
    after_png: Path,
    dest_dir: Path,
    multiplier: int,
    *,
    weights_path: Path,
    device: str,
    interp_backend: str = "rife",
) -> list[Path]:
    """Render the seam mids via the resident interp leg (lazy torch import)."""
    from voyage.augment import load_png_frames_as_tensors, write_tensors_as_png_frames
    from voyage.workers.augment_worker import (
        interpolate_mids,
        interpolate_rife_mids,
        validate_interp_backend,
    )

    validate_interp_backend(interp_backend)
    frames = load_png_frames_as_tensors([before_png, after_png])
    moments = [(position + 1) / multiplier for position in range(multiplier - 1)]
    if interp_backend == "rife":
        mids = interpolate_rife_mids(frames, weights_path, moments=moments, device=device)
    else:
        mids = interpolate_mids(frames, weights_path, moments=moments, device=device)
    return write_tensors_as_png_frames(mids, dest_dir)


def render_seam_once(
    run_dir: Path,
    seam_dir: Path,
    *,
    before_png: Path,
    after_png: Path,
    multiplier: int,
    source_key: str,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    crf: int,
    preset: str,
    weights_path: Path | None = None,
    device: str = "cuda:1",
    interp_backend: str = "rife",
    interp_fn: SeamInterpFn | None = None,
) -> bool:
    """Render one seam's mids (True) or skip the ledger hit (False).

    Ledger-truth plus output-truth resume (DESIGN §§56-57, chunk-poller
    parity): an exact-key `interpolated` record plus its `multiplier - 1`
    complete PNGs on disk is finished work and is never re-rendered; a
    ledger hit whose output is gone, short, or empty is stale (crashed
    publish, manual cleanup, disk loss) and is dropped and re-rendered
    instead of failing loud — same policy as the upscale/interp pollers,
    which rejoin ledgered-but-incomplete chunks to missing. Unledgered
    output is dropped and redone. `multiplier < 2` is a caller bug
    (`multiplier=1` has zero mids — the finalize wiring skips seams
    entirely instead of calling here).
    """
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 2:
        raise ValueError(f"seam multiplier must be an int >= 2 (got {multiplier!r})")
    if not isinstance(seam_dir, Path):
        raise TypeError(f"seam_dir must be a Path (got {type(seam_dir).__name__})")
    expected = multiplier - 1
    final_fps = out_fps * multiplier
    key = ChunkKey(
        chunk_index=SEAM_CHUNK_INDEX,
        start_frame=0,
        source_frames=SEAM_SOURCE_FRAMES,
        expected_frames=expected,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        crf=crf,
        preset=preset,
        source_key=source_key,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=final_fps,
    )
    seam_dir.mkdir(parents=True, exist_ok=True)
    prune_stale_partials(seam_dir)
    ledger_path = seam_dir / "chunks.jsonl"
    records = load_chunk_ledger(ledger_path)
    output_dir = seam_dir / f"interpolated_{SEAM_CHUNK_INDEX:02d}"
    if chunk_cache_hit(records, key, stage=STAGE_INTERPOLATED):
        if chunk_output_complete(output_dir, expected):
            return False
        # Auto-heal: ledger hit but output wrong (stale publish, partial
        # cleanup) — drop the stale dir and fall through to re-render.
        # The fresh render appends a new exact-key record (last-wins);
        # the stale line stays harmlessly superseded, same as the
        # chunk pollers' ledgered-but-incomplete rejoin.
        if output_dir.exists():
            if output_dir.is_dir() and not output_dir.is_symlink():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
    if output_dir.exists():
        # Ledger-truth rule: unledgered output is incomplete — drop it.
        # (The `.partial` dir never survives: `prune_stale_partials` above
        # already removed it, so only a fully-written dir can be here.)
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    partial_dir = seam_dir / f"interpolated_{SEAM_CHUNK_INDEX:02d}.partial"
    if interp_fn is None:
        if weights_path is None:
            raise ValueError("seam default render needs weights_path (interp leg)")
        written = _default_seam_pngs(
            before_png,
            after_png,
            partial_dir,
            multiplier,
            weights_path=weights_path,
            device=device,
            interp_backend=interp_backend,
        )
    else:
        written = interp_fn(before_png, after_png, partial_dir, multiplier)
    if len(written) != expected:
        raise MediaError(
            f"seam rendered {len(written)} mids, expected {expected} "
            f"(pair {before_png.name} + {after_png.name})"
        )
    for frame_file in written:
        if not frame_file.is_file():
            raise MediaError(f"seam output missing: {frame_file}")
    os.replace(partial_dir, output_dir)
    relative = output_dir.relative_to(run_dir).as_posix()
    append_chunk_record(ledger_path, key, stage=STAGE_INTERPOLATED, path=relative)
    return True


# Pair wiring lives in `augment_finalize.run_durable_model_pass`, which
# already holds both sides' source keys and plan dirs and calls
# `seam_plan_dir` + `seam_endpoints` + `render_seam_once` directly.


def maybe_render_seam_joint(
    run_dir: Path,
    *,
    key_a: str,
    key_b: str,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    multiplier: int,
    crf: int,
    preset: str,
    weights_path: Path | None = None,
    device: str = "cuda:1",
    interp_fn: SeamInterpFn | None = None,
    interp_backend: str = "rife",
) -> bool:
    """Attempt one segment-boundary joint early (True) or wait (False).

    Same derivation as the drain's per-boundary wiring (`seam_plan_dir` +
    `render_seam_once` with identical args), so an early render and the
    drain's later attempt share one ledger identity: the drain skips via
    ledger hit. Returns False while either side's interp is incomplete
    (`seam_endpoints` None — the caller waits, it never fails loud here;
    the drain still fail-louds on genuinely stuck boundaries).
    """
    plan_a = plan_dir_for_segment(
        run_dir,
        source_key=key_a,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
    plan_b = plan_dir_for_segment(
        run_dir,
        source_key=key_b,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
    endpoints = seam_endpoints(plan_a, plan_b)
    if endpoints is None:
        return False
    before_png, after_png = endpoints
    seam_dir = seam_plan_dir(
        run_dir,
        key_a=key_a,
        key_b=key_b,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
    return render_seam_once(
        run_dir,
        seam_dir,
        before_png=before_png,
        after_png=after_png,
        multiplier=multiplier,
        source_key=f"{key_a}|{key_b}",
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
        weights_path=weights_path,
        device=device,
        interp_fn=interp_fn,
        interp_backend=interp_backend,
    )

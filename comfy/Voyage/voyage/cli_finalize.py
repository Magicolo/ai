"""Final-video assembly (DESIGN §§54-57).

Library module for `generate`'s finalize step: final-video assembly
plus the SFX dub, with CLI augment multipliers riding the stored
`[augment]` section via `cli_core._augment_overrides`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from voyage.cli_core import _augment_overrides, _load_run, generate_skip_key, get_console
from voyage.cli_paths import resolve_run_ref, warn_if_outside_output_dir
from voyage.config import resolve_config
from voyage.errors import DiskSpaceError, MediaError, StateError
from voyage.media import SfxParallelRequest, finalize_run, presented_frames
from voyage.media import probe as media_probe
from voyage.persistence import read_state, record_final_coverage


def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = resolve_run_ref(run=getattr(args, "run", None), name=getattr(args, "name", None))
    if run_dir is None:
        return 2
    config = _load_run(run_dir)
    try:
        # CLI multipliers ride the stored [augment] section: explicit flags
        # win, otherwise the run TOML rules. Invalid multipliers fail here
        # (exit 2), before any media work. The resolved multipliers ride
        # `config.augment` for the finalize consumer (a later slice reads
        # them).
        config = resolve_config(config, **_augment_overrides(args))
    except (ValidationError, ValueError) as exc:
        print(f"error: invalid augment override: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    # Containment warning (issue 024): warn-only, same policy as init —
    # absolute outside-tree finals are legal, typos should be loud.
    warn_if_outside_output_dir(output, flag="--output")
    sfx_backend = getattr(args, "sfx_backend", None) or config.sfx.backend
    sfx_will_run = not getattr(args, "no_sfx", False) and sfx_backend != "fake"
    console = get_console(args)
    # Shared video-stage kwargs (both finalize paths take the same box):
    # finalize keeps the generation resolution (no downscale) — the run
    # config snapshot carries what the segments rendered at, so old
    # 768x512 runs refinalize natively too. getattr: the stop --finalize
    # handoff reuses the stop-parser namespace (issue 109) — a missing
    # flag must read as off, never AttributeError after the status flips.
    video_kwargs: dict[str, Any] = {
        "skip_bad": getattr(args, "skip_bad", False),
        "min_free_space_gib": config.min_free_space_gib,
        "sample_rate": config.audio.sample_rate,
        "channels": config.audio.channels,
        "overlap_fraction": config.audio.final_overlap_fraction,
        "overlap_cap_seconds": config.audio.final_overlap_cap_seconds,
        "upscale": config.augment.upscale,
        "interpolate": config.augment.interpolate,
        "presentation_fps": config.augment.presentation_fps,
        "models_dir": config.video.models_dir,
        "audio_config": config.audio,
        "seed": config.seed,
        # Generate-only --no-music/--no-audio (getattr: hand-built
        # namespaces + the stop --finalize handoff lack the flag — off).
        "no_music": bool(getattr(args, "no_music", False)),
    }
    # Two-stream finalize: the SFX bed renders inside `finalize_run` while
    # the model pass publishes (stream A ∥ stream B — music, then the bed
    # on cuda:0 after the ACE teardown, then the dub on the main thread),
    # so a dubbed final skips the legacy post-pass below. A bed failure
    # inside finalize ships music-only and the legacy shipped-pixels pass
    # retries as the fallback (the keyed ledger keeps proxy stems from
    # false-hitting there).
    sfx_request = (
        SfxParallelRequest(
            backend=sfx_backend,
            models_dir=config.sfx.models_dir,
            device=getattr(args, "sfx_device", None) or config.sfx.device,
            model_size=getattr(args, "sfx_model_size", None) or config.sfx.model_size,
            num_workers=getattr(args, "sfx_workers", 1),
            fps=config.video.fps,
            caption_override=getattr(args, "sfx_caption", None),
        )
        if sfx_will_run
        else None
    )
    sfx_report: dict[str, Any] = {}
    try:
        finalize_run(
            run_dir,
            output,
            **video_kwargs,
            invoker=getattr(args, "invoker", None),
            progress=console,
            sfx_request=sfx_request,
            sfx_report=sfx_report,
        )
    except (MediaError, StateError, DiskSpaceError) as exc:
        print(f"finalize failed: {exc}", file=sys.stderr)
        return 1
    if sfx_will_run and sfx_report.get("sfx_status") != "dubbed":
        from voyage.sfx_finalize import finalize_sfx_pass

        try:
            finalize_sfx_pass(
                run_dir,
                output,
                backend=sfx_backend,
                models_dir=config.sfx.models_dir,
                device=getattr(args, "sfx_device", None) or config.sfx.device,
                model_size=getattr(args, "sfx_model_size", None) or config.sfx.model_size,
                seed=config.seed,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
                num_workers=getattr(args, "sfx_workers", 1),
                fps=config.video.fps,
                caption_override=getattr(args, "sfx_caption", None),
                progress=console,
            )
        except (MediaError, StateError) as exc:
            print(f"sfx pass failed (music-only kept at {output}): {exc}", file=sys.stderr)
            return 1
    # Freshness stamp for the generate 'nothing to do' gate (redundant
    # finalize fix): records what final.mp4 actually presents, so a revisit
    # compares presented-against-presented. Only on full success — an SFX
    # failure returns above, and its music-only final must stay re-finalizable.
    # The skip_key records the behavior (music/sfx/upscale/interpolate) so
    # a revisit with different generate-only skips re-finalizes instead of
    # reading a music-only diff as fresh. Format is owned by
    # `generate_skip_key` (shared with the generate gate); legacy callers
    # without the flags stamp the all-off key over the resolved
    # multipliers — same shape as the generate gate computes for a
    # no-skip run.
    coverage_frames = presented_frames(output)
    if coverage_frames is not None:
        # Single format source with the generate gate (`generate_skip_key`):
        # args.no_sfx already ORs the manifest policy (generate passes the
        # ORed value; direct callers pass their own), args.upscale/None rode
        # resolve_config above, so the resolved multipliers are effective.
        skip_key = generate_skip_key(
            {
                "skip_music": bool(getattr(args, "no_music", False)),
                "skip_sfx": False,
                "force_upscale_1": False,
                "force_interpolate_1": False,
            },
            manifest_no_sfx=bool(getattr(args, "no_sfx", False)),
            stored_upscale=config.augment.upscale,
            stored_interpolate=config.augment.interpolate,
        )
        record_final_coverage(
            run_dir,
            presented_frames=coverage_frames,
            segments=read_state(run_dir).committed_segments,
            skip_key=skip_key,
        )
    try:
        info = media_probe(output)
        duration = info.get("format", {}).get("duration", "?") if isinstance(info, dict) else "?"
    except MediaError:
        duration = "?"
    try:
        size_mb = output.stat().st_size / (1024**2)
        size_text = f"{size_mb:.1f} MiB"
    except OSError:
        size_text = "unknown size"
    console.ok(f"finalized -> {output} ({duration}s, {size_text})")
    print(f"finalized -> {output}")
    return 0

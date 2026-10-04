"""`finalize` and `sfx` verbs (DESIGN §§54-57).

Verb module of the issue-080 split: final-video assembly plus the
standalone SFX pass. CLI augment floors ride the stored `[augment]`
section via `cli_core._augment_overrides`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from voyage.audio_finalize import is_deferred_backend
from voyage.cli_core import _augment_overrides, _load_run, get_console
from voyage.cli_paths import resolve_run_ref, warn_if_outside_output_dir
from voyage.config import resolve_config
from voyage.errors import DiskSpaceError, MediaError, StateError
from voyage.media import finalize_run, presented_frames
from voyage.media import probe as media_probe
from voyage.persistence import read_state, record_final_coverage


def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = resolve_run_ref(run=getattr(args, "run", None), name=getattr(args, "name", None))
    if run_dir is None:
        return 2
    config = _load_run(run_dir)
    try:
        # CLI floors ride the stored [augment] section: explicit flags win,
        # otherwise the run TOML rules. Invalid floors fail here (exit 2),
        # before any media work. The resolved floors ride `config.augment`
        # for the finalize consumer (a later slice reads them).
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
    # Shared video-stage kwargs (both finalize paths take the same box):
    # finalize keeps the generation resolution (no downscale) — the run
    # config snapshot carries what the segments rendered at, so old
    # 768x512 runs refinalize natively too. getattr: the stop --finalize
    # handoff reuses the stop-parser namespace (issue 109) — a missing
    # flag must read as off, never AttributeError after the status flips.
    video_kwargs: dict[str, Any] = {
        "width": config.video.width,
        "height": config.video.height,
        "fps": config.video.fps,
        "skip_bad": getattr(args, "skip_bad", False),
        "min_free_space_gib": config.min_free_space_gib,
        "sample_rate": config.audio.sample_rate,
        "channels": config.audio.channels,
        "overlap_fraction": config.audio.final_overlap_fraction,
        "overlap_cap_seconds": config.audio.final_overlap_cap_seconds,
        "min_fps": config.augment.min_fps,
        "min_width": config.augment.min_width,
        "min_height": config.augment.min_height,
        "use_model_pass": config.augment.use_model_pass,
        "interp_multiplier": config.augment.interp_multiplier,
        "presentation_fps": config.augment.presentation_fps,
        "models_dir": config.video.models_dir,
        "deferred_audio": is_deferred_backend(config.video.backend),
        "audio_config": config.audio,
        "seed": config.seed,
    }
    # DESIGN §140 GPU defaults: model pass (cuda:1) + SFX dub (cuda:0)
    # run side by side when both stages are live on a 2-GPU box.
    # Deferred ACE also renders on cuda:0 (same card as the SFX bed), so a
    # deferred run always takes the sequential path — Thread A/B forking
    # while takes are pending would collide on the 4060. An explicit
    # presentation fps also forces sequential: Thread B's reference concat
    # walks the source timeline, which a retimed present no longer matches.
    parallel = False
    if (
        sfx_will_run
        and not video_kwargs["deferred_audio"]
        and video_kwargs["presentation_fps"] is None
    ):
        from voyage.finalize_parallel import SfxBedError, run_parallel_finalize, should_run_parallel

        parallel = should_run_parallel(
            sfx_backend=sfx_backend,
            use_model_pass=config.augment.use_model_pass,
            models_dir=config.video.models_dir,
            num_workers=getattr(args, "sfx_workers", 1),
        )
        if parallel:
            try:
                run_parallel_finalize(
                    run_dir,
                    output,
                    **video_kwargs,
                    sfx_backend=sfx_backend,
                    sfx_device=getattr(args, "sfx_device", None) or config.sfx.device,
                    sfx_model_size=getattr(args, "sfx_model_size", None) or config.sfx.model_size,
                    sfx_caption=getattr(args, "sfx_caption", None),
                    invoker=getattr(args, "invoker", None),
                )
            except SfxBedError as exc:
                print(
                    f"sfx pass failed (music-only kept at {output}): {exc}",
                    file=sys.stderr,
                )
                return 1
            except (MediaError, StateError) as exc:
                print(f"finalize failed: {exc}", file=sys.stderr)
                return 1
    if not parallel:
        try:
            finalize_run(run_dir, output, **video_kwargs, invoker=getattr(args, "invoker", None))
        except (MediaError, StateError, DiskSpaceError) as exc:
            print(f"finalize failed: {exc}", file=sys.stderr)
            return 1
        if sfx_will_run:
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
                )
            except (MediaError, StateError) as exc:
                print(f"sfx pass failed (music-only kept at {output}): {exc}", file=sys.stderr)
                return 1
    # Freshness stamp for the generate 'nothing to do' gate (redundant
    # finalize fix): records what final.mp4 actually presents, so a revisit
    # compares presented-against-presented. Only on full success — an SFX
    # failure returns above, and its music-only final must stay re-finalizable.
    coverage_frames = presented_frames(output)
    if coverage_frames is not None:
        record_final_coverage(
            run_dir,
            presented_frames=coverage_frames,
            segments=read_state(run_dir).committed_segments,
        )
    console = get_console(args)
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


def cmd_sfx(args: argparse.Namespace) -> int:
    """Dub SFX onto an existing video (standalone post-pass, no re-finalize).

    Conditions on `--video` (default: the run's `final.mp4`), mixes the
    bed under its audio, and publishes `--output` (default:
    `final-sfx.mp4` beside the input — the input is never modified).
    Caption source is per-segment director captions unless
    `--sfx-caption` overrides (required for runs committed before SFX
    captions existed, e.g. poulah).
    """
    import shutil

    from voyage.sfx_finalize import finalize_sfx_pass

    run_dir = resolve_run_ref(run=getattr(args, "run", None), name=getattr(args, "name", None))
    if run_dir is None:
        return 2
    config = _load_run(run_dir)
    video = Path(args.video).resolve() if args.video else run_dir / "final.mp4"
    if not video.exists():
        print(f"sfx failed: no such video {video}", file=sys.stderr)
        return 1
    output = Path(args.output).resolve() if args.output else video.parent / "final-sfx.mp4"
    if args.output:
        # Containment warning (issue 024) for the explicit write target
        # only: --video is a read-only input (never warned), and the
        # derived default beside it inherits the input's location.
        warn_if_outside_output_dir(output, flag="--output")
    backend = args.sfx_backend or config.sfx.backend
    try:
        if output != video:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, output)
        finalize_sfx_pass(
            run_dir,
            output,
            backend=backend,
            models_dir=config.sfx.models_dir,
            device=args.sfx_device or config.sfx.device,
            model_size=args.sfx_model_size or config.sfx.model_size,
            seed=config.seed,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            num_workers=args.sfx_workers,
            fps=config.video.fps,
            caption_override=args.sfx_caption,
        )
    except (MediaError, StateError, OSError) as exc:
        print(f"sfx dub failed: {exc}", file=sys.stderr)
        return 1
    console = get_console(args)
    try:
        info = media_probe(output)
        duration = info.get("format", {}).get("duration", "?") if isinstance(info, dict) else "?"
    except MediaError:
        duration = "?"
    console.ok(f"sfx dubbed -> {output} ({duration}s, backend {backend})")
    print(f"sfx dubbed -> {output}")
    return 0

"""`generate` one-shot verb (DESIGN §58).

Verb module of the issue-080 split: init → run → validate →
finalize orchestration. Fans out to the sibling verb modules via
function-level imports (the verb-module cycle rule).
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

from pydantic import ValidationError

from voyage.cli_core import _augment_overrides, _load_run, get_console
from voyage.cli_paths import (
    _check_run_id,
    _effective_run_id,
    resolve_run_dir,
    warn_if_outside_output_dir,
)
from voyage.cli_planning import (
    _CUDA_VIDEO_BACKENDS,
    _cuda_stack_error,
    _frames_per_segment,
    _require_cuda_stack,
    segments_for_duration,
)
from voyage.config import ProjectConfig, apply_draft_overrides, default_config_toml
from voyage.errors import DiskSpaceError
from voyage.persistence import read_state
from voyage.seeds import random_master_seed


def _pre_init_override_gate(args: argparse.Namespace, run_id: str, style: str, seed: int) -> int:
    """Dry-run the override resolution against the preset config (110).

    Renders the exact TOML `cmd_init` would write and applies the same
    overrides the post-init path applies — pure, no directory touched.
    Returns 0 when the overrides resolve, else prints the error and
    returns 2. Any exception shape here (ValidationError from the model
    validators, ValueError from unknown presets) maps to exit 2, exactly
    like the post-init gate it guards.
    """
    import tomllib

    try:
        preset_config = ProjectConfig.model_validate(
            tomllib.loads(
                default_config_toml(
                    run_id,
                    style,
                    seed,
                    video_backend=args.backend,
                    director_backend=args.director,
                    director_device=getattr(args, "director_device", None) or "cuda:1",
                )
            )
        )
        apply_draft_overrides(
            preset_config,
            draft=args.draft,
            director=args.director,
            director_device=getattr(args, "director_device", None),
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n_segments=args.drift_every_n,
            music_caption=getattr(args, "music_caption", None),
            video_caption=getattr(args, "video_caption", None),
            **_augment_overrides(args),
        )
    except (ValidationError, ValueError) as exc:
        print(f"error: invalid numeric override: {exc}", file=sys.stderr)
        return 2
    return 0


def _start_stop_key_listener(
    run_dir: Path, console: VoyageConsole, sentinel: str = "s"
) -> threading.Thread:
    """Watch stdin for the stop sentinel; request stop at the next boundary.

    Daemon thread: a line whose first non-space character matches the
    sentinel (case-insensitive) flips the run to STOP_REQUESTED via the
    state.json control plane, so the current segment finishes before
    finalize. Obvious console feedback on arm, on trigger, and on EOF.
    Returns the thread (daemon, never joined — the process exits past it).
    """
    from voyage.persistence import read_state, write_state

    def _watch() -> None:
        import sys

        try:
            for line in sys.stdin:
                if line.strip()[:1].lower() == sentinel:
                    try:
                        state = read_state(run_dir)
                        state.status = "STOP_REQUESTED"
                        write_state(run_dir, state)
                    except Exception as exc:
                        print(
                            f"warning: stop key ignored ({exc})",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        console.ok(
                            "stop key received — finishing the current segment, then finalizing ..."
                        )
                        print(
                            "stop key received — finishing the current segment, then finalizing ..."
                        )
                    except Exception:
                        pass
                    return
        except Exception:
            pass
        try:
            console.warn(
                "stop-key listener ended (stdin closed) — use Ctrl-C or "
                "`voyage stop` to finish instead"
            )
        except Exception:
            pass

    thread = threading.Thread(target=_watch, name="voyage-stop-key", daemon=True)
    thread.start()
    return thread


def cmd_generate(args: argparse.Namespace) -> int:
    """One-shot video: init -> run -> validate -> finalize.

    With --duration the run covers at least that long (rounded up to whole
    segments). Without it the run continues until the operator presses
    's' (finishing the current segment first), Ctrl-C, or `voyage stop`.
    """
    # Seam dispatch (issue 080): orchestration targets + test-patched
    # leaves resolve through the voyage.cli namespace at call time, exactly
    # as when they shared one module — monkeypatching voyage.cli keeps
    # intercepting every fan-out edge below.
    from voyage.cli import (
        _torch_available,
        _warn_if_no_cuda,
        check_ffmpeg,
        check_free_space,
        cmd_finalize,
        cmd_init,
        cmd_run,
        validate_run,
    )

    # Fail before init: generate always writes a fresh config from the
    # backend preset (CUDA video backends pair with ACE-Step audio), so the
    # preset alone decides the stack.
    if args.backend in _CUDA_VIDEO_BACKENDS and not _torch_available():
        print(_cuda_stack_error(args.backend), file=sys.stderr)
        return 1
    run_id = _effective_run_id(args)
    if _check_run_id(run_id) != 0:
        return 2
    # Resolve once: workers spawn with CWD=run_dir, so every downstream path
    # (payloads, takes, slices) must be absolute or they double up.
    output = Path(args.output) if args.output else Path("output") / run_id
    run_dir = resolve_run_dir(str(output))
    # Validate-before-mutate (issue 110): numeric overrides are pure config
    # math knowable before the first byte is written, so gate them against
    # the preset-resolved config BEFORE cmd_init. A bad override exits 2
    # with no orphan run dir for the retry to trip over. Blank styles skip
    # the gate — cmd_init reports those itself, litter-free since 143/185.
    style_value = getattr(args, "style", None)
    seed_value = getattr(args, "seed", None)
    if seed_value is None:
        seed_value = random_master_seed()
        args.seed = seed_value
        print(f"random seed for this run: {seed_value} (re-run with --seed {seed_value})")
    if (
        isinstance(style_value, str)
        and style_value.strip()
        and isinstance(seed_value, int)
        and not isinstance(seed_value, bool)
    ):
        gate_code = _pre_init_override_gate(args, run_id, style_value, seed_value)
        if gate_code != 0:
            return gate_code
    init_args = argparse.Namespace(
        output=str(run_dir),
        run_id=run_id,
        name=run_id,
        style=args.style,
        seed=args.seed,
        force=args.force,
        backend=args.backend,
        director=args.director,
        director_device=getattr(args, "director_device", None),
    )
    code = cmd_init(init_args)
    if code != 0:
        return code
    config, _digest = _load_run(run_dir)
    if not _require_cuda_stack(config):
        return 1
    _warn_if_no_cuda(config)
    # `generate` enables the full stack by default: qwen director drift,
    # beat-grid audio and overlap-blend finalize all ride the run config
    # written at init; explicit flags still win. The parser default for
    # --director is qwen, so args.director carries it directly.
    director = args.director
    try:
        effective = apply_draft_overrides(
            config,
            draft=args.draft,
            director=director,
            director_device=getattr(args, "director_device", None),
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n_segments=args.drift_every_n,
            music_caption=getattr(args, "music_caption", None),
            video_caption=getattr(args, "video_caption", None),
            **_augment_overrides(args),
        )
    except (ValidationError, ValueError) as exc:
        print(f"error: invalid numeric override: {exc}", file=sys.stderr)
        return 2
    frames_per_segment = _frames_per_segment(effective)
    raw_duration = getattr(args, "duration", None)
    if raw_duration is None:
        segments = None
        planned_frames = None
    else:
        segments = segments_for_duration(raw_duration, effective.video.fps, frames_per_segment)
        planned_frames = segments * frames_per_segment
    console = get_console(args)
    sink = getattr(args, "progress_sink", None)
    ffmpeg_ok, ffmpeg_message = check_ffmpeg()
    if not ffmpeg_ok:
        print(f"error: {ffmpeg_message}", file=sys.stderr)
        return 1
    try:
        check_free_space(run_dir, effective.min_free_space_gib)
        stack_dirs = {
            effective.video.models_dir,
            effective.audio.models_dir,
            effective.sfx.models_dir,
        }
        for stack_dir in sorted(stack_dirs):
            # The mount may not exist yet (downloads create it): check the
            # nearest existing ancestor so a missing dir never crashes.
            anchor = Path(stack_dir)
            while not anchor.exists():
                anchor = anchor.parent
            check_free_space(anchor, effective.min_free_space_gib)
    except DiskSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    from voyage.models_ensure import ensure_models

    sfx_enabled = (
        not bool(getattr(args, "no_sfx", False))
        and (getattr(args, "sfx_backend", None) or effective.sfx.backend) != "fake"
    )
    if (
        ensure_models(
            effective,
            sfx_enabled,
            console,
            allow_download=not bool(getattr(args, "no_download", False)),
            augment_enabled=not bool(getattr(args, "no_augment", False))
            and (effective.augment.min_fps > 0 or effective.augment.min_width > 0),
        )
        != 0
    ):
        return 1
    if sink is None:
        console.rule(
            f"voyage generate · {effective.video.backend} "
            f"{effective.video.width}x{effective.video.height} @{effective.video.fps}fps · "
            f"director {effective.director.backend}"
        )
        if planned_frames is None:
            console.info(
                f"plan: open-ended run (no --duration) · {frames_per_segment}f/segment · "
                f"{effective.audio.beats_per_segment} beats/segment · "
                f"drift every {effective.voyage.drift_every_n_segments}"
            )
            print(
                "generating until you press 's' "
                f"({frames_per_segment}f per segment) "
                f"with {effective.video.backend} ..."
            )
            print("PRESS 's' + Enter at any time to finish the current segment,")
            print("then the run validates and finalizes automatically.")
            print("TIP: keep this terminal focused — the 's' key is read from stdin.")
        else:
            console.info(
                f"plan: ~{planned_frames / effective.video.fps:.1f}s "
                f"({segments} segments, {planned_frames} frames) "
                f"· {effective.audio.beats_per_segment} beats/segment · "
                f"drift every {effective.voyage.drift_every_n_segments}"
            )
            print(
                f"generating ~{planned_frames / effective.video.fps:.1f}s "
                f"({segments} segments, {planned_frames} frames) "
                f"with {effective.video.backend} ..."
            )
    if planned_frames is None:
        console.info("stop-key armed: type 's' + Enter to stop after the current segment")
        print("stop-key armed: type 's' + Enter to stop after the current segment")
        _start_stop_key_listener(run_dir, console)
    cmd_run(
        argparse.Namespace(
            run=str(run_dir),
            segments=segments,
            draft=args.draft,
            director=director,
            director_device=getattr(args, "director_device", None),
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n=args.drift_every_n,
            music_caption=getattr(args, "music_caption", None),
            video_caption=getattr(args, "video_caption", None),
            min_fps=getattr(args, "min_fps", None),
            min_resolution=getattr(args, "min_resolution", None),
            no_augment=bool(getattr(args, "no_augment", False)),
            use_model_pass=getattr(args, "use_model_pass", None),
            verbose=console.verbose,
            no_color=getattr(args, "no_color", False),
            progress_sink=sink,
        )
    )
    errors = validate_run(run_dir)
    if errors:
        print("INVALID:")
        for error in errors:
            print(f"  - {error}")
        if not getattr(args, "skip_bad", False):
            print(
                "aborting before finalize (re-run with --skip-bad to salvage)",
                file=sys.stderr,
            )
            return 1
        print("continuing with --skip-bad ...", file=sys.stderr)
    final_video = getattr(args, "final_video", None)
    final = Path(final_video).resolve() if final_video else run_dir / "final.mp4"
    if final_video:
        # Containment warning (issue 024): the run dir already warned via
        # cmd_init's funnel; the final video is this verb's own write.
        warn_if_outside_output_dir(final, flag="--final-video")
    final.parent.mkdir(parents=True, exist_ok=True)
    final_code = cmd_finalize(
        argparse.Namespace(
            run=str(run_dir),
            output=str(final),
            skip_bad=getattr(args, "skip_bad", False),
            no_sfx=getattr(args, "no_sfx", False),
            sfx_backend=getattr(args, "sfx_backend", None),
            sfx_caption=getattr(args, "sfx_caption", None),
            sfx_device=getattr(args, "sfx_device", None),
            sfx_model_size=getattr(args, "sfx_model_size", None),
            sfx_workers=getattr(args, "sfx_workers", 1),
            min_fps=getattr(args, "min_fps", None),
            min_resolution=getattr(args, "min_resolution", None),
            no_augment=bool(getattr(args, "no_augment", False)),
            use_model_pass=getattr(args, "use_model_pass", None),
            # Console context rides both child stages (issue 147): the run
            # call above already forwards these three, the finalize call
            # dropped them — so generate --verbose went silent exactly
            # when the final video assembled.
            verbose=console.verbose,
            no_color=getattr(args, "no_color", False),
            progress_sink=sink,
        )
    )
    if final_code != 0:
        return final_code
    state = read_state(run_dir)
    actual_seconds = state.timeline_frames / effective.video.fps
    if sink is None:
        console.ok(
            f"generated {final} ({state.committed_segments} segments, "
            f"{state.timeline_frames} frames, ~{actual_seconds:.1f}s)"
        )
        print(
            f"generated {final} ({state.committed_segments} segments, "
            f"{state.timeline_frames} frames, ~{actual_seconds:.1f}s)"
        )
    return 0

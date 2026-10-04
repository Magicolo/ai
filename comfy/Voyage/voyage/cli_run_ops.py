"""`run` verb (DESIGN §§58, 70).

Verb module of the issue-080 split: run lifecycle (commit segments).
`generate` is the sole run creator — it writes the manifest carrying
the effective config, and `run` resumes from it. Orchestration stays in
`voyage.cli`'s parser seam; this module owns the verb and nothing else.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from types import FrameType

from pydantic import ValidationError

from voyage.cli_core import _augment_overrides, _load_run, get_console
from voyage.cli_paths import resolve_run_ref
from voyage.cli_planning import _require_cuda_stack
from voyage.config import apply_draft_overrides, is_provided
from voyage.console import RichSegmentProgress
from voyage.supervisor import Supervisor


def maybe_refinalize(args: argparse.Namespace, run_dir: Path, new_commits: list[str]) -> int:
    """Re-finalize `final.mp4` after an extend, iff segments committed.

    No new segments (or `--no-finalize`) → no-op returning 0. SFX /
    augment / skip-bad flags forward into the finalize step (same
    namespace shape `generate` fans out with); console context rides
    along so `--verbose` stays loud through the assembly.
    """
    if not new_commits or bool(getattr(args, "no_finalize", False)):
        return 0
    # Seam dispatch (issue 080): resolve through the voyage.cli namespace
    # at call time, exactly like `cmd_stop --finalize` does.
    from voyage.cli import cmd_finalize

    return cmd_finalize(
        argparse.Namespace(
            run=str(run_dir),
            output=str(run_dir / "final.mp4"),
            skip_bad=bool(getattr(args, "skip_bad", False)),
            no_sfx=bool(getattr(args, "no_sfx", False)),
            sfx_backend=getattr(args, "sfx_backend", None),
            sfx_caption=getattr(args, "sfx_caption", None),
            sfx_device=getattr(args, "sfx_device", None),
            sfx_model_size=getattr(args, "sfx_model_size", None),
            sfx_workers=getattr(args, "sfx_workers", 1),
            min_fps=getattr(args, "min_fps", None),
            min_resolution=getattr(args, "min_resolution", None),
            no_augment=bool(getattr(args, "no_augment", False)),
            use_model_pass=getattr(args, "use_model_pass", None),
            verbose=getattr(args, "verbose", False),
            no_color=getattr(args, "no_color", False),
            progress_sink=getattr(args, "progress_sink", None),
        )
    )


def cmd_run(args: argparse.Namespace) -> int:
    if args.segments is not None and args.segments <= 0:
        print(
            f"error: --segments must be positive, got {args.segments}",
            file=sys.stderr,
        )
        return 2
    run_dir = resolve_run_ref(run=getattr(args, "run", None), name=getattr(args, "name", None))
    if run_dir is None:
        return 2
    config, _digest = _load_run(run_dir)
    # Absent-encoding (issue 045): TUI namespaces carry Unset, argparse
    # carries None — branch on the predicate so an unset TUI field is
    # never mistaken for an explicit override.
    if (
        args.draft
        or is_provided(args.director)
        or is_provided(getattr(args, "director_device", None))
        or is_provided(args.blocks)
        or is_provided(args.take_seconds)
        or is_provided(args.quantization)
        or is_provided(getattr(args, "beats_per_segment", None))
        or is_provided(getattr(args, "drift_every_n", None))
        or is_provided(getattr(args, "music_caption", None))
        or is_provided(getattr(args, "video_caption", None))
        or bool(_augment_overrides(args))
    ):
        try:
            config = apply_draft_overrides(
                config,
                draft=args.draft,
                director=args.director,
                director_device=getattr(args, "director_device", None),
                blocks=args.blocks,
                take_seconds=args.take_seconds,
                quantization=args.quantization,
                beats_per_segment=getattr(args, "beats_per_segment", None),
                drift_every_n_segments=getattr(args, "drift_every_n", None),
                music_caption=getattr(args, "music_caption", None),
                video_caption=getattr(args, "video_caption", None),
                **_augment_overrides(args),
            )
        except (ValidationError, ValueError) as exc:
            print(f"error: invalid numeric override: {exc}", file=sys.stderr)
            return 2
        print(
            "effective settings: "
            f"director={config.director.backend} "
            f"director_device={config.director.device} "
            f"blocks={config.video.blocks_per_segment} "
            f"{config.video.width}x{config.video.height} "
            f"latent={list(config.video.latent_shape)} "
            f"take_seconds={config.audio.take_seconds} "
            f"beats_per_segment={config.audio.beats_per_segment} "
            f"drift_every_n={config.voyage.drift_every_n_segments} "
            f"quantization={config.video.quantization} "
            f"min_fps={config.augment.min_fps} "
            f"min_resolution={config.augment.min_width}x{config.augment.min_height}"
        )
    if not _require_cuda_stack(config):
        return 1
    console = get_console(args)
    # The TUI swaps the progress sink from stdout to its RichLog: when a
    # sink rides the namespace, the console stays silent (the TUI owns
    # the display) and only the sink reports.
    sink = getattr(args, "progress_sink", None)
    progress = sink if sink is not None else RichSegmentProgress(console)
    if sink is None:
        console.rule(
            f"voyage run · {config.video.backend} {config.video.width}x{config.video.height} "
            f"@{config.video.fps}fps · director {config.director.backend} · "
            f"{config.audio.beats_per_segment} beats/segment · "
            f"drift every {config.voyage.drift_every_n_segments}"
        )
    supervisor = Supervisor(run_dir, config, progress=progress)

    def _on_signal(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        supervisor.request_stop()

    previous = None
    try:
        previous = signal.signal(signal.SIGINT, _on_signal)
    except ValueError:
        # Non-main thread (e.g. the TUI worker): no SIGINT handler here —
        # the TUI Stop button drives the file-status control plane instead.
        pass
    try:
        committed = supervisor.run_segments(args.segments)
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if sink is None:
        console.ok(f"run finished · {len(committed)} segment(s) committed")
    if committed:
        final_code = maybe_refinalize(args, run_dir, committed)
        if final_code != 0:
            return final_code
    return 0

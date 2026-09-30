"""`init` and `run` verbs (DESIGN §§58, 70).

Verb module of the issue-080 split: run lifecycle (create + commit
segments). Orchestration stays in `voyage.cli`'s parser seam; this
module owns the two verbs and nothing else.
"""

from __future__ import annotations

import argparse
import signal
import sys
from types import FrameType
from typing import cast

from pydantic import ValidationError

from voyage import paths
from voyage.cli_core import _augment_overrides, _load_run, get_console
from voyage.cli_paths import (
    _check_run_id,
    _effective_run_id,
    _run_dir_arg,
    resolve_run_dir,
    warn_if_outside_output_dir,
)
from voyage.cli_planning import _require_cuda_stack
from voyage.config import (
    BACKEND_REGISTRY,
    VideoBackendName,
    apply_draft_overrides,
    default_config_toml,
    is_provided,
    load_config,
    warn_if_deprecated_backend,
)
from voyage.console import RichSegmentProgress
from voyage.persistence import build_manifest, initial_state, write_manifest, write_state
from voyage.supervisor import Supervisor


def cmd_init(args: argparse.Namespace) -> int:
    # Validate-before-mutate (issues 143/185): every pure-config check
    # below runs BEFORE the first mkdir, so a bad value exits 2 with no
    # partial run dir for the retry to trip over. Hand-built namespaces
    # read via getattr with the same exit-2 shape — never AttributeError.
    run_id = _effective_run_id(args)
    if _check_run_id(run_id) != 0:
        return 2
    output_value = getattr(args, "output", None)
    if not isinstance(output_value, str) or not output_value.strip():
        print("error: --output is required (run directory to create)", file=sys.stderr)
        return 2
    run_dir = resolve_run_dir(output_value)
    # Containment warning (issue 024): an explicit absolute --output is
    # legal but a typo (`--output /`) scatters writes with exit 0, so warn
    # when the resolved dir escapes ./output/. Warn-only: rejecting would
    # break documented /tmp flows and tmp_path-based suites.
    warn_if_outside_output_dir(run_dir, flag="--output")
    if run_dir.exists() and any(run_dir.iterdir()) and not getattr(args, "force", False):
        print(f"refusing to init non-empty directory {run_dir} (use --force)", file=sys.stderr)
        return 2
    style_value = getattr(args, "style", None)
    if not isinstance(style_value, str) or not style_value.strip():
        print("error: --style must be a non-empty human-owned style string", file=sys.stderr)
        return 2
    seed_value = getattr(args, "seed", None)
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        print(f"error: --seed must be an integer, got {seed_value!r}", file=sys.stderr)
        return 2
    backend_value = getattr(args, "backend", None) or "ltxv"
    if backend_value not in BACKEND_REGISTRY:
        known = ", ".join(sorted(BACKEND_REGISTRY))
        print(f"error: unknown video backend {backend_value!r} (known: {known})", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    backend: VideoBackendName = cast(VideoBackendName, backend_value)
    warn_if_deprecated_backend(backend)
    director_backend: str = getattr(args, "director", None) or "qwen"
    director_device: str = getattr(args, "director_device", None) or "cuda:1"
    config_text = default_config_toml(
        run_id,
        style_value,
        seed_value,
        video_backend=backend,
        director_backend=director_backend,
        director_device=director_device,
    )
    (run_dir / paths.CONFIG_FILENAME).write_text(config_text, encoding="utf-8")
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    hardware = {"note": "recorded at init; see `voyage doctor` for live facts"}
    software = {"python": sys.version.split()[0]}
    write_manifest(run_dir, build_manifest(config, digest, hardware, software))
    write_state(run_dir, initial_state(config))
    print(f"initialized voyage run at {run_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.segments is not None and args.segments <= 0:
        print(
            f"error: --segments must be positive, got {args.segments}",
            file=sys.stderr,
        )
        return 2
    run_dir = _run_dir_arg(args.run)
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
    return 0

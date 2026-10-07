"""`voyage` CLI (DESIGN §§58, two-verb CLI).

`configure` plans a run into `output/<NAME>/manifest.json`;
`generate` reconciles the run directory to that plan and runs it.
Every other verb was removed (step 4): `run`, `status`, `pause`,
`resume`, `stop`, `validate`, `finalize`, `sfx`, `benchmark`, `soak`,
`inspect`, `models`, `doctor`, and the bare-launch TUI are gone —
`generate` runs the whole pipeline (segments → validate → finalize)
internally, and the removed verbs' orchestration lives on only where
`generate`/`configure` still need it (see `cli_finalize`,
`models_ensure`, `doctor`).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from voyage.cli_configure import (
    cmd_configure,
)
from voyage.cli_generate import cmd_generate
from voyage.cli_planning import (
    _DURATION_EXAMPLES,
    parse_duration,
)
from voyage.errors import VoyageError

__all__ = [
    "_DURATION_EXAMPLES",
    "_add_augment_args",
    "_add_configure_parser",
    "_add_console_args",
    "_add_generate_parser",
    "_add_generation_overrides",
    "_add_sfx_args",
    "build_parser",
    "cmd_configure",
    "cmd_generate",
    "main",
    "parse_duration",
]


def _add_console_args(parser: argparse.ArgumentParser) -> None:
    """Two verbosity levels + color kill-switch + quiet (console output only)."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="detailed console output (retry feedback, beat math, take decisions, seeds)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="plain console output (no colors or animation; also honors NO_COLOR)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only failures and final paths (exit code callers, log scrapers)",
    )


def _add_generation_overrides(parser: argparse.ArgumentParser) -> None:
    """In-memory run overrides for `configure` (issue 020).

    One helper so the flag set stays in one place; order matches the
    historical layout (help output unchanged).
    """
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="override video blocks per segment (must be positive)",
    )
    parser.add_argument(
        "--take-seconds",
        type=float,
        default=None,
        help="override audio take length in seconds (must be positive)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=("fp8", "bf16"),
        help="override DiT quantization (fp8 default, bf16 for clean highlights)",
    )
    parser.add_argument(
        "--beats-per-segment",
        type=int,
        default=None,
        help="override beats per segment (must be positive; default 4, doubles to hold >=60 BPM)",
    )
    parser.add_argument(
        "--drift-every-n",
        type=int,
        default=None,
        help="director drifts every Nth segment (must be positive; default 1); other segments hold",
    )
    parser.add_argument(
        "--scene-cut-every-n",
        type=int,
        default=None,
        help="fresh scene cut every Nth segment (must be positive; "
        "default 3); other segments continue",
    )
    parser.add_argument(
        "--music-caption",
        default=None,
        help="pin the music caption family (default: director drives + evolves it)",
    )
    parser.add_argument(
        "--video-caption",
        default=None,
        help="pin the video caption family (default: director drives + evolves it)",
    )
    parser.add_argument(
        "--prompt-enhance",
        action="store_true",
        default=None,
        help="expand staged prompts through the llama-server sidecar before "
        "the LTX video render (default on; "
        "stored in the manifest, so generate honors it)",
    )
    parser.add_argument(
        "--no-prompt-enhance",
        action="store_true",
        default=None,
        help="disable the prompt-expansion stage (default on; "
        "mutually exclusive with --prompt-enhance)",
    )
    parser.add_argument(
        "--director-device",
        default=None,
        help="director decider placement (default cuda:1 = second GPU via 4-bit AWQ; "
        "cpu = legacy bf16 CPU path, explicit opt-out for single-GPU/CI boxes)",
    )


def _add_sfx_args(parser: argparse.ArgumentParser) -> None:
    """Finalize-time SFX flags for `configure` (092).

    The SFX pass runs inside `generate`'s finalize step; the stored
    `[sfx]` section plus these overrides decide it.
    """
    parser.add_argument(
        "--no-sfx",
        action="store_true",
        help="skip the finalize-time SFX pass even when [sfx] is configured",
    )
    parser.add_argument(
        "--sfx-backend",
        default=None,
        choices=["fake", "mmaudio"],
        help="SFX backend override (default: [sfx] backend)",
    )
    parser.add_argument(
        "--sfx-caption",
        default=None,
        help="single SFX caption for the whole timeline (default: per-segment "
        "director captions; required for runs committed before SFX captions existed)",
    )
    parser.add_argument(
        "--sfx-device",
        default=None,
        help="SFX worker device override (default: [sfx] device)",
    )
    parser.add_argument(
        "--sfx-model-size",
        default=None,
        choices=["small_44k", "medium_44k", "large_44k_v2"],
        help="MMAudio variant override (default: [sfx] model_size)",
    )
    parser.add_argument(
        "--sfx-workers",
        type=int,
        default=None,
        choices=[1, 2],
        help="SFX worker override: 1 = one worker; 2 = shard small_44k across "
        "cuda:0+cuda:1 (needs 2 visible GPUs, fails fast otherwise; "
        "default: [sfx] num_workers, 1)",
    )
    parser.add_argument(
        "--sfx-dual-pan",
        action="store_true",
        default=None,
        help="render the spatialized SFX pair: two same-caption seeds panned "
        "75%% left/right, mixed with music to stereo (default on; "
        "stored in the manifest, so generate honors it)",
    )
    parser.add_argument(
        "--no-sfx-dual-pan",
        action="store_true",
        default=None,
        help="single SFX bed (legacy; mutually exclusive with --sfx-dual-pan)",
    )


def _add_augment_args(parser: argparse.ArgumentParser) -> None:
    """Finalize-time explicit quality multipliers for `configure`.

    The multipliers ride the run's stored `[augment]` section; `generate`'s
    finalize step consumes them. There are no minimum quality floors:
    quality is specified explicitly (`1/1` ships the source as-is).
    """
    parser.add_argument(
        "--upscale",
        type=int,
        default=None,
        help="resolution multiplier at finalize against the probed source "
        "(default: [augment] upscale 1; 2 doubles width and height)",
    )
    parser.add_argument(
        "--interpolate",
        type=int,
        default=None,
        help="frame-count multiplier at finalize via the configured interpolation "
        "backend (default: [augment] interpolate 1; 2 doubles the frame count)",
    )
    parser.add_argument(
        "--interp-backend",
        default=None,
        choices=["film", "rife"],
        help="interpolation backend at finalize: rife (default, ~16.8x faster "
        "than film at 2048x1152, eyeball-identical) or film (hero/archival)",
    )
    parser.add_argument(
        "--presentation-fps",
        type=int,
        default=None,
        help="pin the shipped frame rate instead of source x interpolate "
        "(default: unset; e.g. 24fps x2 content at 32fps stretches the "
        "timeline 1.5x slow motion)",
    )


def _add_configure_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`configure` verb: init or update a run manifest (all run options live here)."""
    conf = sub.add_parser(
        "configure", help="Init or update a run manifest (all run options live here)"
    )
    conf.add_argument("name", help="run name (flat folder name → output/<name>)")
    conf.add_argument(
        "--backend",
        choices=("fake", "ltxv", "causvid", "ltx25", "ltx23"),
        default=None,
        help="video backend preset written into the run config",
    )
    conf.add_argument(
        "--low-definition",
        action="store_true",
        help="lowest native reasonable resolution for the effective backend "
        "(mutually exclusive with --medium-definition and --high-definition)",
    )
    conf.add_argument(
        "--medium-definition",
        action="store_true",
        help="use the medium native reasonable resolution for the effective backend "
        "(mutually exclusive with --low-definition and --high-definition)",
    )
    conf.add_argument(
        "--high-definition",
        action="store_true",
        help="highest native reasonable resolution for the effective backend "
        "(default on create when no tier flag is passed; mutually "
        "exclusive with --low-definition and --medium-definition)",
    )
    conf.add_argument(
        "--from",
        dest="from_run",
        default=None,
        help="inherit style + tuning + segment count from another run's "
        "manifest (output/<NAME>/manifest.json); only on create, "
        "explicit flags override inherited values, seed stays fresh "
        "unless --seed is passed",
    )
    conf.add_argument(
        "--duration",
        type=parse_duration,
        required=False,
        default=None,
        help=f"target length, e.g. {_DURATION_EXAMPLES} (converts to a segment count; "
        "exactly one of --duration/--segments on first configure)",
    )
    conf.add_argument(
        "--segments",
        type=int,
        default=None,
        help="planned segment count (exactly one of --duration/--segments on first configure)",
    )
    conf.add_argument("--style", default=None, help="permanent style charter for the run")
    conf.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master seed for the run (omit for a fresh random seed, printed at init)",
    )
    conf.add_argument("--force", action="store_true", help="allow init into a non-empty directory")
    conf.add_argument(
        "--final-video",
        default=None,
        help="final mp4 path (default <run>/final.mp4)",
    )
    conf.add_argument(
        "--skip-bad",
        action="store_true",
        help="finalize past corrupt segments instead of aborting",
    )
    conf.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading missing models (verify only)",
    )
    conf.add_argument(
        "--director",
        default=None,
        choices=("qwen", "deterministic", "llama"),
        help="director backend",
    )
    _add_generation_overrides(conf)
    _add_sfx_args(conf)
    _add_augment_args(conf)
    _add_console_args(conf)
    conf.set_defaults(func=cmd_configure)


def _add_generate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`generate` verb: reconcile a configured run to its manifest plan."""
    gen = sub.add_parser(
        "generate",
        help="Generate (or resume) a configured run to its manifest plan",
    )
    gen.add_argument("name", help="run name (flat folder name → output/<name>)")
    gen.add_argument(
        "--segments",
        type=int,
        default=None,
        help="extend the stored plan by this many segments (additive: "
        "new plan = manifest segments + N; exactly one of --segments/--duration)",
    )
    gen.add_argument(
        "--duration",
        type=parse_duration,
        required=False,
        default=None,
        help=f"extend the stored plan by this much video, e.g. {_DURATION_EXAMPLES} "
        "(additive: converts to segments rounding up, like configure; "
        "exactly one of --segments/--duration)",
    )
    gen.add_argument(
        "--no-music",
        action="store_true",
        default=False,
        help="skip the finalize-time music mix for this generate only "
        "(ships silent AAC sized to the timeline; non-persistent)",
    )
    gen.add_argument(
        "--no-sfx",
        action="store_true",
        default=False,
        help="skip the finalize-time SFX pass for this generate only (non-persistent)",
    )
    gen.add_argument(
        "--no-upscale",
        action="store_true",
        default=False,
        help="force upscale multiplier 1 for this generate only (non-persistent)",
    )
    gen.add_argument(
        "--no-interpolate",
        action="store_true",
        default=False,
        help="force interpolate multiplier 1 for this generate only (non-persistent)",
    )
    gen.add_argument(
        "--no-audio",
        action="store_true",
        default=False,
        help="shorthand for --no-music --no-sfx --no-master (this generate only)",
    )
    gen.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="shorthand for --no-upscale --no-interpolate (this generate only)",
    )
    gen.add_argument(
        "--no-master",
        action="store_true",
        default=False,
        help="skip the SonicMaster mastering stage for this generate only (non-persistent)",
    )
    _add_console_args(gen)
    gen.set_defaults(func=cmd_generate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voyage", description="Autonomous infinite audiovisual voyage"
    )
    sub = parser.add_subparsers(dest="command", required=False)

    _add_configure_parser(sub)
    _add_generate_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help(file=sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except VoyageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

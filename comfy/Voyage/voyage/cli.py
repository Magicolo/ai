"""`voyage` CLI (DESIGN §§58, task group J).

init / doctor / models / benchmark / run / generate / status / pause / resume /
stop / validate / finalize / inspect
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from voyage.cli_configure import (
    cmd_configure,
)
from voyage.cli_core import (
    _augment_overrides,
    _load_run,
    get_console,
)
from voyage.cli_finalize import (
    cmd_finalize,
    cmd_sfx,
)
from voyage.cli_generate import cmd_generate
from voyage.cli_models import (
    _download_audio,
    _download_director,
    _download_director_awq,
    _download_director_gguf,
    _download_film,
    _download_inspector,
    _download_realesrgan,
    _download_sfx,
    _models_dir,
    cmd_doctor,
    cmd_models,
)
from voyage.cli_observe import (
    _benchmark_env,
    _benchmark_revisions,
    _check_benchmark_counts,
    _persist_benchmark_report,
    _stage_means,
    _video_geometry_setup,
    cmd_benchmark,
    cmd_inspect,
    cmd_soak,
)
from voyage.cli_paths import (
    _RESERVED_FOLDER_NAMES,
    _check_run_id,
    _effective_run_id,
    _run_dir_arg,
    is_flat_folder_name,
    resolve_run_dir,
    resolve_run_ref,
)
from voyage.cli_planning import (
    _CAUSVID_NOVEL_PER_ROLLOUT,
    _CUDA_AUDIO_BACKENDS,
    _CUDA_BACKENDS,
    _CUDA_SFX_BACKENDS,
    _CUDA_VIDEO_BACKENDS,
    _DURATION_EXAMPLES,
    _DURATION_PATTERN,
    _LTXV_NOVEL_BLOCK_FRAMES,
    _cuda_offenders,
    _cuda_stack_error,
    _frames_per_segment,
    _require_cuda_stack,
    _torch_available,
    _warn_if_no_cuda,
    parse_duration,
    segments_for_duration,
)
from voyage.cli_run_ops import (
    cmd_run,
)
from voyage.cli_status import (
    _format_uptime,
    _last_commit_stages,
    _latest_novelty,
    _read_all_metric_events,
    _set_status,
    _slowest_stage,
    _status_restart_counts,
    cmd_pause,
    cmd_resume,
    cmd_status,
    cmd_stop,
)
from voyage.cli_validate import (
    _ORPHAN_PATTERNS,
    _SEGMENT_ID_PATTERN,
    _check_segment_checksums,
    _check_segment_metrics,
    _collect_transient_orphans,
    cmd_validate,
    validate_run,
)
from voyage.doctor import check_ffmpeg
from voyage.errors import VoyageError
from voyage.media import check_free_space
from voyage.model_registry import (
    download_audio_models,
    download_causvid_models,
    download_director_awq_models,
    download_director_gguf_models,
    download_director_models,
    download_film_models,
    download_inspector_models,
    download_ltx23_models,
    download_ltx25_models,
    download_ltxv_models,
    download_realesrgan_models,
    download_sfx_models,
    verify_audio_models,
    verify_causvid_models,
    verify_director_awq_models,
    verify_director_gguf_models,
    verify_director_models,
    verify_film_models,
    verify_inspector_models,
    verify_ltx23_models,
    verify_ltx25_models,
    verify_ltxv_models,
    verify_realesrgan_models,
    verify_sfx_models,
)

# Backward-compat surface (issue 080): every name below is either defined
# in this seam (parsers/build_parser/main) or re-exported from its verb-group
# home module. mypy reads this list as the explicit export contract, so
# imports from voyage.cli keep working exactly as before the split.
# Dispatch rule: cross-verb calls and test-patched leaves (check_ffmpeg,
# check_free_space, download_*/verify_*) resolve through this namespace at
# call time (function-level `from voyage.cli import ...` in the verb
# modules) — never bind them from their home module, or seam patching
# silently stops intercepting.
__all__ = [
    "_CAUSVID_NOVEL_PER_ROLLOUT",
    "_CUDA_AUDIO_BACKENDS",
    "_CUDA_BACKENDS",
    "_CUDA_SFX_BACKENDS",
    "_CUDA_VIDEO_BACKENDS",
    "_DURATION_EXAMPLES",
    "_DURATION_PATTERN",
    "_LTXV_NOVEL_BLOCK_FRAMES",
    "_ORPHAN_PATTERNS",
    "_RESERVED_FOLDER_NAMES",
    "_SEGMENT_ID_PATTERN",
    "_add_augment_args",
    "_add_benchmark_parser",
    "_add_configure_parser",
    "_add_console_args",
    "_add_doctor_parser",
    "_add_finalize_parser",
    "_add_generate_parser",
    "_add_generation_overrides",
    "_add_inspect_parser",
    "_add_models_parser",
    "_add_pause_parser",
    "_add_resume_parser",
    "_add_run_parser",
    "_add_run_ref",
    "_add_sfx_args",
    "_add_sfx_parser",
    "_add_soak_parser",
    "_add_status_parser",
    "_add_stop_parser",
    "_add_validate_parser",
    "_augment_overrides",
    "_benchmark_env",
    "_benchmark_revisions",
    "_check_benchmark_counts",
    "_check_run_id",
    "_check_segment_checksums",
    "_check_segment_metrics",
    "_collect_transient_orphans",
    "_cuda_offenders",
    "_cuda_stack_error",
    "_download_audio",
    "_download_director",
    "_download_director_awq",
    "_download_director_gguf",
    "_download_film",
    "_download_inspector",
    "_download_realesrgan",
    "_download_sfx",
    "_effective_run_id",
    "_format_uptime",
    "_frames_per_segment",
    "_last_commit_stages",
    "_latest_novelty",
    "_load_run",
    "_models_dir",
    "_persist_benchmark_report",
    "_read_all_metric_events",
    "_require_cuda_stack",
    "_run_dir_arg",
    "_set_status",
    "_slowest_stage",
    "_stage_means",
    "_status_restart_counts",
    "_torch_available",
    "_video_geometry_setup",
    "_warn_if_no_cuda",
    "build_parser",
    "check_ffmpeg",
    "check_free_space",
    "cmd_benchmark",
    "cmd_configure",
    "cmd_doctor",
    "cmd_finalize",
    "cmd_generate",
    "cmd_inspect",
    "cmd_models",
    "cmd_pause",
    "cmd_resume",
    "cmd_run",
    "cmd_sfx",
    "cmd_soak",
    "cmd_status",
    "cmd_stop",
    "cmd_validate",
    "download_audio_models",
    "download_causvid_models",
    "download_director_awq_models",
    "download_director_gguf_models",
    "download_director_models",
    "download_film_models",
    "download_inspector_models",
    "download_ltx23_models",
    "download_ltx25_models",
    "download_ltxv_models",
    "download_realesrgan_models",
    "download_sfx_models",
    "get_console",
    "is_flat_folder_name",
    "launch_tui",
    "main",
    "parse_duration",
    "resolve_run_dir",
    "resolve_run_ref",
    "segments_for_duration",
    "validate_run",
    "verify_audio_models",
    "verify_causvid_models",
    "verify_director_awq_models",
    "verify_director_gguf_models",
    "verify_director_models",
    "verify_film_models",
    "verify_inspector_models",
    "verify_ltx23_models",
    "verify_ltx25_models",
    "verify_ltxv_models",
    "verify_realesrgan_models",
    "verify_sfx_models",
]


def _add_console_args(parser: argparse.ArgumentParser) -> None:
    """Two verbosity levels + color kill-switch (console output only)."""
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


def launch_tui() -> int:
    """Bare-command launcher (indirection so tests can monkeypatch).

    The Textual import stays lazy so every CLI verb works without the
    display extra installed.
    """
    from voyage.tui import run_tui

    return run_tui()


def _add_doctor_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`doctor` verb: probe hardware and environment."""
    doctor = sub.add_parser("doctor", help="Probe hardware and environment")
    doctor.set_defaults(func=cmd_doctor)


def _add_models_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`models` verb: model management."""
    models = sub.add_parser("models", help="Model management")
    models.add_argument(
        "models_action",
        choices=["list", "download", "verify", "info"],
        help="list backends, download weights, verify files, or show pointers",
    )
    models.add_argument(
        "models_target",
        nargs="?",
        default="ltx25",
        choices=[
            "ltxv-2b",
            "causvid",
            "ltx25",
            "ltx23",
            "director-qwen8b",
            "director-qwen35-gguf",
            "audio-acestep",
            "sfx-mmaudio",
            "film",
            "realesrgan-anime",
            "inspector-qwen35",
        ],
        help="weight bundle for download (default ltx25)",
    )
    models.add_argument("--models-dir", default=None, help="model root (default /models)")
    models.set_defaults(func=cmd_models)


def _add_generation_overrides(parser: argparse.ArgumentParser) -> None:
    """In-memory run overrides shared by `run` and `generate` (issue 020).

    One helper so the flag set cannot drift between the two verbs; order
    matches the historical layout (help output unchanged).
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
        "--director-device",
        default=None,
        help="director decider placement (default cuda:1 = second GPU via 4-bit AWQ; "
        "cpu = legacy bf16 CPU path, explicit opt-out for single-GPU/CI boxes)",
    )


def _add_sfx_args(parser: argparse.ArgumentParser, *, include_no_sfx: bool = True) -> None:
    """Finalize-time SFX flags, shared by every verb that finalizes (092).

    `finalize` owns the pass; `generate`/`stop --finalize` forward into
    it; the standalone `sfx` verb dubs an existing video (no --no-sfx
    there — the verb IS the pass). One helper so the flags (and their
    defaults) cannot drift apart across verbs — a missing flag on any
    finalizing verb is an AttributeError at finalize time.
    """
    if include_no_sfx:
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
        default=1,
        choices=[1, 2],
        help="1 = one worker (default); 2 = shard small_44k across cuda:0+cuda:1 "
        "(needs 2 visible GPUs, fails fast otherwise)",
    )


def _add_augment_args(parser: argparse.ArgumentParser) -> None:
    """Finalize-time augmentation floors, shared by verbs carrying them (Track A).

    `finalize` owns the floors (defaults ride the run's [augment] TOML
    section); `generate`/`run` forward overrides into resolve_config so
    the effective config carries them. One helper so the flags (and their
    defaults) cannot drift apart across verbs — mirrors `_add_sfx_args`.
    """
    parser.add_argument(
        "--min-fps",
        type=int,
        default=None,
        help="floor output fps at finalize (default: [augment] min_fps 24; 0 disables)",
    )
    parser.add_argument(
        "--min-resolution",
        default=None,
        help='floor output resolution at finalize, WxH e.g. "1216x704" '
        '(default: [augment] 1216x704; "0" disables)',
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="disable all finalize augmentation floors (fps + resolution floors to 0)",
    )
    parser.add_argument(
        "--use-model-pass",
        action="store_true",
        default=None,
        help="run the Real-ESRGAN + FILM model pass at finalize when provisioned "
        "(default: [augment] use_model_pass on, pinned to cuda:1 when two GPUs show; "
        "--no-augment turns it off with the floors)",
    )
    parser.add_argument(
        "--interp-multiplier",
        type=int,
        default=None,
        help="FILM interpolation multiplier for the model pass "
        "(default: [augment] interp_multiplier 4; 1 = upscale only, no interpolation)",
    )
    parser.add_argument(
        "--presentation-fps",
        type=int,
        default=None,
        help="pin the shipped frame rate instead of the floors rule "
        "(default: unset; e.g. 24fps x2 content at 32fps stretches the "
        "timeline 1.5x slow motion)",
    )


def _add_run_ref(parser: argparse.ArgumentParser, *, noun: str) -> None:
    """`--run` vs `--name` selector shared by every run verb.

    `--name jango` resolves to the default output path (`output/jango`);
    passing both flags (or a non-flat name) is exit 2 at resolve time.
    `--run` stays accepted everywhere for scripts and tmp_path flows.
    """
    parser.add_argument(
        "--run",
        required=False,
        default=None,
        help=f"run directory (absolute or relative; exactly one of --run/--name) — {noun}",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="run name: shortcut for output/<name> (exactly one of --run/--name)",
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


def _add_run_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`run` verb: generate segments (infinite unless --segments)."""
    run = sub.add_parser("run", help="Generate segments (infinite unless --segments)")
    _add_run_ref(run, noun="run directory to extend")
    run.add_argument(
        "--segments",
        type=int,
        default=None,
        help="segments to generate (must be positive; omit to run until pause/stop/SIGINT)",
    )
    run.add_argument(
        "--no-finalize",
        action="store_true",
        help="skip the automatic re-finalize of final.mp4 after new segments commit",
    )
    run.add_argument(
        "--skip-bad",
        action="store_true",
        help="finalize past corrupt segments instead of aborting "
        "(forwarded to the re-finalize step)",
    )
    run.add_argument(
        "--director",
        default=None,
        choices=("qwen", "deterministic", "llama"),
        help="override the director backend",
    )
    _add_generation_overrides(run)
    _add_augment_args(run)
    _add_sfx_args(run)
    _add_console_args(run)
    run.set_defaults(func=cmd_run)


def _add_generate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`generate` verb: reconcile a configured run to its manifest plan."""
    gen = sub.add_parser(
        "generate",
        help="Generate (or resume) a configured run to its manifest plan",
    )
    gen.add_argument("name", help="run name (flat folder name → output/<name>)")
    _add_console_args(gen)
    gen.set_defaults(func=cmd_generate)


def _add_status_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`status` verb: show run status."""
    status = sub.add_parser("status", help="Show run status")
    _add_run_ref(status, noun="run directory to report on")
    status.set_defaults(func=cmd_status)


def _add_pause_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`pause` verb: request a safe pause."""
    pause = sub.add_parser("pause", help="Request a safe pause")
    _add_run_ref(pause, noun="run directory to pause")
    pause.set_defaults(func=cmd_pause)


def _add_resume_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`resume` verb: resume from last commit."""
    resume = sub.add_parser("resume", help="Resume from last commit")
    _add_run_ref(resume, noun="run directory to resume")
    resume.set_defaults(func=cmd_resume)


def _add_stop_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`stop` verb: safely stop generation."""
    stop = sub.add_parser("stop", help="Safely stop generation")
    _add_run_ref(stop, noun="run directory to stop")
    stop.add_argument(
        "--finalize",
        action="store_true",
        help="run the finalizer inline after requesting stop",
    )
    stop.add_argument(
        "--skip-bad",
        action="store_true",
        help="skip corrupt segments with a warning instead of aborting",
    )
    _add_sfx_args(stop)
    _add_augment_args(stop)
    _add_console_args(stop)
    stop.set_defaults(func=cmd_stop)


def _add_validate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`validate` verb: offline consistency check (read-only)."""
    validate = sub.add_parser("validate", help="Offline consistency check (read-only)")
    _add_run_ref(validate, noun="run directory to check")
    validate.set_defaults(func=cmd_validate)


def _add_finalize_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`finalize` verb: assemble the final MP4."""
    finalize = sub.add_parser("finalize", help="Assemble the final MP4")
    _add_run_ref(finalize, noun="run directory to finalize")
    finalize.add_argument("--output", required=True, help="final mp4 path to write")
    finalize.add_argument(
        "--skip-bad",
        action="store_true",
        help="skip corrupt segments with a warning instead of aborting",
    )
    _add_sfx_args(finalize)
    _add_augment_args(finalize)
    _add_console_args(finalize)
    finalize.set_defaults(func=cmd_finalize)


def _add_sfx_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`sfx` verb: dub SFX onto an existing video (no re-finalize)."""
    sfx = sub.add_parser("sfx", help="Dub SFX onto an existing video")
    _add_run_ref(sfx, noun="run directory (captions + ledger + seeds)")
    sfx.add_argument(
        "--video",
        default=None,
        help="existing mp4 to condition on (default: <run>/final.mp4)",
    )
    sfx.add_argument(
        "--output",
        default=None,
        help="dubbed mp4 to write (default: final-sfx.mp4 beside the input)",
    )
    _add_sfx_args(sfx, include_no_sfx=False)
    _add_console_args(sfx)
    sfx.set_defaults(func=cmd_sfx)


def _add_benchmark_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`benchmark` verb: performance probes."""
    benchmark = sub.add_parser("benchmark", help="Performance probes")
    benchmark.add_argument(
        "benchmark_target",
        choices=["audio", "augment", "end-to-end", "sfx", "video"],
        help="which probe to run",
    )
    benchmark.add_argument(
        "--run",
        default=None,
        help="run directory (required for video/audio; sfx/augment use the run's "
        "config when given — fake sfx and the augment ffmpeg probe need no run)",
    )
    benchmark.add_argument(
        "--name",
        default=None,
        help="run name: shortcut for output/<name> (exactly one of --run/--name)",
    )
    benchmark.add_argument("--warmup", type=int, default=1, help="warmup iterations (must be >= 0)")
    benchmark.add_argument(
        "--measured", type=int, default=3, help="measured iterations (must be >= 1)"
    )
    benchmark.add_argument(
        "--segments",
        type=int,
        default=2,
        help="segments for the end-to-end target (must be positive)",
    )
    benchmark.set_defaults(func=cmd_benchmark)


def _add_soak_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`soak` verb: stability run with a resource-trend report."""
    soak = sub.add_parser("soak", help="Stability run with a resource-trend report")
    _add_run_ref(soak, noun="run directory to soak-test")
    soak.add_argument(
        "--segments", type=int, required=True, help="segments to run (must be positive)"
    )
    _add_console_args(soak)
    soak.set_defaults(func=cmd_soak)


def _add_inspect_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`inspect` verb: inspect run artifacts."""
    inspect = sub.add_parser("inspect", help="Inspect run artifacts")
    inspect.add_argument(
        "inspect_target",
        choices=["concepts", "segments", "media", "metrics", "scoreboard"],
        help="which artifact view to print",
    )
    _add_run_ref(inspect, noun="run directory to inspect")
    inspect.set_defaults(func=cmd_inspect)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voyage", description="Autonomous infinite audiovisual voyage"
    )
    # No `required=True`: the bare command (no verb) launches the
    # interactive launcher TUI (see main), which configures `generate`.
    sub = parser.add_subparsers(dest="command", required=False)

    _add_doctor_parser(sub)
    _add_models_parser(sub)
    _add_configure_parser(sub)
    _add_run_parser(sub)
    _add_generate_parser(sub)

    _add_status_parser(sub)
    _add_pause_parser(sub)
    _add_resume_parser(sub)
    _add_stop_parser(sub)
    _add_validate_parser(sub)
    _add_finalize_parser(sub)
    _add_sfx_parser(sub)
    _add_benchmark_parser(sub)
    _add_soak_parser(sub)
    _add_inspect_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        # Bare `voyage`: interactive launcher TUI (TTY + Textual required;
        # pipes and missing extras get guidance, exit 2). CLI verbs below
        # stay fully usable non-interactively.
        try:
            return int(launch_tui())
        except VoyageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        return int(args.func(args))
    except VoyageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

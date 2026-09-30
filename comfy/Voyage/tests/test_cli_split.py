"""CLI split surface tests (issues 080 + 085).

Behavior-preservation contract for the `voyage/cli.py` verb-group split:
every moved callable stays importable from `voyage.cli` (backward compat)
and is the identical object as the new home module (single source, not a
copy). Parser + planning math are pinned so the split cannot drift them.
"""

from __future__ import annotations

import argparse
from typing import cast

import pytest

import voyage.cli as cli
import voyage.cli_core as cli_core
import voyage.cli_finalize as cli_finalize
import voyage.cli_generate as cli_generate
import voyage.cli_models as cli_models
import voyage.cli_observe as cli_observe
import voyage.cli_paths as cli_paths
import voyage.cli_planning as cli_planning
import voyage.cli_run_ops as cli_run_ops
import voyage.cli_status as cli_status
import voyage.cli_validate as cli_validate
from voyage.config import BACKEND_REGISTRY, VideoBackendName

EXPECTED_VERBS = (
    "init",
    "doctor",
    "models",
    "run",
    "generate",
    "status",
    "pause",
    "resume",
    "stop",
    "validate",
    "finalize",
    "sfx",
    "benchmark",
    "soak",
    "inspect",
)


def test_paths_helpers_are_single_sourced() -> None:
    """Re-exports are the same objects, not copies (issue 080)."""
    assert cli._run_dir_arg is cli_paths._run_dir_arg
    assert cli.resolve_run_dir is cli_paths.resolve_run_dir
    assert cli.is_flat_folder_name is cli_paths.is_flat_folder_name
    assert cli._check_run_id is cli_paths._check_run_id
    assert cli._effective_run_id is cli_paths._effective_run_id
    assert cli._RESERVED_FOLDER_NAMES is cli_paths._RESERVED_FOLDER_NAMES


def test_planning_helpers_are_single_sourced() -> None:
    """Planning + preflight re-exports are the same objects (080 + 085)."""
    assert cli.parse_duration is cli_planning.parse_duration
    assert cli._frames_per_segment is cli_planning._frames_per_segment
    assert cli.segments_for_duration is cli_planning.segments_for_duration
    assert cli._require_cuda_stack is cli_planning._require_cuda_stack
    assert cli._warn_if_no_cuda is cli_planning._warn_if_no_cuda
    assert cli._torch_available is cli_planning._torch_available
    assert cli._cuda_offenders is cli_planning._cuda_offenders
    assert cli._CUDA_BACKENDS is cli_planning._CUDA_BACKENDS


def test_verb_commands_are_single_sourced() -> None:
    """Every cmd_* re-export is the verb-module object (issue 080)."""
    assert cli.cmd_init is cli_run_ops.cmd_init
    assert cli.cmd_run is cli_run_ops.cmd_run
    assert cli.cmd_generate is cli_generate.cmd_generate
    assert cli.cmd_status is cli_status.cmd_status
    assert cli.cmd_pause is cli_status.cmd_pause
    assert cli.cmd_resume is cli_status.cmd_resume
    assert cli.cmd_stop is cli_status.cmd_stop
    assert cli.validate_run is cli_validate.validate_run
    assert cli.cmd_validate is cli_validate.cmd_validate
    assert cli.cmd_finalize is cli_finalize.cmd_finalize
    assert cli.cmd_sfx is cli_finalize.cmd_sfx
    assert cli.cmd_models is cli_models.cmd_models
    assert cli.cmd_doctor is cli_models.cmd_doctor
    assert cli.cmd_benchmark is cli_observe.cmd_benchmark
    assert cli.cmd_soak is cli_observe.cmd_soak
    assert cli.cmd_inspect is cli_observe.cmd_inspect
    assert cli._load_run is cli_core._load_run
    assert cli.get_console is cli_core.get_console


def _verb_choices(parser: argparse.ArgumentParser) -> list[str]:
    """Subcommand names behind the build_parser seam (typed, no privates)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    raise AssertionError("build_parser exposes no subparsers")


def test_build_parser_lists_all_verbs() -> None:
    """The build_parser seam still exposes every verb (issue 080)."""
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert _verb_choices(parser) == sorted(EXPECTED_VERBS)


def test_build_parser_is_repeatable() -> None:
    """Two builds expose the same verb set (no registration-order drift)."""
    first = _verb_choices(cli.build_parser())
    second = _verb_choices(cli.build_parser())
    assert first == second == sorted(EXPECTED_VERBS)


@pytest.mark.parametrize(
    "backend",
    ["fake", "longlive2", "ltxv", "causvid"],
)
def test_frames_per_segment_matches_registry_at_single_block(backend: str) -> None:
    """Steady-state planning equals the registry row at blocks=1 (085)."""
    from voyage.config import ProjectConfig, with_video_backend

    name = cast(VideoBackendName, backend)
    # Resolved config (preset applied — the same path cmd_generate plans
    # with): an unresolved VideoConfig carries the ltxv default
    # segment_frames regardless of backend, which is not plannable.
    config = with_video_backend(ProjectConfig(style="planning"), name)
    assert cli_planning._frames_per_segment(config) == BACKEND_REGISTRY[name].segment_frames


def test_segments_for_duration_rounds_up() -> None:
    """Duration math still rounds up with a minimum of one segment."""
    assert cli_planning.segments_for_duration(5.0, 24, 96) == 2
    assert cli_planning.segments_for_duration(4.0, 24, 96) == 1
    assert cli_planning.segments_for_duration(0.1, 24, 96) == 1


def test_run_id_helpers_reject_reserved_names() -> None:
    """Reserved basenames stay rejected after the paths extraction."""
    assert cli_paths.is_flat_folder_name("voyage") is True
    assert cli_paths.is_flat_folder_name("con") is False
    assert cli_paths.is_flat_folder_name("COM1") is False
    assert cli_paths.is_flat_folder_name("a/b") is False
    assert cli_paths.is_flat_folder_name("..") is False


def test_seam_dispatch_names_are_single_sourced() -> None:
    """Patched leaves resolve through the seam (issue 080 dispatch rule)."""
    import voyage.doctor as doctor
    import voyage.media as media
    import voyage.model_registry as model_registry

    assert cli.check_free_space is media.check_free_space
    assert cli.check_ffmpeg is doctor.check_ffmpeg
    assert cli.download_film_models is model_registry.download_film_models
    assert cli.download_realesrgan_models is model_registry.download_realesrgan_models
    assert cli.download_ltxv_models is model_registry.download_ltxv_models
    assert cli.verify_film_models is model_registry.verify_film_models
    assert cli.verify_longlive2_bf16 is model_registry.verify_longlive2_bf16


def test_cli_all_covers_surface() -> None:
    """The seam __all__ is the explicit export contract (mypy reads it)."""
    surface = {name for name in dir(cli) if not name.startswith("__")}
    assert set(cli.__all__) <= surface
    for name in (
        "cmd_generate",
        "cmd_init",
        "cmd_run",
        "build_parser",
        "main",
        "parse_duration",
        "validate_run",
        "_frames_per_segment",
        "_CUDA_BACKENDS",
    ):
        assert name in cli.__all__


def test_resolve_run_dir_returns_absolute() -> None:
    """Run-dir resolution still absolutizes (worker CWD invariant)."""
    resolved = cli_paths.resolve_run_dir("output/voyage")
    assert resolved.is_absolute()

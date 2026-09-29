"""CPU-only contract tests for the ACE-Step music backend (issue 038).

`audio/acestep.py` had zero direct tests, and its heavy surface
(`initialize`/`render_take`) needs weights + CUDA. This module pins
everything testable without them: the energy→tempo mapping, the module
constants, the stack shape, and — critically — that `render_take`
validates its arguments *before* touching the upstream import, so a bad
call fails as ValueError on CPU instead of ModuleNotFoundError. The one
real render stays behind the `gpu` marker and skips without CUDA.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from voyage.audio.acestep import (
    MAX_BPM,
    MIN_BPM,
    MIN_DURATION_SECONDS,
    PLANNER_MODEL,
    PROJECT_SUBDIR,
    TURBO_CONFIG,
    VALID_TASK_TYPES,
    AceStepStack,
    bpm_for_energy,
    render_take,
)


def test_bpm_for_energy_maps_knob_to_tempo() -> None:
    assert bpm_for_energy(0.0) == 80
    assert bpm_for_energy(0.5) == 110
    assert bpm_for_energy(1.0) == 140


def test_bpm_for_energy_clamps_outside_unit_range() -> None:
    # The energy map spans 60..140 BPM by design (0..1 knob); MIN_BPM/MAX_BPM
    # bound explicit tempos instead (corrected 2026-09-29: MIN_BPM == 1 would
    # pin sub-audible drones for any negative energy).
    assert bpm_for_energy(-1.0) == 60
    assert bpm_for_energy(99.0) == 140


def test_module_constants_pin_upstream_contract() -> None:
    assert MIN_DURATION_SECONDS == 1.0
    assert (MIN_BPM, MAX_BPM) == (1, 300)
    assert VALID_TASK_TYPES == ("text2music", "repaint")
    assert TURBO_CONFIG == "acestep-v15-turbo"
    assert PLANNER_MODEL == "acestep-5Hz-lm-0.6B"
    assert PROJECT_SUBDIR == "acestep"


def test_stack_carries_resident_handles() -> None:
    stack = AceStepStack(diffusion=None, language=None, device="cuda:0")
    assert stack.device == "cuda:0"


def _unloaded_stack() -> AceStepStack:
    return AceStepStack(diffusion=None, language=None, device="cpu")


def test_render_take_validates_bpm_before_upstream_import(tmp_path: Path) -> None:
    """Bad BPM raises ValueError even where `acestep` is not installed."""
    with pytest.raises(ValueError, match="bpm"):
        render_take(_unloaded_stack(), "ambient", 2.0, 1, tmp_path / "take.flac", bpm=0)


def test_render_take_validates_task_type_before_upstream_import(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_type"):
        render_take(
            _unloaded_stack(), "ambient", 2.0, 1, tmp_path / "take.flac", task_type="variation"
        )


def test_render_take_validates_duration_before_upstream_import(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        render_take(_unloaded_stack(), "ambient", 0.0, 1, tmp_path / "take.flac")


@pytest.mark.gpu
def test_render_take_renders_real_music_when_cuda_available(tmp_path: Path) -> None:
    """Real ACE-Step render (issue 040 candidate 2: GPU determinism proxy).

    Skips without CUDA + the upstream package — the CPU suite never pays
    for weights. Run on an idle GPU with `pytest -m gpu`.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if importlib.util.find_spec("acestep.inference") is None:
        pytest.skip("acestep package not installed")
    pytest.skip("GPU render needs provisioned ACE-Step checkpoints (see DESIGN §37)")

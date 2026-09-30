"""TUI untouched-form inheritance (issue 023).

The CLI spells absence as ``None`` (stored TOML wins) but the TUI form
prefills ``quantization``/``min_fps``/``min_resolution`` with concrete
defaults, so pressing Generate on an untouched form silently reverted a
run customized to ``bf16``/``60``/``1920x1080``. These tests pin the
absent-encoding: untouched-at-default fields emit ``Unset`` and resolve
to the stored config unchanged, while explicit non-defaults still win.
"""

from __future__ import annotations

from voyage.config import AugmentConfig, ProjectConfig, Unset, VideoConfig, resolve_config
from voyage.tui_state import GenerateFormState, to_generate_namespace


def _customized_base() -> ProjectConfig:
    """Stored config differing from every TUI form default."""
    base = ProjectConfig(style="inherit-probe")
    video = VideoConfig(**{**base.video.model_dump(), "quantization": "bf16"})
    augment = AugmentConfig(min_fps=60, min_width=1920, min_height=1080)
    return base.model_copy(update={"video": video, "augment": augment})


def test_untouched_form_emits_unset_for_quantization_and_floors() -> None:
    """Default-valued TUI fields encode absence, never explicit values."""
    namespace = to_generate_namespace(GenerateFormState(style="x", name="probe"))
    assert namespace.quantization is Unset
    assert namespace.min_fps is Unset
    assert namespace.min_resolution is Unset


def test_untouched_form_preserves_stored_quantization_and_floors() -> None:
    """An untouched form resolves to the stored config unchanged (023)."""
    namespace = to_generate_namespace(GenerateFormState(style="x", name="probe"))
    resolved = resolve_config(
        _customized_base(),
        quantization=namespace.quantization,
        min_fps=namespace.min_fps,
        min_resolution=namespace.min_resolution,
    )
    assert resolved.video.quantization == "bf16"
    assert (resolved.augment.min_fps, resolved.augment.min_width, resolved.augment.min_height) == (
        60,
        1920,
        1080,
    )


def test_explicit_non_default_tui_values_still_win() -> None:
    """Deliberate TUI choices override the stored config as before."""
    state = GenerateFormState(
        style="x",
        name="probe",
        quantization="bf16",
        min_fps="60",
        min_resolution="1920x1080",
    )
    namespace = to_generate_namespace(state)
    assert namespace.quantization == "bf16"
    assert namespace.min_fps == 60
    assert namespace.min_resolution == "1920x1080"
    resolved = resolve_config(
        ProjectConfig(style="probe"),
        quantization=namespace.quantization,
        min_fps=namespace.min_fps,
        min_resolution=namespace.min_resolution,
    )
    assert resolved.video.quantization == "bf16"
    assert (resolved.augment.min_fps, resolved.augment.min_width, resolved.augment.min_height) == (
        60,
        1920,
        1080,
    )


def test_blank_tui_floors_still_emit_unset() -> None:
    """Actively cleared floor fields keep the pre-existing blank→Unset path."""
    namespace = to_generate_namespace(
        GenerateFormState(style="x", name="probe", min_fps="", min_resolution="")
    )
    assert namespace.min_fps is Unset
    assert namespace.min_resolution is Unset

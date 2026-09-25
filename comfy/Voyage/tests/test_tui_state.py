"""Last-settings persistence + GPU warning tests (pure, CPU-only, no Textual).

Covers the ``voyage.tui_state`` prefill contract Stream B consumes:
save/load round-trip, missing/corrupt files, per-field fallback, silent
save failures, and the per-backend GPU notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.tui_state import (
    LAST_SETTINGS_PATH,
    GenerateFormState,
    field_errors,
    gpu_warning,
    load_last_settings,
    plan_counts,
    save_last_settings,
)


def _filled_state() -> GenerateFormState:
    return GenerateFormState(
        backend="fake",
        duration="1m30s",
        style='neon "glow"\nline art',
        name="test-run",
        seed="42",
        force=True,
        skip_bad=True,
        draft=True,
        director="deterministic",
        blocks="3",
        take_seconds="30",
        quantization="bf16",
        beats_per_segment="8",
        drift_every_n="2",
        verbose=True,
        no_color=True,
    )


def test_last_settings_path_default() -> None:
    assert LAST_SETTINGS_PATH == Path.home() / ".config" / "voyage" / "tui-last.toml"


def test_save_load_round_trip(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    state = _filled_state()
    save_last_settings(state, settings_file)
    assert load_last_settings(settings_file) == state


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    settings_file = tmp_path / "nested" / "dir" / "tui-last.toml"
    save_last_settings(_filled_state(), settings_file)
    assert settings_file.exists()
    assert load_last_settings(settings_file) == _filled_state()


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_last_settings(tmp_path / "absent.toml") == GenerateFormState()


def test_load_corrupt_toml_returns_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text("backend = [unclosed\n", encoding="utf-8")
    assert load_last_settings(settings_file) == GenerateFormState()


def test_load_wrong_types_fall_back_while_valid_fields_survive(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text(
        'backend = 123\nforce = "yes"\nstyle = "kept style"\nname = "kept-run"\n',
        encoding="utf-8",
    )
    loaded = load_last_settings(settings_file)
    defaults = GenerateFormState()
    assert loaded.backend == defaults.backend
    assert loaded.force is False
    assert loaded.style == "kept style"
    assert loaded.name == "kept-run"


def test_load_legacy_run_id_migrates_to_name(tmp_path: Path) -> None:
    """Pre-rework settings files stored run_id; they prefill Name now."""
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text('run_id = "old-run"\nstyle = "kept style"\n', encoding="utf-8")
    loaded = load_last_settings(settings_file)
    assert loaded.name == "old-run"
    settings_file.write_text('run_id = "old-run"\nname = "new-name"\n', encoding="utf-8")
    assert load_last_settings(settings_file).name == "new-name"


def test_load_invalid_choices_fall_back_to_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text(
        'backend = "nope"\ndirector = "nope"\nquantization = "int8"\nduration = "9s"\n',
        encoding="utf-8",
    )
    loaded = load_last_settings(settings_file)
    defaults = GenerateFormState()
    assert loaded.backend == defaults.backend
    assert loaded.director == defaults.director
    assert loaded.quantization == defaults.quantization
    assert loaded.duration == "9s"


def test_load_ignores_unknown_keys(tmp_path: Path) -> None:
    settings_file = tmp_path / "tui-last.toml"
    settings_file.write_text('future_option = "x"\nstyle = "kept style"\n', encoding="utf-8")
    loaded = load_last_settings(settings_file)
    assert loaded.style == "kept style"
    assert loaded == GenerateFormState(style="kept style")


def test_save_failure_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    save_last_settings(GenerateFormState(style="x"), blocker / "tui-last.toml")


def test_gpu_warning_empty_for_fake_and_unknown() -> None:
    assert gpu_warning("fake") == ""
    assert gpu_warning("nope") == ""


@pytest.mark.parametrize("backend", ["ltxv", "longlive2", "causvid"])
def test_gpu_warning_names_image_and_gpu_flag(backend: str) -> None:
    warning = gpu_warning(backend)
    assert warning
    assert "\n" not in warning
    assert "VOYAGE_IMAGE" in warning
    assert "voyage-video" in warning
    assert "--gpus" in warning


def test_plan_counts_causvid_uses_72_novel_per_block_at_16fps() -> None:
    state = GenerateFormState(style="x", backend="causvid", duration="5s")
    assert plan_counts(state) == (2, 144, pytest.approx(9.0))


def test_plan_counts_causvid_scales_with_blocks() -> None:
    state = GenerateFormState(style="x", backend="causvid", duration="5s", blocks="2")
    assert plan_counts(state) == (1, 144, pytest.approx(9.0))


def test_field_errors_keyed_by_field() -> None:
    errors = field_errors(GenerateFormState())
    assert set(errors) == {"style"}
    assert "style" in errors["style"]
    errors = field_errors(GenerateFormState(style="x", duration="soon", name="a/b"))
    assert "duration" in errors
    assert "name" in errors
    assert "5s" in errors["duration"]


def test_name_must_be_flat() -> None:
    assert "name" in field_errors(GenerateFormState(style="x", name=""))
    assert "name" in field_errors(GenerateFormState(style="x", name="a/b"))
    assert "name" not in field_errors(GenerateFormState(style="x", name="my-run_01"))

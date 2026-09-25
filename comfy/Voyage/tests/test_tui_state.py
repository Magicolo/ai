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
    _flat_folder_name,
    field_errors,
    gpu_warning,
    load_last_settings,
    plan_counts,
    plan_summary,
    save_last_settings,
    to_generate_namespace,
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


def test_plan_counts_match_cli_truth_all_backends_and_blocks() -> None:
    """Single-source planning (issue 024): TUI == cli._frames_per_segment math.

    Cross-test over every backend × blocks 1..3 against a planning-only
    ProjectConfig with preset fps — the same inputs cmd_generate plans
    with. Any drift on either side fails here.
    """
    from voyage.cli import _frames_per_segment, segments_for_duration
    from voyage.config import ProjectConfig, VideoConfig, _video_preset

    for backend in ("ltxv", "longlive2", "causvid", "fake"):
        preset = _video_preset(backend)
        raw_fps = preset.get("fps", 24)
        assert isinstance(raw_fps, int)
        for blocks in (1, 2, 3):
            state = GenerateFormState(style="x", backend=backend, duration="5s", blocks=str(blocks))
            config = ProjectConfig(
                style="planning",
                video=VideoConfig(backend=backend, blocks_per_segment=blocks, fps=raw_fps),
            )
            frames_per_segment = _frames_per_segment(config)
            segments = segments_for_duration(5.0, raw_fps, frames_per_segment)
            assert plan_counts(state) == (
                segments,
                segments * frames_per_segment,
                pytest.approx(segments * frames_per_segment / raw_fps),
            )


def test_plan_summary_uses_single_source_struct() -> None:
    """plan_summary formats plan_counts (no independent frame math)."""
    for backend, expected_fragment in (
        ("ltxv", "96f/segment @ 24fps"),
        ("longlive2", "29f/segment @ 24fps"),
        ("causvid", "72f/segment @ 16fps"),
        ("fake", "48f/segment @ 24fps"),
    ):
        state = GenerateFormState(style="x", backend=backend, duration="5s")
        counts = plan_counts(state)
        assert counts is not None
        segments, planned_frames, _seconds = counts
        summary = plan_summary(state)
        assert expected_fragment in summary
        assert f"{segments} segment(s)" in summary
        assert f"{planned_frames} frames" in summary


def test_gpu_warning_derives_from_shared_cuda_set() -> None:
    """gpu_warning agrees with cli._CUDA_BACKENDS (issue 024)."""
    from voyage.cli import _CUDA_BACKENDS

    for backend in ("ltxv", "longlive2", "causvid", "fake", "nope"):
        assert bool(gpu_warning(backend)) == (backend in _CUDA_BACKENDS)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NAN", "Infinity"])
def test_take_seconds_rejects_non_finite(raw: str) -> None:
    """Non-finite take lengths never reach planning or the ACE payload."""
    assert "take_seconds" in field_errors(GenerateFormState(style="x", take_seconds=raw))
    with pytest.raises(ValueError, match="take-seconds"):
        to_generate_namespace(GenerateFormState(style="x", take_seconds=raw))


@pytest.mark.parametrize("raw", ["-5", "0", "-0.5"])
def test_take_seconds_rejects_non_positive(raw: str) -> None:
    assert "take_seconds" in field_errors(GenerateFormState(style="x", take_seconds=raw))


def test_take_seconds_accepts_positive_finite() -> None:
    assert "take_seconds" not in field_errors(GenerateFormState(style="x", take_seconds="30"))
    assert to_generate_namespace(
        GenerateFormState(style="x", take_seconds="30")
    ).take_seconds == pytest.approx(30.0)


@pytest.mark.parametrize(
    "raw",
    [
        ".",
        "con",
        "CON",
        "prn",
        "aux",
        "nul",
        "NUL",
        "com1",
        "COM9",
        "lpt1",
        "LPT9",
        "con.txt",
        "a/b",
        "..",
        "",
    ],
)
def test_flat_folder_name_rejects_dot_and_reserved(raw: str) -> None:
    """Single dot targets shared output/; reserved names are never folders."""
    assert _flat_folder_name(raw) is False
    assert "name" in field_errors(GenerateFormState(style="x", name=raw))


@pytest.mark.parametrize("raw", ["my-run_01", "voyage", "run 2", "a.b", "comet", "null"])
def test_flat_folder_name_accepts_ordinary_names(raw: str) -> None:
    assert _flat_folder_name(raw) is True
    assert "name" not in field_errors(GenerateFormState(style="x", name=raw))


@pytest.mark.parametrize("raw", ["a\x01b", "hello\x00world", "x\x0by\x0cz", "tab\there", "q\x1f"])
def test_last_settings_round_trip_control_characters(raw: str, tmp_path: Path) -> None:
    """Every C0 control survives save/load; the file holds no raw control."""
    settings_file = tmp_path / "tui-last.toml"
    state = GenerateFormState(style=raw, backend="fake", name="myrun")
    save_last_settings(state, settings_file)
    file_bytes = settings_file.read_bytes()
    assert all(byte >= 0x20 or byte == 0x0A for byte in file_bytes)
    assert load_last_settings(settings_file) == state

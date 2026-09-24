"""`voyage generate` one-shot tests: duration parsing, segment math, backend
presets, and the full init -> run -> validate -> finalize pipeline.

Fake backends (real media, no GPU) in-container; GPU presets are asserted
at the TOML level only, never executed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.cli import main, parse_duration, segments_for_duration, validate_run
from voyage.config import with_video_backend


def test_no_cuda_warning_for_cpu_device(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from voyage.cli import _warn_if_no_cuda
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    _warn_if_no_cuda(config)
    assert capsys.readouterr().err == ""


def test_cuda_warning_when_no_gpu_visible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.doctor
    from voyage.cli import _warn_if_no_cuda
    from voyage.config import default_config_toml, load_config

    monkeypatch.setattr(voyage.doctor, "probe", lambda: {"nvidia_smi": None, "gpus": []})
    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    _warn_if_no_cuda(with_video_backend(config, "ltxv"))
    assert "no GPU is visible" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5s", 5.0),
        ("90", 90.0),
        ("0.5s", 0.5),
        ("2m", 120.0),
        ("1m30s", 90.0),
        ("1h", 3600.0),
        ("1h2m3.5s", 3723.5),
    ],
)
def test_parse_duration_human_readable(raw: str, expected: float) -> None:
    assert parse_duration(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "abc", "5x", "-5s", "0s", "0m", "1s2m", "m", "s", "1h2h"])
def test_parse_duration_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(raw)


@pytest.mark.parametrize(
    ("duration", "fps", "frames_per_segment", "expected"),
    [
        (5.0, 24, 25, 5),  # 120 frames, ltxv native blocks -> rounds up
        (2.0, 24, 48, 1),  # exact fit, no extra segment
        (4.0, 24, 48, 2),
        (0.1, 24, 48, 1),  # minimum one segment
        (5.0, 24, 49, 3),  # ltxv two-block segments round up
    ],
)
def test_segments_for_duration_rounds_up(
    duration: float, fps: int, frames_per_segment: int, expected: int
) -> None:
    assert segments_for_duration(duration, fps, frames_per_segment) == expected


def test_ltxv_preset_mirrors_verified_e2e_toml(tmp_path: Path) -> None:
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    ltxv = with_video_backend(config, "ltxv")
    assert ltxv.video.backend == "ltxv"
    assert ltxv.video.profile == "ltxv-512p"
    assert (ltxv.video.width, ltxv.video.height) == (768, 512)
    assert ltxv.video.device == "cuda:0"
    # Source config untouched (pure function).
    assert config.video.backend == "fake"


def test_longlive2_preset_keeps_default_geometry(tmp_path: Path) -> None:
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    longlive = with_video_backend(config, "longlive2")
    assert longlive.video.backend == "longlive2"
    assert longlive.video.device == "cuda:0"
    assert (longlive.video.width, longlive.video.height) == (
        config.video.width,
        config.video.height,
    )


def test_unknown_backend_rejected(tmp_path: Path) -> None:
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    with pytest.raises(ValueError, match="unknown video backend"):
        with_video_backend(config, "framepack")


def test_cuda_presets_select_acestep_audio(tmp_path: Path) -> None:
    """CUDA video presets must pair with real ACE-Step music, not fake sine."""
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    for backend in ("ltxv", "longlive2"):
        applied = with_video_backend(config, backend)
        assert applied.audio.backend == "acestep"
        assert applied.audio.device == "cuda:0"
        assert applied.audio.models_dir == "/models"
    # Source config untouched (pure function).
    assert config.audio.backend == "fake"


def test_fake_preset_keeps_fake_audio(tmp_path: Path) -> None:
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    assert with_video_backend(config, "fake").audio.backend == "fake"


def test_init_toml_carries_audio_preset(tmp_path: Path) -> None:
    """`init --backend ltxv` writes the audio preset into the toml directly."""
    from voyage.config import default_config_toml, load_config

    (tmp_path / "voyage.toml").write_text(
        default_config_toml("preset", "pastel neon line-art, peaceful", 11, video_backend="ltxv"),
        encoding="utf-8",
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    assert config.audio.backend == "acestep"
    assert config.audio.device == "cuda:0"


def _generate_args(output: Path, *extra: str) -> list[str]:
    return [
        "generate",
        "--backend",
        "fake",
        "--duration",
        "4s",
        "--style",
        "pastel neon line-art, peaceful",
        "--output",
        str(output),
        "--run-id",
        "gen",
        "--seed",
        "11",
        *extra,
    ]


def test_generate_fake_end_to_end_validated_finalized(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(_generate_args(run_dir)) == 0
    final = run_dir / "final.mp4"
    assert final.exists() and final.stat().st_size > 0
    assert validate_run(run_dir) == []
    from voyage.persistence import read_state

    state = read_state(run_dir)
    assert state.committed_segments == 2  # 4s @24fps, 48f segments
    assert state.timeline_frames >= 4 * 24  # rounded-up, never short


def test_generate_refuses_nonempty_dir_without_force(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("other agent's data", encoding="utf-8")
    assert main(_generate_args(run_dir)) == 2
    assert not (run_dir / "final.mp4").exists()


def test_generate_defaults_to_output_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "generate",
                "--backend",
                "fake",
                "--duration",
                "2s",
                "--style",
                "pastel neon line-art, peaceful",
                "--run-id",
                "gen-default",
                "--seed",
                "11",
            ]
        )
        == 0
    )
    assert (tmp_path / "output" / "gen-default" / "final.mp4").exists()


def test_cuda_guard_passes_for_fake_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    from voyage.cli import _require_cuda_stack
    from voyage.config import default_config_toml, load_config

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    (tmp_path / "voyage.toml").write_text(
        default_config_toml("guard", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    assert _require_cuda_stack(config) is True


def test_cuda_guard_fails_for_ltxv_without_torch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    from voyage.cli import _require_cuda_stack
    from voyage.config import default_config_toml, load_config

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    (tmp_path / "voyage.toml").write_text(
        default_config_toml("guard", "pastel neon line-art, peaceful", 11), encoding="utf-8"
    )
    config, _ = load_config(tmp_path / "voyage.toml")
    assert _require_cuda_stack(with_video_backend(config, "ltxv")) is False
    assert "voyage-video" in capsys.readouterr().err


def test_generate_aborts_before_init_without_cuda_stack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "generate",
            "--backend",
            "ltxv",
            "--duration",
            "5s",
            "--style",
            "pastel neon line-art, peaceful",
        ]
    )
    assert code == 1
    assert not (tmp_path / "output" / "voyage").exists()
    assert "voyage-video" in capsys.readouterr().err

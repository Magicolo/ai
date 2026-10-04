"""`voyage generate` helper tests: duration parsing, segment math, backend
presets, CUDA guards, and the stop-key listener.

Creation-flow coverage lives in `test_configure.py` (manifest init) and
reconcile coverage in `test_generate_reconcile.py`; this file keeps only
tests that never invoke the old create-flow `cmd_generate`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.cli import (
    _frames_per_segment,
    _run_dir_arg,
    parse_duration,
    segments_for_duration,
)
from voyage.config import VideoConfig, with_video_backend


def test_no_cuda_warning_for_cpu_device(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from voyage.cli import _warn_if_no_cuda
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11, video_backend="fake")
    _warn_if_no_cuda(config)
    assert capsys.readouterr().err == ""


def test_cuda_warning_when_no_gpu_visible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.doctor
    from voyage.cli import _warn_if_no_cuda
    from voyage.config import preset_config

    monkeypatch.setattr(voyage.doctor, "probe", lambda: {"nvidia_smi": None, "gpus": []})
    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
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
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    ltxv = with_video_backend(config, "ltxv")
    assert ltxv.video.backend == "ltxv"
    assert ltxv.video.profile == "ltxv-512p"
    assert (ltxv.video.width, ltxv.video.height) == (768, 512)
    assert ltxv.video.device == "cuda:0"
    # Source config untouched (pure function) — the default preset is ltx25.
    assert config.video.backend == "ltx25"


def test_unknown_video_preset_rejected(tmp_path: Path) -> None:
    """Unknown video backends fail fast with the known set, never remap."""
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    with pytest.raises(ValueError, match="unknown video backend"):
        with_video_backend(config, "nope")  # type: ignore[arg-type]


def test_run_dir_arg_resolves_absolute(tmp_path: Path) -> None:
    """Worker CWD is run_dir: relative dirs double up downstream (qual-leg)."""
    assert _run_dir_arg(str(tmp_path / "some-run")) == (tmp_path / "some-run").resolve()
    assert _run_dir_arg("output/some-run") == (Path.cwd() / "output/some-run").resolve()


def test_frames_per_segment_ltxv_uses_novel_minimum(tmp_path: Path) -> None:
    """ltxv duration math must use the 96-novel steady state, not 121 fresh."""
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    one_block = with_video_backend(config, "ltxv")
    assert _frames_per_segment(one_block) == 96
    two_blocks = one_block.model_copy(
        update={"video": VideoConfig(**{**one_block.video.model_dump(), "blocks_per_segment": 2})}
    )
    assert _frames_per_segment(two_blocks) == 192


def test_causvid_preset_pins_native_geometry(tmp_path: Path) -> None:
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    causvid = with_video_backend(config, "causvid")
    assert causvid.video.backend == "causvid"
    assert causvid.video.profile == "causvid-480p"
    # Native worker geometry (832x480 @ 16 fps — the worker rejects
    # anything else).
    assert (causvid.video.width, causvid.video.height) == (832, 480)
    assert causvid.video.device == "cuda:0"
    # Native 16 fps end-to-end (the worker refuses relabeled timelines).
    assert causvid.video.fps == 16
    assert causvid.video.latent_shape == [1, 21, 16, 60, 104]
    # CUDA video backends pair with ACE-Step music.
    assert causvid.audio.backend == "acestep"
    # Source config untouched (pure function) — the default preset is ltx25.
    assert config.video.backend == "ltx25"


def test_frames_per_segment_causvid_uses_novel_minimum(tmp_path: Path) -> None:
    """causvid duration math must use the 72-novel steady state, not 81 rollout."""
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    one_block = with_video_backend(config, "causvid")
    assert _frames_per_segment(one_block) == 72
    two_blocks = one_block.model_copy(
        update={"video": VideoConfig(**{**one_block.video.model_dump(), "blocks_per_segment": 2})}
    )
    assert _frames_per_segment(two_blocks) == 144


def test_unknown_backend_rejected(tmp_path: Path) -> None:
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    with pytest.raises(ValueError, match="unknown video backend"):
        with_video_backend(config, "framepack")  # type: ignore[arg-type]


def test_cuda_presets_select_acestep_audio(tmp_path: Path) -> None:
    """CUDA video presets must pair with real ACE-Step music, not fake sine."""
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    for backend in ("ltxv", "causvid", "ltx25", "ltx23"):
        applied = with_video_backend(config, backend)
        assert applied.audio.backend == "acestep"
        assert applied.audio.device == "cuda:0"
        assert applied.audio.models_dir == "/models"
    # Source config untouched (pure function) — the default preset is
    # ltx25 with ACE-Step music for continuous mood.
    assert config.audio.backend == "acestep"


def test_fake_preset_keeps_fake_audio(tmp_path: Path) -> None:
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11)
    assert with_video_backend(config, "fake").audio.backend == "fake"


def test_preset_carries_audio_preset(tmp_path: Path) -> None:
    """`preset_config(..., video_backend="ltxv")` pairs the ACE audio preset directly."""
    from voyage.config import preset_config

    config = preset_config("preset", "pastel neon line-art, peaceful", 11, video_backend="ltxv")
    assert config.audio.backend == "acestep"
    assert config.audio.device == "cuda:0"

"""Finalize GPU defaults: MMAudio SFX on cuda:0, model pass on cuda:1, sequential.

TDD contract for the user-approved defaults change (DESIGN §140): the
streaming backends defer ACE-Step music to finalize (timeline-exact
silent stubs at commit, takes rendered after the video worker stops —
no joint-audio GPU contention on the 4060), then dub MMAudio SFX over
the rendered music at finalize on the 4060 (cuda:0), while the
Real-ESRGAN + FILM model pass runs on the 2060 (cuda:1) — overlapping
the deferred music takes on 2-GPU boxes (`voyage/media.py`
fork-join; 1-GPU finalizes stay sequential).
"""

from __future__ import annotations

from typing import cast

import pytest

from voyage.config import ProjectConfig, VideoBackendName, with_video_backend


def _config_with_style() -> ProjectConfig:
    return ProjectConfig(style="pastel neon line-art, peaceful")


def test_ltx_backends_pair_acestep_music_and_sfx_mmaudio_on_cuda0() -> None:
    """ltx25/ltx23: ACE-Step music (deferred to finalize, planner long
    takes) and the finalize SFX dub runs MMAudio on cuda:0 by default."""
    for backend in ("ltx25", "ltx23"):
        config = with_video_backend(_config_with_style(), cast(VideoBackendName, backend))
        assert config.audio.backend == "acestep"
        assert config.audio.device == "cuda:0"
        assert config.sfx.backend == "mmaudio"
        assert config.sfx.device == "cuda:0"


@pytest.mark.parametrize("backend", ["ltxv", "causvid", "ltx25", "ltx23"])
def test_all_cuda_backends_use_sequential_devices(backend: str) -> None:
    """Every CUDA backend pairs SFX dub on cuda:0 with the model pass on cuda:1.

    The model pass owns cuda:1 (the 2060) while the SFX dub and the
    deferred ACE music own cuda:0 (the 4060); the model pass and the
    music takes overlap on 2-GPU boxes, everything else stays ordered.
    `fake` never routes (SFX off).
    """
    from voyage.augment import model_pass_devices

    config = with_video_backend(_config_with_style(), cast(VideoBackendName, backend))
    assert config.sfx.backend == "mmaudio"
    assert config.sfx.device == "cuda:0"
    assert config.audio.backend == "acestep"
    assert config.audio.device == "cuda:0"
    assert model_pass_devices(devices=("cuda:0", "cuda:1")) == ("cuda:1",)


def test_fake_backend_never_routes_sequential_cpu() -> None:
    """Fake stays CPU-only: no SFX dub, no model-pass device routing."""
    config = with_video_backend(_config_with_style(), "fake")
    assert config.sfx.backend == "fake"
    assert config.audio.backend == "fake"


def test_ltx_backends_require_sfx_stack_with_acestep() -> None:
    """`required_specs` for an ltx backend with SFX enabled pulls both the
    SFX stack and ACE-Step (continuous planner music, not the worker's
    joint track)."""
    from voyage.models_ensure import required_specs

    for backend, spec in (("ltx25", "ltx25"), ("ltx23", "ltx23")):
        config = _config_with_style()
        config.video.backend = cast(VideoBackendName, backend)
        config.audio.backend = "acestep"
        config.sfx.backend = "mmaudio"
        specs = {item.spec for item in required_specs(config, sfx_enabled=True)}
        assert "sfx-mmaudio" in specs
        assert "audio-acestep" in specs
        assert spec in specs


def test_explicit_quality_defaults_ship_source() -> None:
    """Explicit quality ships the source by default: config dataclass and stored agree."""
    from voyage.config import AugmentConfig, preset_config

    assert AugmentConfig().upscale == 1
    assert AugmentConfig().interpolate == 1
    assert AugmentConfig().presentation_fps is None
    assert preset_config("story", "style", 7).augment.upscale == 1
    assert preset_config("story", "style", 7).augment.interpolate == 1


def test_model_pass_devices_pins_secondary_gpu() -> None:
    """Two visible GPUs pin the whole model pass to cuda:1 (the 2060);
    fewer GPUs keep the legacy single-device selection."""
    from voyage import augment as augment_module
    from voyage.augment import model_pass_devices

    assert model_pass_devices(devices=("cuda:0", "cuda:1")) == ("cuda:1",)
    assert model_pass_devices(devices=("cuda:0",)) == ("cuda:0",)
    assert model_pass_devices(devices=()) == ()
    # Default probes live visibility (patched here to two GPUs).
    monkeypatched = [("cuda:0", "cuda:1")]
    original = augment_module.augment_devices
    try:
        augment_module.augment_devices = lambda **_: monkeypatched[0]  # type: ignore[method-assign]
        assert model_pass_devices() == ("cuda:1",)
    finally:
        augment_module.augment_devices = original  # type: ignore[method-assign]

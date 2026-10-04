"""SFX worker contract: fake backend + validators + config (slice 2a, TDD).

The SFX worker renders video-synced effects windows at finalize time
behind a `generate_sfx` op. The fake backend draws deterministic
seeded noise through ffmpeg so the windowing/sharding/mix path is
genuine with no GPU, no weights, no network. Real MMAudio behavior is
pinned by the ladder (slice 2c); this module pins the contract both
backends share: op shape, validation-before-side-effects, benchmark
count guard, uniform evict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.audio.mmaudio_sfx import (
    SFX_MODEL_SIZES,
    validate_duration_seconds,
    validate_model_size,
)
from voyage.workers import sfx as fake_sfx_worker


def test_sfx_model_sizes_cover_ladder() -> None:
    assert SFX_MODEL_SIZES == ("small_44k", "medium_44k", "large_44k_v2")
    for size in SFX_MODEL_SIZES:
        validate_model_size(size)
    with pytest.raises(ValueError, match="model_size"):
        validate_model_size("tiny_8k")


def test_sfx_duration_bounds() -> None:
    for good in (0.5, 8.0, 30.0):
        validate_duration_seconds(good)
    for bad in (0.0, -1.0, float("nan"), float("inf"), 61.0):
        with pytest.raises(ValueError, match="duration_seconds"):
            validate_duration_seconds(bad)


def test_fake_generate_sfx_validates_before_side_effects(tmp_path: Path) -> None:
    out = tmp_path / "nope" / "sfx.wav"
    with pytest.raises(ValueError, match="duration_seconds"):
        fake_sfx_worker.handle_generate_sfx(
            {
                "window_id": "w0000",
                "caption": "rain",
                "video_path": "/nonexistent.mp4",
                "start_seconds": 0.0,
                "duration_seconds": 0.0,
                "seed": 1,
                "output_path": str(out),
                "sample_rate": 48000,
                "channels": 2,
            }
        )
    assert not out.exists()


def test_fake_generate_sfx_renders_valid_wav(tmp_path: Path) -> None:
    out = tmp_path / "sfx.wav"
    result = fake_sfx_worker.handle_generate_sfx(
        {
            "window_id": "w0000",
            "caption": "glass chimes in gusts",
            "video_path": str(tmp_path / "video.mp4"),
            "start_seconds": 0.0,
            "duration_seconds": 2.0,
            "seed": 7,
            "output_path": str(out),
            "sample_rate": 48000,
            "channels": 2,
        }
    )
    assert out.exists() and out.stat().st_size > 1000
    assert result["artifacts"] == [str(out)]
    assert result["sfx"]["backend"] == "fake"
    # Deterministic in seed: same seed renders byte-identical bytes.
    out2 = tmp_path / "sfx2.wav"
    fake_sfx_worker.handle_generate_sfx(
        {
            "window_id": "w0000",
            "caption": "glass chimes in gusts",
            "video_path": str(tmp_path / "video.mp4"),
            "start_seconds": 0.0,
            "duration_seconds": 2.0,
            "seed": 7,
            "output_path": str(out2),
            "sample_rate": 48000,
            "channels": 2,
        }
    )
    assert out.read_bytes() == out2.read_bytes()


def test_fake_sfx_benchmark_counts_guarded() -> None:
    with pytest.raises(ValueError, match="measured"):
        fake_sfx_worker.handle_benchmark({"warmup": 0, "measured": 0})


def test_fake_sfx_evict_is_uniform() -> None:
    assert fake_sfx_worker.handle_evict_gpu({}) == {"evicted": True}


def test_sfx_pairs_by_backend_cuda_on_fake_off(tmp_path: Path) -> None:
    from voyage.config import preset_config, with_video_backend

    # Default preset is ltx25 (CUDA): the SFX dub pairs on GPU by
    # default and music comes from ACE-Step long takes (audio acestep).
    config = preset_config("demo", "pastel neon line-art, peaceful", 1)
    assert config.sfx.backend == "mmaudio"
    assert config.sfx.device == "cuda:0"
    assert config.sfx.model_size == "large_44k_v2"
    assert config.audio.backend == "acestep"
    # Fake stays CPU-only (gates never touch weights); SfxConfig defaults
    # keep byte-identical behavior for old runs.
    fake_config = preset_config("demo", "pastel neon line-art, peaceful", 1, video_backend="fake")
    assert fake_config.sfx.backend == "fake"
    from voyage.config import SfxConfig

    assert SfxConfig().backend == "fake"
    assert SfxConfig().device == "cpu"
    switched = with_video_backend(config, "fake")
    assert switched.sfx.backend == "fake"
    assert switched.sfx.device == "cpu"
    back = with_video_backend(switched, "ltxv")
    assert back.sfx.backend == "mmaudio"
    assert back.sfx.device == "cuda:0"


def test_caption_overrides_pin_families() -> None:
    from voyage.config import preset_config, resolve_config
    from voyage.supervisor import effective_music_caption, effective_video_stages

    config = preset_config("demo", "pastel neon line-art, peaceful", 1)
    # Director drives by default (None pins nothing).
    assert config.audio.music_caption is None
    assert config.video.video_caption is None
    assert effective_music_caption(None, "director music", "style") == "director music"
    assert effective_video_stages(None, ["a", "b"]) == ["a", "b"]
    # Explicit flags pin the family (no drift), style still fallback.
    pinned = resolve_config(config, music_caption="brass fanfare", video_caption="red dune")
    assert pinned.audio.music_caption == "brass fanfare"
    assert pinned.video.video_caption == "red dune"
    assert (
        effective_music_caption(pinned.audio.music_caption, "director music", "style")
        == "brass fanfare"
    )
    assert effective_video_stages(pinned.video.video_caption, ["a", "b"]) == ["red dune"]
    assert effective_music_caption("", "", "style") == "style"


def test_sfx_config_rejects_unknown_backend_and_size() -> None:
    from pydantic import ValidationError

    from voyage.config import SfxConfig

    with pytest.raises(ValidationError):
        SfxConfig(backend="sdxl")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SfxConfig(model_size="tiny_8k")  # type: ignore[arg-type]


def test_registry_carries_sfx_mmaudio_spec() -> None:
    from voyage.model_registry import MODEL_SPECS

    spec = MODEL_SPECS["sfx-mmaudio"]
    assert spec.manifest_key == "sfx"
    repos = [file.repo_id for file in spec.files]
    repos += [snap.repo_id for snap in spec.snapshots]
    assert "hkchengrex/MMAudio" in repos

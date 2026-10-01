"""Backend registry: one Literal vocabulary + one table (issue 022).

CPU-only: config resolution only — no workers, no ffmpeg, no GPU. Pins
that the backend fields, the state-mode map, and the streaming set all
derive from config.BACKEND_REGISTRY, plus the per-row geometry and audio
pairing the preset dicts used to restate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest

from voyage.backends import BACKEND_STATE_MODES, VideoBackendAdapter, VideoSegmentRequest
from voyage.config import (
    BACKEND_REGISTRY,
    AudioBackendName,
    AudioConfig,
    ProjectConfig,
    VideoBackendName,
    VideoConfig,
    default_config_toml,
    load_config,
    resolve_config,
    with_video_backend,
)
from voyage.errors import ConfigurationError


def _transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"video": {"frames": 48, "fps": 24}}


def _base_config() -> ProjectConfig:
    return ProjectConfig(style="registry-probe")


def _request(state_mode: str) -> VideoSegmentRequest:
    return VideoSegmentRequest(
        segment_id="000001",
        prompt="pastel neon line-art, peaceful",
        seed=7,
        width=768,
        height=432,
        fps=24,
        segment_seconds=2.0,
        state_mode=state_mode,  # type: ignore[arg-type]
    )


def test_backend_name_vocabularies() -> None:
    assert get_args(VideoBackendName) == ("fake", "ltxv", "causvid", "ltx25", "ltx23")
    assert get_args(AudioBackendName) == ("fake", "acestep")


def test_registry_covers_exactly_the_video_vocabulary() -> None:
    assert set(BACKEND_REGISTRY) == set(get_args(VideoBackendName))


def test_video_config_defaults_equal_ltxv_row() -> None:
    row = BACKEND_REGISTRY["ltxv"]
    defaults = VideoConfig()
    assert defaults.backend == "ltxv"
    assert defaults.profile == row.profile
    assert (defaults.width, defaults.height) == (row.width, row.height)
    assert defaults.fps == row.fps
    assert defaults.device == row.device
    assert defaults.latent_shape == list(row.latent_shape)


def test_audio_config_default_backend_is_fake() -> None:
    assert AudioConfig().backend == "fake"


def test_state_modes_derive_from_registry() -> None:
    assert BACKEND_STATE_MODES == {
        name: record.state_mode for name, record in BACKEND_REGISTRY.items()
    }
    assert BACKEND_STATE_MODES["fake"] == "independent_clip"
    assert BACKEND_STATE_MODES["ltxv"] == "reconstructable_prefix"
    assert BACKEND_STATE_MODES["causvid"] == "reconstructable_prefix"


def test_streaming_set_derives_from_registry() -> None:
    for name in get_args(VideoBackendName):
        adapter = VideoBackendAdapter(_transport, name, VideoConfig(backend=name))
        assert adapter.streaming == BACKEND_REGISTRY[name].streaming
    streaming = {name for name in BACKEND_REGISTRY if BACKEND_REGISTRY[name].streaming}
    assert streaming == {"ltxv", "causvid", "ltx25", "ltx23"}


def test_capabilities_carry_typed_backend_per_row() -> None:
    for name in get_args(VideoBackendName):
        adapter = VideoBackendAdapter(_transport, name, VideoConfig(backend=name))
        capabilities = adapter.capabilities()
        assert capabilities.backend == name
        assert capabilities.state_mode == BACKEND_REGISTRY[name].state_mode
        assert capabilities.streaming == BACKEND_REGISTRY[name].streaming


def test_registry_dispatch_applies_geometry_and_audio_pairing() -> None:
    for name in get_args(VideoBackendName):
        row = BACKEND_REGISTRY[name]
        resolved = with_video_backend(_base_config(), name)
        assert resolved.video.backend == name
        assert resolved.video.profile == row.profile
        assert (resolved.video.width, resolved.video.height) == (row.width, row.height)
        assert resolved.video.fps == row.fps
        assert resolved.video.device == row.device
        assert resolved.video.latent_shape == list(row.latent_shape)
        assert resolved.audio.backend == row.audio_backend
        assert resolved.audio.device == row.audio_device


def test_unknown_backend_rejected_with_known_list() -> None:
    with pytest.raises(ValueError, match="unknown video backend"):
        with_video_backend(_base_config(), "framepack")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown video backend"):
        resolve_config(_base_config(), backend="framepack")  # type: ignore[arg-type]


def test_unknown_adapter_backend_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown video backend"):
        VideoBackendAdapter(_transport, "imaginary", VideoConfig())  # type: ignore[arg-type]


def test_state_mode_mismatch_rejected_before_transport() -> None:
    calls: list[str] = []

    def recording(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(operation)
        return {}

    adapter = VideoBackendAdapter(recording, "fake", VideoConfig())
    with pytest.raises(ConfigurationError, match="state_mode"):
        adapter.build_payload(_request("persistent_kv"), Path("seg/video.mp4"))
    assert calls == []


def test_default_toml_carries_registry_geometry_per_backend(tmp_path: Path) -> None:
    for name in get_args(VideoBackendName):
        row = BACKEND_REGISTRY[name]
        toml_path = tmp_path / f"voyage-{name}.toml"
        toml_path.write_text(
            default_config_toml("registry", "pastel neon line-art, peaceful", 3, name),
            encoding="utf-8",
        )
        config, _digest = load_config(toml_path)
        assert config.video.backend == name
        assert (config.video.width, config.video.height) == (row.width, row.height)
        assert config.video.fps == row.fps
        assert config.video.latent_shape == list(row.latent_shape)
        assert config.audio.backend == row.audio_backend


def test_with_video_backend_is_pure() -> None:
    base = _base_config()
    before = base.model_dump()
    with_video_backend(base, "ltxv")
    assert base.model_dump() == before

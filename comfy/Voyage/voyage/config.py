"""Project configuration (DESIGN task group B).

TOML loader + validation + sha256 hash. A snapshot of the config is
written into each run directory at init time; the hash is recorded in
the run manifest so runs are traceable to exact configuration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python 3.10 worker image (upstream env)
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from pydantic import BaseModel, Field, field_validator

from voyage.errors import ConfigurationError


class VideoConfig(BaseModel):
    backend: str = "fake"
    profile: str = "fake-432p"
    width: int = 768
    height: int = 432
    fps: int = 24
    segment_frames: int = 48
    device: str = "cpu"
    # LongLive backend only: host path (or /models mount in the worker
    # image) holding wan_models/ + longlive2/, and the latent shape the
    # pipeline denoises. [1,8,48,44,80] decodes to 1280x704 (x16 spatial;
    # 8 latents -> 8 frames chunked, 29 causal).
    models_dir: str = "/models"
    latent_shape: list[int] = Field(default_factory=lambda: [1, 8, 48, 44, 80])

    @field_validator("width", "height", "fps", "segment_frames")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class AudioConfig(BaseModel):
    backend: str = "fake"
    sample_rate: int = 44100
    channels: int = 2
    music_style: str = "ambient electronic"
    energy: float = 0.5

    @field_validator("sample_rate", "channels")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("energy")
    @classmethod
    def unit_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("energy must be within [0, 1]")
        return value


class DirectorConfig(BaseModel):
    backend: str = "deterministic"
    model_id: str = "Qwen/Qwen3-8B"
    temperature: float = 0.7


class ProjectConfig(BaseModel):
    schema_version: int = 1
    run_id: str = "voyage"
    style: str = ""
    seed: int = 0
    min_free_space_gib: float = 20.0
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    director: DirectorConfig = Field(default_factory=DirectorConfig)

    @field_validator("style")
    @classmethod
    def style_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("style must be a non-empty human-owned style string")
        return value


def default_config_toml(run_id: str, style: str, seed: int) -> str:
    return f"""\
schema_version = 1
run_id = "{run_id}"
style = "{style}"
seed = {seed}
min_free_space_gib = 5.0

[video]
backend = "fake"
profile = "fake-432p"
width = 768
height = 432
fps = 24
segment_frames = 48
device = "cpu"
models_dir = "/models"
latent_shape = [1, 8, 48, 44, 80]

[audio]
backend = "fake"
sample_rate = 44100
channels = 2
music_style = "ambient electronic"
energy = 0.5

[director]
backend = "deterministic"
model_id = "Qwen/Qwen3-8B"
temperature = 0.7
"""


def load_config(path: Path) -> tuple[ProjectConfig, str]:
    """Load TOML config, returning (config, sha256_of_file_bytes)."""
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot parse config {path}: {exc}") from exc
    try:
        config = ProjectConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"invalid config {path}: {exc}") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, digest

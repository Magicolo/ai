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
    # Phase 2: DiT blocks per committed segment (1 block = 8 latents).
    # The stream session holds caches across blocks, so memory stays flat;
    # only wall time grows. Fake backend ignores this (renders segment_frames).
    blocks_per_segment: int = 1

    @field_validator("width", "height", "fps", "segment_frames", "blocks_per_segment")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class AudioConfig(BaseModel):
    backend: str = "fake"
    sample_rate: int = 48000
    channels: int = 2
    music_style: str = "ambient electronic"
    energy: float = 0.5
    # Slow-loop music takes (§35): each take covers `take_seconds` of video
    # time; a new take renders when coverage drops within `ahead_seconds`
    # (audio-ahead, §40); take boundaries inside a segment join with a
    # `crossfade_seconds` acrossfade (clamped to half the shortest slice).
    take_seconds: float = 45.0
    ahead_seconds: float = 20.0
    crossfade_seconds: float = 2.0
    # ACE-Step backend only: mirrors VideoConfig — host path (or /models
    # mount in the GPU image) holding acestep/checkpoints/, and the torch
    # device the resident stack loads on. Fake backend ignores both.
    models_dir: str = "/models"
    device: str = "cpu"

    @field_validator("sample_rate", "channels")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("take_seconds", "ahead_seconds", "crossfade_seconds")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be non-negative")
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
    # Qwen worker: non-thinking mode (no <think> parsing), JSON-only output.
    enable_thinking: bool = False
    max_new_tokens: int = 1024
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    # VLM inspector (Phase 5): Qwen3.5-9B lives in the director image
    # (transformers 5.x); a local path works for E2E (/models/Qwen3.5-9B).
    inspector_model_id: str = "Qwen/Qwen3.5-9B"


class VoyageConfig(BaseModel):
    """Evolution policy (DESIGN §14 [voyage] table, §§21.2-21.3)."""

    allow_concept_revisit: bool = False
    major_transition_min_seconds: float = 30.0
    major_transition_max_seconds: float = 120.0
    world_decision_interval_seconds: float = 16.0
    blocks_per_prompt_stage: int = 3
    novelty_threshold: float = 0.85
    novelty_max_attempts: int = 3

    @field_validator("blocks_per_prompt_stage", "novelty_max_attempts")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("novelty_threshold")
    @classmethod
    def unit_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("novelty_threshold must be within [0, 1]")
        return value


class ExperimentalConfig(BaseModel):
    """Explicit feature flags for experimental behavior (DESIGN §132)."""

    visual_inspector: bool = False


class DraftConfig(BaseModel):
    """Cheap iteration profile (fast loop): quarter-res spatial latents.

    Applied only when the run requests it (`voyage run --draft`); the
    stored TOML keeps full-quality values. Spatial dims are halved
    (640x352, latent [1,8,48,22,40]); the temporal dim is untouched.
    Draft renders are for iteration, never for finals.
    """

    width: int = 640
    height: int = 352
    latent_shape: list[int] = Field(default_factory=lambda: [1, 8, 48, 22, 40])
    blocks_per_segment: int = 1
    take_seconds: float = 15.0

    @field_validator("width", "height", "blocks_per_segment")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("take_seconds")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be non-negative")
        return value


class ProjectConfig(BaseModel):
    schema_version: int = 1
    run_id: str = "voyage"
    style: str = ""
    seed: int = 0
    min_free_space_gib: float = 20.0
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    director: DirectorConfig = Field(default_factory=DirectorConfig)
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)
    draft: DraftConfig = Field(default_factory=DraftConfig)

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
blocks_per_segment = 1

[audio]
backend = "fake"
sample_rate = 48000
channels = 2
music_style = "ambient electronic"
energy = 0.5
take_seconds = 45.0
ahead_seconds = 20.0
crossfade_seconds = 2.0
models_dir = "/models"
device = "cpu"

[director]
backend = "deterministic"
model_id = "Qwen/Qwen3-8B"
temperature = 0.7
enable_thinking = false
max_new_tokens = 1024
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
inspector_model_id = "Qwen/Qwen3.5-9B"

[voyage]
allow_concept_revisit = false
major_transition_min_seconds = 30.0
major_transition_max_seconds = 120.0
world_decision_interval_seconds = 16.0
blocks_per_prompt_stage = 3
novelty_threshold = 0.85
novelty_max_attempts = 3

[experimental]
visual_inspector = false

[draft]
width = 640
height = 352
latent_shape = [1, 8, 48, 22, 40]
blocks_per_segment = 1
take_seconds = 15.0
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


def apply_draft_overrides(
    config: ProjectConfig,
    *,
    draft: bool = False,
    director: str | None = None,
    blocks: int | None = None,
    take_seconds: float | None = None,
) -> ProjectConfig:
    """Apply the draft profile + targeted run overrides (fast loop).

    Pure: returns a new config, never mutates. Rebuilds submodels through
    their constructors so invalid overrides (blocks=0, negative takes)
    raise ValidationError instead of silently corrupting the run.
    """
    video = config.video
    audio = config.audio
    director_cfg = config.director
    if draft:
        profile = config.draft
        video = VideoConfig(
            **{
                **video.model_dump(),
                "width": profile.width,
                "height": profile.height,
                "latent_shape": list(profile.latent_shape),
                "blocks_per_segment": profile.blocks_per_segment,
            }
        )
        audio = AudioConfig(**{**audio.model_dump(), "take_seconds": profile.take_seconds})
    if director is not None:
        director_cfg = DirectorConfig(**{**director_cfg.model_dump(), "backend": director})
    if blocks is not None:
        video = VideoConfig(**{**video.model_dump(), "blocks_per_segment": blocks})
    if take_seconds is not None:
        audio = AudioConfig(**{**audio.model_dump(), "take_seconds": take_seconds})
    return config.model_copy(update={"video": video, "audio": audio, "director": director_cfg})

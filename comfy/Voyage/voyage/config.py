"""Project configuration (DESIGN task group B).

TOML loader + validation + sha256 hash. A snapshot of the config is
written into each run directory at init time; the hash is recorded in
the run manifest so runs are traceable to exact configuration.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ImportError:  # Python 3.10 worker image (upstream env)
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from pydantic import BaseModel, Field, field_validator

from voyage.errors import ConfigurationError

VideoBackendName = Literal["fake", "longlive2", "ltxv", "causvid"]
"""Video backend vocabulary (issue 022): every backend field, the registry,
and the streaming set are keyed by this — a typo fails at typecheck
instead of after GPU init."""

AudioBackendName = Literal["fake", "acestep"]
"""Audio backend vocabulary (issue 022): fake sine vs the ACE-Step music stack."""

StateMode = Literal["persistent_kv", "reconstructable_prefix", "independent_clip"]
"""Continuation-state vocabulary (DESIGN §5.1): how a backend resumes work.

Single home (issue 022); voyage.backends re-exports it for the adapter
contract so the two modules never spell the modes apart.
"""


@dataclass(frozen=True)
class BackendRecord:
    """One row of BACKEND_REGISTRY: a video backend's geometry, audio
    pairing, continuation mode, and streaming shape (issues 022 + 025).

    Frozen so the single source of truth cannot drift at runtime; every
    preset dict, state-mode map, and streaming set below derives from
    the registry instead of restating these values.
    """

    profile: str
    width: int
    height: int
    fps: int
    latent_shape: tuple[int, ...]
    device: str
    audio_backend: AudioBackendName
    audio_device: str
    state_mode: StateMode
    streaming: bool


BACKEND_REGISTRY: dict[VideoBackendName, BackendRecord] = {
    # Single source of truth for `generate --backend` presets (also used
    # by default_config_toml). ltxv renders native 768x512 on CUDA (both
    # /32 and /64 clean for the two-stage multiscale pipeline; 1024x576
    # was tried 2026-09-24 but needs ~15.6 GB in the forward — beyond the
    # 16 GB card even via the dynamic-fp8 fallback — so it stays reverted
    # until a memory-optimization pass lands); longlive2 renders native
    # 1280x704 (latent_shape x16 spatial — the worker ignores the request
    # geometry), so the preset pins that geometry: anything else fails the
    # commit-time resolution check (qual-longlive2, 2026-09-24); fake is
    # the config default, spelled out for explicitness; causvid renders
    # native 832x480 @ 16 fps (the worker rejects anything else — same
    # native-geometry rule as longlive2).
    # fps + latent_shape ride the row too (default_config_toml writes them
    # into voyage.toml — hardcoding 24/[1,8,48,44,80] there made the
    # causvid preset a lie: the supervisor sent fps 24 and the worker
    # refused).
    "fake": BackendRecord(
        profile="fake-432p",
        width=768,
        height=432,
        fps=24,
        latent_shape=(1, 8, 48, 44, 80),
        device="cpu",
        audio_backend="fake",
        audio_device="cpu",
        state_mode="independent_clip",
        streaming=False,
    ),
    "longlive2": BackendRecord(
        profile="longlive2-704p",
        width=1280,
        height=704,
        fps=24,
        latent_shape=(1, 8, 48, 44, 80),
        device="cuda:0",
        audio_backend="acestep",
        audio_device="cuda:0",
        state_mode="persistent_kv",
        streaming=True,
    ),
    "ltxv": BackendRecord(
        profile="ltxv-512p",
        width=768,
        height=512,
        fps=24,
        latent_shape=(1, 8, 48, 44, 80),
        device="cuda:0",
        audio_backend="acestep",
        audio_device="cuda:0",
        state_mode="reconstructable_prefix",
        streaming=True,
    ),
    "causvid": BackendRecord(
        profile="causvid-480p",
        width=832,
        height=480,
        # Native 16 fps end-to-end (the worker refuses anything else —
        # DESIGN §5.4: never relabel 16 fps media as 24; the 24 fps
        # presentation resample is a separate finalize-stage slice).
        fps=16,
        latent_shape=(1, 21, 16, 60, 104),
        device="cuda:0",
        audio_backend="acestep",
        audio_device="cuda:0",
        state_mode="reconstructable_prefix",
        streaming=True,
    ),
}
"""Backend name → full row (issues 022 + 025): geometry + audio pairing +
state mode + streaming shape. This table IS the BACKEND_GEOMETRY table
(geometry columns live in each row) and the preset/state registries —
every dict below is a derived view, never a second source."""

_FAKE_ROW: BackendRecord = BACKEND_REGISTRY["fake"]
"""VideoConfig defaults spell this row (issue 025: defaults = fake row)."""


class VideoConfig(BaseModel):
    # Defaults ARE the fake registry row (issue 025) — change the row,
    # not these references. Pinned by tests/test_backend_registry.py.
    backend: VideoBackendName = "fake"
    profile: str = _FAKE_ROW.profile
    width: int = _FAKE_ROW.width
    height: int = _FAKE_ROW.height
    fps: int = _FAKE_ROW.fps
    segment_frames: int = 48
    device: str = _FAKE_ROW.device
    # LongLive backend only: host path (or /models mount in the worker
    # image) holding wan_models/ + longlive2/, and the latent shape the
    # pipeline denoises. [1,8,48,44,80] decodes to 1280x704 (x16 spatial;
    # 8 latents -> 8 frames chunked, 29 causal).
    models_dir: str = "/models"
    latent_shape: list[int] = Field(default_factory=lambda: list(_FAKE_ROW.latent_shape))
    # Phase 2: DiT blocks per committed segment (1 block = 8 latents).
    # The stream session holds caches across blocks, so memory stays flat;
    # only wall time grows. Fake backend ignores this (renders segment_frames).
    blocks_per_segment: int = 1
    # Slice 4: DiT weight precision. fp8 W8A8 dynamic activation
    # quantization proved to be the highlight-blowout amplifier (bf16
    # probe renders clean); bf16 keeps full precision at +~4.5GB VRAM.
    # The worker derives its recovery profile from this, so tapes never
    # resume across numerics.
    quantization: Literal["fp8", "bf16"] = "fp8"
    # Continuity investigation (§140): KV attention window in frames. The
    # cache must hold sink + one block (capacity floor), else every chunk
    # attends only to its own window and each block regenerates a fresh
    # scene (~6x frame-diff jumps at every boundary, measured). 16 = sink
    # 8F + 8F rolling (chunk-1 boundary structurally invisible, chunk-2
    # flicker-level; draft-validated metric + eyeball). Full-res fits via
    # the worker's VAE-offload-for-generate (breakdown: 13.16 + 2.59 -
    # 1.31 = 14.44 GiB peak). Upstream uses 32 (needs >24GB VRAM).
    local_attn_size: int = 16

    @field_validator(
        "width", "height", "fps", "segment_frames", "blocks_per_segment", "local_attn_size"
    )
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class AudioConfig(BaseModel):
    backend: AudioBackendName = "fake"
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
    # Rhythm grid (§35): each committed segment spans `beats_per_segment`
    # beats; the take BPM derives from the segment duration (adaptive k:
    # 4 → 8 → 16 … until BPM >= 60 — see voyage.audio.beat). Takes chain
    # on segment-aligned boundaries so cuts land on beats (approximately:
    # ACE honors tempo as a hint, not a sample-exact grid).
    beats_per_segment: int = 4
    # Finalize-only joint blend (§56): overlap = min(fraction × shortest
    # adjoining segment, cap). Previews (per-segment audio.wav) stay
    # hard-cut; the blend applies at finalize from take re-slices, so no
    # commit-format change and no A/V drift (overlap content comes from
    # the takes, not by shortening the timeline).
    final_overlap_fraction: float = 0.10
    final_overlap_cap_seconds: float = 0.5
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
        if not math.isfinite(value) or value < 0:
            raise ValueError("must be a finite non-negative number")
        return value

    @field_validator("beats_per_segment")
    @classmethod
    def positive_beats(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("final_overlap_fraction")
    @classmethod
    def overlap_fraction_range(cls, value: float) -> float:
        if not 0.0 <= value <= 0.5:
            raise ValueError("final_overlap_fraction must be within [0, 0.5]")
        return value

    @field_validator("final_overlap_cap_seconds")
    @classmethod
    def overlap_cap_non_negative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("must be a finite non-negative number")
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
    max_worker_restarts: int = 3
    rpc_timeout_seconds: float = 600.0
    # Thematic drift cadence: the director must propose a novel destination
    # every Nth segment (1 = drift each segment). Non-drift segments hold
    # the current concept via the deterministic fallback (still recorded).
    drift_every_n_segments: int = 1

    @field_validator("blocks_per_prompt_stage", "novelty_max_attempts", "drift_every_n_segments")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("max_worker_restarts")
    @classmethod
    def non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("rpc_timeout_seconds")
    @classmethod
    def positive_seconds(cls, value: float) -> float:
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

    This model IS the named draft overlay (issue 025): the stored [draft]
    TOML table rides ProjectConfig, and resolve_config applies it on top
    of the backend preset. Kept as a model (not a bare dict) because
    stored configs and existing callers read config.draft with validation;
    the canonical default values are pinned by tests/test_draft.py.
    Applied only when the run requests it (`voyage run --draft`); the
    stored TOML keeps full-quality values. Spatial dims are halved
    (640x352, latent [1,8,48,22,40]); the temporal dim is untouched.
    Draft renders are for iteration, never for finals.
    """

    width: int = 640
    height: int = 352
    latent_shape: list[int] = Field(default_factory=lambda: [1, 8, 48, 22, 40])
    blocks_per_segment: int = 1
    # Same take length as full quality: a take shorter than the audio-ahead
    # window forces a render + full GPU swap on EVERY segment (draft E2E:
    # take 15 < ahead 20 → swap every segment, ~85s audio stage). At 45s the
    # swap happens ~every 21 segments and amortizes to ~7s/segment.
    take_seconds: float = 45.0

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


def _preset_int(preset: dict[str, str | int | list[int]], key: str, default: int) -> int:
    """Narrow a preset value to int (mypy-strict: the dict also holds lists)."""
    value = preset.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"video preset key {key!r} must be an int (got {value!r})")
    return value


def _toml_basic_string(raw_value: str) -> str:
    """Quote free-text as a TOML basic string.

    Why a local escaper: style/run_id are creator free-text persisted into
    the run charter, so a quote or newline would otherwise break the
    generated TOML or inject live tables (issue 009). Mirrors the TUI
    escaper; kept here (not imported) because the TUI helper belongs to a
    later batch.
    """
    escaped_value = (
        raw_value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped_value}"'


def default_config_toml(
    run_id: str, style: str, seed: int, video_backend: VideoBackendName = "fake"
) -> str:
    preset = _video_preset(video_backend)
    backend = str(preset.get("backend", video_backend))
    profile = str(preset.get("profile", "fake-432p"))
    width = _preset_int(preset, "width", 768)
    height = _preset_int(preset, "height", 432)
    fps = _preset_int(preset, "fps", 24)
    raw_latent = preset.get("latent_shape", [1, 8, 48, 44, 80])
    latent_dims = (
        [int(dim) for dim in raw_latent] if isinstance(raw_latent, list) else [1, 8, 48, 44, 80]
    )
    latent_toml = "[" + ", ".join(str(dim) for dim in latent_dims) + "]"
    device = str(preset.get("device", "cpu"))
    audio_preset = _audio_preset(video_backend)
    audio_backend = audio_preset["backend"]
    audio_device = audio_preset["device"]
    escaped_run_id = _toml_basic_string(run_id)
    escaped_style = _toml_basic_string(style)
    return f"""\
schema_version = 1
run_id = {escaped_run_id}
style = {escaped_style}
seed = {seed}
min_free_space_gib = 5.0

[video]
# "fake" (built-in testsrc) | "longlive2" (CUDA) | "ltxv" (CUDA) | "causvid" (CUDA, 16 fps)
backend = "{backend}"
profile = "{profile}"
width = {width}
height = {height}
fps = {fps}
segment_frames = 48
device = "{device}"
models_dir = "/models"
latent_shape = {latent_toml}
blocks_per_segment = 1
quantization = "fp8"

[audio]
backend = "{audio_backend}"
sample_rate = 48000
channels = 2
music_style = "ambient electronic"
energy = 0.5
take_seconds = 45.0
ahead_seconds = 20.0
crossfade_seconds = 2.0
beats_per_segment = 4
final_overlap_fraction = 0.1
final_overlap_cap_seconds = 0.5
models_dir = "/models"
device = "{audio_device}"

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
max_worker_restarts = 3
rpc_timeout_seconds = 600.0
drift_every_n_segments = 1

[experimental]
visual_inspector = false

[draft]
width = 640
height = 352
latent_shape = [1, 8, 48, 22, 40]
blocks_per_segment = 1
take_seconds = 45.0
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


def resolve_config(
    config: ProjectConfig,
    *,
    backend: VideoBackendName | None = None,
    draft: bool = False,
    director: str | None = None,
    blocks: int | None = None,
    take_seconds: float | None = None,
    quantization: str | None = None,
    beats_per_segment: int | None = None,
    drift_every_n_segments: int | None = None,
) -> ProjectConfig:
    """Single configuration resolver (issues 022 + 025): backend preset,
    then the stored [draft] overlay, then targeted overrides — in that
    order, so explicit flags always win over profiles.

    Pure: returns a new config, never mutates. Rebuilds submodels through
    their constructors so invalid overrides (blocks=0, negative takes)
    raise ValidationError instead of silently corrupting the run. Unknown
    backends raise ValueError (same message as the old preset lookup).
    This replaces the apply_draft_overrides + with_video_backend pair —
    both survive below as thin wrappers for their existing callers.
    """
    video = config.video
    audio = config.audio
    director_config = config.director
    voyage_config = config.voyage
    if backend is not None:
        video = VideoConfig(**{**video.model_dump(), **_video_preset(backend)})
        audio = AudioConfig(
            **{**audio.model_dump(), **_audio_preset(backend), "models_dir": "/models"}
        )
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
        director_config = DirectorConfig(**{**director_config.model_dump(), "backend": director})
    if blocks is not None:
        video = VideoConfig(**{**video.model_dump(), "blocks_per_segment": blocks})
    if quantization is not None:
        video = VideoConfig(**{**video.model_dump(), "quantization": quantization})
    if take_seconds is not None:
        audio = AudioConfig(**{**audio.model_dump(), "take_seconds": take_seconds})
    if beats_per_segment is not None:
        audio = AudioConfig(**{**audio.model_dump(), "beats_per_segment": beats_per_segment})
    if drift_every_n_segments is not None:
        voyage_config = VoyageConfig(
            **{**voyage_config.model_dump(), "drift_every_n_segments": drift_every_n_segments}
        )
    return config.model_copy(
        update={
            "video": video,
            "audio": audio,
            "director": director_config,
            "voyage": voyage_config,
        }
    )


def apply_draft_overrides(
    config: ProjectConfig,
    *,
    draft: bool = False,
    director: str | None = None,
    blocks: int | None = None,
    take_seconds: float | None = None,
    quantization: str | None = None,
    beats_per_segment: int | None = None,
    drift_every_n_segments: int | None = None,
) -> ProjectConfig:
    """Apply the draft profile + targeted run overrides (fast loop).

    Thin wrapper over resolve_config (issue 025) — kept for the CLI and
    existing tests. New code should call resolve_config directly.
    """
    return resolve_config(
        config,
        draft=draft,
        director=director,
        blocks=blocks,
        take_seconds=take_seconds,
        quantization=quantization,
        beats_per_segment=beats_per_segment,
        drift_every_n_segments=drift_every_n_segments,
    )


# Audio backend paired with each video row: CUDA video backends get the
# real ACE-Step music stack (models + device mirror the video row), while
# fake video keeps the fake sine backend for CPU-only test runs. The
# pairing lives in BackendRecord.audio_backend/audio_device — these dicts
# are derived views (issue 022), never a second source.
_VIDEO_BACKEND_PRESETS: dict[str, dict[str, str | int | list[int]]] = {
    name: {
        "backend": name,
        "profile": record.profile,
        "width": record.width,
        "height": record.height,
        "fps": record.fps,
        "latent_shape": list(record.latent_shape),
        "device": record.device,
    }
    for name, record in BACKEND_REGISTRY.items()
}

_AUDIO_BACKEND_PRESETS: dict[str, dict[str, str]] = {
    name: {"backend": record.audio_backend, "device": record.audio_device}
    for name, record in BACKEND_REGISTRY.items()
}


def _video_preset(backend: str) -> dict[str, str | int | list[int]]:
    """Video preset row as a plain dict (derived from BACKEND_REGISTRY).

    Takes plain str on purpose: external callers such as
    tui_state._planning_frames_and_fps pass unchecked strings, so the
    boundary validates at runtime while typed cores take VideoBackendName.
    """
    try:
        return _VIDEO_BACKEND_PRESETS[backend]
    except KeyError:
        known = ", ".join(sorted(_VIDEO_BACKEND_PRESETS))
        raise ValueError(f"unknown video backend {backend!r} (known: {known})") from None


def _audio_preset(backend: str) -> dict[str, str]:
    """Audio pairing row as a plain dict (derived from BACKEND_REGISTRY)."""
    try:
        return _AUDIO_BACKEND_PRESETS[backend]
    except KeyError:
        known = ", ".join(sorted(_AUDIO_BACKEND_PRESETS))
        raise ValueError(f"unknown video backend {backend!r} (known: {known})") from None


def with_video_backend(config: ProjectConfig, backend: VideoBackendName) -> ProjectConfig:
    """Return a copy of config with the video-backend preset applied.

    Thin wrapper over resolve_config (issue 025) — kept for existing
    callers and tests, which pin its behavior (preset geometry + audio
    pairing + purity + ValueError on unknown backends).
    """
    return resolve_config(config, backend=backend)

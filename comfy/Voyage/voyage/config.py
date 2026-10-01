"""Project configuration (DESIGN task group B).

TOML loader + validation + sha256 hash. A snapshot of the config is
written into each run directory at init time; the hash is recorded in
the run manifest so runs are traceable to exact configuration.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeGuard, TypeVar

try:
    import tomllib
except ImportError:  # Python 3.10 worker image (upstream env)
    import tomli as tomllib

from pydantic import BaseModel, Field, field_validator, model_validator

from voyage.errors import ConfigurationError

VideoBackendName = Literal["fake", "ltxv", "causvid"]
"""Video backend vocabulary (issue 022): every backend field, the registry,
and the streaming set are keyed by this — a typo fails at typecheck
instead of after GPU init."""

REMOVED_VIDEO_BACKENDS: tuple[str, ...] = ("longlive2",)
"""Video backends removed outright (issue 079, full delete).

`longlive2` was the heaviest legacy (1264L worker, own `.pt` tape,
checkpoint-load OOM in archived issue 096). Silent remap to `ltxv` is
FORBIDDEN: it changes geometry mid-run (1280x704/29f -> 768x512/96f),
invalidating committed segment durations, while longlive `.pt` tapes can
never resume on the ltxv JSON-tape path. Stored `backend = "longlive2"`
runs must fail fast with the migration hint below.
"""


def removed_backend_suffix(backend: str) -> str:
    """Migration-hint suffix for removed backends (issue 079), else empty.

    Pure string helper so every unknown-backend site (`_video_preset`,
    `_audio_preset`, `_sfx_preset`, `load_config`, `video_worker_module`,
    `cmd_init`) shares one hint instead of restating it.
    """
    if backend in REMOVED_VIDEO_BACKENDS:
        return (
            ' — video backend "longlive2" was removed (issue 079); '
            "re-init with --backend ltxv (tapes do not transfer; "
            "existing segments stay valid media, only continuation stops)"
        )
    return ""


AudioBackendName = Literal["fake", "acestep"]
"""Audio backend vocabulary (issue 022): fake sine vs the ACE-Step music stack."""

SfxBackendName = Literal["fake", "mmaudio"]
"""SFX backend vocabulary: fake noise vs the MMAudio effects stack."""

SfxModelSize = Literal["small_44k", "medium_44k", "large_44k_v2"]
"""MMAudio 44 kHz variant vocabulary (mirrors audio.mmaudio_sfx)."""

StateMode = Literal["persistent_kv", "reconstructable_prefix", "independent_clip"]
"""Continuation-state vocabulary (DESIGN §5.1): how a backend resumes work.

Single home (issue 022); voyage.backends re-exports it for the adapter
contract so the two modules never spell the modes apart.
"""


class UnsetType:
    """Sentinel type for "option not provided" (issue 045).

    The CLI spells absence as `None` (argparse defaults) while the TUI
    form spells it as `""` — two encodings for one meaning, forcing every
    consumer to agree on which emptiness means "default". New option
    plumbing uses `Unset` instead; `None` stays a tolerated legacy alias
    so existing callers keep working. Never construct directly.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "Unset"

    def __bool__(self) -> bool:
        return False


Unset = UnsetType()
"""The single "not provided" value for option plumbing (issue 045)."""


_ProvidedT = TypeVar("_ProvidedT")


def is_provided(value: _ProvidedT | UnsetType | None) -> TypeGuard[_ProvidedT]:
    """Whether an option was explicitly provided (issue 045).

    The one absent-check for option plumbing: `Unset` (new code) and
    `None` (argparse legacy) both mean absent; anything else passes.
    A TypeGuard (not identity checks inline) so every typechecker
    narrows `X | UnsetType | None` to `X` in the true branch —
    singleton `is`-narrowing is not reliable across checkers.
    Consumers only ever branch on this predicate, never on `is not
    None` (which would mistake Unset for a value).
    """
    return value is not None and not isinstance(value, UnsetType)


@dataclass(frozen=True)
class BackendRecord:
    """One row of BACKEND_REGISTRY: a video backend's geometry, audio
    pairing, SFX pairing, continuation mode, and streaming shape
    (issues 022 + 025).

    Frozen so the single source of truth cannot drift at runtime; every
    preset dict, state-mode map, and streaming set below derives from
    the registry instead of restating these values.
    """

    profile: str
    width: int
    height: int
    fps: int
    # Natural novel frames per committed segment at blocks_per_segment=1
    # (the duration-math source for presets/toml; the per-backend segment
    # duration formulas in cli stay authoritative for planning).
    segment_frames: int
    latent_shape: tuple[int, ...]
    device: str
    audio_backend: AudioBackendName
    audio_device: str
    sfx_backend: SfxBackendName
    sfx_device: str
    state_mode: StateMode
    streaming: bool


BACKEND_REGISTRY: dict[VideoBackendName, BackendRecord] = {
    # Single source of truth for `generate --backend` presets (also used
    # by default_config_toml). ltxv renders native 768x512 on CUDA (both
    # /32 and /64 clean for the two-stage multiscale pipeline; 1024x576
    # was tried 2026-09-24 but needs ~15.6 GB in the forward — beyond the
    # 16 GB card even via the dynamic-fp8 fallback — so it stays reverted
    # until a memory-optimization pass lands).
    # ltxv is the config default (2026-09-29 backend decision), spelled
    # out for explicitness; causvid renders
    # native 832x480 @ 16 fps (the worker rejects anything else — same
    # native-geometry rule a removed backend once enforced).
    # fps + latent_shape ride the row too (default_config_toml writes them
    # into voyage.toml — hardcoding 24/[1,8,48,44,80] there made the
    # causvid preset a lie: the supervisor sent fps 24 and the worker
    # refused).
    "fake": BackendRecord(
        profile="fake-432p",
        width=768,
        height=432,
        fps=24,
        segment_frames=48,
        latent_shape=(1, 8, 48, 44, 80),
        device="cpu",
        audio_backend="fake",
        audio_device="cpu",
        sfx_backend="fake",
        sfx_device="cpu",
        state_mode="independent_clip",
        streaming=False,
    ),
    "ltxv": BackendRecord(
        profile="ltxv-512p",
        width=768,
        height=512,
        fps=24,
        segment_frames=96,
        latent_shape=(1, 8, 48, 44, 80),
        device="cuda:0",
        audio_backend="acestep",
        audio_device="cuda:0",
        sfx_backend="mmaudio",
        sfx_device="cuda:0",
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
        segment_frames=72,
        latent_shape=(1, 21, 16, 60, 104),
        device="cuda:0",
        audio_backend="acestep",
        audio_device="cuda:0",
        sfx_backend="mmaudio",
        sfx_device="cuda:0",
        state_mode="reconstructable_prefix",
        streaming=True,
    ),
}
"""Backend name → full row (issues 022 + 025): geometry + audio pairing +
state mode + streaming shape. This table IS the BACKEND_GEOMETRY table
(geometry columns live in each row) and the preset/state registries —
every dict below is a derived view, never a second source."""

_DEFAULT_ROW: BackendRecord = BACKEND_REGISTRY["ltxv"]
"""VideoConfig defaults spell this row (issues 025 + 2026-09-29 ltxv decision)."""


class VideoConfig(BaseModel):
    # Defaults ARE the ltxv registry row (issue 025) — change the row,
    # not these references. Pinned by tests/test_backend_registry.py.
    backend: VideoBackendName = "ltxv"
    profile: str = _DEFAULT_ROW.profile
    width: int = _DEFAULT_ROW.width
    height: int = _DEFAULT_ROW.height
    fps: int = _DEFAULT_ROW.fps
    segment_frames: int = _DEFAULT_ROW.segment_frames
    device: str = _DEFAULT_ROW.device
    # Host path (or /models mount in the worker image) holding the
    # backend's weights, and the latent shape the pipeline denoises.
    models_dir: str = "/models"
    latent_shape: list[int] = Field(default_factory=lambda: list(_DEFAULT_ROW.latent_shape))
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
    # Explicit video-caption pin (CLI --video-caption, in-memory only —
    # never written to voyage.toml). When set, every segment's staged
    # prompt uses it instead of the director's evolving stages (no drift
    # for this family, still style-checked against the charter); when
    # None the director drives (stages evolve with the general prompt).
    video_caption: str | None = None

    @field_validator(
        "width", "height", "fps", "segment_frames", "blocks_per_segment", "local_attn_size"
    )
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


#: Ceiling for `AudioConfig.final_overlap_fraction` (finalize blend): at most
#: half a segment may be re-sliced into the overlap — beyond that the "joint"
#: would swallow the take itself instead of joining two takes.
MAX_FINAL_OVERLAP_FRACTION = 0.5


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
    # Repaint gate (Stage B): a caption change repaints the unconsumed take
    # region only when Jaccard token-set similarity drops below this.
    # Boba-calibrated 0.5: reword-repaints scored 0.72–1.00 (suppressed),
    # the one defensible shift scored 0.444 (still repaints).
    repaint_similarity_threshold: float = 0.5
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
    # Explicit music-caption pin (CLI --music-caption, in-memory only —
    # never written to voyage.toml). When set, every take uses it instead
    # of the director's evolving caption (no drift for this family);
    # when None the director drives (captions evolve with the general
    # prompt). Empty string is falsy → falls back like an absent pin.
    music_caption: str | None = None

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

    @field_validator("repaint_similarity_threshold")
    @classmethod
    def similarity_unit_range(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("repaint_similarity_threshold must be within [0, 1]")
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
        if not 0.0 <= value <= MAX_FINAL_OVERLAP_FRACTION:
            raise ValueError(
                f"final_overlap_fraction must be within [0, {MAX_FINAL_OVERLAP_FRACTION}]"
            )
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

    @model_validator(mode="after")
    def take_covers_ahead(self) -> AudioConfig:
        """Refuse take lengths that force a GPU swap on every segment (issue 013).

        A take no longer than the audio-ahead window expires before the
        coverage check ever keeps it, so every segment pays a full
        video-evict + take-render + audio-evict + video-rebuild cycle
        (the dominant wall-clock cost on a 16 GiB card). Steady state
        needs take_seconds >> ahead_seconds (defaults 45.0/20.0).
        """
        if not self.take_seconds > self.ahead_seconds:
            raise ValueError(
                f"take_seconds ({self.take_seconds}) must exceed ahead_seconds "
                f"({self.ahead_seconds}): a take no longer than the audio-ahead "
                "window forces a full video↔audio GPU swap on EVERY segment — "
                "keep take_seconds >> ahead_seconds (defaults 45.0/20.0)"
            )
        return self


class SfxConfig(BaseModel):
    """Finalize-time effects config (SFX slice 2, three-caption doctrine).

    `backend="fake"` keeps old runs byte-identical (finalize without the
    SFX pass); `"mmaudio"` renders director-captioned windows at finalize
    (slice 3 wiring). `model_size` rides the ladder vocabulary — the 2060
    ladder (slice 2c) locks the deployed value.
    """

    backend: SfxBackendName = "fake"
    device: str = "cpu"
    models_dir: str = "/models"
    model_size: SfxModelSize = "large_44k_v2"


class AugmentConfig(BaseModel):
    """Finalize-time augmentation floors (Track A: knobs only).

    `min_fps = 0` disables the fps floor; `min_width = min_height = 0`
    disables the resolution floor. Geometry must be both-zero or
    both-positive — a half-disabled floor (0 wide x 720 high) is
    meaningless, so the model rejects it. The media consumer that reads
    these floors lands in a later slice; this track only plumbs them
    through TOML + CLI + TUI.
    """

    min_fps: int = 32
    min_width: int = 1280
    min_height: int = 720

    @field_validator("min_fps", "min_width", "min_height")
    @classmethod
    def non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative (0 disables the floor)")
        return value

    @model_validator(mode="after")
    def geometry_both_or_neither(self) -> AugmentConfig:
        if (self.min_width == 0) != (self.min_height == 0):
            raise ValueError("min_width and min_height must both be 0 or both positive")
        return self


def parse_min_resolution(raw: str) -> tuple[int, int]:
    """Parse a resolution floor: WxH like "1280x720", or "0" to disable.

    Returns (width, height); "0" (and the equivalent "0x0") returns
    (0, 0). Anything else — wrong shape, non-digits, or a half-disabled
    pair like "0x720" — raises ValueError so CLI/TUI/resolve paths share
    one error source. The full both-or-neither invariant also lives on
    AugmentConfig for direct construction and TOML loads.
    """
    text = raw.strip().lower()
    if text == "0":
        return (0, 0)
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", text)
    if match is None:
        raise ValueError(
            f'invalid resolution {raw!r} (expected WxH like "1280x720" or "0" to disable)'
        )
    width, height = int(match.group(1)), int(match.group(2))
    if (width == 0) != (height == 0):
        raise ValueError(
            f"invalid resolution {raw!r}: width and height must both be 0 or both positive"
        )
    return (width, height)


class DirectorConfig(BaseModel):
    backend: str = "qwen"
    model_id: str = "Qwen/Qwen3-8B"
    # Decider placement: the unified worker image runs the Qwen decider on
    # cuda:1 (second GPU) via a 4-bit AWQ model; "cpu" keeps the legacy bf16
    # path (explicit opt-out for single-GPU / CI boxes). The worker falls
    # back to CPU with a loud warning when the device is absent.
    device: str = "cuda:1"
    temperature: float = 0.7
    # Qwen worker: non-thinking mode (no <think> parsing), JSON-only output.
    enable_thinking: bool = False
    max_new_tokens: int = 1024
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    # VLM inspector (Phase 5): Qwen3.5-9B lives in the director image
    # (transformers 5.x); a local path works for E2E (/models/Qwen3.5-9B).
    inspector_model_id: str = "Qwen/Qwen3.5-9B"

    @field_validator("device")
    @classmethod
    def _placement(cls, value: str) -> str:
        # Fail fast at the CLI/config layer: without this, garbage survives
        # into TOML, ensures the AWQ weights, then dies deep in the worker.
        # Mirrors workers/director._normalize_device across the boundary —
        # the supervisor package must not import the worker (GPU ban), so
        # the one-line predicate is deliberately duplicated, not shared.
        if value != "cpu" and not value.startswith("cuda"):
            raise ValueError(f"director device must be 'cpu' or 'cuda[:N]' (got {value!r})")
        return value


class VoyageConfig(BaseModel):
    """Evolution policy (DESIGN §14 [voyage] table, §§21.2-21.3)."""

    allow_concept_revisit: bool = False
    major_transition_min_seconds: float = 30.0
    major_transition_max_seconds: float = 120.0
    world_decision_interval_seconds: float = 16.0
    blocks_per_prompt_stage: int = 3
    novelty_threshold: float = 0.85
    novelty_max_attempts: int = 3
    # Novelty leniency: after this many novelty rejections the last
    # generation is accepted anyway (novelty_accepted=False) instead of
    # burning the remaining attempts toward a deterministic fallback.
    # Schema/empty-stages/style rejections stay hard — only novelty goes
    # lenient, so the style charter still always wins.
    novelty_max_rejections: int = 2
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

    @field_validator("novelty_max_rejections")
    @classmethod
    def non_negative_rejections(cls, value: int) -> int:
        if value < 0:
            raise ValueError("novelty_max_rejections must be non-negative")
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
        if not math.isfinite(value) or value <= 0:
            raise ValueError("must be a finite positive number")
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
        if not math.isfinite(value) or value < 0:
            raise ValueError("must be a finite non-negative number")
        return value


SPEC_MIN_FREE_SPACE_GIB = 20.0
"""Spec default reserve (DESIGN §70/§140): hand-written TOML omitting the key gets this."""

DEV_MIN_FREE_SPACE_GIB = 5.0
"""Init-generated TOML default (issue 052): small dev boxes validate at 5 GiB."""


class ProjectConfig(BaseModel):
    schema_version: int = 1
    run_id: str = "voyage"
    style: str = ""
    seed: int = 0
    min_free_space_gib: float = SPEC_MIN_FREE_SPACE_GIB
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    sfx: SfxConfig = Field(default_factory=SfxConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
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
    """Quote free-text as a TOML basic string (single shared escaper).

    Why one escaper: style/run_id are creator free-text persisted into
    the run charter, so a quote or newline would otherwise break the
    generated TOML or inject live tables (issue 009). Short escapes
    cover backslash/quote/newline/return/tab; every other C0 control
    plus DEL becomes ``\\uXXXX`` so one stray byte can never emit a
    file our own reader rejects (issue 020). The TUI helper delegates
    here (lazy import, same single source) to preserve its stdlib-only
    import time.
    """
    escaped_value = (
        raw_value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    for code in range(0x20):
        control = chr(code)
        if control in ("\n", "\r", "\t"):
            continue
        escaped_value = escaped_value.replace(control, f"\\u{code:04X}")
    escaped_value = escaped_value.replace("\x7f", "\\u007F")
    return f'"{escaped_value}"'


def default_config_toml(
    run_id: str,
    style: str,
    seed: int,
    video_backend: VideoBackendName = "ltxv",
    director_backend: str = "qwen",
    director_device: str = "cuda:1",
) -> str:
    preset = _video_preset(video_backend)
    # Fallbacks below read `_DEFAULT_ROW` (issue 085 single source), never
    # restated literals: preset dicts are complete projections of
    # BACKEND_REGISTRY, so these defaults never fire for known backends —
    # they only pin the shape for hypothetical partial rows. (The old
    # `device` fallback said "cpu" while the default row is cuda:0; that
    # literal was unreachable for the same reason, so deriving it changes
    # no generated TOML.)
    backend = str(preset.get("backend", video_backend))
    profile = str(preset.get("profile", _DEFAULT_ROW.profile))
    width = _preset_int(preset, "width", _DEFAULT_ROW.width)
    height = _preset_int(preset, "height", _DEFAULT_ROW.height)
    fps = _preset_int(preset, "fps", _DEFAULT_ROW.fps)
    segment_frames = _preset_int(preset, "segment_frames", _DEFAULT_ROW.segment_frames)
    raw_latent = preset.get("latent_shape", list(_DEFAULT_ROW.latent_shape))
    latent_dims = (
        [int(dim) for dim in raw_latent]
        if isinstance(raw_latent, list)
        else list(_DEFAULT_ROW.latent_shape)
    )
    latent_toml = "[" + ", ".join(str(dim) for dim in latent_dims) + "]"
    device = str(preset.get("device", _DEFAULT_ROW.device))
    audio_preset = _audio_preset(video_backend)
    audio_backend = audio_preset["backend"]
    audio_device = audio_preset["device"]
    sfx_preset = _sfx_preset(video_backend)
    sfx_backend = sfx_preset["backend"]
    sfx_device = sfx_preset["device"]
    escaped_run_id = _toml_basic_string(run_id)
    escaped_style = _toml_basic_string(style)
    return f"""\
schema_version = 1
run_id = {escaped_run_id}
style = {escaped_style}
seed = {seed}
min_free_space_gib = {DEV_MIN_FREE_SPACE_GIB}

[video]
# "ltxv" (CUDA) | "causvid" (CUDA, 16 fps) | "fake" (built-in testsrc)
backend = "{backend}"
profile = "{profile}"
width = {width}
height = {height}
fps = {fps}
segment_frames = {segment_frames}
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
repaint_similarity_threshold = 0.5
beats_per_segment = 4
final_overlap_fraction = 0.1
final_overlap_cap_seconds = 0.5
models_dir = "/models"
device = "{audio_device}"

[sfx]
# "mmaudio" (CUDA) | "fake" (built-in noise — CPU-only/test runs)
backend = "{sfx_backend}"
device = "{sfx_device}"
models_dir = "/models"
model_size = "large_44k_v2"

[augment]
# Finalize-time floors: 0 disables a floor (min_fps = 0, or 0x0 geometry).
min_fps = 32
min_width = 1280
min_height = 720

[director]
backend = "{director_backend}"
model_id = "Qwen/Qwen3-8B"
device = "{director_device}"
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
novelty_max_rejections = 2
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
    video_section = raw.get("video")
    if isinstance(video_section, dict):
        stored_backend = video_section.get("backend")
        if isinstance(stored_backend, str) and stored_backend in REMOVED_VIDEO_BACKENDS:
            known = ", ".join(sorted(BACKEND_REGISTRY))
            raise ConfigurationError(
                f"invalid config {path}: unknown video backend {stored_backend!r} "
                f"(known: {known}){removed_backend_suffix(stored_backend)}"
            )
    try:
        config = ProjectConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"invalid config {path}: {exc}") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, digest


def resolve_config(
    config: ProjectConfig,
    *,
    backend: VideoBackendName | None | UnsetType = Unset,
    draft: bool = False,
    director: str | None | UnsetType = Unset,
    director_device: str | None | UnsetType = Unset,
    blocks: int | None | UnsetType = Unset,
    take_seconds: float | None | UnsetType = Unset,
    quantization: str | None | UnsetType = Unset,
    beats_per_segment: int | None | UnsetType = Unset,
    drift_every_n_segments: int | None | UnsetType = Unset,
    music_caption: str | None | UnsetType = Unset,
    video_caption: str | None | UnsetType = Unset,
    min_fps: int | None | UnsetType = Unset,
    min_resolution: str | None | UnsetType = Unset,
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

    Absent-encoding (issue 045): defaults are `Unset`; `None` (argparse
    legacy) is tolerated as absent too — every branch below goes through
    `is_provided`, so the body never checks `is not None` (which would
    mistake Unset for a value).
    """
    video = config.video
    audio = config.audio
    sfx = config.sfx
    director_config = config.director
    voyage_config = config.voyage
    if is_provided(backend):
        video = VideoConfig(**{**video.model_dump(), **_video_preset(backend)})
        audio = AudioConfig(
            **{**audio.model_dump(), **_audio_preset(backend), "models_dir": "/models"}
        )
        sfx = SfxConfig(**{**sfx.model_dump(), **_sfx_preset(backend), "models_dir": "/models"})
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
    if is_provided(director):
        director_config = DirectorConfig(**{**director_config.model_dump(), "backend": director})
    if is_provided(director_device):
        director_config = DirectorConfig(
            **{**director_config.model_dump(), "device": director_device}
        )
    if is_provided(blocks):
        video = VideoConfig(**{**video.model_dump(), "blocks_per_segment": blocks})
    if is_provided(quantization):
        video = VideoConfig(**{**video.model_dump(), "quantization": quantization})
    if is_provided(take_seconds):
        audio = AudioConfig(**{**audio.model_dump(), "take_seconds": take_seconds})
    if is_provided(beats_per_segment):
        audio = AudioConfig(**{**audio.model_dump(), "beats_per_segment": beats_per_segment})
    if is_provided(drift_every_n_segments):
        voyage_config = VoyageConfig(
            **{**voyage_config.model_dump(), "drift_every_n_segments": drift_every_n_segments}
        )
    if is_provided(music_caption):
        audio = AudioConfig(**{**audio.model_dump(), "music_caption": music_caption})
    if is_provided(video_caption):
        video = VideoConfig(**{**video.model_dump(), "video_caption": video_caption})
    augment = config.augment
    if is_provided(min_fps) or is_provided(min_resolution):
        resolved_fps = augment.min_fps
        resolved_width = augment.min_width
        resolved_height = augment.min_height
        if is_provided(min_fps):
            resolved_fps = min_fps
        if is_provided(min_resolution):
            resolved_width, resolved_height = parse_min_resolution(min_resolution)
        augment = AugmentConfig(
            min_fps=resolved_fps,
            min_width=resolved_width,
            min_height=resolved_height,
        )
    return config.model_copy(
        update={
            "video": video,
            "audio": audio,
            "sfx": sfx,
            "augment": augment,
            "director": director_config,
            "voyage": voyage_config,
        }
    )


def apply_draft_overrides(
    config: ProjectConfig,
    *,
    draft: bool = False,
    director: str | None | UnsetType = Unset,
    director_device: str | None | UnsetType = Unset,
    blocks: int | None | UnsetType = Unset,
    take_seconds: float | None | UnsetType = Unset,
    quantization: str | None | UnsetType = Unset,
    beats_per_segment: int | None | UnsetType = Unset,
    drift_every_n_segments: int | None | UnsetType = Unset,
    music_caption: str | None | UnsetType = Unset,
    video_caption: str | None | UnsetType = Unset,
    min_fps: int | None | UnsetType = Unset,
    min_resolution: str | None | UnsetType = Unset,
) -> ProjectConfig:
    """Apply the draft profile + targeted run overrides (fast loop).

    Thin wrapper over resolve_config (issue 025) — kept for the CLI and
    existing tests. New code should call resolve_config directly.
    Absent-encoding (issue 045): defaults are `Unset`, `None` tolerated.
    """
    return resolve_config(
        config,
        draft=draft,
        director=director,
        director_device=director_device,
        blocks=blocks,
        take_seconds=take_seconds,
        quantization=quantization,
        beats_per_segment=beats_per_segment,
        drift_every_n_segments=drift_every_n_segments,
        music_caption=music_caption,
        video_caption=video_caption,
        min_fps=min_fps,
        min_resolution=min_resolution,
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
        "segment_frames": record.segment_frames,
        "latent_shape": list(record.latent_shape),
        "device": record.device,
    }
    for name, record in BACKEND_REGISTRY.items()
}

_AUDIO_BACKEND_PRESETS: dict[str, dict[str, str]] = {
    name: {"backend": record.audio_backend, "device": record.audio_device}
    for name, record in BACKEND_REGISTRY.items()
}

_SFX_BACKEND_PRESETS: dict[str, dict[str, str]] = {
    name: {"backend": record.sfx_backend, "device": record.sfx_device}
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
        raise ValueError(
            f"unknown video backend {backend!r} (known: {known}){removed_backend_suffix(backend)}"
        ) from None


def _audio_preset(backend: str) -> dict[str, str]:
    """Audio pairing row as a plain dict (derived from BACKEND_REGISTRY)."""
    try:
        return _AUDIO_BACKEND_PRESETS[backend]
    except KeyError:
        known = ", ".join(sorted(_AUDIO_BACKEND_PRESETS))
        raise ValueError(
            f"unknown video backend {backend!r} (known: {known}){removed_backend_suffix(backend)}"
        ) from None


def _sfx_preset(backend: str) -> dict[str, str]:
    """SFX pairing row as a plain dict (derived from BACKEND_REGISTRY).

    CUDA video backends pair the MMAudio stack on cuda:0 (SFX/music on
    by default on GPU); fake keeps the fake-noise backend on CPU so
    CPU-only test runs never touch weights.
    """
    try:
        return _SFX_BACKEND_PRESETS[backend]
    except KeyError:
        known = ", ".join(sorted(_SFX_BACKEND_PRESETS))
        raise ValueError(
            f"unknown video backend {backend!r} (known: {known}){removed_backend_suffix(backend)}"
        ) from None


def with_video_backend(config: ProjectConfig, backend: VideoBackendName) -> ProjectConfig:
    """Return a copy of config with the video-backend preset applied.

    Thin wrapper over resolve_config (issue 025) — kept for existing
    callers and tests, which pin its behavior (preset geometry + audio
    pairing + purity + ValueError on unknown backends).
    """
    return resolve_config(config, backend=backend)

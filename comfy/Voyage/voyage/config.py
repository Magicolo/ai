"""Project configuration (DESIGN task group B).

CLI-is-config: the effective `ProjectConfig` is built directly from the
`generate` flags (`preset_config` + `resolve_config`) and persisted as
structured data inside `run_manifest.json` — there is no TOML layer.
Validation lives in the model validators; the manifest digest traces
runs to their exact configuration.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, TypeGuard, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from voyage.models import DEFAULT_MUSIC_STYLE

VideoBackendName = Literal["fake", "ltxv", "causvid", "ltx25", "ltx23"]
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
    `_audio_preset`, `_sfx_preset`, `video_worker_module`) shares one hint
    instead of restating it.
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
    # (the duration-math source for registry presets; the per-backend segment
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
    # by preset_config). ltxv renders native 768x512 on CUDA (both
    # /32 and /64 clean for the two-stage multiscale pipeline; 1024x576
    # was tried 2026-09-24 but needs ~15.6 GB in the forward — beyond the
    # 16 GB card even via the dynamic-fp8 fallback — so it stays reverted
    # until a memory-optimization pass lands).
    # ltxv is the config default (2026-09-29 backend decision), spelled
    # out for explicitness; causvid renders
    # native 832x480 @ 16 fps (the worker rejects anything else — same
    # native-geometry rule a removed backend once enforced).
    # fps + latent_shape ride the row too (preset_config resolves them
    # through the row — hardcoding 24/[1,8,48,44,80] once made the
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
    # ltx25/ltx23 (DESIGN §140 ltx plan, Voyage/LTX2.md campaign):
    # LTX-2.5 / LTX-2.3 distilled GGUF via in-process ComfyUI
    # (voyage/workers/video_ltx25.py / video_ltx23.py). Committed
    # segments are Mode A two-stage 1216x704@24 (stage-1
    # 608x352x121 8-sigma distilled, 2x latent upscale, stage-2
    # 3-step refine); 121-frame windows with a 25-frame frozen
    # prefix carry commit 96 novel (same 25+96 math as ltxv).
    # latent_shape is the stage-1 video latent [B, C, T, H, W] =
    # [1, 128, (121-1)//8+1, 608//32, 352//32] (EmptyLTXVLatentVideo
    # convention); the workers enforce the fixed native geometry
    # themselves. Quantization, text encoder, and VAE are implicit
    # per family (Q3_K_M DiT only — OOM is a clean failure, no
    # fallback rung). Both pair ACE-Step music takes (acestep/cuda:0,
    # sequential residency with the video worker — evict video, render
    # take, evict audio, rebuild video): the ACE planner's long
    # caption-driven takes carry the continuous mood across segments
    # (DESIGN §140 audio continuity), while the finalize SFX dub pairs
    # mmaudio/cuda:0 (MMAudio dubs effects under the soundtrack at
    # finalize, after the video worker has stopped — DESIGN §140 GPU
    # defaults). The workers' own joint audio.wav files are ignored.
    "ltx25": BackendRecord(
        profile="ltx25-704p",
        width=1216,
        height=704,
        fps=24,
        segment_frames=96,
        latent_shape=(1, 128, 16, 19, 11),
        device="cuda:0",
        audio_backend="acestep",
        audio_device="cuda:0",
        sfx_backend="mmaudio",
        sfx_device="cuda:0",
        state_mode="reconstructable_prefix",
        streaming=True,
    ),
    "ltx23": BackendRecord(
        profile="ltx23-704p",
        width=1216,
        height=704,
        fps=24,
        segment_frames=96,
        latent_shape=(1, 128, 16, 19, 11),
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

_DEFAULT_ROW: BackendRecord = BACKEND_REGISTRY["ltx25"]
"""VideoConfig defaults spell this row (issues 025 + 2026-10-02 ltx25 decision)."""


class VideoConfig(BaseModel):
    # Defaults ARE the ltx25 registry row (issue 025) — change the row,
    # not these references. Pinned by tests/test_backend_registry.py.
    backend: VideoBackendName = "ltx25"
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
    # never persisted to the run manifest). When set, every segment's staged
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
    music_style: str = DEFAULT_MUSIC_STYLE
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
    # never persisted to the run manifest). When set, every take uses it instead
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
    both-positive — a half-disabled floor (0 wide x 704 high) is
    meaningless, so the model rejects it. The media consumer that reads
    these floors lands in a later slice; this track only plumbs them
    through TOML + CLI + TUI.

    `use_model_pass` (issue 166) is the model pass: Real-ESRGAN
    upscale + FILM interpolate via `resolve_augment_weights` when
    provisioned (default on — DESIGN §140 GPU defaults pins it to
    cuda:1), ffmpeg floors only when off or when the legs are absent.

    `interp_multiplier` (DESIGN §56) is the FILM frame multiplier for
    the model pass (default 4, matching the validated Comfy
    `video_export.json` recipe); 1 keeps the frame count
    (`(n-1)*1+1 = n`, the interp worker passes frames through), so the
    pass upscales without interpolating.

    `presentation_fps` (slow-mo finalize) pins the shipped frame rate
    instead of the floors rule: with `interp_multiplier=2` on 24fps
    content presented at 32fps, the timeline stretches 1.5x (slow
    motion) instead of lifting fps with minterpolate. `None` (default)
    keeps the floors behavior; TOML `0` also means unset.
    """

    min_fps: int = 24
    min_width: int = 1216
    min_height: int = 704
    use_model_pass: bool = True
    interp_multiplier: int = 4
    presentation_fps: int | None = Field(default=None, ge=1)

    @field_validator("presentation_fps", mode="before")
    @classmethod
    def unset_presentation_zero(cls, value: object) -> object:
        if value == 0:
            return None
        return value

    @field_validator("min_fps", "min_width", "min_height")
    @classmethod
    def non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative (0 disables the floor)")
        return value

    @field_validator("interp_multiplier")
    @classmethod
    def multiplier_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("interp_multiplier must be >= 1 (1 = upscale only, no interpolation)")
        return value

    @model_validator(mode="after")
    def geometry_both_or_neither(self) -> AugmentConfig:
        if (self.min_width == 0) != (self.min_height == 0):
            raise ValueError("min_width and min_height must both be 0 or both positive")
        return self


def parse_min_resolution(raw: str) -> tuple[int, int]:
    """Parse a resolution floor: WxH like "1216x704", or "0" to disable.

    Returns (width, height); "0" (and the equivalent "0x0") returns
    (0, 0). Anything else — wrong shape, non-digits, or a half-disabled
    pair like "0x704" — raises ValueError so CLI/TUI/resolve paths share
    one error source. The full both-or-neither invariant also lives on
    AugmentConfig for direct construction and TOML loads.
    """
    text = raw.strip().lower()
    if text == "0":
        return (0, 0)
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", text)
    if match is None:
        raise ValueError(
            f'invalid resolution {raw!r} (expected WxH like "1216x704" or "0" to disable)'
        )
    width, height = int(match.group(1)), int(match.group(2))
    if (width == 0) != (height == 0):
        raise ValueError(
            f"invalid resolution {raw!r}: width and height must both be 0 or both positive"
        )
    return (width, height)


DirectorBackendName = Literal["qwen", "deterministic", "llama"]
"""Director backend vocabulary (the issue-022 Literal-vocabulary pattern):
the in-process Qwen decider, the weight-free deterministic fallback, and
the llama-server sidecar (DESIGN §140 llama entry — the supervisor spawns
the loopback sidecar and injects its endpoint; the worker routes on
endpoint presence, so the wire backend stays `qwen`)."""

DEFAULT_LLAMA_ENDPOINT = "http://127.0.0.1:8080"
"""Fixed loopback sidecar base URL (contract — isolated to containers)."""


class DirectorConfig(BaseModel):
    # Default decider (A/B-proven 2026-10-01: 10.7s/directive at 67-70
    # tok/s vs 23.4s AWQ): the loopback llama-server sidecar serving the
    # Qwen3.5 GGUF. "qwen" keeps the in-process AWQ path (explicit opt-in
    # for single-GPU/CI boxes without a second GPU); the worker routes on
    # endpoint presence, so the wire backend stays `qwen` either way.
    backend: DirectorBackendName = "llama"
    model_id: str = "Qwen/Qwen3-8B"
    # Decider placement: the unified worker image runs the Qwen decider on
    # cuda:1 (second GPU) via a 4-bit AWQ model; "cpu" keeps the legacy bf16
    # path (explicit opt-out for single-GPU / CI boxes). The worker falls
    # back to CPU with a loud warning when the device is absent.
    device: str = "cuda:1"
    # llama-server sidecar base URL (DESIGN §140 llama entry): the
    # supervisor spawns the loopback sidecar when backend is "llama" and
    # injects this string into the director init + every decide payload
    # (`llama_endpoint` key — the worker's opt-in switch). Fixed loopback
    # by contract (isolated to containers); other backends never read it.
    llama_endpoint: str = DEFAULT_LLAMA_ENDPOINT
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

    @field_validator("llama_endpoint")
    @classmethod
    def _loopback_http(cls, value: str) -> str:
        # Fail fast at the CLI/config layer: without this, a typo survives
        # into TOML, ensures the GGUF, then dies deep in the worker's HTTP
        # client. Scheme check only — the fixed loopback default above is
        # the contract, but the shape (not the host) is what config owns.
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"llama_endpoint must be an http(s) URL (got {value!r})")
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
    # Deprecated (item 1): novelty never rejects, so no cap is read —
    # kept (with validator and TOML line) so older run dirs still load.
    novelty_max_rejections: int = 2
    max_worker_restarts: int = 3
    rpc_timeout_seconds: float = 600.0
    # Thematic drift cadence: the director must propose a novel destination
    # every Nth segment (1 = drift each segment). Non-drift segments hold
    # the current concept via the deterministic fallback (still recorded).
    drift_every_n_segments: int = 1
    # Gauge sampling cadence (Stage C): resource snapshots cost one worker
    # health round-trip per tail (~5 s when a probe times out), so long
    # runs can thin them out. 1 keeps every-segment sampling.
    resource_gauge_interval_segments: int = 1

    @field_validator(
        "blocks_per_prompt_stage",
        "novelty_max_attempts",
        "drift_every_n_segments",
        "resource_gauge_interval_segments",
    )
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


def preset_config(
    run_id: str,
    style: str,
    seed: int,
    video_backend: VideoBackendName = "ltx25",
    director_backend: str = "llama",
    director_device: str = "cuda:1",
) -> ProjectConfig:
    """Build the stock run config directly from creator inputs (no TOML).

    CLI-is-config: `generate` calls this, then layers the flag overrides
    via `resolve_config`. The base carries the dev-box reserve
    (`DEV_MIN_FREE_SPACE_GIB`); every other stock value is the model
    default, and the backend row (geometry + audio/SFX pairing +
    models_dir) plus the director selection resolve through the same
    `resolve_config` every other override path uses — so the preset can
    never drift from the resolver. Raises ValidationError on empty
    style / unknown backend, exactly like the old file round-trip did.
    """
    base = ProjectConfig(
        run_id=run_id,
        style=style,
        seed=seed,
        min_free_space_gib=DEV_MIN_FREE_SPACE_GIB,
    )
    return resolve_config(
        base,
        backend=video_backend,
        director=director_backend,
        director_device=director_device,
    )


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
    use_model_pass: bool | None | UnsetType = Unset,
    interp_multiplier: int | None | UnsetType = Unset,
    presentation_fps: int | None | UnsetType = Unset,
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
    if (
        is_provided(min_fps)
        or is_provided(min_resolution)
        or is_provided(use_model_pass)
        or is_provided(interp_multiplier)
        or is_provided(presentation_fps)
    ):
        resolved_fps = augment.min_fps
        resolved_width = augment.min_width
        resolved_height = augment.min_height
        resolved_model_pass = augment.use_model_pass
        resolved_multiplier = augment.interp_multiplier
        resolved_presentation = augment.presentation_fps
        if is_provided(min_fps):
            resolved_fps = min_fps
        if is_provided(min_resolution):
            resolved_width, resolved_height = parse_min_resolution(min_resolution)
        if is_provided(use_model_pass):
            resolved_model_pass = use_model_pass
        if is_provided(interp_multiplier):
            resolved_multiplier = interp_multiplier
        if is_provided(presentation_fps):
            resolved_presentation = presentation_fps
        augment = AugmentConfig(
            min_fps=resolved_fps,
            min_width=resolved_width,
            min_height=resolved_height,
            use_model_pass=resolved_model_pass,
            interp_multiplier=resolved_multiplier,
            presentation_fps=resolved_presentation,
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
    use_model_pass: bool | None | UnsetType = Unset,
    interp_multiplier: int | None | UnsetType = Unset,
    presentation_fps: int | None | UnsetType = Unset,
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
        use_model_pass=use_model_pass,
        interp_multiplier=interp_multiplier,
        presentation_fps=presentation_fps,
    )


# Audio backend paired with each video row: ltxv/causvid/ltx25/ltx23 get
# the real ACE-Step music stack (models + device mirror the video row —
# the ACE planner's long caption-driven takes carry the continuous mood);
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

    ltxv/causvid pair the MMAudio stack on cuda:0 (SFX/music on by
    default on GPU); ltx25/ltx23 pair ACE-Step music (continuous takes)
    plus MMAudio SFX on cuda:0 (the finalize dub runs after the video
    worker stops — DESIGN §140 GPU defaults); fake keeps the fake-noise
    backend on CPU so CPU-only test runs never touch weights.
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

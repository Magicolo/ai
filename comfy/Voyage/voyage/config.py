"""Project configuration (DESIGN task group B).

CLI-is-config: the effective `ProjectConfig` is built directly from the
`configure` flags (`preset_config` + `resolve_config`) and persisted as
structured data inside `manifest.json` — there is no TOML layer.
Validation lives in the model validators; the manifest digest traces
runs to their exact configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeGuard, TypeVar, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from voyage.models import DEFAULT_MUSIC_STYLE

VideoBackendName = Literal["fake", "ltxv", "causvid", "ltx25", "ltx23"]
"""Video backend vocabulary (issue 022): every backend field, the registry,
and the streaming set are keyed by this — a typo fails at typecheck
instead of after GPU init."""

AudioBackendName = Literal["fake", "acestep"]
"""Audio backend vocabulary (issue 022): fake sine vs the ACE-Step music stack."""

SfxBackendName = Literal["fake", "mmaudio"]
"""SFX backend vocabulary: fake noise vs the MMAudio effects stack."""

SfxModelSize = Literal["small_44k", "medium_44k", "large_44k_v2"]
"""MMAudio 44 kHz variant vocabulary (mirrors audio.mmaudio_sfx)."""

InterpBackendName = Literal["film", "rife"]
"""Interpolation backend vocabulary: FILM (flow-once per pair, 7-level
pyramid) vs RIFE v4.25 (per-moment IFNet forward, ~5.7x less activation
memory). RIFE is the default (Phase-0 A/B: ~16.8x per-pair at 2048x1152,
eyeball-identical on line art); FILM stays for hero/archival renders."""

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
    # (voyage/workers/video_ltx25.py / video_ltx23.py). ltx25 committed
    # segments are Mode A two-stage 1216x704@24 (stage-1
    # 608x352x257 8-sigma distilled, 2x latent upscale, stage-2
    # 3-step refine); 257-frame windows with a 25-frame frozen
    # prefix carry commit 232 novel (257 is the native upstream ceiling
    # and a 2026-10-06 GPU sweep proved it fits VRAM at ~13.4 GiB peak;
    # ltx23 stays 121f/96-novel on its own row below).
    # latent_shape is the stage-1 video latent [B, C, T, H, W] =
    # [1, 128, (257-1)//8+1, 608//32, 352//32] (EmptyLTXVLatentVideo
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
    # defaults). The workers commit video only (all-deferred audio):
    # segment audio comes from the ACE-Step music takes + finalize SFX
    # dub, never from the worker.
    "ltx25": BackendRecord(
        profile="ltx25-704p",
        width=1216,
        height=704,
        fps=24,
        segment_frames=232,
        latent_shape=(1, 128, 33, 19, 11),
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


@dataclass(frozen=True)
class DefinitionTier:
    """One definition tier for a backend: native geometry + profile name.

    `latent_shape=None` keeps the backend row's latent (fake/ltxv/causvid
    derive or ignore it); a tuple overrides it (ltx25/ltx23 low tier
    halves both stage-1 axes, so the stage-1 video latent shrinks from
    [1, 128, 33, 19, 11] to [1, 128, 33, 12, 7] — the T=33 frame axis
    rides the 257f production window on every tier, only the spatial
    axes shrink).
    """

    width: int
    height: int
    profile: str
    latent_shape: tuple[int, ...] | None


DEFINITION_TIERS: dict[VideoBackendName, dict[str, DefinitionTier]] = {
    # Lowest / medium / highest native reasonable resolution per backend
    # (`voyage configure --low-definition / --medium-definition /
    # --high-definition`): tiers stay on divisor-clean native geometry
    # the worker accepts (fake /32-able, ltxv /32, ltx25/ltx23 /64);
    # causvid is fixed 832x480, so all tiers are identical by design.
    "fake": {
        "low": DefinitionTier(width=512, height=288, profile="fake-288p", latent_shape=None),
        "medium": DefinitionTier(width=1024, height=576, profile="fake-576p", latent_shape=None),
        "high": DefinitionTier(width=768, height=432, profile="fake-432p", latent_shape=None),
    },
    "ltxv": {
        "low": DefinitionTier(width=512, height=320, profile="ltxv-320p", latent_shape=None),
        "medium": DefinitionTier(width=1024, height=576, profile="ltxv-576p", latent_shape=None),
        "high": DefinitionTier(width=768, height=512, profile="ltxv-512p", latent_shape=None),
    },
    "causvid": {
        "low": DefinitionTier(width=832, height=480, profile="causvid-480p", latent_shape=None),
        "medium": DefinitionTier(width=832, height=480, profile="causvid-480p", latent_shape=None),
        "high": DefinitionTier(width=832, height=480, profile="causvid-480p", latent_shape=None),
    },
    "ltx25": {
        "low": DefinitionTier(
            width=768,
            height=448,
            profile="ltx25-448p",
            latent_shape=(1, 128, 33, 12, 7),
        ),
        "medium": DefinitionTier(
            width=1024,
            height=576,
            profile="ltx25-576p",
            latent_shape=(1, 128, 33, 16, 9),
        ),
        "high": DefinitionTier(
            width=1216,
            height=704,
            profile="ltx25-704p",
            latent_shape=(1, 128, 33, 19, 11),
        ),
    },
    "ltx23": {
        "low": DefinitionTier(
            width=768,
            height=448,
            profile="ltx23-448p",
            latent_shape=(1, 128, 16, 12, 7),
        ),
        "medium": DefinitionTier(
            width=1024,
            height=576,
            profile="ltx23-576p",
            latent_shape=(1, 128, 16, 16, 9),
        ),
        "high": DefinitionTier(
            width=1216,
            height=704,
            profile="ltx23-704p",
            latent_shape=(1, 128, 16, 19, 11),
        ),
    },
}
"""Backend name → low/medium/high definition tiers (geometry + profile + latent
override). High tiers equal their BACKEND_REGISTRY rows (pinned by
tests/test_backend_registry.py); the resolver applies the tier AFTER
the backend preset, so `--backend` then tier always agrees."""


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
    # Track B enhancer (DESIGN §140, default ON since 2026-10-05: seams
    # stay invisible and adherence is no worse with expansion): when True,
    # the supervisor expands the staged prompts through the llama-server
    # sidecar (`voyage.prompt_enhancer`, free-form text, no
    # response_format) at the top of `_render_video` — main commit thread,
    # before the worker RPC, never overlapping the video forward. False
    # keeps payloads byte-identical (`--no-prompt-enhance` opt-out).
    # Needs the sidecar: deterministic-director runs degrade to input
    # text (fail-soft, zero tokens, recorded in `prompt_enhanced`).
    prompt_enhance: bool = True

    @field_validator(
        "width", "height", "fps", "segment_frames", "blocks_per_segment", "local_attn_size"
    )
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


#: Ceiling for `AudioConfig.final_overlap_fraction` (finalize blend): at most
#: one whole shortest segment may be re-sliced into the overlap — the blend
#: extends each window by half the overlap per side, so 1.0 centers a full
#: segment-length crossfade on the joint. Takes are 30-60s of continuous
#: music, so take joints (which land on segment boundaries) stay
#: well-defined even at generous overlaps; beyond 1.0 the "joint" would
#: swallow neighboring windows instead of joining them.
MAX_FINAL_OVERLAP_FRACTION = 1.0


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
    # adjoining segment, cap). The blend applies at finalize from take
    # re-slices, so no commit-format change and no A/V drift (overlap
    # content comes from the takes, not by shortening the timeline).
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
    ladder (slice 2c) locks the deployed value. `dual_pan` renders the
    spatialized pair (two same-caption seeds, ±75% constant-power pan,
    mixed with music to stereo); the second channel completes a legacy
    single-bed run instead of rebuilding it. `sfx_caption` pins one
    caption for the whole timeline (whole-timeline repair for runs
    committed before SFX captions existed; None = per-window director
    captions). `num_workers` shards small_44k across cuda:0+cuda:1
    when 2 (needs 2 visible GPUs, fails fast otherwise).
    """

    backend: SfxBackendName = "fake"
    device: str = "cpu"
    models_dir: str = "/models"
    model_size: SfxModelSize = "large_44k_v2"
    dual_pan: bool = True
    sfx_caption: str | None = None
    num_workers: int = 1

    @field_validator("num_workers", mode="before")
    @classmethod
    def workers_pair(cls, value: object) -> object:
        # Shard vocabulary (mirrors the worker gate in sfx_finalize):
        # 1 = one worker, 2 = shard small_44k across cuda:0+cuda:1.
        # `before` on purpose: an `after` validator sees the parsed int,
        # so a stored `true` would coerce to 1 silently — a bool is never
        # a worker count, reject it before parsing.
        if isinstance(value, bool) or value not in (1, 2):
            raise ValueError(f"sfx workers must be 1 or 2 (got {value!r})")
        return value


class AugmentConfig(BaseModel):
    """Finalize-time explicit quality multipliers.

    No minimum floors: quality is specified explicitly per run.
    `upscale` scales the shipped resolution against the probed source
    (`2` doubles width and height); `interpolate` scales the shipped
    frame count the same way FILM always has (`(n-1)*m+1`). Both
    default to `1` (ship the source geometry as-is, ffmpeg only).

    The model pass (Real-ESRGAN upscale + interpolate via
    `resolve_augment_weights` when provisioned) runs exactly when the
    multipliers demand work — `upscale > 1 or interpolate > 1` — and
    the legs are present; otherwise finalize ships via ffmpeg. There
    is no `use_model_pass` / `--no-augment` knob: `1/1` means no work.

    `interp_backend` selects the interpolation engine: `"rife"` (default,
    RIFE v4.25 — ~16.8x faster per pair than FILM at 2048x1152, identical
    eyeball on line art) or `"film"` (legacy FILM port, kept for
    hero/archival renders). The backend rides the sidecar weights key,
    so switching backends re-renders the interp leg by design.

    `presentation_fps` pins the shipped frame rate: with
    `interpolate=2` on 24fps content presented at 32fps, the timeline
    stretches 1.5x (slow motion) instead of lifting the frame rate.
    `None` (default, unset) ships `round(source_fps * interpolate)`.
    """

    upscale: int = 1
    interpolate: int = 1
    interp_backend: InterpBackendName = "rife"
    presentation_fps: int | None = Field(default=None, ge=1)

    @field_validator("upscale", "interpolate")
    @classmethod
    def multiplier_in_worker_vocab(cls, value: int) -> int:
        # Fail fast at configure time: the upscale worker + poller only
        # serve (1, 2, 4) (one x4 SRVGG pass covers 4/2/1), and FILM
        # multipliers outside small ints are never sensible. Without this,
        # a 3 slips into TOML and dies deep in the worker.
        if value not in (1, 2, 4):
            raise ValueError(
                f"upscale/interpolate must be 1, 2, or 4 (1 = no work on that axis; got {value})"
            )
        return value


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
    blocks_per_prompt_stage: int = 3
    novelty_threshold: float = 0.85
    novelty_max_attempts: int = 3
    max_worker_restarts: int = 3
    rpc_timeout_seconds: float = 600.0
    # Thematic drift cadence: the director must propose a novel destination
    # every Nth segment (1 = drift each segment). Non-drift segments hold
    # the current concept via the deterministic fallback (still recorded).
    drift_every_n_segments: int = 1
    # Scene-cut cadence: every Nth segment renders fresh (a strong visual
    # change) instead of continuing from the worker tail. 1 cuts every
    # segment (no continuation at all); larger N holds shots longer.
    scene_cut_every_n_segments: int = 3
    # Gauge sampling cadence (Stage C): resource snapshots cost one worker
    # health round-trip per tail (~5 s when a probe times out), so long
    # runs can thin them out. 1 keeps every-segment sampling.
    resource_gauge_interval_segments: int = 1

    @field_validator(
        "blocks_per_prompt_stage",
        "novelty_max_attempts",
        "drift_every_n_segments",
        "scene_cut_every_n_segments",
        "resource_gauge_interval_segments",
    )
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
        if not math.isfinite(value) or value <= 0:
            raise ValueError("must be a finite positive number")
        return value

    @field_validator("novelty_threshold")
    @classmethod
    def unit_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("novelty_threshold must be within [0, 1]")
        return value


SPEC_MIN_FREE_SPACE_GIB = 20.0
"""Spec default reserve (DESIGN §70/§140): hand-written TOML omitting the key gets this."""

DEV_MIN_FREE_SPACE_GIB = 5.0
"""Init-generated TOML default (issue 052): small dev boxes validate at 5 GiB."""


class ProjectConfig(BaseModel):
    name: str = "voyage"
    style: str = ""
    seed: int = 0
    min_free_space_gib: float = SPEC_MIN_FREE_SPACE_GIB
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    sfx: SfxConfig = Field(default_factory=SfxConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    director: DirectorConfig = Field(default_factory=DirectorConfig)
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)

    @field_validator("style")
    @classmethod
    def style_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("style must be a non-empty human-owned style string")
        return value


def preset_config(
    name: str,
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
        name=name,
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
    director: str | None | UnsetType = Unset,
    director_device: str | None | UnsetType = Unset,
    blocks: int | None | UnsetType = Unset,
    take_seconds: float | None | UnsetType = Unset,
    quantization: str | None | UnsetType = Unset,
    beats_per_segment: int | None | UnsetType = Unset,
    definition: str | None | UnsetType = Unset,
    drift_every_n_segments: int | None | UnsetType = Unset,
    scene_cut_every_n_segments: int | None | UnsetType = Unset,
    music_caption: str | None | UnsetType = Unset,
    video_caption: str | None | UnsetType = Unset,
    prompt_enhance: bool | None | UnsetType = Unset,
    upscale: int | None | UnsetType = Unset,
    interpolate: int | None | UnsetType = Unset,
    presentation_fps: int | None | UnsetType = Unset,
    interp_backend: InterpBackendName | None | UnsetType = Unset,
    sfx_backend: SfxBackendName | None | UnsetType = Unset,
    sfx_device: str | None | UnsetType = Unset,
    sfx_model_size: SfxModelSize | None | UnsetType = Unset,
    sfx_caption: str | None | UnsetType = Unset,
    sfx_workers: int | None | UnsetType = Unset,
    sfx_dual_pan: bool | None | UnsetType = Unset,
) -> ProjectConfig:
    """Single configuration resolver (issue 022): backend preset,
    then targeted overrides — in that order, so explicit flags always
    win over profiles.

    Pure: returns a new config, never mutates. Rebuilds submodels through
    their constructors so invalid overrides (blocks=0, negative takes)
    raise ValidationError instead of silently corrupting the run. Unknown
    backends raise ValueError (same message as the old preset lookup).
    `with_video_backend` survives below as a thin wrapper for its
    existing callers.

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
        # Issue 244: the "/models" roots below are the image-fixed mount
        # (no --models-dir CLI surface exists and every default is already
        # "/models", so this is a no-op today) — but a customized root is
        # silently clobbered on every backend switch. Preserve custom
        # roots here before adding any models-dir override path.
        audio = AudioConfig(
            **{**audio.model_dump(), **_audio_preset(backend), "models_dir": "/models"}
        )
        sfx = SfxConfig(**{**sfx.model_dump(), **_sfx_preset(backend), "models_dir": "/models"})
    if is_provided(definition):
        # Definition tier resolves against the already-preset backend
        # (--backend first, then tier), so the geometry always belongs
        # to the effective backend. Unknown tiers raise ValueError.
        video = VideoConfig(
            **{**video.model_dump(), **_definition_preset(video.backend, definition)}
        )
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
    if is_provided(scene_cut_every_n_segments):
        voyage_config = VoyageConfig(
            **{
                **voyage_config.model_dump(),
                "scene_cut_every_n_segments": scene_cut_every_n_segments,
            }
        )
    if is_provided(music_caption):
        audio = AudioConfig(**{**audio.model_dump(), "music_caption": music_caption})
    if is_provided(video_caption):
        video = VideoConfig(**{**video.model_dump(), "video_caption": video_caption})
    if is_provided(prompt_enhance):
        video = VideoConfig(**{**video.model_dump(), "prompt_enhance": prompt_enhance})
    if is_provided(sfx_dual_pan):
        sfx = SfxConfig(**{**sfx.model_dump(), "dual_pan": sfx_dual_pan})
    if is_provided(sfx_backend):
        sfx = SfxConfig(**{**sfx.model_dump(), "backend": sfx_backend})
    if is_provided(sfx_device):
        sfx = SfxConfig(**{**sfx.model_dump(), "device": sfx_device})
    if is_provided(sfx_model_size):
        sfx = SfxConfig(**{**sfx.model_dump(), "model_size": sfx_model_size})
    if is_provided(sfx_caption):
        sfx = SfxConfig(**{**sfx.model_dump(), "sfx_caption": sfx_caption})
    if is_provided(sfx_workers):
        sfx = SfxConfig(**{**sfx.model_dump(), "num_workers": sfx_workers})
    augment = config.augment
    if (
        is_provided(upscale)
        or is_provided(interpolate)
        or is_provided(presentation_fps)
        or is_provided(interp_backend)
    ):
        resolved_upscale = augment.upscale
        resolved_interpolate = augment.interpolate
        resolved_presentation = augment.presentation_fps
        resolved_backend = augment.interp_backend
        if is_provided(upscale):
            resolved_upscale = upscale
        if is_provided(interpolate):
            resolved_interpolate = interpolate
        if is_provided(presentation_fps):
            resolved_presentation = presentation_fps
        if is_provided(interp_backend):
            if interp_backend in ("film", "rife"):
                resolved_backend = cast(InterpBackendName, interp_backend)
            else:
                raise ValueError(
                    f"unknown interp backend {interp_backend!r} (expected 'film' or 'rife')"
                )
        augment = AugmentConfig(
            upscale=resolved_upscale,
            interpolate=resolved_interpolate,
            presentation_fps=resolved_presentation,
            interp_backend=resolved_backend,
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


def _definition_preset(tier_backend: str, tier: str) -> dict[str, str | int | list[int]]:
    """Geometry override for a definition tier (derived from DEFINITION_TIERS).

    Applied after the backend preset, keyed by the resolved backend so
    `--backend X --low-definition` (or --medium-definition /
    --high-definition) always yields X's tier geometry.
    A `None` latent keeps the backend row's latent_shape.
    """
    if tier not in ("low", "medium", "high"):
        raise ValueError(f"unknown definition tier {tier!r} (expected 'low', 'medium' or 'high')")
    try:
        tiers = DEFINITION_TIERS[tier_backend]  # type: ignore[index]
    except KeyError:
        raise ValueError(f"unknown backend {tier_backend!r} for definition tier") from None
    tier_geometry = tiers[tier]
    row = BACKEND_REGISTRY[tier_backend]  # type: ignore[index]
    latent = tier_geometry.latent_shape
    if latent is None:
        latent = row.latent_shape
    return {
        "width": tier_geometry.width,
        "height": tier_geometry.height,
        "profile": tier_geometry.profile,
        "latent_shape": list(latent),
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
        raise ValueError(f"unknown video backend {backend!r} (known: {known})") from None


def with_video_backend(config: ProjectConfig, backend: VideoBackendName) -> ProjectConfig:
    """Return a copy of config with the video-backend preset applied.

    Thin wrapper over resolve_config (issue 025) — kept for existing
    callers and tests, which pin its behavior (preset geometry + audio
    pairing + purity + ValueError on unknown backends).
    """
    return resolve_config(config, backend=backend)

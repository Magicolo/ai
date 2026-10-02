"""ffmpeg/ffprobe wrappers + validation + finalizer (DESIGN §§54-57, I).

All invocations use argument lists — never shell strings. The finalizer
never mutates source segment files; it publishes the final path
atomically.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from voyage.config import AudioConfig

from voyage import paths
from voyage.atomic import atomic_copy, atomic_write_json, read_json
from voyage.augment import CRF_MAXIMUM as _AUGMENT_CRF_MAXIMUM
from voyage.augment import CRF_MINIMUM as _AUGMENT_CRF_MINIMUM
from voyage.augment import interpolated_frame_count as interpolated_frame_count
from voyage.errors import MediaError
from voyage.media_audio import ABSORPTION_EPSILON_SECONDS as ABSORPTION_EPSILON_SECONDS
from voyage.media_audio import AV_ALIGNMENT_TOLERANCE_SECONDS as AV_ALIGNMENT_TOLERANCE_SECONDS
from voyage.media_audio import (
    DURATION_FRAME_ESTIMATE_SLACK_FRAMES as DURATION_FRAME_ESTIMATE_SLACK_FRAMES,
)
from voyage.media_audio import FFMPEG_TIMEOUT_SECONDS as FFMPEG_TIMEOUT_SECONDS
from voyage.media_audio import FPS_MATCH_TOLERANCE as FPS_MATCH_TOLERANCE
from voyage.media_audio import MAX_SLICES_PER_WINDOW as MAX_SLICES_PER_WINDOW
from voyage.media_audio import MIN_FADE_GRAPH_SECONDS as MIN_FADE_GRAPH_SECONDS
from voyage.media_audio import MIN_OVERLAP_BLEND_SECONDS as MIN_OVERLAP_BLEND_SECONDS
from voyage.media_audio import MIN_SLICE_PIECE_SECONDS as MIN_SLICE_PIECE_SECONDS
from voyage.media_audio import PAIR_BLEND_INPUT_COUNT as PAIR_BLEND_INPUT_COUNT
from voyage.media_audio import _audio_duration_seconds as _audio_duration_seconds
from voyage.media_audio import _blend_fade_seconds as _blend_fade_seconds
from voyage.media_audio import _blend_pair as _blend_pair
from voyage.media_audio import _cached_slice_take as _cached_slice_take
from voyage.media_audio import _check_segment_committed as _check_segment_committed
from voyage.media_audio import _concat_fallback_audio as _concat_fallback_audio
from voyage.media_audio import _join_audio_single_graph as _join_audio_single_graph
from voyage.media_audio import _probe_video_fps as _probe_video_fps
from voyage.media_audio import _segment_timeline as _segment_timeline
from voyage.media_audio import _sha256_file as _sha256_file
from voyage.media_audio import _slice_cache_key as _slice_cache_key
from voyage.media_audio import _take_joint_fade as _take_joint_fade
from voyage.media_audio import _verify_segment as _verify_segment
from voyage.media_audio import assemble_segment_audio as assemble_segment_audio
from voyage.media_audio import av_drift_seconds as av_drift_seconds
from voyage.media_audio import build_final_audio as build_final_audio
from voyage.media_audio import check_av_alignment as check_av_alignment
from voyage.media_audio import check_free_space as check_free_space
from voyage.media_audio import probe as probe
from voyage.media_audio import probed_take_seconds as probed_take_seconds
from voyage.media_audio import run_capture as run_capture
from voyage.media_audio import slice_take as slice_take
from voyage.media_audio import validate_audio as validate_audio
from voyage.media_audio import validate_video as validate_video

# Moved to voyage.media_audio (issue 036): AV_ALIGNMENT_TOLERANCE_SECONDS,
# FPS_MATCH_TOLERANCE, MIN_FADE_GRAPH_SECONDS, MIN_OVERLAP_BLEND_SECONDS,
# ABSORPTION_EPSILON_SECONDS, MIN_SLICE_PIECE_SECONDS, MAX_SLICES_PER_WINDOW,
# DURATION_FRAME_ESTIMATE_SLACK_FRAMES, PAIR_BLEND_INPUT_COUNT,
# FFMPEG_TIMEOUT_SECONDS + av_drift_seconds, check_av_alignment,
# check_free_space, run_capture, probe, validate_video, _probe_video_fps,
# validate_audio, slice_take, _take_joint_fade, probed_take_seconds,
# assemble_segment_audio, _sha256_file, _verify_segment,
# _check_segment_committed, _segment_timeline, _slice_cache_key,
# _cached_slice_take, _concat_fallback_audio, _audio_duration_seconds,
# _blend_fade_seconds, _blend_pair, _join_audio_single_graph,
# build_final_audio - facades above re-export them verbatim.


# Floor for the final presentation frame rate (user decision 2026-09-25:
# the shipped video is always >= 24fps). Sub-24fps sources (CausVid native
# 16fps) are motion-interpolated up; sources already at/above the floor
# keep the plain fps filter (no behavior change).
PRESENTATION_MIN_FPS = 24

#: Presentation floors for the Track B augment path (coming config): the
#: shipped video is always >= 24fps and covers 1216x704. `finalize_run`
#: and `FinalizeOptions` default to these so headless/legacy callers get
#: the same presentation without a config round-trip.
AUGMENT_DEFAULT_MIN_FPS = 24
AUGMENT_DEFAULT_MIN_WIDTH = 1216
AUGMENT_DEFAULT_MIN_HEIGHT = 704


@dataclass
class AugmentPlan:
    """Presentation geometry/fps the finalizer must produce (Track B).

    Why this exists: segment videos render at backend-native geometry
    (CausVid 832x480@16, LTXV 768x512@24, fake 768x432@24) but the
    shipped video must always cover the presentation floors. The plan is
    pure math over probed source + requested target + floors, so unit
    tests pin it without ffmpeg and `finalize_run` just renders it.
    """

    out_w: int
    out_h: int
    out_fps: int
    needs_reencode: bool
    needs_minterpolate: bool


def plan_augmentation(
    source_w: int,
    source_h: int,
    source_fps: float,
    target_w: int,
    target_h: int,
    requested_fps: int,
    min_fps: int | None,
    min_width: int | None,
    min_height: int | None,
    interp_multiplier: int = 1,
) -> AugmentPlan:
    """Compute the presentation box/fps for one finalize (pure, Track B).

    `effective_fps = max(requested, min_fps or 0, PRESENTATION_MIN_FPS)`;
    geometry is `max(target, min, source)` per axis — the output box always
    covers the requested target, the floors, AND the probed source, preserving
    aspect downstream via the scale-to-fit + pad vf (never stretched). Floors
    are a minimum quality requirement, never a ceiling: a target already above
    them is kept as-is ("minimal upscale"), segments already above
    target+floors ship at native spec (no downscaling — rendered compute is
    never thrown away), and a 0/None floor disables that axis (the 24fps
    `PRESENTATION_MIN_FPS` still applies — 0 disables the 24fps floor,
    not the shipped-video guarantee).

    `needs_minterpolate` is True only for an fps lift (source + 0.5 <
    out — motion interpolation); an fps drop uses the plain fps filter.
    When the model pass interpolates (`interp_multiplier > 1`), the lift
    is measured against the interpolated rate (`source * multiplier`):
    presenting 24fps x2 content at 32fps stretches the timeline (slow
    motion) instead of synthesizing more frames.
    `needs_reencode` covers any pixel/timing change (dims differ, fps
    differs past 0.5 either way, lift, or unknown source fps) and gates
    the stream-copy fast path off.
    """
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"target geometry must be positive (got {target_w}x{target_h})")
    if requested_fps <= 0:
        raise ValueError(f"requested fps must be positive (got {requested_fps})")
    if interp_multiplier < 1:
        raise ValueError(f"interp_multiplier must be >= 1 (got {interp_multiplier})")
    floor_fps = int(min_fps or 0)
    floor_w = int(min_width or 0)
    floor_h = int(min_height or 0)
    if floor_fps < 0 or floor_w < 0 or floor_h < 0:
        raise ValueError(f"augment floors must be >= 0 (got {min_fps}/{min_width}/{min_height})")
    out_fps = max(int(requested_fps), floor_fps, PRESENTATION_MIN_FPS)
    out_w = max(int(target_w), floor_w, int(source_w))
    out_h = max(int(target_h), floor_h, int(source_h))
    source_fps_value = float(source_fps)
    interpolated_rate = source_fps_value * interp_multiplier
    needs_minterpolate = source_fps_value > 0 and out_fps > interpolated_rate + FPS_MATCH_TOLERANCE
    fps_mismatch = source_fps_value <= 0 or abs(out_fps - source_fps_value) > FPS_MATCH_TOLERANCE
    needs_reencode = bool(
        needs_minterpolate or fps_mismatch or int(source_w) != out_w or int(source_h) != out_h
    )
    return AugmentPlan(
        out_w=out_w,
        out_h=out_h,
        out_fps=out_fps,
        needs_reencode=needs_reencode,
        needs_minterpolate=needs_minterpolate,
    )


def slowmo_factor(source_fps: float, interp_multiplier: int, out_fps: int) -> float:
    """Timeline stretch of a slow-motion present (pure).

    Interpolating `source_fps` content by `interp_multiplier` and
    presenting at `out_fps` stretches wall-clock by
    `source_fps * multiplier / out_fps` (24fps x2 at 32fps = 1.5x).
    1.0 means no stretch (today's behavior).
    """
    if source_fps <= 0 or interp_multiplier < 1 or out_fps <= 0:
        raise ValueError(
            f"slowmo inputs must be positive (got {source_fps}/{interp_multiplier}/{out_fps})"
        )
    return float(source_fps) * interp_multiplier / out_fps


def presentation_stretch(source_fps: float, interp_multiplier: int, out_fps: int) -> float:
    """Stretch for one finalize, tolerating an unprobable source (pure).

    `_probe_video_fps` returns 0.0 when the first segment's fps is
    absent/unparseable, and `plan_augmentation` handles that gracefully
    (fps mismatch → re-encode). `slowmo_factor` rightly rejects
    non-positive inputs, so this wrapper maps them to 1.0 (no stretch):
    the tensor path can never arm on fps 0.0, and an unconditional call
    would crash a finalize that previously proceeded.
    """
    if source_fps <= 0:
        return 1.0
    return slowmo_factor(source_fps, interp_multiplier, out_fps)


#: Stretch values within this of 1.0 count as no-stretch (float noise
#: from `slowmo_factor` must not arm the slow-mo path).
SLOWMO_STRETCH_TOLERANCE = 1e-9


def slowmo_video_active(tensor: bool, presentation_fps: int | None, stretch: float) -> bool:
    """True when the tensor present must retime instead of decimate (pure).

    All three must hold: the model pass actually interpolated (tensor
    path), the user explicitly requested a presentation fps (opt-in —
    default runs keep the legacy fps-filter timeline byte-identical), and
    the stretch differs from 1.0. A bare `fps=` filter drops frames to
    hold the duration; slow-mo keeps every FILM frame via setpts.
    """
    return tensor and presentation_fps is not None and abs(stretch - 1.0) > SLOWMO_STRETCH_TOLERANCE


def tensor_presentation_vf(
    out_w: int, out_h: int, out_fps: int, stretch: float, *, slowmo: bool
) -> str:
    """Presentation vf for the tensor intermediate (pure).

    Slow-mo prefixes `setpts=<stretch>*PTS` (uniform retime — every
    interpolated frame survives, the timeline stretches); otherwise the
    legacy scale/pad/setsar/fps chain (fps filter holds the duration).
    """
    head = f"setpts={stretch}*PTS," if slowmo else ""
    return (
        f"{head}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={out_fps}"
    )


def tensor_path_armed(
    *,
    model_selected: bool,
    weights_present: bool,
    source_fps: float,
    needs_reencode: bool,
    devices_available: bool,
) -> bool:
    """Tensor-path gate shared by `finalize_run` and the parallel pre-fork.

    Pure: the model pass interpolates iff the knob selected legs, weights
    resolved, the source has a real fps, the plan flags work, and a model
    device is visible. One predicate in one place — the pre-fork ACE
    render must predict exactly what Thread A will decide, or the takes
    ledger lands on the wrong timeline.
    """
    return bool(
        model_selected
        and weights_present
        and source_fps > 0
        and needs_reencode
        and devices_available
    )


def presentation_setup_facts(
    plan: AugmentPlan,
    *,
    min_fps: int | None,
    min_width: int | None,
    min_height: int | None,
) -> dict[str, object]:
    """§104 setup facts for the finalize presentation floors (issue 194).

    Pure: the floor triple as given plus the resolved presentation plan,
    so benchmark/soak reports can record the dominant finalize variable
    instead of leaving re-encode-vs-stream-copy unexplained. HOOK FOR THE
    OBSERVE TRACK (`voyage/cli_observe.py` is out of this change's scope):
    spread these facts into `_benchmark_env()` (or alongside
    `_video_geometry_setup()`) at the `cmd_benchmark`/`cmd_soak` setup
    sites so every §104 setup block carries them.
    """
    return {
        "min_fps": min_fps,
        "min_width": min_width,
        "min_height": min_height,
        "out_w": plan.out_w,
        "out_h": plan.out_h,
        "out_fps": plan.out_fps,
        "needs_reencode": plan.needs_reencode,
        "needs_minterpolate": plan.needs_minterpolate,
    }


def _probe_video_geometry(info: dict[str, Any]) -> tuple[int, int]:
    """Source WxH from ffprobe info; (0, 0) when absent/unparseable.

    Unknown geometry forces the augment re-encode path (the plan treats
    0 as "differs from any positive out box"), so callers never
    stream-copy blind.
    """
    streams = info.get("streams", [])
    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None,
    )
    if video is None:
        return (0, 0)
    try:
        return (int(video.get("width", 0) or 0), int(video.get("height", 0) or 0))
    except (ValueError, TypeError):
        return (0, 0)


JointStyle = Literal["blend", "hard-splice"]
"""Audio-joint rendering for the final mix (issues 045, 046).

`blend` re-slices the takes ledger with a proportional overlap crossfade
(the default); `hard-splice` concatenates the per-segment previews with
no blend (the legacy behavior, clicks included). An explicit style
replaces the old `overlap_fraction=0` encoding, so call sites state the
intent instead of smuggling it through a zero.
"""

#: Default h264 quality for the finalize encode (issues 050). Matches the
#: validated Comfy `video_export.json` recipe (crf 15) and the chunk
#: encoder in `voyage.augment.ffmpeg_encode_chunk` (same default), so the
#: shipped video never silently uses ffmpeg's default CRF 23.
FINALIZE_CRF_DEFAULT = 15

#: Lowest/highest h264 CRF (issues 050, 083). Single ladder home is
#: `voyage.augment` (`CRF_MINIMUM`/`CRF_MAXIMUM`); these aliases keep the
#: `FINALIZE_CRF_*` names for existing imports (the old "stated here so
#: `media` stays stdlib-only without importing" comment is stale —
#: `voyage.augment` is stdlib-only too, so the import is free).
FINALIZE_CRF_MINIMUM = _AUGMENT_CRF_MINIMUM
FINALIZE_CRF_MAXIMUM = _AUGMENT_CRF_MAXIMUM

#: Default x264 speed/quality trade-off (issues 050). Keeps the validated
#: `veryfast` recipe; slower presets are opt-in via `FinalizeOptions`.
FINALIZE_PRESET_DEFAULT = "veryfast"

#: Allowed x264 presets (issues 050). The full ffmpeg `-preset` vocabulary
#: for libx264, so validation rejects typos before an ffmpeg spawn fails.
FINALIZE_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)


def validate_crf(value: int) -> int:
    """Validate a finalize CRF (issues 050).

    Ints only (bools rejected — `True` is `1` but never a quality knob);
    range is the h264 0..51 ladder. Returns the value for `__post_init__`
    and scalar-override paths to share.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"crf must be an int (got {value!r})")
    if value < FINALIZE_CRF_MINIMUM or value > FINALIZE_CRF_MAXIMUM:
        raise ValueError(
            f"crf must be within [{FINALIZE_CRF_MINIMUM}, {FINALIZE_CRF_MAXIMUM}] (got {value})"
        )
    return value


def validate_preset(value: str) -> str:
    """Validate a finalize x264 preset (issues 050)."""
    if not isinstance(value, str):
        raise TypeError(f"preset must be a str (got {value!r})")
    if value not in FINALIZE_PRESETS:
        raise ValueError(f"preset must be one of {sorted(FINALIZE_PRESETS)} (got {value!r})")
    return value


def escape_concat_path(path: Path | str) -> str:
    """Escape a path for an ffmpeg concat-demuxer `file '...'` line (053).

    A single quote inside the single-quoted value closes the quoting, so
    it becomes `'\\''` (close, escaped literal, reopen) per the ffmpeg
    concat-demuxer docs. Spaces/`$`/double quotes need no escaping inside
    the single quotes with `-safe 0`; arg-lists already keep them safe on
    the supervisor side.
    """
    return str(path).replace("'", "'\\''")


def write_concat_list(entries: list[Path], dest: Path) -> Path:
    """Write a concat-demuxer list with quoting-safe entries (053).

    Every entry goes through `escape_concat_path`, so adversarial run
    directories (`o'brien`, spaces) produce a parseable list instead of a
    truncated `file '...'` line. Returns `dest` for call-site chaining.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(f"file '{escape_concat_path(entry)}'\n" for entry in entries),
        encoding="utf-8",
    )
    return dest


@dataclass
class FinalizeOptions:
    """Explicit finalize knobs (issue 045).

    Twelve positional scalars made `finalize_run`'s semantics depend on
    hand-audited call sites; this dataclass groups the five semantic
    knobs (skip policy, audio shape, joint rendering) under names. New
    callers should pass `options=`; the legacy scalars stay as the
    default path and build an equivalent instance internally, so existing
    callers are untouched.

    Track B augment fields (`min_fps`/`min_width`/`min_height`): the
    presentation floors `finalize_run` enforces via `plan_augmentation`
    (defaults 24/1216/704 to match the coming config; 0 disables that
    axis — the 24fps `PRESENTATION_MIN_FPS` still applies).

    Encode fields (`crf`/`preset`, issues 050): the single vf encode
    quality (defaults crf 15 + veryfast match the validated Comfy
    `video_export.json` recipe and `augment.ffmpeg_encode_chunk`).

    Model-pass field (`use_model_pass`, issue 166): Real-ESRGAN +
    FILM pass when provisioned (default True — DESIGN §140 GPU defaults;
    absent legs fall back to the same vf path, so off == on-absent
    byte-for-byte).

    Interpolation multiplier (`interp_multiplier`, DESIGN §56): FILM
    frames per pair for the model pass (default 4); 1 keeps the frame
    count — upscale without interpolating.

    Presentation fps (`presentation_fps`, slow-mo finalize): pins the
    shipped frame rate instead of the floors rule — 24fps x2 content
    presented at 32fps stretches the timeline 1.5x (slow motion).
    `None` (default) keeps the floors behavior; 0 also means unset.

    Deferred audio (`deferred_audio`, DESIGN §140 slow-mo finalize):
    ltxv/causvid commit no ACE takes, so finalize renders them via
    `ensure_deferred_for_finalize` before the mix (full move, never at
    commit). `False` (default) keeps the takes-ledger behavior; joint
    backends never set this.
    """

    skip_bad: bool = False
    sample_rate: int = 48000
    channels: int = 2
    overlap_fraction: float = 0.10
    overlap_cap_seconds: float = 0.5
    joint_style: JointStyle = "blend"
    min_fps: int = 24
    min_width: int = 1216
    min_height: int = 704
    crf: int = FINALIZE_CRF_DEFAULT
    preset: str = FINALIZE_PRESET_DEFAULT
    use_model_pass: bool = True
    interp_multiplier: int = 4
    presentation_fps: int | None = None
    deferred_audio: bool = False

    def __post_init__(self) -> None:
        if self.joint_style not in ("blend", "hard-splice"):
            raise ValueError(
                f"joint_style must be 'blend' or 'hard-splice' (got {self.joint_style!r})"
            )
        if self.overlap_fraction < 0:
            raise ValueError(f"overlap_fraction must be >= 0 (got {self.overlap_fraction})")
        if self.overlap_cap_seconds < 0:
            raise ValueError(f"overlap_cap_seconds must be >= 0 (got {self.overlap_cap_seconds})")
        if self.min_fps < 0:
            raise ValueError(f"min_fps must be >= 0 (got {self.min_fps})")
        if self.min_width < 0:
            raise ValueError(f"min_width must be >= 0 (got {self.min_width})")
        if self.min_height < 0:
            raise ValueError(f"min_height must be >= 0 (got {self.min_height})")
        if not isinstance(self.use_model_pass, bool):
            raise TypeError(f"use_model_pass must be a bool (got {self.use_model_pass!r})")
        if self.interp_multiplier < 1:
            raise ValueError(f"interp_multiplier must be >= 1 (got {self.interp_multiplier!r})")
        if self.presentation_fps is not None and self.presentation_fps < 1:
            raise ValueError(f"presentation_fps must be >= 1 (got {self.presentation_fps!r})")
        if not isinstance(self.deferred_audio, bool):
            raise TypeError(f"deferred_audio must be a bool (got {self.deferred_audio!r})")
        validate_crf(self.crf)
        validate_preset(self.preset)

    def effective_overlap_fraction(self) -> float:
        """Overlap the mixer actually uses: hard-splice forces zero."""
        return 0.0 if self.joint_style == "hard-splice" else self.overlap_fraction


def _segment_video_matches_target(segment: Path, width: int, height: int, fps: int) -> bool:
    """True when a committed segment video can stream-copy into the final (031).

    Probe-only, never raises: any mismatch or probe failure falls back to
    the re-encode path. Requires h264 + yuv420p + matching WxH/fps so
    `ffmpeg -f concat -c copy` yields a valid final without touching pixels.
    """
    try:
        info = probe(segment / "video.mp4")
        streams = [s for s in info.get("streams", []) if isinstance(s, dict)]
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video is None:
            return False
        if int(video.get("width", -1)) != width or int(video.get("height", -1)) != height:
            return False
        if str(video.get("codec_name", "")) != "h264":
            return False
        if str(video.get("pix_fmt", "")) != "yuv420p":
            return False
        return abs(_probe_video_fps(info) - fps) <= FPS_MATCH_TOLERANCE
    except (MediaError, ValueError, TypeError, KeyError):
        return False


@dataclass(frozen=True)
class ResolvedFinalizeSettings:
    """One contract for `finalize_run` knob resolution (issue 083).

    The twelve positional scalars are the legacy shim; `options=` is the
    canonical knob. Both spell the same thing — this struct is what
    `finalize_run` actually consumes — so scalar-built and options-built
    calls with matching values resolve identically (pinned by
    `tests/test_media_augment_unified_083.py`). `settings` carries the
    audio/joint/skip policy; the `effective_*` fields carry the
    presentation floors + encode quality after the
    "explicit scalar wins over `options`, `None` means use `options`"
    rule (0 disables a floor axis — the 24fps `PRESENTATION_MIN_FPS`
    still applies downstream in `plan_augmentation`).
    """

    settings: FinalizeOptions
    min_fps: int
    min_width: int
    min_height: int
    crf: int
    preset: str
    use_model_pass: bool
    interp_multiplier: int
    presentation_fps: int | None
    deferred_audio: bool


def resolve_finalize_settings(
    *,
    options: FinalizeOptions | None,
    skip_bad: bool | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    min_fps: int | None,
    min_width: int | None,
    min_height: int | None,
    crf: int | None,
    preset: str | None,
    use_model_pass: bool | None = None,
    interp_multiplier: int | None = None,
    presentation_fps: int | None = None,
    deferred_audio: bool | None = None,
) -> ResolvedFinalizeSettings:
    """Resolve the scalar/`options=` split into one settings struct (pure).

    One uniform rule for every knob (issue 190): an explicit scalar wins
    over `options`, `None` means "use the `options` value" (0 disables a
    floor axis — the 24fps `PRESENTATION_MIN_FPS` still applies downstream
    in `plan_augmentation`). No filesystem, no ffmpeg: `finalize_run`
    calls this first, then runs the §53 preflight + encode off the result.
    Extracted (not duplicated) so the shim and the canonical path can
    never drift.
    """
    if options is None:
        effective_overlap = (
            FinalizeOptions.overlap_fraction if overlap_fraction is None else overlap_fraction
        )
        settings = FinalizeOptions(
            skip_bad=False if skip_bad is None else skip_bad,
            sample_rate=48000 if sample_rate is None else sample_rate,
            channels=2 if channels is None else channels,
            overlap_fraction=effective_overlap,
            overlap_cap_seconds=0.5 if overlap_cap_seconds is None else overlap_cap_seconds,
            joint_style="hard-splice" if effective_overlap <= 0 else "blend",
            min_fps=AUGMENT_DEFAULT_MIN_FPS if min_fps is None else min_fps,
            min_width=AUGMENT_DEFAULT_MIN_WIDTH if min_width is None else min_width,
            min_height=AUGMENT_DEFAULT_MIN_HEIGHT if min_height is None else min_height,
            crf=FINALIZE_CRF_DEFAULT if crf is None else crf,
            preset=FINALIZE_PRESET_DEFAULT if preset is None else preset,
            use_model_pass=True if use_model_pass is None else use_model_pass,
            interp_multiplier=4 if interp_multiplier is None else interp_multiplier,
            presentation_fps=presentation_fps,
            deferred_audio=False if deferred_audio is None else deferred_audio,
        )
    else:
        settings = options
        if skip_bad is not None:
            settings = replace(settings, skip_bad=skip_bad)
        if sample_rate is not None:
            settings = replace(settings, sample_rate=sample_rate)
        if channels is not None:
            settings = replace(settings, channels=channels)
        if overlap_fraction is not None:
            settings = replace(settings, overlap_fraction=overlap_fraction)
        if overlap_cap_seconds is not None:
            settings = replace(settings, overlap_cap_seconds=overlap_cap_seconds)
        if use_model_pass is not None:
            settings = replace(settings, use_model_pass=use_model_pass)
        if interp_multiplier is not None:
            settings = replace(settings, interp_multiplier=interp_multiplier)
        if presentation_fps is not None:
            settings = replace(settings, presentation_fps=presentation_fps)
        if deferred_audio is not None:
            settings = replace(settings, deferred_audio=deferred_audio)
    return ResolvedFinalizeSettings(
        settings=settings,
        min_fps=min_fps if min_fps is not None else settings.min_fps,
        min_width=min_width if min_width is not None else settings.min_width,
        min_height=min_height if min_height is not None else settings.min_height,
        crf=validate_crf(crf if crf is not None else settings.crf),
        preset=validate_preset(preset if preset is not None else settings.preset),
        use_model_pass=settings.use_model_pass,
        interp_multiplier=(
            interp_multiplier if interp_multiplier is not None else settings.interp_multiplier
        ),
        presentation_fps=(
            presentation_fps if presentation_fps is not None else settings.presentation_fps
        ),
        deferred_audio=(deferred_audio if deferred_audio is not None else settings.deferred_audio),
    )


def _record_final_geometry(
    run_dir: Path,
    width: int,
    height: int,
    fps: int,
    min_fps: int,
    min_width: int,
    min_height: int,
) -> bool:
    """Best-effort provenance write-back (issue 141): stamp the shipped box.

    `build_manifest` records the run's `[augment]` floors plus a null
    `final_geometry` at init; the first finalize overwrites both with the
    effective floors and the validated output box, so the manifest never
    claims the stale 768x432 source hint as shipped geometry. Returns False
    (never raises — a provenance write must not fail a finalize) when the
    run has no manifest, e.g. throwaway/legacy dirs.
    """
    try:
        manifest_path = run_dir / paths.MANIFEST_FILENAME
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            return False
        manifest["presentation"] = {
            "min_fps": min_fps,
            "min_width": min_width,
            "min_height": min_height,
        }
        manifest["final_geometry"] = {"width": width, "height": height, "fps": fps}
        atomic_write_json(manifest_path, manifest)
    except (OSError, ValueError):
        return False
    return True


def committed_usable_segments(run_dir: Path, skip_bad: bool) -> list[Path]:
    """Committed segment dirs in order, triaged per §56 steps 4-6 (issues 138/188).

    Shared by `finalize_run` and the parallel finalize path (DESIGN §140
    GPU defaults): numbering gaps raise unless `skip_bad` (then the first
    gap warns and stops the scan), and every segment passes
    `_check_segment_committed` (failures raise, or print a loud skip line
    under `skip_bad`). An empty usable set raises either way.
    """
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    segment_dirs = (
        sorted(p for p in segments_root.iterdir() if p.is_dir()) if segments_root.exists() else []
    )
    committed = [d for d in segment_dirs if (d / paths.DONE_MARKER).exists()]
    if not committed:
        raise MediaError(f"no committed segments in {run_dir}")
    if skip_bad:
        for position, segment in enumerate(committed):
            if segment.name != f"{position:06d}":
                print(
                    "finalize: skipping segment numbering gap: "
                    f"expected {position:06d}, found {segment.name}"
                )
                break
    else:
        for position, segment in enumerate(committed):
            if segment.name != f"{position:06d}":
                raise MediaError(
                    f"segment numbering gap: expected {position:06d}, found {segment.name}"
                )
    usable: list[Path] = []
    for segment in committed:
        try:
            _check_segment_committed(segment)
        except MediaError as exc:
            if not skip_bad:
                raise
            print(f"finalize: skipping {segment.name} ({exc})")
            continue
        usable.append(segment)
    if not usable:
        raise MediaError(f"no usable segments in {run_dir}")
    return usable


def finalize_run(
    run_dir: Path,
    output_path: Path,
    width: int = 768,
    height: int = 432,
    fps: int = 24,
    skip_bad: bool | None = None,
    min_free_space_gib: float = 0.0,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    min_fps: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    crf: int | None = None,
    preset: str | None = None,
    use_model_pass: bool | None = None,
    interp_multiplier: int | None = None,
    presentation_fps: int | None = None,
    models_dir: Path | str | None = None,
    options: FinalizeOptions | None = None,
    deferred_audio: bool | None = None,
    seed: int = 0,
    audio_config: AudioConfig | None = None,
) -> Path:
    """Concat committed segments → single normalized MP4 (DESIGN §56).

    The presentation box/fps come from `plan_augmentation` (Track B):
    `max(requested, floors, 24fps)` for fps and `max(target, floors)`
    per axis for geometry, so backend-native segments (CausVid
    832x480@16, LTXV 768x512@24) ship at >= 1216x704@24 by default.
    One uniform knob rule (issue 190): an explicit scalar wins over
    `options`, `None` means "use the `options` value" (which defaults to
    24/1216/704 floors, crf 15 + veryfast, blend joints at 0.10 overlap);
    pass 0 to disable a floor axis (the 24fps `PRESENTATION_MIN_FPS`
    still applies). An explicit `overlap_fraction=0` behaves as a hard
    splice (the blend falls back to concat below the audibility floor).

    Native-geometry runs (presentation already matches) stream-copy the
    committed videos with zero video re-encodes (issue 031 fast path);
    anything the plan flags (`needs_reencode`) takes a single
    concat-demuxer + vf encode (issues 050: minterpolate-when-lifting +
    scale/pad/fps, one libx264 pass over the originals — no intermediate
    per-segment parts). Then mux audio, validate against the presentation
    box/fps, atomically publish, and append a `finalize_completed` event
    (effective crf/preset + `parts_encode_ms`/`audio_blend_ms`/
    `final_encode_ms`) to `logs/metrics.jsonl` for soak trending. With
    skip_bad, corrupt segments are skipped with a warning instead of
    aborting the whole finalize — input triage only (issues 138/188):
    missing artifacts, checksum/metrics/alignment failures, and numbering
    gaps each print a `finalize: skipping ...` line and continue, while
    the post-assembly `validate_video` stays strict (a corrupt stage
    still aborts even under skip_bad). A positive `min_free_space_gib`
    runs the §53 preflight first so a full disk fails fast instead of
    mid-encode.

    Native-geometry runs (presentation already matches) stream-copy the
    committed videos with zero video re-encodes (issue 031 fast path);
    anything the plan flags (`needs_reencode`) takes a single
    concat-demuxer + vf encode (issues 050: minterpolate-when-lifting +
    scale/pad/fps, one libx264 pass over the originals — no intermediate
    per-segment parts). Then mux audio, validate against the presentation
    box/fps, atomically publish, and append a `finalize_completed` event
    (effective crf/preset + `parts_encode_ms`/`audio_blend_ms`/
    `final_encode_ms`) to `logs/metrics.jsonl` for soak trending. With
    skip_bad, corrupt segments are skipped with a warning instead of
    aborting the whole finalize. A positive `min_free_space_gib` runs
    the §53 preflight first so a full disk fails fast instead of
    mid-encode.

    Audio joints get a proportional overlap crossfade (re-sliced from
    the takes ledger — previews untouched); pass overlap_fraction=0 to
    keep the legacy hard splice. The audio timeline stays on the source
    fps (frame counts / requested fps = seconds) — the fps lift touches
    video only, never the mix.

    Knob contract (issue 083): `options=` is canonical; the scalars are a
    tested shim resolved by `resolve_finalize_settings` (scalar wins over
    `options`, `None` means use `options`). The `width`/`height`/`fps`
    defaults (768/432/24) are the legacy fake-native fallback — kept (not
    raised to the presentation floors) because zero-floor callers rely on
    them for the stream-copy fast path (see `test_finalize_fastpath`);
    default-floor callers are lifted to 1216x704@24 by `plan_augmentation`
    either way, so either default ships the same presentation.

    Model pass (issue 166): `use_model_pass=True` consults
    `resolve_augment_weights(models_dir)` via `model_pass_active` — absent
    legs (or no `models_dir`) read as ffmpeg fallback, never an error, so
    knob-off == knob-on-absent byte-for-byte. Present legs select the
    tensor chunk encode (`run_finalize_model_pass`: SRVGG upscale + FILM
    mids chunked, then the presentation vf without minterpolate) only when
    the augment plan flags work (`needs_reencode`); sources already at the
    presentation box/fps skip it for the stream-copy fast path, so ltx25
    native 1216x704@24 is not augmented by default.
    """
    resolved = resolve_finalize_settings(
        options=options,
        skip_bad=skip_bad,
        sample_rate=sample_rate,
        channels=channels,
        overlap_fraction=overlap_fraction,
        overlap_cap_seconds=overlap_cap_seconds,
        min_fps=min_fps,
        min_width=min_width,
        min_height=min_height,
        crf=crf,
        preset=preset,
        use_model_pass=use_model_pass,
        interp_multiplier=interp_multiplier,
        presentation_fps=presentation_fps,
        deferred_audio=deferred_audio,
    )
    settings = resolved.settings
    effective_min_fps = resolved.min_fps
    effective_min_width = resolved.min_width
    effective_min_height = resolved.min_height
    effective_crf = resolved.crf
    effective_preset = resolved.preset
    effective_use_model_pass = resolved.use_model_pass
    effective_interp_multiplier = resolved.interp_multiplier
    effective_presentation_fps = resolved.presentation_fps
    effective_deferred_audio = resolved.deferred_audio
    resolved_weights = None
    model_selected = False
    if effective_use_model_pass and models_dir is not None:
        from voyage.augment import model_pass_active, resolve_augment_weights

        # Present legs select the tensor chunk encode below; absent legs
        # fall back to the ffmpeg paths (byte-identical to knob-off).
        resolved_weights = resolve_augment_weights(models_dir)
        model_selected = model_pass_active(effective_use_model_pass, resolved_weights)
    if min_free_space_gib > 0:
        check_free_space(run_dir, min_free_space_gib)
    # §56 steps 4-6 per segment, before any encoding work. skip_bad is
    # input triage (issues 138/188): missing artifacts fold into the same
    # skippable loop as checksum/metrics/alignment failures, and numbering
    # gaps warn instead of vanishing silently — while the post-assembly
    # validate_video below stays strict under both settings.
    usable = committed_usable_segments(run_dir, settings.skip_bad)

    # Presentation box/fps via the pure augment plan (Track B): sources
    # below the floors (CausVid 16fps, sub-704p natives) are lifted with
    # motion interpolation + upscale; the audio timeline stays on the
    # requested (== source) fps — frame counts / source fps = seconds.
    source_info = probe(usable[0] / "video.mp4")
    source_fps = _probe_video_fps(source_info)
    source_w, source_h = _probe_video_geometry(source_info)
    # The plan only credits the FILM multiplier when the model pass will
    # actually interpolate (legs present + knob on); otherwise a missing
    # leg must fall back to the minterpolate lift, never to slow motion.
    plan_multiplier = effective_interp_multiplier if model_selected else 1
    plan = plan_augmentation(
        source_w,
        source_h,
        source_fps,
        width,
        height,
        (effective_presentation_fps if effective_presentation_fps is not None else fps),
        effective_min_fps,
        effective_min_width,
        effective_min_height,
        interp_multiplier=plan_multiplier,
    )
    out_w, out_h, out_fps = plan.out_w, plan.out_h, plan.out_fps
    stretch = presentation_stretch(source_fps, effective_interp_multiplier, out_fps)
    lift = ""
    if plan.needs_minterpolate:
        lift = f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"

    with tempfile.TemporaryDirectory(prefix="voyage-final-", dir=run_dir) as tmp:
        tmpdir = Path(tmp)
        # Single-pass shape (issues 050): no intermediate per-segment
        # libx264 parts — the concat demuxer feeds one vf encode over the
        # originals. `parts_encode_ms` stays 0.0 so soak trending keeps a
        # stable schema across the old double-encode and the new path.
        parts_encode_ms = 0.0
        staged = tmpdir / "final.mp4"
        # Present-legs tensor path (issue 166): chunked SRVGG + FILM to one
        # intermediate, then the presentation vf without minterpolate (FILM
        # already interpolated). Absent/off keeps the ffmpeg paths below.
        # Trigger rule: the tensor pass only runs when the augment plan
        # flags work (`needs_reencode` — a geometry/fps lift). Sources
        # already meeting the presentation box/fps skip it even when legs
        # are present, so ltx25 native 1216x704@24 takes the stream-copy
        # fast path instead of a no-op enhance.
        tensor_intermediate: Path | None = None
        # Phase timings for the elapsed-time report (DESIGN §56): the
        # durable entry fills this in; the legacy flow and the no-model
        # paths leave it empty (only wall time is known there).
        model_pass_timings: dict[str, float] = {}
        # One shared gate (`tensor_path_armed`): the parallel pre-fork
        # predicts exactly this, or the ACE takes ledger lands on the
        # wrong timeline.
        tensor_devices: tuple[str, ...] = ()
        if model_selected and resolved_weights is not None:
            from voyage.augment import augment_devices, model_pass_devices

            if augment_devices():
                tensor_devices = model_pass_devices()
        if (
            resolved_weights is not None
            and tensor_devices
            and tensor_path_armed(
                model_selected=model_selected,
                weights_present=True,
                source_fps=source_fps,
                needs_reencode=plan.needs_reencode,
                devices_available=True,
            )
        ):
            # DESIGN §140 GPU defaults: the whole model pass is pinned
            # to cuda:1 (the 2060) so the MMAudio SFX stack owns cuda:0
            # (the 4060) — `model_pass_devices` collapses to cuda:0 on
            # a 1-GPU box, so the sequential path is unchanged there.
            model_work = tmpdir / "model_pass"
            if resolved_weights.film is not None and resolved_weights.realesrgan is not None:
                # Durable sidecar path (independent workers): poll the
                # upscale + interp workers to completion under
                # run_dir/augment/<plan-hash>/, drain one intermediate
                # per usable segment, concat in segment order. A crashed
                # finalize retries only the missing chunks.
                from voyage.augment_finalize import run_durable_model_pass

                tensor_intermediate, _ = run_durable_model_pass(
                    run_dir,
                    usable,
                    out_width=out_w,
                    out_height=out_h,
                    source_fps=source_fps,
                    weights=resolved_weights,
                    multiplier=effective_interp_multiplier,
                    crf=effective_crf,
                    preset=effective_preset,
                    device=tensor_devices[0],
                    work_dir=model_work,
                    timings=model_pass_timings,
                )
            else:
                # Partial legs keep the legacy all-or-nothing tmpdir
                # flow (unchanged behavior for film-only/ESRGAN-only).
                from voyage.augment import run_finalize_model_pass

                tensor_intermediate, _ = run_finalize_model_pass(
                    [segment / "video.mp4" for segment in usable],
                    resolved_weights,
                    source_fps=source_fps,
                    multiplier=effective_interp_multiplier,
                    crf=effective_crf,
                    preset=effective_preset,
                    work_dir=model_work,
                    devices=tensor_devices,
                )
        # Deferred ACE music (DESIGN §140 slow-mo finalize): ltxv/causvid
        # commit no takes, so the takes render here — AFTER the model pass
        # (upscale → interp → music → SFX order) and BEFORE the mix below.
        # The stretch covers the slow-mo timeline only when the present
        # actually retimes (tensor path + explicit presentation fps +
        # stretch != 1 — `slowmo_video_active`); otherwise the mix stays on
        # the source timeline. A complete ledger (re-finalize) no-ops
        # inside the helper with no worker spawned.
        audio_start = time.monotonic()
        slowmo = slowmo_video_active(
            tensor_intermediate is not None, effective_presentation_fps, stretch
        )
        if effective_deferred_audio:
            from voyage.audio_finalize import ensure_deferred_for_finalize

            ace_models_dir = (
                getattr(audio_config, "models_dir", None) if audio_config is not None else None
            )
            if ace_models_dir is None:
                ace_models_dir = models_dir
            audio_stretch = stretch if slowmo else 1.0
            ensure_deferred_for_finalize(
                run_dir=run_dir,
                usable=usable,
                source_fps=source_fps,
                stretch=audio_stretch,
                run_seed=seed,
                models_dir=ace_models_dir,
                device="cuda:0",
                music_style=(
                    getattr(audio_config, "music_style", "") if audio_config is not None else ""
                ),
                explicit_caption=(
                    getattr(audio_config, "music_caption", None)
                    if audio_config is not None
                    else None
                ),
                take_seconds=(
                    getattr(audio_config, "take_seconds", 45.0)
                    if audio_config is not None
                    else 45.0
                ),
                ahead_seconds=(
                    getattr(audio_config, "ahead_seconds", 20.0)
                    if audio_config is not None
                    else 20.0
                ),
                beats_per_segment=(
                    getattr(audio_config, "beats_per_segment", 4) if audio_config is not None else 4
                ),
                sample_rate=(
                    getattr(audio_config, "sample_rate", 48000)
                    if audio_config is not None
                    else 48000
                ),
                channels=(getattr(audio_config, "channels", 2) if audio_config is not None else 2),
            )
        else:
            audio_stretch = 1.0
        # Blended final mix (overlap re-sliced from takes; previews untouched).
        final_audio = build_final_audio(
            run_dir,
            usable,
            tmpdir,
            fps,
            settings.sample_rate,
            settings.channels,
            settings.effective_overlap_fraction(),
            settings.overlap_cap_seconds,
            deferred=effective_deferred_audio,
            stretch=audio_stretch,
        )
        audio_blend_ms = (time.monotonic() - audio_start) * 1000.0
        # Issue 031 fast path: every committed video already matches the
        # presentation geometry/pix_fmt/fps, so concat the originals with a
        # stream copy and mux the final audio — zero video re-encodes. The
        # augment plan gates it off whenever an upscale or fps lift is
        # required (needs_reencode covers both, plus any fps mismatch).
        native = (
            lift == ""
            and not plan.needs_reencode
            and source_fps > 0
            and abs(out_fps - source_fps) <= FPS_MATCH_TOLERANCE
            and all(
                _segment_video_matches_target(segment, out_w, out_h, out_fps) for segment in usable
            )
        )
        if tensor_intermediate is not None:
            tensor_vf = tensor_presentation_vf(out_w, out_h, out_fps, stretch, slowmo=slowmo)
            final_start = time.monotonic()
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(tensor_intermediate),
                    "-i",
                    str(final_audio),
                    "-vf",
                    tensor_vf,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    effective_preset,
                    "-crf",
                    str(effective_crf),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-shortest",
                    str(staged),
                ]
            )
            final_encode_ms = (time.monotonic() - final_start) * 1000.0
            if proc.returncode != 0:
                raise MediaError(f"model-pass final encode failed: {proc.stderr[-2000:]}")
        elif native:
            from voyage.augment_morph import (
                assemble_morphed_timeline,
                morph_backend_for_run,
                resolve_morph_device,
            )

            # Always-on morph-cut leg (ltx25/ltx23): count-preserving 2+2
            # FILM joints replace the hard cuts between committed segments.
            # Frame total is unchanged, so the audio timeline below needs no
            # work. Any missing precondition (foreign backend, single
            # segment, no FILM leg) keeps the plain stream-copy concat.
            morph_video: Path | None = None
            if (
                morph_backend_for_run(run_dir) is not None
                and len(usable) > 1
                and resolved_weights is not None
                and resolved_weights.film is not None
            ):
                morph_start = time.monotonic()
                morph_video = assemble_morphed_timeline(
                    [segment / "video.mp4" for segment in usable],
                    joint_root=run_dir / "augment" / "morph_native",
                    fps=int(round(source_fps)),
                    crf=effective_crf,
                    preset=effective_preset,
                    pix_fmt="yuv420p",
                    interp_fn=None,
                    weights=resolved_weights.film,
                    device=resolve_morph_device(tensor_devices[0] if tensor_devices else None),
                )
                model_pass_timings.setdefault("morph_s", 0.0)
                model_pass_timings.setdefault("morphs_done", 0.0)
                model_pass_timings["morph_s"] += time.monotonic() - morph_start
                model_pass_timings["morphs_done"] += float(len(usable) - 1)
            if morph_video is not None:
                video_inputs = ["-i", str(morph_video)]
            else:
                concat_list = write_concat_list(
                    [segment / "video.mp4" for segment in usable], tmpdir / "concat.txt"
                )
                video_inputs = ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
            final_start = time.monotonic()
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    *video_inputs,
                    "-i",
                    str(final_audio),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-shortest",
                    str(staged),
                ]
            )
            final_encode_ms = (time.monotonic() - final_start) * 1000.0
            if proc.returncode != 0:
                raise MediaError(f"final concat copy failed: {proc.stderr[-2000:]}")
        else:
            # Single vf encode over the concat demuxer (issues 050): the
            # originals feed `scale/pad/fps/minterpolate` once — the old
            # per-segment `libx264/veryfast/-an` parts were a wasted first
            # pass with no vf. Concat entries go through the quoting-safe
            # helper (issues 053) so adversarial paths stay parseable.
            concat_list = write_concat_list(
                [segment / "video.mp4" for segment in usable], tmpdir / "concat.txt"
            )
            vf = (
                f"{lift}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={out_fps}"
            )
            final_start = time.monotonic()
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-i",
                    str(final_audio),
                    "-vf",
                    vf,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    effective_preset,
                    "-crf",
                    str(effective_crf),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                    "-shortest",
                    str(staged),
                ]
            )
            final_encode_ms = (time.monotonic() - final_start) * 1000.0
            if proc.returncode != 0:
                raise MediaError(f"final encode failed: {proc.stderr[-2000:]}")
        validate_video(staged, out_w, out_h, out_fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(staged, output_path)
        _record_final_geometry(
            run_dir,
            out_w,
            out_h,
            out_fps,
            effective_min_fps,
            effective_min_width,
            effective_min_height,
        )
        from voyage.logrotate import append_line

        append_line(
            run_dir / paths.LOGS_DIRNAME / "metrics.jsonl",
            json.dumps(
                {
                    "ts": time.time(),
                    "event": "finalize_completed",
                    "segments": len(usable),
                    "out_w": out_w,
                    "out_h": out_h,
                    "out_fps": out_fps,
                    "crf": effective_crf,
                    "preset": effective_preset,
                    "parts_encode_ms": round(parts_encode_ms, 1),
                    "audio_blend_ms": round(audio_blend_ms, 1),
                    "final_encode_ms": round(final_encode_ms, 1),
                    "fast_path": native and tensor_intermediate is None,
                    "interp_multiplier": effective_interp_multiplier,
                    "presentation_fps": effective_presentation_fps,
                    "slowmo_factor": round(stretch, 4),
                    "deferred_audio": effective_deferred_audio,
                    "model_pass_timings_s": {
                        key: round(value, 3)
                        for key, value in model_pass_timings.items()
                        if key.endswith("_s")
                    },
                    "model_pass_chunks": {
                        key: value
                        for key, value in model_pass_timings.items()
                        if not key.endswith("_s")
                    },
                }
            ),
        )
    return output_path

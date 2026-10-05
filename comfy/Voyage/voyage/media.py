"""ffmpeg/ffprobe wrappers + validation + finalizer (DESIGN §§54-57, I).

All invocations use argument lists — never shell strings. The finalizer
never mutates source segment files; it publishes the final path
atomically.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from voyage.config import AudioConfig
    from voyage.console import VoyageConsole

from voyage import paths
from voyage.atomic import atomic_copy
from voyage.augment import CRF_MAXIMUM as _AUGMENT_CRF_MAXIMUM
from voyage.augment import CRF_MINIMUM as _AUGMENT_CRF_MINIMUM
from voyage.augment import interpolated_frame_count as interpolated_frame_count
from voyage.console import ParallelFinalizeDisplay, optional_stage
from voyage.errors import MediaError, StateError
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
# _cached_slice_take, _audio_duration_seconds,
# _blend_fade_seconds, _blend_pair, _join_audio_single_graph,
# build_final_audio - facades above re-export them verbatim.


@dataclass
class AugmentPlan:
    """Resolution/fps the finalizer must produce (explicit quality).

    Why this exists: segment videos render at backend-native geometry
    (CausVid 832x480@16, LTXV 768x512@24, fake 768x432@24) but the
    shipped video is `source x upscale` at the interpolated rate unless
    `--presentation-fps` pins it. The plan is pure math over probed
    source + explicit multipliers, so unit tests pin it without ffmpeg
    and `finalize_run` just renders it.
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
    *,
    upscale: int = 1,
    interpolate: int = 1,
    presentation_fps: int | None = None,
    model_interpolate: int = 1,
) -> AugmentPlan:
    """Compute the output box/fps for one finalize (pure).

    `out = source x upscale` per axis (exact integer multiply — the
    model SRVGG leg upscales 2x exactly, any residual lands in the vf);
    `out_fps` is `presentation_fps` when set, else
    `round(source_fps * interpolate)` (`--interpolate 2` doubles the
    frame count). There are no minimum quality floors: quality is
    specified explicitly via the multipliers, so a CausVid 16fps
    source ships at 16fps unless pinned otherwise.

    `needs_minterpolate` is True only for an fps lift the model pass
    did not already produce (`model_interpolate <= 1` and source + 0.5
    < out — ffmpeg motion interpolation); an fps drop uses the plain
    fps filter. When the model pass interpolates
    (`model_interpolate > 1`), the lift is measured against the
    interpolated rate (`source * model_interpolate`): presenting 24fps
    x2 content at 32fps stretches the timeline (slow motion) instead
    of synthesizing more frames.
    `needs_reencode` covers any pixel/timing change (dims differ, fps
    differs past 0.5 either way, lift, or unknown source fps) and gates
    the stream-copy fast path off.
    """
    if upscale not in (1, 2, 4):
        raise ValueError(f"upscale must be 1, 2, or 4 (got {upscale})")
    if interpolate < 1:
        raise ValueError(f"interpolate must be >= 1 (got {interpolate})")
    if model_interpolate < 1:
        raise ValueError(f"model_interpolate must be >= 1 (got {model_interpolate})")
    if source_w <= 0 or source_h <= 0:
        raise MediaError(f"cannot plan finalize: unprobable source dims ({source_w}x{source_h})")
    source_fps_value = float(source_fps)
    if source_fps_value <= 0 and presentation_fps is None:
        raise MediaError("cannot plan finalize: unprobable source fps and no --presentation-fps")
    out_w = int(source_w) * upscale
    out_h = int(source_h) * upscale
    if presentation_fps:
        out_fps = int(presentation_fps)
    else:
        out_fps = int(round(source_fps_value * interpolate))
    if out_fps <= 0:
        raise MediaError(f"cannot plan finalize: non-positive output fps ({out_fps})")
    interpolated_rate = source_fps_value * model_interpolate
    needs_minterpolate = (
        model_interpolate <= 1
        and source_fps_value > 0
        and out_fps > interpolated_rate + FPS_MATCH_TOLERANCE
    )
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


def slowmo_factor(source_fps: float, interpolate: int, out_fps: int) -> float:
    """Timeline stretch of a slow-motion present (pure).

    Interpolating `source_fps` content by `interpolate` and
    presenting at `out_fps` stretches wall-clock by
    `source_fps * interpolate / out_fps` (24fps x2 at 32fps = 1.5x).
    1.0 means no stretch (today's behavior).
    """
    if source_fps <= 0 or interpolate < 1 or out_fps <= 0:
        raise ValueError(
            f"slowmo inputs must be positive (got {source_fps}/{interpolate}/{out_fps})"
        )
    return float(source_fps) * interpolate / out_fps


def presentation_stretch(source_fps: float, interpolate: int, out_fps: int) -> float:
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
    return slowmo_factor(source_fps, interpolate, out_fps)


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


#: Card the deferred ACE music worker renders on (the `device=` at the
#: `ensure_deferred_for_finalize` call site). The parallel fork gate
#: below must follow this literal — overlap is only safe while the
#: model pass owns a different card.
DEFERRED_MUSIC_DEVICE = "cuda:0"


def model_music_parallel_armed(
    *,
    tensor_path: bool,
    deferred_pending: bool,
    model_devices: tuple[str, ...],
) -> bool:
    """Fork gate: upscale phase + deferred music may overlap (DESIGN §140).

    Pure: the model branch must actually take the tensor path, the dry
    walk must show takes still to render, and every model device must
    differ from `DEFERRED_MUSIC_DEVICE` (`upscale_pass_devices`
    collapses to cuda:0 on a 1-GPU box, so single-GPU finalizes stay
    sequential). One predicate in one place, mirroring
    `tensor_path_armed` — the pre-fork ACE render must predict exactly
    what Thread A will decide, or the takes ledger lands on the wrong
    timeline.
    """
    return bool(
        tensor_path
        and deferred_pending
        and model_devices
        and all(device != DEFERRED_MUSIC_DEVICE for device in model_devices)
    )


def run_model_pass_and_music_parallel(
    *,
    model_work: Callable[[], None],
    music_work: Callable[[], None],
) -> None:
    """Fork-join the upscale phase and the deferred music takes.

    Thread A runs the upscale leg (cuda:1), thread B renders ACE takes
    (cuda:0) — distinct cards, distinct ledgers (`chunks.jsonl` vs the
    takes ledger), no shared console writes (both branches run with
    progress None under one outer stage). Both threads are always
    joined: a branch exception propagates only after the join, so a
    music failure never orphans a running upscale phase and vice versa.
    Fail-soft: whatever rendered stays ledgered, so retrying the
    finalize resumes instead of redoing.
    """
    errors: dict[str, BaseException] = {}

    def _guard(key: str, work: Callable[[], None]) -> None:
        try:
            work()
        except Exception as exc:  # noqa: BLE001 — branch errors re-raise after join, never swallowed
            errors[key] = exc

    model_thread = threading.Thread(target=_guard, args=("model", model_work), daemon=True)
    music_thread = threading.Thread(target=_guard, args=("music", music_work), daemon=True)
    model_thread.start()
    music_thread.start()
    model_thread.join()
    music_thread.join()
    model_error = errors.get("model")
    if model_error is not None:
        raise model_error
    music_error = errors.get("music")
    if music_error is not None:
        raise music_error


@dataclass
class SfxParallelRequest:
    """SFX bed work eligible to overlap the publish encode (two-stream finalize).

    Carries everything the legacy `finalize_sfx_pass` takes except the
    published final itself: the bed conditions on a stream-copy proxy of
    the committed segments (same source timeline, no upscale/interp/morph)
    and dubs onto the staged publish on the main thread. `fps` is the
    source fps the SFX bounds walk (mirrors the legacy pass's `fps`).
    """

    backend: str
    models_dir: str
    device: str
    model_size: str
    num_workers: int
    fps: int
    caption_override: str | None = None


def sfx_parallel_armed(sfx_request: SfxParallelRequest | None) -> bool:
    """Bed∥publish gate: overlap only for real SFX backends (DESIGN §140).

    Pure: the fake backend renders test tones offline, so parallelizing
    it would only add threads without saving GPU time — and offline
    fake-stack tests must keep the exact legacy path.
    """
    return sfx_request is not None and sfx_request.backend != "fake"


def presentation_setup_facts(
    plan: AugmentPlan,
    *,
    upscale: int,
    interpolate: int,
    presentation_fps: int | None,
) -> dict[str, object]:
    """§104 setup facts for the finalize quality multipliers (issue 194).

    Pure: the explicit multiplier triple as given plus the resolved
    output plan, so benchmark/soak reports can record the dominant
    finalize variable instead of leaving re-encode-vs-stream-copy
    unexplained. HOOK FOR THE OBSERVE TRACK (`voyage/cli_observe.py`
    is out of this change's scope): spread these facts into
    `_benchmark_env()` (or alongside `_video_geometry_setup()`) at the
    `cmd_benchmark`/`cmd_soak` setup sites so every §104 setup block
    carries them.
    """
    return {
        "upscale": upscale,
        "interpolate": interpolate,
        "presentation_fps": presentation_fps,
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

    Track B augment fields (`upscale`/`interpolate`): the explicit
    quality multipliers `finalize_run` renders via `plan_augmentation`
    (defaults 1/1 — ship at source resolution and frame count).

    Encode fields (`crf`/`preset`, issues 050): the single vf encode
    quality (defaults crf 15 + veryfast match the validated Comfy
    `video_export.json` recipe and `augment.ffmpeg_encode_chunk`).

    Presentation fps (`presentation_fps`, slow-mo finalize): pins the
    shipped frame rate instead of the interpolate rule — 24fps x2 content
    presented at 32fps stretches the timeline 1.5x (slow motion).
    `None` (default) keeps the interpolate behavior.
    """

    skip_bad: bool = False
    sample_rate: int = 48000
    channels: int = 2
    overlap_fraction: float = 0.10
    overlap_cap_seconds: float = 0.5
    joint_style: JointStyle = "blend"
    upscale: int = 1
    interpolate: int = 1
    crf: int = FINALIZE_CRF_DEFAULT
    preset: str = FINALIZE_PRESET_DEFAULT
    presentation_fps: int | None = None

    def __post_init__(self) -> None:
        if self.joint_style not in ("blend", "hard-splice"):
            raise ValueError(
                f"joint_style must be 'blend' or 'hard-splice' (got {self.joint_style!r})"
            )
        if self.overlap_fraction < 0:
            raise ValueError(f"overlap_fraction must be >= 0 (got {self.overlap_fraction})")
        if self.overlap_cap_seconds < 0:
            raise ValueError(f"overlap_cap_seconds must be >= 0 (got {self.overlap_cap_seconds})")
        if self.upscale not in (1, 2, 4):
            raise ValueError(f"upscale must be 1, 2, or 4 (got {self.upscale!r})")
        if self.interpolate < 1:
            raise ValueError(f"interpolate must be >= 1 (got {self.interpolate!r})")
        if self.presentation_fps is not None and self.presentation_fps < 1:
            raise ValueError(f"presentation_fps must be >= 1 (got {self.presentation_fps!r})")
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

    `options=` is the canonical knob; the scalar overrides spell the
    same thing — this struct is what `finalize_run` actually consumes —
    so scalar-built and options-built calls with matching values
    resolve identically (pinned by
    `tests/test_media_augment_unified_083.py`). `settings` carries the
    audio/joint/skip policy; the `effective_*` fields carry the
    explicit quality multipliers + encode quality after the
    "explicit scalar wins over `options`, `None` means use `options`"
    rule.
    """

    settings: FinalizeOptions
    upscale: int
    interpolate: int
    crf: int
    preset: str
    presentation_fps: int | None


def resolve_finalize_settings(
    *,
    options: FinalizeOptions | None,
    skip_bad: bool | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    upscale: int | None,
    interpolate: int | None,
    crf: int | None,
    preset: str | None,
    presentation_fps: int | None = None,
) -> ResolvedFinalizeSettings:
    """Resolve the scalar/`options=` split into one settings struct (pure).

    One uniform rule for every knob (issue 190): an explicit scalar wins
    over `options`, `None` means "use the `options` value". No
    filesystem, no ffmpeg: `finalize_run` calls this first, then runs
    the §53 preflight + encode off the result. Extracted (not
    duplicated) so the shim and the canonical path can never drift.
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
            upscale=1 if upscale is None else upscale,
            interpolate=1 if interpolate is None else interpolate,
            crf=FINALIZE_CRF_DEFAULT if crf is None else crf,
            preset=FINALIZE_PRESET_DEFAULT if preset is None else preset,
            presentation_fps=presentation_fps,
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
        if upscale is not None:
            settings = replace(settings, upscale=upscale)
        if interpolate is not None:
            settings = replace(settings, interpolate=interpolate)
        if presentation_fps is not None:
            settings = replace(settings, presentation_fps=presentation_fps)
    return ResolvedFinalizeSettings(
        settings=settings,
        upscale=upscale if upscale is not None else settings.upscale,
        interpolate=interpolate if interpolate is not None else settings.interpolate,
        crf=validate_crf(crf if crf is not None else settings.crf),
        preset=validate_preset(preset if preset is not None else settings.preset),
        presentation_fps=(
            presentation_fps if presentation_fps is not None else settings.presentation_fps
        ),
    )


def committed_usable_segments(
    run_dir: Path, skip_bad: bool, progress: Any | None = None
) -> list[Path]:
    """Committed segment dirs in order, triaged per §56 steps 4-6 (issues 138/188).

    Shared by `finalize_run` and the parallel finalize path (DESIGN §140
    GPU defaults): numbering gaps raise unless `skip_bad` (then the first
    gap warns and stops the scan), and every segment passes
    `_check_segment_committed` (failures raise, or warn a skip line
    under `skip_bad`). Skip lines go through the progress sink when one
    is present (quiet-aware, TTY-consistent) and keep the historical
    bare `print` otherwise. An empty usable set raises either way.
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
                message = (
                    "finalize: skipping segment numbering gap: "
                    f"expected {position:06d}, found {segment.name}"
                )
                if progress is not None:
                    progress.warn(message)
                else:
                    print(message)
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
            message = f"finalize: skipping {segment.name} ({exc})"
            if progress is not None:
                progress.warn(message)
            else:
                print(message)
            continue
        usable.append(segment)
    if not usable:
        raise MediaError(f"no usable segments in {run_dir}")
    return usable


def presented_frames(video: Path) -> int | None:
    """ffprobe presented-frame count, None when the file is unreadable.

    Shared by the generate freshness gate and the finalize coverage stamp
    — one probe shape, so recorded and compared counts always agree.
    """
    try:
        proc = run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nw=1:nk=1",
                str(video),
            ]
        )
    except (OSError, MediaError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def _model_pass_stage_rows(
    model_pass_timings: dict[str, float], model_start: float
) -> dict[str, float]:
    """Timing-table rows for the model pass: per-leg seconds when known.

    The durable path records `upscale_poll_s` / `interp_poll_s`
    separately, so the table shows upscale vs interp instead of one
    aggregate. Legacy/tmpdir paths only know wall time — they keep the
    single `model pass` row.
    """
    up_seconds = float(model_pass_timings.get("upscale_poll_s", 0.0) or 0.0)
    ip_seconds = float(model_pass_timings.get("interp_poll_s", 0.0) or 0.0)
    if up_seconds > 0 or ip_seconds > 0:
        return {"model pass · upscale": up_seconds, "model pass · interp": ip_seconds}
    return {"model pass": time.monotonic() - model_start}


FINALIZE_TMPDIR_PREFIX = "voyage-final-"
"""Staging tmpdir prefix for `finalize_run` (DESIGN §56).

`TemporaryDirectory(prefix=...)` appends random characters to this prefix
inside the run dir. A SIGKILL between staging and cleanup orphans the whole
directory (the validate orphan scan only covers `*.partial`/`*.tmp*`
files, never directories), so finalize prunes stale siblings on entry.
"""


def prune_stale_finalize_tmpdirs(run_dir: Path) -> int:
    """Remove orphaned `voyage-final-*` staging dirs under `run_dir` (DESIGN §56).

    Why this exists: finalize stages everything in
    `TemporaryDirectory(prefix="voyage-final-", dir=run_dir)` — a SIGKILL
    leaves `run/voyage-final-*` forever and the next finalize would
    accumulate another one. Best-effort and narrow: only directories whose
    name starts with the exact prefix are removed (never files, never
    symlinks, never anything else); any error is swallowed and the dir is
    left for the operator. Returns the number of removed directories.
    """
    removed = 0
    try:
        children = sorted(run_dir.iterdir())
    except OSError:
        return 0
    for child in children:
        try:
            if not child.name.startswith(FINALIZE_TMPDIR_PREFIX):
                continue
            if not child.is_dir() or child.is_symlink():
                continue
            shutil.rmtree(child)
            removed += 1
        except OSError:
            continue
    return removed


def finalize_run(
    run_dir: Path,
    output_path: Path,
    skip_bad: bool | None = None,
    min_free_space_gib: float = 0.0,
    sample_rate: int | None = None,
    channels: int | None = None,
    overlap_fraction: float | None = None,
    overlap_cap_seconds: float | None = None,
    upscale: int | None = None,
    interpolate: int | None = None,
    crf: int | None = None,
    preset: str | None = None,
    presentation_fps: int | None = None,
    models_dir: Path | str | None = None,
    options: FinalizeOptions | None = None,
    seed: int = 0,
    audio_config: AudioConfig | None = None,
    invoker: str | None = None,
    progress: VoyageConsole | None = None,
    sfx_request: SfxParallelRequest | None = None,
    sfx_report: dict[str, Any] | None = None,
    no_music: bool = False,
) -> Path:
    """Concat committed segments → single normalized MP4 (DESIGN §56).

    The output box/fps come from `plan_augmentation`: `source x upscale`
    per axis and `round(source_fps x interpolate)` for fps, unless
    `presentation_fps` pins the shipped rate — so backend-native
    segments ship at native quality by default (upscale=1/interpolate=1)
    and are lifted only when asked. One uniform knob rule (issue 190):
    an explicit scalar wins over `options`, `None` means "use the
    `options` value" (which defaults to 1/1 multipliers, crf 15 +
    veryfast, blend joints at 0.10 overlap). An explicit
    `overlap_fraction=0` behaves as a hard splice (the blend falls back
    to concat below the audibility floor).

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

    Audio is always deferred (DESIGN §140): segments commit video only,
    so finalize renders the takes ledger via `ensure_deferred_for_finalize`
    (ACE-Step on cuda:0, or the fake sine worker when
    `audio_config.backend == "fake"`) before the ledger-only mix below.
    `seed` + `audio_config` size the replay (take seconds/ahead/beats,
    music style/caption, rate/channels); a complete ledger (re-finalize)
    no-ops with no worker spawned.

    Audio joints get a proportional overlap crossfade (re-sliced from
    the takes ledger); pass overlap_fraction=0 to
    keep the legacy hard splice. The audio timeline stays on the source
    fps (frame counts / requested fps = seconds) — the fps lift touches
    video only, never the mix.

    Knob contract (issue 083): `options=` is canonical; the scalars are a
    tested shim resolved by `resolve_finalize_settings` (scalar wins over
    `options`, `None` means use `options`). The output derives from the
    probed source (unprobable dims fail loud; unprobable fps fails loud
    unless `presentation_fps` pins the rate).

    Model pass (issue 166): present legs (via `model_pass_active` over
    `resolve_augment_weights(models_dir)`) select the tensor chunk encode
    (SRVGG upscale + FILM mids chunked, then the presentation vf without
    minterpolate) only when work is demanded (`upscale > 1 or
    interpolate > 1`) and the augment plan flags work (`needs_reencode`);
    1/1 sources skip it for the stream-copy fast path. Absent legs (or no
    `models_dir`) with demanded work fall back to the ffmpeg vf path
    (scale/minterpolate), never an error. Both legs provisioned splits
    the pass into Phase A (upscale-only on cuda:1, overlapping the
    deferred music takes on cuda:0) and Phase C (SFX bed, then FILM
    interp on cuda:0, then publish) — the 4060 runs music -> SFX ->
    interp sequentially and the 2060 never shares a card.

    `no_music` (generate-only `--no-music`): skips the deferred ACE takes
    render and the ledger mix entirely — finalize ships silent AAC sized
    to the stretched timeline via `anullsrc`. SFX is independent: with a
    live `sfx_request` the bed still dubs over the silence below.
    """
    resolved = resolve_finalize_settings(
        options=options,
        skip_bad=skip_bad,
        sample_rate=sample_rate,
        channels=channels,
        overlap_fraction=overlap_fraction,
        overlap_cap_seconds=overlap_cap_seconds,
        upscale=upscale,
        interpolate=interpolate,
        crf=crf,
        preset=preset,
        presentation_fps=presentation_fps,
    )
    settings = resolved.settings
    effective_upscale = resolved.upscale
    effective_interpolate = resolved.interpolate
    effective_crf = resolved.crf
    effective_preset = resolved.preset
    effective_presentation_fps = resolved.presentation_fps
    resolved_weights = None
    model_selected = False
    if (effective_upscale > 1 or effective_interpolate > 1) and models_dir is not None:
        from voyage.augment import model_pass_active, resolve_augment_weights

        # Present legs select the tensor chunk encode below; absent legs
        # fall back to the ffmpeg paths (same vf encode).
        resolved_weights = resolve_augment_weights(models_dir)
        model_selected = model_pass_active(resolved_weights)
    if min_free_space_gib > 0:
        check_free_space(run_dir, min_free_space_gib)
    # §56 steps 4-6 per segment, before any encoding work. skip_bad is
    # input triage (issues 138/188): missing artifacts fold into the same
    # skippable loop as checksum/metrics/alignment failures, and numbering
    # gaps warn instead of vanishing silently — while the post-assembly
    # validate_video below stays strict under both settings.
    final_stages: dict[str, float] = {}
    triage_cm = optional_stage(progress, "triage segments")
    triage_start = time.monotonic()
    with triage_cm:
        usable = committed_usable_segments(run_dir, settings.skip_bad, progress)

        # Output box/fps via the pure augment plan: `source x upscale`
        # per axis, `round(source_fps x interpolate)` for fps unless
        # `presentation_fps` pins it. The audio timeline stays on the
        # source fps — frame counts / source fps = seconds.
        source_info = probe(usable[0] / "video.mp4")
        source_fps = _probe_video_fps(source_info)
        source_w, source_h = _probe_video_geometry(source_info)
        # The plan only credits the FILM multiplier when the model pass will
        # actually interpolate (legs present + work demanded); otherwise a
        # missing leg must fall back to the minterpolate lift, never to
        # slow motion.
        plan_multiplier = effective_interpolate if model_selected else 1
        plan = plan_augmentation(
            source_w,
            source_h,
            source_fps,
            upscale=effective_upscale,
            interpolate=effective_interpolate,
            presentation_fps=effective_presentation_fps,
            model_interpolate=plan_multiplier,
        )
        out_w, out_h, out_fps = plan.out_w, plan.out_h, plan.out_fps
        stretch = presentation_stretch(source_fps, effective_interpolate, out_fps)
        lift = ""
        if plan.needs_minterpolate:
            lift = f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
    final_stages["triage"] = time.monotonic() - triage_start

    # SIGKILL orphans the staging dir (never covered by the orphan scan),
    # so prune stale siblings before creating the new tmpdir — best-effort,
    # never fails finalize when the run dir is unreadable.
    prune_stale_finalize_tmpdirs(run_dir)
    with tempfile.TemporaryDirectory(prefix=FINALIZE_TMPDIR_PREFIX, dir=run_dir) as tmp:
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
        upscale_devices: tuple[str, ...] = ()
        interp_devices: tuple[str, ...] = ()
        if model_selected and resolved_weights is not None:
            from voyage.augment import (
                augment_devices,
                interp_pass_devices,
                model_pass_devices,
                upscale_pass_devices,
            )

            if augment_devices():
                tensor_devices = model_pass_devices()
                # DESIGN §140 A/V stream: the 2060 (cuda:1) owns the
                # upscale leg only, the 4060 (cuda:0) owns music takes ->
                # SFX bed -> FILM interp sequentially. 1-GPU boxes
                # collapse all three onto cuda:0 (sequential there too).
                upscale_devices = upscale_pass_devices()
                interp_devices = interp_pass_devices()
        model_start = time.monotonic()
        # One shared gate (parallel pre-fork contract): the model branch
        # takes the tensor path exactly when this holds, and both
        # model-pass entries always return a real intermediate — so the
        # pre-fork slow-mo prediction below equals the post-join value
        # and the ACE takes ledger always lands on the right timeline.
        tensor_path = bool(
            resolved_weights is not None
            and tensor_devices
            and tensor_path_armed(
                model_selected=model_selected,
                weights_present=True,
                source_fps=source_fps,
                needs_reencode=plan.needs_reencode,
                devices_available=True,
            )
        )
        # Always-deferred music inputs resolve before the fork so the dry
        # walk and the render use identical sizing (unchanged logic, moved
        # up from below the model pass).
        from voyage.audio_finalize import deferred_render_pending, ensure_deferred_for_finalize

        resolved_audio_config = audio_config
        if resolved_audio_config is None:
            try:
                from voyage.persistence import read_effective_config

                resolved_audio_config = read_effective_config(run_dir).audio
            except StateError:
                resolved_audio_config = None
        ace_models_dir = (
            getattr(resolved_audio_config, "models_dir", None)
            if resolved_audio_config is not None
            else None
        )
        if ace_models_dir is None:
            ace_models_dir = models_dir
        music_backend = (
            getattr(resolved_audio_config, "backend", "fake")
            if resolved_audio_config is not None
            else "fake"
        )
        music_style_value = (
            getattr(resolved_audio_config, "music_style", "")
            if resolved_audio_config is not None
            else ""
        )
        music_caption_value = (
            getattr(resolved_audio_config, "music_caption", None)
            if resolved_audio_config is not None
            else None
        )
        music_take_seconds = (
            getattr(resolved_audio_config, "take_seconds", 45.0)
            if resolved_audio_config is not None
            else 45.0
        )
        music_ahead_seconds = (
            getattr(resolved_audio_config, "ahead_seconds", 20.0)
            if resolved_audio_config is not None
            else 20.0
        )
        music_beats_per_segment = (
            getattr(resolved_audio_config, "beats_per_segment", 4)
            if resolved_audio_config is not None
            else 4
        )
        music_sample_rate = (
            getattr(resolved_audio_config, "sample_rate", 48000)
            if resolved_audio_config is not None
            else 48000
        )
        music_channels = (
            getattr(resolved_audio_config, "channels", 2)
            if resolved_audio_config is not None
            else 2
        )
        # Precomputed stretch: `tensor_path` predicts `tensor_intermediate
        # is not None` exactly, so this equals the old post-hoc value in
        # every success case — Thread B renders on the right timeline.
        # The stretch covers the slow-mo timeline only when the present
        # actually retimes (tensor path + explicit presentation fps +
        # stretch != 1 — `slowmo_video_active`); otherwise the mix stays
        # on the source timeline. A complete ledger (re-finalize) no-ops
        # inside the helper with no worker spawned. Phased or legacy,
        # the interp leg returns non-None exactly when the gate holds.
        slowmo = slowmo_video_active(tensor_path, effective_presentation_fps, stretch)
        audio_stretch = stretch if slowmo else 1.0
        # Dry walk (no worker, no writes): True when the takes ledger
        # still owes the timeline — the fork gate needs the same answer
        # `ensure_deferred_for_finalize` derives first thing.
        music_pending = (
            False
            if no_music
            else deferred_render_pending(
                run_dir=run_dir,
                usable=usable,
                source_fps=source_fps,
                run_seed=seed,
                music_style=music_style_value,
                explicit_caption=music_caption_value,
                stretch=audio_stretch,
                take_seconds=music_take_seconds,
                ahead_seconds=music_ahead_seconds,
                beats_per_segment=music_beats_per_segment,
                sample_rate=music_sample_rate,
                channels=music_channels,
            )
        )
        music_rendered = False
        branch_progress: VoyageConsole | None = progress
        music_branch_progress: VoyageConsole | None = progress

        # DESIGN §140 A/V stream: both legs provisioned splits the model
        # pass into Phase A (upscale-only on the 2060, overlapping the
        # music takes on the 4060) and Phase C (FILM interp on the 4060
        # after the SFX bed, picking up the Phase A upscale ledger).
        # Partial legs keep the legacy all-or-nothing flow (unchanged).
        both_legs = (
            resolved_weights is not None
            and resolved_weights.film is not None
            and resolved_weights.realesrgan is not None
        )
        phased = bool(tensor_path and both_legs and upscale_devices and interp_devices)

        def _do_upscale_phase() -> None:
            if not phased or resolved_weights is None or not upscale_devices:
                return
            # Phase A (2060, cuda:1): upscale sweep only — publishes
            # `upscaled_NN` chunk dirs + ledger records, never
            # interpolating (interp + seam/morph FILM work is Phase C).
            from voyage.augment_finalize import run_upscale_phase

            run_upscale_phase(
                run_dir,
                weights=resolved_weights,
                out_width=out_w,
                out_height=out_h,
                source_fps=source_fps,
                upscale_factor=effective_upscale,
                crf=effective_crf,
                preset=effective_preset,
                device=upscale_devices[0],
                timings=model_pass_timings,
                progress=branch_progress,
            )

        def _do_interp_phase() -> None:
            nonlocal tensor_intermediate
            if not phased or resolved_weights is None or not interp_devices:
                return
            # Phase C (4060, cuda:0): the SFX bed finished, so FILM owns
            # the card alone — interp sweep over the Phase A ledger, then
            # drain (+ seam/morph joints, same device) to one intermediate.
            from voyage.augment_finalize import run_interp_phase

            model_work = tmpdir / "model_pass"
            tensor_intermediate, _ = run_interp_phase(
                run_dir,
                usable,
                weights=resolved_weights,
                out_width=out_w,
                out_height=out_h,
                source_fps=source_fps,
                upscale_factor=effective_upscale,
                multiplier=effective_interpolate,
                crf=effective_crf,
                preset=effective_preset,
                device=interp_devices[0],
                work_dir=model_work,
                timings=model_pass_timings,
                # The Phase A fork display closed at the join — Phase C
                # reports on the main progress (sequential bars, one Live).
                progress=progress,
            )
            if tensor_intermediate is not None:
                final_stages.update(_model_pass_stage_rows(model_pass_timings, model_start))

        def _do_model_pass() -> None:
            nonlocal tensor_intermediate
            # Legacy flow (partial legs, or tensor path without the
            # phased split): the whole pass runs here, exactly as before.
            # The extra conjuncts repeat the `tensor_path` contract for
            # the type checker (True already implies both); they never
            # change the branch outcome.
            if tensor_path and resolved_weights is not None and tensor_devices:
                model_work = tmpdir / "model_pass"
                if resolved_weights.film is not None and resolved_weights.realesrgan is not None:
                    # Durable sidecar path (independent workers): poll the
                    # upscale + interp workers to completion under
                    # run_dir/augment/<plan-hash>/, drain one intermediate
                    # per usable segment, concat in segment order. A crashed
                    # finalize retries only the missing chunks.
                    from voyage.augment_finalize import run_durable_model_pass

                    # The durable legs upscale by `upscale` exactly (SRVGG
                    # 2x, Lanczos to exact below) and interpolate by
                    # `interpolate`; upscale=1 passes geometry through.
                    tensor_intermediate, _ = run_durable_model_pass(
                        run_dir,
                        usable,
                        out_width=out_w,
                        out_height=out_h,
                        source_fps=source_fps,
                        weights=resolved_weights,
                        upscale_factor=effective_upscale,
                        multiplier=effective_interpolate,
                        crf=effective_crf,
                        preset=effective_preset,
                        device=tensor_devices[0],
                        work_dir=model_work,
                        timings=model_pass_timings,
                        progress=branch_progress,
                    )
                else:
                    # Partial legs keep the legacy all-or-nothing tmpdir
                    # flow (unchanged behavior for film-only/ESRGAN-only).
                    from voyage.augment import run_finalize_model_pass

                    tensor_intermediate, _ = run_finalize_model_pass(
                        [segment / "video.mp4" for segment in usable],
                        resolved_weights,
                        source_fps=source_fps,
                        upscale_factor=effective_upscale,
                        multiplier=effective_interpolate,
                        crf=effective_crf,
                        preset=effective_preset,
                        work_dir=model_work,
                        devices=tensor_devices,
                    )
            if tensor_intermediate is not None:
                final_stages.update(_model_pass_stage_rows(model_pass_timings, model_start))

        def _do_music_takes() -> None:
            # Always-deferred music (DESIGN §140): segments commit video
            # only, so the takes render here — overlapping the upscale
            # Phase A on two cards (upscale on cuda:1, music on cuda:0)
            # or after it on one — and BEFORE the mix below. With
            # `no_music` this is a no-op (silent AAC ships instead).
            nonlocal music_rendered
            if no_music:
                return
            music_start = time.monotonic()
            music_rendered = ensure_deferred_for_finalize(
                run_dir=run_dir,
                usable=usable,
                source_fps=source_fps,
                stretch=audio_stretch,
                run_seed=seed,
                models_dir=ace_models_dir,
                device=DEFERRED_MUSIC_DEVICE,
                audio_backend=music_backend,
                music_style=music_style_value,
                explicit_caption=music_caption_value,
                take_seconds=music_take_seconds,
                ahead_seconds=music_ahead_seconds,
                beats_per_segment=music_beats_per_segment,
                sample_rate=music_sample_rate,
                channels=music_channels,
                progress=music_branch_progress,
            )
            if music_rendered:
                final_stages["music takes"] = time.monotonic() - music_start

        if model_music_parallel_armed(
            tensor_path=tensor_path,
            deferred_pending=music_pending,
            # Phased: the overlapping branch is the upscale leg (cuda:1
            # on 2-GPU, cuda:0 on 1-GPU so the gate disarms). Legacy:
            # the whole model pass, exactly as before.
            model_devices=upscale_devices if phased else tensor_devices,
        ):
            # Two cards: the upscale leg owns cuda:1 while ACE renders
            # takes on cuda:0 — one outer stage, with the two branches
            # sharing one N-stream display so upscale (model leg)
            # and ACE takes (music leg) report live bars concurrently
            # (DESIGN §59). The display degrades to plain lines when
            # progress is None or the console is not a TTY. The blend
            # clock covers the whole fork-join: upscale and music overlap,
            # so this is fork-to-mix, not music+mix.
            audio_start = time.monotonic()
            model_music_display: ParallelFinalizeDisplay | None = None
            span_progress: VoyageConsole | None = progress
            if progress is not None:
                model_music_display = ParallelFinalizeDisplay(progress)
                span_progress = model_music_display.stream_view("model pass + music takes")
                branch_progress = model_music_display.stream_view("model pass")
                music_branch_progress = model_music_display.stream_view("music takes")
            else:
                branch_progress = None
                music_branch_progress = None
            try:
                with optional_stage(span_progress, "model pass + music takes"):
                    run_model_pass_and_music_parallel(
                        model_work=_do_upscale_phase if phased else _do_model_pass,
                        music_work=_do_music_takes,
                    )
            finally:
                if model_music_display is not None:
                    model_music_display.close()
        else:
            if phased:
                _do_upscale_phase()
            else:
                _do_model_pass()
            # Sequential keeps the historical music+mix meaning of
            # `audio_blend_ms` (model time excluded, as before the fork).
            audio_start = time.monotonic()
            _do_music_takes()
        # Blended final mix (overlap re-sliced from takes).
        mix_cm = optional_stage(progress, "mix final audio")
        mix_start = time.monotonic()
        with mix_cm:
            # The audio timeline stays on the probed source fps (frame
            # counts / source fps = seconds); an unprobable fps falls back
            # to the pinned presentation rate (plan_augmentation refuses to
            # plan without one of the two).
            audio_fps = int(source_fps) if source_fps > 0 else out_fps
            music_digest = "silent"
            if no_music:
                # Silent AAC sized to the stretched timeline (same
                # frame-count math as the ledger mix, no takes needed).
                _, _, silent_timeline = _segment_timeline(usable, audio_fps / audio_stretch)
                silent_dest = tmpdir / "final_audio.wav"
                channel_layout = "mono" if settings.channels == 1 else "stereo"
                silent_proc = run_capture(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-nostdin",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        f"anullsrc=r={settings.sample_rate}:cl={channel_layout}",
                        "-t",
                        f"{silent_timeline:.6f}",
                        "-ac",
                        f"{settings.channels}",
                        "-c:a",
                        "pcm_s16le",
                        str(silent_dest),
                    ]
                )
                if silent_proc.returncode != 0:
                    raise MediaError(f"silent audio render failed: {silent_proc.stderr[-2000:]}")
                final_audio = silent_dest
            else:
                from voyage.final_mix_cache import (
                    load_music_cache,
                    music_fingerprint,
                    store_music_cache,
                )

                # Durable single-slot music cache (DESIGN §56): a no-change
                # resume reuses the last published mix instead of re-slicing
                # the takes ledger and re-joining. A later publish
                # overwrites the same slot, so the cache never grows.
                music_digest = music_fingerprint(
                    run_dir,
                    usable,
                    sample_rate=settings.sample_rate,
                    channels=settings.channels,
                    overlap_fraction=settings.effective_overlap_fraction(),
                    overlap_cap_seconds=settings.overlap_cap_seconds,
                    audio_stretch=audio_stretch,
                    audio_fps=float(audio_fps),
                )
                cached_music = load_music_cache(run_dir, music_digest)
                if cached_music is not None:
                    final_audio = tmpdir / "final_audio.wav"
                    shutil.copyfile(cached_music, final_audio)
                    if progress is not None:
                        progress.info(
                            f"music: cache hit — reusing last mix ({len(usable)} segments)"
                        )
                else:
                    final_audio = build_final_audio(
                        run_dir,
                        usable,
                        tmpdir,
                        audio_fps,
                        settings.sample_rate,
                        settings.channels,
                        settings.effective_overlap_fraction(),
                        settings.overlap_cap_seconds,
                        stretch=audio_stretch,
                    )
                    store_music_cache(run_dir, music_digest, final_audio)
        final_stages["mix audio"] = time.monotonic() - mix_start
        audio_blend_ms = (time.monotonic() - audio_start) * 1000.0
        # DESIGN §140 A/V stream: on the phased path the 4060 runs
        # music takes -> SFX bed -> FILM interp sequentially, while the
        # 2060 owned Phase A upscale only. The bed therefore renders
        # synchronously here (before Phase C interp), not overlapped
        # with the publish below. The legacy path keeps the two-stream
        # bed∥publish overlap exactly as before.
        #
        # Two-stream SFX (legacy): the bed conditions on a
        # stream-copy proxy of the committed segments, so it needs
        # nothing from the mix or the publish — start it now (cuda:0 is
        # free: the ACE worker stopped at take-render end) and let it
        # overlap the publish encode below. The main thread dubs after
        # the join; a bed failure ships music-only and the caller falls
        # back to the legacy shipped-pixels pass.
        sfx_armed = sfx_parallel_armed(sfx_request)
        sfx_display: ParallelFinalizeDisplay | None = None
        publish_progress = progress
        bed_outcome: dict[str, Any] = {}
        bed_start = 0.0
        bed_thread: threading.Thread | None = None
        bed_done_sync = False
        if sfx_armed and sfx_request is not None:
            sfx_request_snapshot = sfx_request
            # DESIGN §140 A/V stream: the phased path reuses the same
            # coordinator sequentially (bed first, publish lines later —
            # never concurrent, so no Live garble) so the sync bed keeps
            # its live BarTracker progress instead of plain lines.
            if progress is not None:
                sfx_display = ParallelFinalizeDisplay(progress)
                bed_view: VoyageConsole | None = sfx_display.stream_view("sfx bed")
                publish_progress = sfx_display.stream_view("publish")
            else:
                bed_view = None

            def _do_sfx_bed() -> None:
                try:
                    from voyage.final_mix_cache import (
                        bed_fingerprint,
                        load_bed_cache,
                        store_bed_cache,
                    )
                    from voyage.sfx_finalize import (
                        SFX_CONDITIONING_PROXY,
                        build_proxy_reference,
                        render_sfx_bed,
                        segment_sfx_bounds,
                    )

                    # Durable single-slot bed cache (DESIGN §56): a
                    # no-change resume reuses the last bed instead of
                    # re-rendering stems and re-joining. Overwrites the
                    # same slot on a miss, so the cache never grows.
                    bed_digest = bed_fingerprint(
                        run_dir,
                        usable,
                        sample_rate=settings.sample_rate,
                        channels=settings.channels,
                        backend=sfx_request_snapshot.backend,
                        model_size=sfx_request_snapshot.model_size,
                        seed=seed,
                        caption_override=sfx_request_snapshot.caption_override,
                        music_digest=music_digest,
                    )
                    cached_bed = load_bed_cache(run_dir, bed_digest)
                    if cached_bed is not None:
                        cached_wav, cached_seconds = cached_bed
                        bed = tmpdir / "sfx_bed.wav"
                        shutil.copyfile(cached_wav, bed)
                        bed_outcome["bed"] = bed
                        bed_outcome["source_seconds"] = cached_seconds
                        if bed_view is not None:
                            bed_view.info(
                                f"sfx: cache hit — reusing last bed ({len(usable)} segments)"
                            )
                        return
                    proxy_ref, source_seconds = build_proxy_reference(run_dir, usable, tmpdir)
                    bounds = segment_sfx_bounds(
                        run_dir,
                        usable,
                        sfx_request_snapshot.fps,
                        sfx_request_snapshot.caption_override,
                    )
                    bed = render_sfx_bed(
                        run_dir,
                        proxy_ref,
                        source_seconds,
                        bounds,
                        tmpdir,
                        sfx_request_snapshot.backend,
                        sfx_request_snapshot.models_dir,
                        sfx_request_snapshot.device,
                        sfx_request_snapshot.model_size,
                        seed,
                        settings.sample_rate,
                        settings.channels,
                        sfx_request_snapshot.num_workers,
                        progress=bed_view,
                        conditioning_source=SFX_CONDITIONING_PROXY,
                        conditioning_timeline=source_seconds,
                    )
                    bed_outcome["bed"] = bed
                    bed_outcome["source_seconds"] = source_seconds
                    store_bed_cache(run_dir, bed_digest, bed, source_seconds=source_seconds)
                except Exception as exc:  # noqa: BLE001 — recorded for the music-only fallback, raised never
                    bed_outcome["error"] = exc

            if phased:
                # Sequential 4060 stream: the bed owns the card alone,
                # then Phase C interp picks it up.
                bed_start = time.monotonic()
                _do_sfx_bed()
                bed_done_sync = True
                final_stages["sfx bed"] = time.monotonic() - bed_start
            else:
                bed_start = time.monotonic()
                bed_thread = threading.Thread(
                    target=_do_sfx_bed, name="voyage-sfx-bed", daemon=True
                )
                bed_thread.start()
        else:
            bed_thread = None
        if phased:
            # Phase C runs after the mix AND the SFX bed (both 4060
            # residents finished) — publish below consumes its output.
            _do_interp_phase()
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
        publish_label = (
            "encode model-pass video"
            if tensor_intermediate is not None
            else "concat native video"
            if native
            else "encode presentation video"
        )
        publish_cm = optional_stage(
            publish_progress, publish_label, f"{out_w}x{out_h}@{out_fps}fps"
        )
        publish_start = time.monotonic()
        with publish_cm:
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
                        # DESIGN §140 A/V stream: all FILM work owns the
                        # 4060 (cuda:0) — even the tiny native-path joints,
                        # so the 2060 stays upscale-only.
                        device=resolve_morph_device(
                            interp_devices[0]
                            if interp_devices
                            else (tensor_devices[0] if tensor_devices else None)
                        ),
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
            # Two-stream SFX join + dub (main thread): the bed overlapped
            # the encode above; stretch it onto the shipped timeline and
            # dub it here so `validate_video` below covers the dub and
            # `output_frames` stays honest. A bed failure ships the
            # music-only staged file and reports for the legacy fallback.
            ship = staged
            if sfx_armed and (bed_thread is not None or bed_done_sync):
                if bed_thread is not None:
                    bed_thread.join()
                    if sfx_display is not None:
                        sfx_display.close()
                    final_stages["sfx bed"] = time.monotonic() - bed_start
                # Phased path: the bed already rendered (and timed) before
                # Phase C interp — join/dub only.
                bed_error = bed_outcome.get("error")
                if bed_error is None:
                    staged_info = probe(staged)
                    shipped_seconds = float(staged_info.get("format", {}).get("duration", 0.0))
                    source_seconds = float(bed_outcome.get("source_seconds", 0.0))
                    if shipped_seconds <= 0.0 or source_seconds <= 0.0:
                        raise MediaError(
                            "sfx dub needs positive durations "
                            f"(got shipped={shipped_seconds}/{source_seconds})"
                        )
                    from voyage.sfx_finalize import stretch_and_dub_sfx_bed

                    if publish_progress is not None:
                        publish_progress.info("dub sfx bed onto final")
                    dubbed = tmpdir / "final_sfx.mp4"
                    stretch_and_dub_sfx_bed(
                        staged,
                        bed_outcome["bed"],
                        final_audio,
                        dubbed,
                        settings.sample_rate,
                        settings.channels,
                        shipped_seconds / source_seconds,
                    )
                    ship = dubbed
                    if sfx_report is not None:
                        sfx_report.update(
                            {
                                "sfx_status": "dubbed",
                                "sfx_stretch": round(shipped_seconds / source_seconds, 4),
                            }
                        )
                elif sfx_report is not None:
                    sfx_report.update({"sfx_status": "music-only", "sfx_error": str(bed_error)})
            elif sfx_report is not None:
                sfx_report.update({"sfx_status": "skipped"})
            if bed_done_sync and sfx_display is not None:
                # Phased path: the bed already rendered synchronously, so
                # stop the shared display before validate — the publish
                # lines below degrade to plain lines, same as the legacy
                # path after its join-time close.
                sfx_display.close()
            published = validate_video(ship, out_w, out_h, out_fps)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_copy(ship, output_path)
        final_stages["publish video"] = time.monotonic() - publish_start
        output_frames = published.get("frames")
        if isinstance(output_frames, int) and not isinstance(output_frames, bool):
            if publish_progress is not None:
                publish_progress.info(
                    f"final video {output_frames}f {out_w}x{out_h}@{out_fps}fps "
                    f"({len(usable)} segments)"
                )
        else:
            output_frames = None
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
                    "output_frames": output_frames,
                    "crf": effective_crf,
                    "preset": effective_preset,
                    "parts_encode_ms": round(parts_encode_ms, 1),
                    "audio_blend_ms": round(audio_blend_ms, 1),
                    "final_encode_ms": round(final_encode_ms, 1),
                    "fast_path": native and tensor_intermediate is None,
                    "upscale": effective_upscale,
                    "interpolate": effective_interpolate,
                    "presentation_fps": effective_presentation_fps,
                    "slowmo_factor": round(stretch, 4),
                    "invoker": invoker,
                    "no_music": no_music,
                    "sfx_status": (
                        sfx_report.get("sfx_status", "skipped")
                        if sfx_report is not None
                        else "skipped"
                    ),
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
        if progress is not None:
            progress.timing_table("finalize", final_stages)
    return output_path

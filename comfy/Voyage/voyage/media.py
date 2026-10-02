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
from typing import Any, Literal

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
    `needs_reencode` covers any pixel/timing change (dims differ, fps
    differs past 0.5 either way, lift, or unknown source fps) and gates
    the stream-copy fast path off.
    """
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"target geometry must be positive (got {target_w}x{target_h})")
    if requested_fps <= 0:
        raise ValueError(f"requested fps must be positive (got {requested_fps})")
    floor_fps = int(min_fps or 0)
    floor_w = int(min_width or 0)
    floor_h = int(min_height or 0)
    if floor_fps < 0 or floor_w < 0 or floor_h < 0:
        raise ValueError(f"augment floors must be >= 0 (got {min_fps}/{min_width}/{min_height})")
    out_fps = max(int(requested_fps), floor_fps, PRESENTATION_MIN_FPS)
    out_w = max(int(target_w), floor_w, int(source_w))
    out_h = max(int(target_h), floor_h, int(source_h))
    source_fps_value = float(source_fps)
    needs_minterpolate = source_fps_value > 0 and out_fps > source_fps_value + FPS_MATCH_TOLERANCE
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
    return ResolvedFinalizeSettings(
        settings=settings,
        min_fps=min_fps if min_fps is not None else settings.min_fps,
        min_width=min_width if min_width is not None else settings.min_width,
        min_height=min_height if min_height is not None else settings.min_height,
        crf=validate_crf(crf if crf is not None else settings.crf),
        preset=validate_preset(preset if preset is not None else settings.preset),
        use_model_pass=settings.use_model_pass,
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
    models_dir: Path | str | None = None,
    options: FinalizeOptions | None = None,
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
    )
    settings = resolved.settings
    effective_min_fps = resolved.min_fps
    effective_min_width = resolved.min_width
    effective_min_height = resolved.min_height
    effective_crf = resolved.crf
    effective_preset = resolved.preset
    effective_use_model_pass = resolved.use_model_pass
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
    plan = plan_augmentation(
        source_w,
        source_h,
        source_fps,
        width,
        height,
        fps,
        effective_min_fps,
        effective_min_width,
        effective_min_height,
    )
    out_w, out_h, out_fps = plan.out_w, plan.out_h, plan.out_fps
    lift = ""
    if plan.needs_minterpolate:
        lift = f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"

    with tempfile.TemporaryDirectory(prefix="voyage-final-", dir=run_dir) as tmp:
        tmpdir = Path(tmp)
        # Blended final mix (overlap re-sliced from takes; previews untouched).
        audio_start = time.monotonic()
        final_audio = build_final_audio(
            run_dir,
            usable,
            tmpdir,
            fps,
            settings.sample_rate,
            settings.channels,
            settings.effective_overlap_fraction(),
            settings.overlap_cap_seconds,
        )
        audio_blend_ms = (time.monotonic() - audio_start) * 1000.0
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
        if (
            model_selected
            and resolved_weights is not None
            and source_fps > 0
            and plan.needs_reencode
        ):
            from voyage.augment import augment_devices, model_pass_devices, run_finalize_model_pass

            if augment_devices():
                # DESIGN §140 GPU defaults: the whole model pass is pinned
                # to cuda:1 (the 2060) so the MMAudio SFX stack owns cuda:0
                # (the 4060) — `model_pass_devices` collapses to cuda:0 on
                # a 1-GPU box, so the sequential path is unchanged there.
                model_work = tmpdir / "model_pass"
                tensor_intermediate, _ = run_finalize_model_pass(
                    [segment / "video.mp4" for segment in usable],
                    resolved_weights,
                    source_fps=source_fps,
                    crf=effective_crf,
                    preset=effective_preset,
                    work_dir=model_work,
                    devices=model_pass_devices(),
                )
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
            tensor_vf = (
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={out_fps}"
            )
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
            concat_list = write_concat_list(
                [segment / "video.mp4" for segment in usable], tmpdir / "concat.txt"
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
                }
            ),
        )
    return output_path

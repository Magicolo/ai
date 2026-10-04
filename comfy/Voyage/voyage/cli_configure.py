"""`configure` verb: init or update a run manifest (DESIGN §58, two-verb CLI).

All run options live here. `configure <NAME>` creates `output/<NAME>/`
with `manifest.json` (full effective config + planned segment count),
or updates the stored config in place touching only provided flags.
`--from <OTHER>` (create only) inherits style + tuning + segment count
from another run's manifest — explicit flags override, the seed stays
fresh unless `--seed` is passed. `generate <NAME>` later reconciles
the directory against the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from voyage.cli_core import _augment_overrides, get_console
from voyage.cli_paths import _check_run_id, output_root
from voyage.cli_planning import _frames_per_segment, segments_for_duration
from voyage.config import (
    ProjectConfig,
    VideoBackendName,
    is_provided,
    preset_config,
    resolve_config,
)
from voyage.errors import DiskSpaceError, MediaError, StateError
from voyage.persistence import (
    build_manifest,
    read_effective_config,
    read_manifest,
    read_state,
    write_manifest,
    write_state,
)


def _manifest_path(run_dir: Path) -> Path:
    from voyage import paths

    return run_dir / paths.MANIFEST_FILENAME


def _resolve_segments(args: argparse.Namespace, fps: int, frames_per_segment: int) -> int | None:
    """Planned segment count from --segments XOR --duration (None if neither)."""
    segments = getattr(args, "segments", None)
    duration = getattr(args, "duration", None)
    if segments is not None and duration is not None:
        print("error: pass only one of --segments or --duration", file=sys.stderr)
        return None
    if segments is not None:
        if isinstance(segments, bool) or not isinstance(segments, int) or segments <= 0:
            print(f"error: --segments must be positive, got {segments}", file=sys.stderr)
            return None
        return segments
    if duration is not None:
        try:
            return segments_for_duration(float(duration), fps, frames_per_segment)
        except (TypeError, ValueError) as exc:
            print(f"error: bad --duration: {exc}", file=sys.stderr)
            return None
    return None


def trim_overflow_segments(run_dir: Path, keep: int, fps: int) -> int:
    """Delete segment dirs at index >= keep; recompute video counters.

    Only overflow beyond the new plan is removed — committed history
    below `keep` is never touched. Returns the recomputed timeline
    frames (sum over surviving DONE segments). Takes covering at/after
    the new end are dropped from the ledger with their wav files.
    """
    from voyage import paths
    from voyage.segment_manifest import load_segment_manifest

    segments_dir = run_dir / paths.SEGMENTS_DIRNAME
    if segments_dir.is_dir():
        for child in sorted(segments_dir.iterdir()):
            if len(child.name) == 6 and child.name.isdigit() and int(child.name) >= keep:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    frames = 0
    if segments_dir.is_dir():
        for index in range(keep):
            segment = segments_dir / f"{index:06d}"
            if not (segment / "DONE").is_file():
                continue
            try:
                metrics = load_segment_manifest(segment).get("metrics")
            except (OSError, ValueError, MediaError):
                metrics = None
            count = metrics.get("frames", 0) if isinstance(metrics, dict) else 0
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise StateError(f"segment {segment.name} has no frame count — cannot recount")
            frames += count
    state = read_state(run_dir)
    state.committed_segments = min(state.committed_segments, keep)
    state.next_segment_number = min(state.next_segment_number, keep)
    state.timeline_frames = frames
    write_state(run_dir, state)
    new_end = frames / fps if fps else 0.0
    ledger = run_dir / "audio" / "takes.jsonl"
    if ledger.is_file() and new_end >= 0:
        from voyage.audio.planner import TAKES_FILENAME, load_takes

        assert ledger.name == TAKES_FILENAME
        takes = load_takes(ledger)
        kept = [take for take in takes if take.covers_from < new_end]
        if len(kept) < len(takes):
            from voyage.atomic import fsync_dir

            partial = ledger.with_name(f"{ledger.name}.partial")
            with partial.open("w", encoding="utf-8") as handle:
                for take in kept:
                    handle.write(json.dumps(take.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, ledger)
            fsync_dir(ledger.parent)
            dropped_ids = {take.take_id for take in takes} - {take.take_id for take in kept}
            for take in takes:
                if take.take_id in dropped_ids:
                    try:
                        take.resolved_path(run_dir).unlink(missing_ok=True)
                    except OSError:
                        pass
    return frames


def cmd_configure(args: argparse.Namespace) -> int:
    """Init (`manifest.json` absent) or update (present) a run manifest."""
    from voyage.doctor import check_ffmpeg
    from voyage.media import check_free_space
    from voyage.seeds import random_master_seed

    name = getattr(args, "name", "")
    if not isinstance(name, str) or _check_run_id(name.strip()) != 0:
        return 2
    run_dir = (output_root() / name.strip()).resolve()
    manifest_path = _manifest_path(run_dir)
    from_name = getattr(args, "from_run", None)
    if manifest_path.is_file() and is_provided(from_name):
        print(
            "error: --from only applies when creating a run "
            "(update touches only provided flags instead)",
            file=sys.stderr,
        )
        return 2
    stored_config = None
    if manifest_path.is_file():
        try:
            stored_config = read_effective_config(run_dir)
        except StateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    source_config: ProjectConfig | None = None
    source_segments: int | None = None
    if stored_config is None and is_provided(from_name):
        if not isinstance(from_name, str) or not from_name.strip():
            print("error: --from needs a run name (output/<NAME>)", file=sys.stderr)
            return 2
        if _check_run_id(from_name.strip()) != 0:
            return 2
        src_dir = (output_root() / from_name.strip()).resolve()
        try:
            source_config = read_effective_config(src_dir)
            raw_segments = read_manifest(src_dir).get("segments")
        except StateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if (
            isinstance(raw_segments, int)
            and not isinstance(raw_segments, bool)
            and raw_segments > 0
        ):
            source_segments = raw_segments
    if stored_config is None:
        style_value = getattr(args, "style", None)
        if source_config is not None and not is_provided(style_value):
            style_value = source_config.style
        if not isinstance(style_value, str) or not style_value.strip():
            print("error: --style must be a non-empty human-owned style string", file=sys.stderr)
            return 2
        if run_dir.exists() and any(run_dir.iterdir()) and not bool(getattr(args, "force", False)):
            print(
                f"refusing to configure into non-empty directory {run_dir} (use --force)",
                file=sys.stderr,
            )
            return 2
        seed_value = getattr(args, "seed", None)
        if seed_value is None:
            seed_value = random_master_seed()
            print(f"random seed for this run: {seed_value} (re-run with --seed {seed_value})")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            print(f"error: --seed must be an integer, got {seed_value!r}", file=sys.stderr)
            return 2
        backend_value = getattr(args, "backend", None)
        director_value = getattr(args, "director", None)
        director_device_value = getattr(args, "director_device", None)
        from voyage.config import BACKEND_REGISTRY

        if source_config is None:
            backend_value = backend_value or "ltx25"
            director_value = director_value or "llama"
            director_device_value = director_device_value or "cuda:1"
        if is_provided(backend_value) and backend_value not in BACKEND_REGISTRY:
            known = ", ".join(sorted(BACKEND_REGISTRY))
            print(
                f"error: unknown video backend {backend_value!r} (known: {known})",
                file=sys.stderr,
            )
            return 2
        try:
            if source_config is not None:
                base = source_config.model_copy(
                    update={"name": name.strip(), "style": style_value, "seed": seed_value}
                )
            else:
                base = preset_config(
                    name.strip(),
                    style_value,
                    seed_value,
                    video_backend=cast(VideoBackendName, backend_value),
                    director_backend=cast(str, director_value),
                    director_device=cast(str, director_device_value),
                )
            effective = resolve_config(
                base,
                backend=getattr(args, "backend", None),
                director=getattr(args, "director", None),
                director_device=getattr(args, "director_device", None),
                blocks=getattr(args, "blocks", None),
                take_seconds=getattr(args, "take_seconds", None),
                quantization=getattr(args, "quantization", None),
                beats_per_segment=getattr(args, "beats_per_segment", None),
                drift_every_n_segments=getattr(args, "drift_every_n", None),
                music_caption=getattr(args, "music_caption", None),
                video_caption=getattr(args, "video_caption", None),
                **_augment_overrides(args),
            )
        except (ValidationError, ValueError) as exc:
            print(f"error: invalid numeric override: {exc}", file=sys.stderr)
            return 2
        planned = _resolve_segments(args, effective.video.fps, _frames_per_segment(effective))
        if (
            planned is None
            and source_segments is not None
            and not is_provided(getattr(args, "segments", None))
            and not is_provided(getattr(args, "duration", None))
        ):
            planned = source_segments
        if planned is None:
            print(
                "error: pass one of --segments or --duration to configure a new run",
                file=sys.stderr,
            )
            return 2
    else:
        if is_provided(getattr(args, "backend", None)):
            try:
                committed_now = read_state(run_dir).committed_segments
            except StateError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            if committed_now > 0 and getattr(args, "backend", None) != stored_config.video.backend:
                print(
                    "error: --backend change on a committed run is refused "
                    "(geometries would mix — configure a fresh NAME instead)",
                    file=sys.stderr,
                )
                return 2
        try:
            effective = resolve_config(
                stored_config,
                backend=getattr(args, "backend", None),
                director=getattr(args, "director", None),
                director_device=getattr(args, "director_device", None),
                blocks=getattr(args, "blocks", None),
                take_seconds=getattr(args, "take_seconds", None),
                quantization=getattr(args, "quantization", None),
                beats_per_segment=getattr(args, "beats_per_segment", None),
                drift_every_n_segments=getattr(args, "drift_every_n", None),
                music_caption=getattr(args, "music_caption", None),
                video_caption=getattr(args, "video_caption", None),
                **_augment_overrides(args),
            )
            if is_provided(getattr(args, "style", None)):
                style_value = getattr(args, "style", None)
                assert isinstance(style_value, str)
                effective = effective.model_copy(update={"style": style_value})
            if is_provided(getattr(args, "seed", None)):
                effective = effective.model_copy(update={"seed": args.seed})
        except (ValidationError, ValueError) as exc:
            print(f"error: invalid numeric override: {exc}", file=sys.stderr)
            return 2
        planned = _resolve_segments(args, effective.video.fps, _frames_per_segment(effective))
        if planned is None and (
            getattr(args, "segments", None) is not None
            or getattr(args, "duration", None) is not None
        ):
            return 2
        if planned is None:
            try:
                previous = read_manifest(run_dir).get("segments")
            except StateError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            planned = previous if isinstance(previous, int) and previous > 0 else None
            if planned is None:
                print(
                    "error: stored manifest has no segment count — pass --segments",
                    file=sys.stderr,
                )
                return 2
    # Configure is offline-safe: it writes the manifest and ensures model
    # files, but never spawns workers — so no CUDA-stack gate here (the
    # slim image configures CUDA runs; `generate` fails fast without torch).
    console = get_console(args)
    ffmpeg_ok, ffmpeg_message = check_ffmpeg()
    if not ffmpeg_ok:
        print(f"error: {ffmpeg_message}", file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        check_free_space(run_dir, effective.min_free_space_gib)
    except DiskSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    from voyage.models_ensure import ensure_models

    sfx_enabled = not bool(getattr(args, "no_sfx", False)) and effective.sfx.backend != "fake"
    if (
        ensure_models(
            effective,
            sfx_enabled,
            console,
            allow_download=not bool(getattr(args, "no_download", False)),
            augment_enabled=not bool(getattr(args, "no_augment", False))
            and (effective.augment.min_fps > 0 or effective.augment.min_width > 0),
        )
        != 0
    ):
        return 1
    return _commit_manifest(
        args, run_dir, name.strip(), effective, planned, stored_config is not None
    )


def _commit_manifest(
    args: argparse.Namespace,
    run_dir: Path,
    name: str,
    effective: Any,
    planned: int,
    is_update: bool,
) -> int:
    """Write the manifest (+ fresh state on create); trim overflow on shrink."""
    from voyage import paths
    from voyage.persistence import create_run_dir

    if not is_update:
        create_run_dir(
            run_dir,
            effective,
            segments=planned,
            final_video=getattr(args, "final_video", None),
            skip_bad=bool(getattr(args, "skip_bad", False)),
            no_sfx=bool(getattr(args, "no_sfx", False)),
        )
        print(f"configured {name}: {planned} segments at {run_dir}")
        return 0
    try:
        previous = read_manifest(run_dir)
        state = read_state(run_dir)
    except StateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if planned < state.committed_segments:
        try:
            trim_overflow_segments(run_dir, planned, effective.video.fps)
        except (StateError, OSError, ValueError) as exc:
            print(f"error: cannot trim to {planned} segments: {exc}", file=sys.stderr)
            return 1
        print(f"trimmed overflow segments to {planned} (state.json ruled)")
    final_video_value = getattr(args, "final_video", None) or previous.get("final_video")
    manifest = build_manifest(
        effective,
        segments=planned,
        final_video=final_video_value if isinstance(final_video_value, str) else None,
        skip_bad=bool(getattr(args, "skip_bad", False)),
        no_sfx=bool(getattr(args, "no_sfx", False)),
    )

    write_manifest(run_dir, manifest)
    _ = paths.SEGMENTS_DIRNAME
    print(f"reconfigured {name}: {planned} segments at {run_dir}")
    return 0

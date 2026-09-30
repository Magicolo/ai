"""`validate` verb + read-only consistency check (DESIGN §70).

Verb module of the issue-080 split: checksum/metrics/orphan scans.
`validate_run` never mutates the run — the commit path and `finalize`
share it as a library call.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from voyage import paths
from voyage.cli_paths import _run_dir_arg
from voyage.concepts import validate_concepts
from voyage.errors import MediaError, StateError
from voyage.media import AV_ALIGNMENT_TOLERANCE_SECONDS
from voyage.media import av_drift_seconds as _av_drift_seconds
from voyage.persistence import read_manifest, read_state
from voyage.supervisor import sha256_file

_SEGMENT_ID_PATTERN = re.compile(r"^\d{6}$")


def _check_segment_checksums(segment: Path) -> list[str]:
    """Recompute sha256.json entries (DESIGN §70: checksum mismatches).

    Media entries are required; metadata entries (issue 095) verify when
    recorded and stay silent when absent, so pre-fix two-entry manifests
    keep validating (additive, never a new error on legacy runs).
    """
    errors: list[str] = []
    checksums_path = segment / "sha256.json"
    if not checksums_path.exists():
        return errors  # missing file already reported by the caller
    try:
        expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    except (ValueError, OSError, RecursionError):
        return [f"{segment.name} has unreadable sha256.json"]
    if not isinstance(expected, dict):
        return [f"{segment.name} has malformed sha256.json"]
    for artifact in ("video.mp4", "audio.wav"):
        recorded = expected.get(artifact)
        target = segment / artifact
        if not target.exists():
            continue  # missing artifact already reported by the caller
        if not target.is_file():
            errors.append(f"{segment.name} {artifact} is not a file")
            continue
        if not isinstance(recorded, str) or not recorded:
            errors.append(f"{segment.name} sha256.json missing entry for {artifact}")
            continue
        try:
            digest = sha256_file(target)
        except OSError as exc:
            errors.append(f"{segment.name} {artifact} is unreadable ({exc})")
            continue
        if digest != recorded:
            errors.append(f"{segment.name} checksum mismatch for {artifact}")
    for artifact in (
        "metrics.json",
        "transition.json",
        "prompt_plan.json",
        "audio_state.json",
        "world_state.json",
    ):
        recorded = expected.get(artifact)
        if not isinstance(recorded, str) or not recorded:
            continue  # legacy manifest: media only means "not covered"
        target = segment / artifact
        if not target.exists():
            continue  # missing artifact already reported by the caller
        if not target.is_file():
            errors.append(f"{segment.name} {artifact} is not a file")
            continue
        try:
            digest = sha256_file(target)
        except OSError as exc:
            errors.append(f"{segment.name} {artifact} is unreadable ({exc})")
            continue
        if digest != recorded:
            errors.append(f"{segment.name} checksum mismatch for {artifact}")
    return errors


def _check_segment_metrics(segment: Path, run_dir: Path | None = None) -> tuple[list[str], int]:
    """Frame ranges, durations, A/V drift, recovery tapes (DESIGN §70).

    The drift check reads the stored `metrics.video/audio.duration` —
    no probe needed — so `validate` enforces the same 0.6 s budget the
    finalizer does (issue 003). Recovery tapes resolve run-relative
    (issue 016 consumer side); `run_dir=None` keeps the legacy
    as-is check for callers without a run context.

    Returns (errors, frames).
    """
    errors: list[str] = []
    metrics_path = segment / "metrics.json"
    if not metrics_path.exists():
        return [f"{segment.name} DONE but missing metrics.json"], 0
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frames = int(metrics.get("frames", 0))
    except (ValueError, KeyError, AttributeError, TypeError, OSError, RecursionError):
        return [f"{segment.name} has unreadable metrics.json"], 0
    if frames <= 0:
        errors.append(f"{segment.name} has non-positive frame count {frames}")
    durations: dict[str, float] = {}
    for key in ("video", "audio"):
        block = metrics.get(key)
        if isinstance(block, dict):
            try:
                duration = float(block.get("duration", 0.0))
            except (TypeError, ValueError):
                duration = 0.0
            durations[key] = duration
            if duration <= 0:
                errors.append(f"{segment.name} has non-positive {key} duration")
    video_duration = durations.get("video", 0.0)
    audio_duration = durations.get("audio", 0.0)
    if video_duration > 0 and audio_duration > 0:
        drift = _av_drift_seconds(video_duration, audio_duration)
        if drift > AV_ALIGNMENT_TOLERANCE_SECONDS:
            errors.append(
                f"{segment.name} A/V alignment drift {drift:.3f}s "
                f"exceeds {AV_ALIGNMENT_TOLERANCE_SECONDS:.1f}s"
            )
    tape = metrics.get("recovery_tape")
    if isinstance(tape, str) and tape:
        if run_dir is None:
            tape_path: Path | None = Path(tape)
        else:
            try:
                tape_path = paths.resolve_stored_path(run_dir, tape)
            except MediaError as exc:
                errors.append(
                    f"{segment.name} references outside-the-run recovery checkpoint {tape} ({exc})"
                )
                return errors, frames
        if tape_path is not None and not tape_path.exists():
            errors.append(f"{segment.name} references missing recovery checkpoint {tape}")
    return errors, frames


_ORPHAN_PATTERNS = ("*.partial", "*.tmp.npy", "*.tmp*")
"""Transient-file globs for the validate orphan scan (issue 058).

`*.partial` covers atomic-write staging; `*.tmp*` (which subsumes the
explicit `*.tmp.npy`) covers concept-vector temps
(`concept_vectors.npy.<pid>.tmp.npy`) and any future pid-suffixed
staging. Legit artifacts never use these suffixes.
"""


def _collect_transient_orphans(root: Path, base: Path) -> list[str]:
    """Sorted base-relative transient files under `root` (issue 058)."""
    found: set[str] = set()
    if root.exists():
        for pattern in _ORPHAN_PATTERNS:
            for candidate in root.rglob(pattern):
                if candidate.is_file():
                    found.add(str(candidate.relative_to(base)))
    return sorted(found)


def validate_run(run_dir: Path) -> list[str]:
    """Read-only consistency check (DESIGN §70). Never mutates the run."""
    errors: list[str] = []
    try:
        state = read_state(run_dir)
        read_manifest(run_dir)
    except StateError as exc:
        return [str(exc)]
    segments_root = run_dir / paths.SEGMENTS_DIRNAME
    committed = (
        sorted(
            p for p in segments_root.iterdir() if p.is_dir() and (p / paths.DONE_MARKER).exists()
        )
        if segments_root.exists()
        else []
    )
    if len(committed) != state.committed_segments:
        errors.append(
            f"state claims {state.committed_segments} segments, found {len(committed)} DONE"
        )
    for position, segment in enumerate(committed):
        if not _SEGMENT_ID_PATTERN.match(segment.name):
            errors.append(f"invalid segment numbering: {segment.name}")
        elif segment.name != f"{position:06d}":
            errors.append(f"segment numbering gap: expected {position:06d}, found {segment.name}")
    expected_frames = 0
    for segment in committed:
        for name in ("video.mp4", "audio.wav", "world_state.json", "sha256.json"):
            if not (segment / name).exists():
                errors.append(f"{segment.name} DONE but missing {name}")
        errors.extend(_check_segment_checksums(segment))
        metric_errors, frames = _check_segment_metrics(segment, run_dir)
        errors.extend(metric_errors)
        expected_frames += frames
    if expected_frames != state.timeline_frames:
        errors.append(
            f"timeline frames {state.timeline_frames} != sum of segment frames {expected_frames}"
        )
    orphans = _collect_transient_orphans(segments_root, segments_root)
    orphans.extend(_collect_transient_orphans(run_dir / "novelty", run_dir))
    orphans = sorted(set(orphans))
    if orphans:
        errors.append(f"orphan transient files: {orphans}")
    novelty_dir = run_dir / "novelty"
    if novelty_dir.exists() or (run_dir / paths.CONCEPTS_FILENAME).exists():
        errors.extend(validate_concepts(novelty_dir))
    if not isinstance(state.fps, int) or state.fps <= 0:
        errors.append(
            f"state fps is corrupt: {state.fps!r} (expected a positive integer; "
            "repair with the run config's video fps)"
        )
    fps = state.fps if isinstance(state.fps, int) and state.fps > 0 else 24
    timeline = state.timeline_frames / fps
    if timeline > 0.0:
        from voyage.sfx_finalize import validate_sfx_ledger

        errors.extend(validate_sfx_ledger(run_dir, timeline))
    return errors


def cmd_validate(args: argparse.Namespace) -> int:
    """Read-only consistency check (DESIGN §70). Never mutates the run."""
    run_dir = _run_dir_arg(args.run)
    errors = validate_run(run_dir)
    if errors:
        print("INVALID:")
        for error in errors:
            print(f"  - {error}")
        return 1
    committed = (
        len(
            [
                p
                for p in (run_dir / paths.SEGMENTS_DIRNAME).iterdir()
                if p.is_dir() and (p / paths.DONE_MARKER).exists()
            ]
        )
        if (run_dir / paths.SEGMENTS_DIRNAME).exists()
        else 0
    )
    try:
        frames = read_state(run_dir).timeline_frames
    except StateError:
        frames = 0
    print(f"VALID: {committed} committed segments, {frames} frames")
    return 0

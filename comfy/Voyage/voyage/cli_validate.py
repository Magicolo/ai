"""`validate` verb + read-only consistency check (DESIGN §70).

Verb module of the issue-080 split: checksum/metrics/orphan scans.
`validate_run` never mutates the run — the commit path and `finalize`
share it as a library call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from voyage import paths
from voyage.concepts import validate_concepts
from voyage.errors import MediaError, StateError
from voyage.persistence import read_effective_config, read_manifest, read_state
from voyage.segment_manifest import load_segment_manifest
from voyage.supervisor import sha256_file

_SEGMENT_ID_PATTERN = re.compile(r"^\d{6}$")


def _check_segment_checksums(segment: Path) -> list[str]:
    """Recompute manifest checksums (DESIGN §70: checksum mismatches).

    Video-only since the all-deferred audio cleanup: only `video.mp4`
    is required. Any other recorded entry verifies when recorded and
    stays silent when absent — except a recorded `audio.wav` entry,
    which is skipped silently for old runs (extra file ignored, extra
    checksum entry ignored) — so old runs keep validating (additive,
    never a new error on legacy runs).
    """
    errors: list[str] = []
    manifest_file = segment / paths.SEGMENT_MANIFEST_FILENAME
    legacy_file = segment / "sha256.json"
    if manifest_file.exists():
        source = "manifest.json"
        try:
            manifest = load_segment_manifest(segment)
        except (ValueError, OSError, RecursionError, MediaError):
            return [f"{segment.name} has unreadable manifest.json"]
        expected = manifest.get("checksums")
        if not isinstance(expected, dict):
            return [f"{segment.name} has malformed manifest.json"]
    elif legacy_file.exists():
        source = "sha256.json"
        try:
            expected = json.loads(legacy_file.read_text(encoding="utf-8"))
        except (ValueError, OSError, RecursionError):
            return [f"{segment.name} has unreadable sha256.json"]
        if not isinstance(expected, dict):
            return [f"{segment.name} has malformed sha256.json"]
    else:
        return errors  # missing file already reported by the caller
    for artifact in ("video.mp4",):
        recorded = expected.get(artifact)
        target = segment / artifact
        if not target.exists():
            continue  # missing artifact already reported by the caller
        if not target.is_file():
            errors.append(f"{segment.name} {artifact} is not a file")
            continue
        if not isinstance(recorded, str) or not recorded:
            errors.append(f"{segment.name} {source} missing entry for {artifact}")
            continue
        try:
            digest = sha256_file(target)
        except OSError as exc:
            errors.append(f"{segment.name} {artifact} is unreadable ({exc})")
            continue
        if digest != recorded:
            errors.append(f"{segment.name} checksum mismatch for {artifact}")
    for artifact, recorded in sorted(expected.items()):
        if artifact in ("video.mp4", "audio.wav"):
            continue  # video already verified; audio.wav silently skipped (old runs)
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
    """Frame counts, video duration, recovery tapes (DESIGN §70).

    Video-only since the all-deferred audio cleanup: only
    `metrics.video.duration` / `frames` are checked (no audio duration,
    no A/V drift gate — audio finalizes from takes at finalize time).
    Recovery tapes resolve run-relative (issue 016 consumer side);
    `run_dir=None` keeps the legacy as-is check for callers without a
    run context.

    Returns (errors, frames).
    """
    errors: list[str] = []
    manifest_file = segment / paths.SEGMENT_MANIFEST_FILENAME
    legacy_metrics = segment / "metrics.json"
    manifest_present = manifest_file.exists()
    if not manifest_present and not legacy_metrics.exists():
        if (segment / "sha256.json").exists():
            return [f"{segment.name} DONE but missing metrics.json"], 0
        return [f"{segment.name} DONE but missing manifest.json"], 0
    try:
        manifest = load_segment_manifest(segment)
    except (ValueError, OSError, RecursionError, MediaError):
        return [f"{segment.name} has unreadable manifest.json"], 0
    metrics_any = manifest.get("metrics")
    metrics = dict(metrics_any) if isinstance(metrics_any, dict) else {}
    if not metrics:
        if manifest_present:
            return [f"{segment.name} has unreadable manifest.json"], 0
        if legacy_metrics.exists():
            return [f"{segment.name} has unreadable metrics.json"], 0
        return [f"{segment.name} DONE but missing metrics.json"], 0
    try:
        frames = int(metrics.get("frames", 0))
    except (ValueError, KeyError, AttributeError, TypeError, OSError, RecursionError):
        if manifest_present:
            return [f"{segment.name} has unreadable manifest.json"], 0
        return [f"{segment.name} has unreadable metrics.json"], 0
    if frames <= 0:
        errors.append(f"{segment.name} has non-positive frame count {frames}")
    video_block = metrics.get("video")
    if isinstance(video_block, dict):
        try:
            video_duration = float(video_block.get("duration", 0.0))
        except (TypeError, ValueError):
            video_duration = 0.0
        if video_duration <= 0:
            errors.append(f"{segment.name} has non-positive video duration")
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


_ORPHAN_PATTERNS = ("*.partial", "*.partial.*", "*.tmp.npy", "*.tmp*")
"""Transient-file globs for the validate orphan scan (issue 058).

`*.partial` covers atomic-write staging (`state.json.*.partial`,
`record.json.partial`); `*.partial.*` covers suffixed temps that keep
their real extension for format inference (SFX `<name>.partial.wav`,
drain `<name>.partial.mp4` — same convention as `prune_stale_partials`);
`*.tmp*` (which subsumes the explicit `*.tmp.npy`) covers
concept-vector temps (`concept_vectors.npy.<pid>.tmp.npy`) and any future
pid-suffixed staging. Legit artifacts never use these suffixes.
"""

ORPHAN_PATTERNS = _ORPHAN_PATTERNS
"""Unified orphan glob vocabulary (Track C single source).

Run-root, segment, novelty, audio, and augment scans share this tuple —
`_collect_transient_orphans` is the single collector. Kept as an alias
(rather than renaming `_ORPHAN_PATTERNS` outright) so existing importers
and tests hold while new code imports the public name.
"""


def _collect_transient_orphans(root: Path, base: Path) -> list[str]:
    """Sorted base-relative transient files under `root` (issue 058)."""
    found: set[str] = set()
    if root.exists():
        for pattern in ORPHAN_PATTERNS:
            for candidate in root.rglob(pattern):
                if candidate.is_file():
                    found.add(str(candidate.relative_to(base)))
    return sorted(found)


def _collect_final_tmpdirs(run_dir: Path) -> list[str]:
    """Sorted run-relative `voyage-final-*` leftovers (crashed finalize staging).

    Why a separate helper: `TemporaryDirectory(prefix="voyage-final-",
    dir=run_dir)` cleans up on success, but a kill/crash strands the dir —
    it holds chunk intermediates, never final artifacts, so flagging it as
    an orphan is read-only hygiene (validate never deletes). Only direct
    children match (the staging call always uses `dir=run_dir`).
    """
    found: list[str] = []
    for candidate in sorted(run_dir.glob("voyage-final-*")):
        if candidate.is_dir():
            found.append(str(candidate.relative_to(run_dir)))
    return found


def _check_sidecar_plan_consistency(run_dir: Path) -> list[str]:
    """Read-only ledger-vs-output notice for durable augment plan dirs.

    Why read-only: a ledgered stage whose PNG dir is gone or short means
    the next poller pass will heal it (ledger-truth + output-truth rejoin)
    — finalize has not run yet, so validate reports instead of healing.
    `chunk_mp4`-stage groups check the durable mp4 file instead of a PNG
    dir, and PNG-stage findings are suppressed for chunks with a complete
    durable mp4 (their dirs were pruned by design after the mp4
    superseded them). Skips `morph_joints` (record.json ledger, not
    sidecar) and plan dirs with no ledger (empty/forked dirs are GC's
    concern, not corruption). Torn ledger tails are already skipped by
    the sidecar loader.
    """
    from voyage import augment_sidecar as sidecar

    errors: list[str] = []
    augment_root = run_dir / sidecar.AUGMENT_DIRNAME
    if not augment_root.is_dir():
        return []
    for plan_dir in sorted(augment_root.iterdir()):
        if not plan_dir.is_dir():
            continue
        if plan_dir.name == "morph_joints":
            continue
        ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
        if not ledger.is_file():
            continue
        records = sidecar.load_chunk_ledger(ledger)
        latest: dict[tuple[int, str], dict[str, object]] = {}
        for record in records:
            stage = record.get("stage")
            index = record.get("chunk_index")
            if stage not in (
                sidecar.STAGE_UPSCALED,
                sidecar.STAGE_INTERPOLATED,
                sidecar.STAGE_CHUNK_MP4,
            ):
                continue
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            latest[(index, str(stage))] = record
        # Chunks with a complete durable mp4 keep ledgered PNG stages
        # whose dirs were pruned by design — suppress their findings
        # (mirrors the heal exemption in `augment_sidecar`).
        mp4_ok = {
            index
            for (index, stage) in latest
            if stage == sidecar.STAGE_CHUNK_MP4 and sidecar.chunk_mp4_file_complete(plan_dir, index)
        }
        for (index, stage), record in sorted(latest.items()):
            if stage == sidecar.STAGE_CHUNK_MP4:
                if index not in mp4_ok:
                    errors.append(
                        f"augment plan {plan_dir.name} chunk {index} stage {stage} "
                        "ledgered but output missing or empty"
                    )
                continue
            if index in mp4_ok:
                continue
            expected = record.get("expected_frames")
            if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
                continue
            png_dir = plan_dir / f"{stage}_{index:02d}"
            try:
                found = sorted(png_dir.glob("frame_*.png")) if png_dir.is_dir() else []
                complete = len(found) == expected and all(
                    frame.is_file() and frame.stat().st_size > 0 for frame in found
                )
            except OSError:
                complete = False
                found = []
            if not complete:
                errors.append(
                    f"augment plan {plan_dir.name} chunk {index} stage {stage} "
                    f"ledgered but output missing or short "
                    f"(expected {expected}, found {len(found)})"
                )
    return errors


def _check_morph_joints_ledger_vs_output(run_dir: Path) -> list[str]:
    """Read-only morph-joint ledger-vs-output notice (Track C).

    Mirrors `_check_sidecar_plan_consistency` for the `morph_joints`
    record.json ledger (skipped there): every ledgered joint video must
    exist non-empty, else the next finalize re-renders it. Torn lines
    are skipped by the loader, never fatal here.
    """
    errors: list[str] = []
    morph_root = run_dir / "augment" / "morph_joints"
    ledger = morph_root / "record.json"
    if not morph_root.is_dir() or not ledger.is_file():
        return []
    try:
        import json as _json

        raw = _json.loads(ledger.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"augment morph_joints ledger unreadable: {ledger}"]
    records = (
        raw if isinstance(raw, list) else raw.get("joints", []) if isinstance(raw, dict) else []
    )
    if not isinstance(records, list):
        return []
    for record in records:
        if not isinstance(record, dict):
            continue
        rel = record.get("path", record.get("video", ""))
        if not isinstance(rel, str) or not rel:
            continue
        if rel.startswith("/") or ".." in rel.split("/"):
            continue
        # Accept either run-relative or morph-dir-local resolution.
        candidates = [run_dir / rel, morph_root / Path(rel).name]
        if not any(c.is_file() and c.stat().st_size > 0 for c in candidates if _safe_is_file(c)):
            errors.append(f"augment morph_joints ledgered but output missing or empty: {rel}")
    return errors


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


ADVISORY_PREFIXES = ("warning:", "note:")
"""Prefixes marking hygiene info inside `validate_run` output (never INVALID).

`warning:` = stranded-dir hints, `note:` = disposable scratch size. Gates
that abort on validation output must filter these via `is_advisory`
(configure's trim gate and generate's pre-finalize gate both do) instead
of treating every line as fatal.
"""


def is_advisory(line: str) -> bool:
    """True when a `validate_run` line is hygiene info, not a hard error."""
    return line.startswith(ADVISORY_PREFIXES)


def _warn_done_less_numeric_dirs(segments_root: Path) -> list[str]:
    """Warn on numeric dirs without DONE (Track C hygiene, never fatal).

    A `NNNNNN/` dir without DONE is a crashed/stranded commit attempt —
    not corruption (validate only counts DONE dirs), but silent strands
    accumulate. Warns sorted, capped at 20 entries to keep output stable.
    """
    warnings: list[str] = []
    if not segments_root.exists():
        return []
    try:
        children = sorted(segments_root.iterdir())
    except OSError:
        return []
    for child in children:
        try:
            is_dir = child.is_dir() and not child.is_symlink()
        except OSError:
            continue
        if is_dir and _SEGMENT_ID_PATTERN.match(child.name) and not (child / "DONE").exists():
            warnings.append(f"{child.name} numeric dir without DONE (stranded commit attempt?)")
    if len(warnings) > 20:
        warnings = warnings[:20] + [f"... and {len(warnings) - 20} more"]
    return [f"warning: {warning}" for warning in warnings]


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
        # Video-only: a present-but-unrequired audio.wav (old runs, or a
        # supervisor that still writes previews) is ignored, never an error.
        if not (segment / "video.mp4").exists():
            errors.append(f"{segment.name} DONE but missing video.mp4")
        if (segment / paths.SEGMENT_MANIFEST_FILENAME).exists():
            pass  # metadata lives inside the manifest
        else:
            for name in ("world_state.json", "sha256.json"):
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
    orphans.extend(_collect_transient_orphans(run_dir / "audio", run_dir))
    orphans.extend(_collect_transient_orphans(run_dir / "augment", run_dir))
    orphans.extend(_collect_final_tmpdirs(run_dir))
    orphans.extend(
        sorted(
            str(path.relative_to(run_dir)) for path in run_dir.glob("*.partial") if path.is_file()
        )
    )
    orphans = sorted(set(orphans))
    if orphans:
        errors.append(f"orphan transient files: {orphans}")
    errors.extend(_check_sidecar_plan_consistency(run_dir))
    errors.extend(_check_morph_joints_ledger_vs_output(run_dir))
    errors.extend(_warn_done_less_numeric_dirs(segments_root))
    # Track C tmp/ visibility: report-only size, never an error (scratch
    # is disposable by design — a leaked session root shows up as hygiene
    # in the message, not as INVALID).
    try:
        tmp_bytes = paths.scratch_tmp_size_bytes(run_dir)
    except (OSError, ValueError):
        tmp_bytes = None
    if tmp_bytes is not None and tmp_bytes > 0:
        errors.append(f"note: run tmp/ scratch holds {tmp_bytes} bytes (disposable, not an error)")
    novelty_dir = run_dir / "novelty"
    if novelty_dir.exists() or (run_dir / paths.CONCEPTS_FILENAME).exists():
        errors.extend(validate_concepts(novelty_dir))
    try:
        effective = read_effective_config(run_dir)
        fps = effective.video.fps
        # The right track only exists when the SFX pass runs: dual-pan on,
        # a real effects backend, and no manifest opt-out. Fake-backend
        # (CPU-only test) runs never render SFX, so expecting the second
        # channel there would INVALIDate every one of them; same for the
        # manifest no-sfx policy (the pass is skipped entirely).
        try:
            manifest_no_sfx = bool(read_manifest(run_dir).get("no_sfx", False))
        except StateError:
            manifest_no_sfx = False
        expect_right = (
            bool(effective.sfx.dual_pan) and effective.sfx.backend != "fake" and not manifest_no_sfx
        )
    except StateError as exc:
        errors.append(str(exc))
        fps = 0
        expect_right = False
    if not isinstance(fps, int) or fps <= 0:
        errors.append(f"config video fps is corrupt: {fps!r} (expected a positive integer)")
        fps = 0
    timeline = state.timeline_frames / fps if fps else 0
    if timeline > 0.0:
        from voyage.sfx_finalize import validate_sfx_ledger

        errors.extend(validate_sfx_ledger(run_dir, timeline, expect_right=expect_right))
    return errors

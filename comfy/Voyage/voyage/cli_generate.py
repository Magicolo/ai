"""`generate <NAME>` verb (DESIGN §58, two-verb CLI).

The manifest plans; `state.json` rules. Generate reconciles the run
directory against both: manifest-only means a new run, segment folders
beyond the committed count are removed as uncommitted work before
resuming, and a validated run whose `final.mp4` already covers the
timeline does no work.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

from voyage.cli_core import generate_skip_key, get_console, resolve_generate_skips
from voyage.cli_paths import _check_run_id, output_root
from voyage.cli_planning import _frames_per_segment, _require_cuda_stack, segments_for_duration
from voyage.cli_validate import validate_run
from voyage.console import RichSegmentProgress
from voyage.errors import DiskSpaceError, MediaError, StateError, VoyageError
from voyage.media import presented_frames
from voyage.persistence import read_effective_config, read_manifest, read_state, write_manifest
from voyage.supervisor import Supervisor, acquire_run_lock


def _log_discard_before_delete(segment: Path, segment_id: str) -> None:
    """Log checksums/size before a discard delete (Track A forensics).

    Best-effort, never fails the discard: records what is about to be
    destroyed so an operator can distinguish an empty torn dir from a
    real render. Failures degrade to `unknown`.
    """
    from voyage.hashing import sha256_file

    video = segment / "video.mp4"
    try:
        size = video.stat().st_size if video.is_file() else -1
    except OSError:
        size = -1
    try:
        digest = sha256_file(video)[:16] if video.is_file() else "missing"
    except OSError:
        digest = "unreadable"
    print(
        f"discard {segment_id}: video.mp4 size={size} sha16={digest} (unverifiable, removing)",
        file=sys.stderr,
    )


def _discard_uncommitted_segments(
    run_dir: Path, committed: int, effective: Any | None = None
) -> int:
    """Delete DONE-less segment folders at/after the committed count + transients.

    `state.json` rules with a crash-window exception (DESIGN §58, issue 013):
    folders beyond the committed count are uncommitted work by definition —
    EXCEPT a folder already carrying a DONE marker. DONE is written
    atomically before the state advance, so a DONE-bearing dir is a
    crash-window orphan the locked commit must adopt (checksum-verify) or
    refuse loudly via `Supervisor._adopt_unaccounted_segment` — deleting it
    here would silently destroy the first render and its provenance. Such
    dirs are skipped (not counted). Only DONE-less numeric segment dirs are
    removed, plus `*.partial`/`*.tmp*` transients; everything else is left
    for `validate_run` to report. Returns the number of removed entries.

    DONE-less adoption (Track A): before any `rmtree`, a dir carrying
    `video.mp4` + `manifest.json` is verified via
    `verify_doneless_segment_for_adoption` (manifest parses + checksum
    matches + `validate_video` passes + tail check). Verifiable dirs get a
    DONE marker instead of deletion (the locked commit adopts them);
    only unverifiable dirs are deleted, after logging checksums/size.
    Manifest-read failures fail loud with the inspect-or-remove adoption
    directive — never silent deletion. Re-stats DONE under the run lock
    to close the TOCTOU between scan and delete.

    Why existence, not size: DONE is empty-by-design (`b""` in
    `_commit_segment`), so the adoptable signal is the file's presence,
    not a non-zero size.
    """
    from voyage import paths
    from voyage.atomic import atomic_write_bytes
    from voyage.supervisor import verify_doneless_segment_for_adoption

    # Geometry for adoption verification: explicit `effective` wins
    # (`cmd_generate` passes its already-loaded config); legacy 2-arg
    # callers fall back to reading the run manifest. A torn manifest
    # fails loud with the adoption directive (never silent deletion);
    # a missing manifest (raw test scaffolds with segments only) falls
    # back to legacy delete-without-adoption so those callers keep
    # working — production always passes `effective`.
    resolved_effective = effective
    if resolved_effective is None:
        try:
            resolved_effective = read_effective_config(run_dir)
        except StateError as exc:
            from voyage import paths as _paths_for_missing_check

            if not (run_dir / _paths_for_missing_check.MANIFEST_FILENAME).exists():
                resolved_effective = None
            else:
                print(
                    f"error: cannot verify DONE-less segments in {run_dir}: {exc} — "
                    f"inspect {run_dir / 'segments'} manually (adoption or remove)",
                    file=sys.stderr,
                )
                raise
    width: int | None = None
    height: int | None = None
    fps_value: int | None = None
    segment_frames_value: int | None = None
    if resolved_effective is not None:
        try:
            width = int(resolved_effective.video.width)
            height = int(resolved_effective.video.height)
            fps_value = int(resolved_effective.video.fps)
            segment_frames_value = int(resolved_effective.video.segment_frames)
        except (AttributeError, TypeError, ValueError) as exc:
            print(
                f"error: cannot verify DONE-less segments in {run_dir}: bad geometry ({exc}) — "
                f"inspect {run_dir / 'segments'} manually (adoption or remove)",
                file=sys.stderr,
            )
            raise StateError(
                f"cannot verify DONE-less segments in {run_dir}: bad geometry"
            ) from exc

    removed = 0
    with acquire_run_lock(run_dir):
        segments_dir = run_dir / paths.SEGMENTS_DIRNAME
        if segments_dir.is_dir():
            for child in sorted(segments_dir.iterdir()):
                if len(child.name) == 6 and child.name.isdigit() and int(child.name) >= committed:
                    if child.is_dir() and not child.is_symlink():
                        try:
                            done_now = (child / paths.DONE_MARKER).exists()
                        except OSError:
                            done_now = False
                        if done_now:
                            continue
                        if (
                            width is not None
                            and height is not None
                            and fps_value is not None
                            and segment_frames_value is not None
                        ):
                            try:
                                verifiable = verify_doneless_segment_for_adoption(
                                    run_dir,
                                    child,
                                    width=width,
                                    height=height,
                                    fps=fps_value,
                                    segment_frames=segment_frames_value,
                                )
                            except (MediaError, VoyageError, OSError, ValueError) as exc:
                                # Manifest-read failures fail loud (Track A):
                                # a torn manifest may still cover real media —
                                # never delete, direct the operator instead.
                                print(
                                    f"error: segment {child.name} metadata unreadable ({exc}); "
                                    f"refusing to delete — inspect or remove {child} manually",
                                    file=sys.stderr,
                                )
                                raise
                            if verifiable:
                                try:
                                    atomic_write_bytes(child / paths.DONE_MARKER, b"")
                                except OSError as exc:
                                    print(
                                        f"error: cannot mark {child.name} DONE ({exc}); "
                                        f"inspect or remove {child} manually",
                                        file=sys.stderr,
                                    )
                                    raise
                                print(
                                    f"adopt: DONE-less {child.name} verified, "
                                    "marked DONE for commit",
                                    file=sys.stderr,
                                )
                                continue
                        _log_discard_before_delete(child, child.name)
                        shutil.rmtree(child)
                        removed += 1
                    elif child.is_file():
                        child.unlink()
                        removed += 1
            for pattern in ("*.partial", "*.tmp*"):
                for stray in sorted(segments_dir.glob(pattern)):
                    if stray.is_file():
                        stray.unlink()
                        removed += 1
    return removed


def _final_is_fresh(
    run_dir: Path,
    final: Path,
    committed: int,
    expected_skip_key: str | None = None,
) -> bool:
    """True when final.mp4 matches the recorded finalize coverage.

    Compares presented-against-presented: the manifest's `final_coverage`
    (segments + presented frames, stamped at finalize success) against the
    current commit count + a fresh ffprobe count. Any mismatch — new
    segments, a missing/unreadable final, a re-configure (which wipes the
    stamp), or a changed present — reads as stale and re-finalizes.

    `expected_skip_key` (optional, generate skip flags): when provided, the
    stamped `skip_key` must match too — a music-only or augment-only diff
    re-finalizes even when segments + frames match. Omitted keeps the
    legacy segments+frames comparison (old callers stay green); a stamp
    without the key reads as stale once a key is expected (one re-stamp).
    """
    if committed <= 0 or not final.is_file():
        return False
    try:
        coverage = read_manifest(run_dir).get("final_coverage")
    except StateError:
        return False
    if not isinstance(coverage, dict):
        return False
    if coverage.get("segments") != committed:
        return False
    if expected_skip_key is not None and coverage.get("skip_key") != expected_skip_key:
        return False
    presented = presented_frames(final)
    return presented is not None and presented == coverage.get("presented_frames")


def _expected_skip_key(
    args: argparse.Namespace,
    manifest: dict[str, object],
    effective: Any,
) -> str:
    """Canonical behavior key for this generate's finalize (freshness gate)."""
    skips = resolve_generate_skips(args)
    augment = getattr(effective, "augment", None)
    sfx_section = getattr(effective, "sfx", None)
    audio_section = getattr(effective, "audio", None)
    return generate_skip_key(
        skips,
        manifest_no_sfx=bool(manifest.get("no_sfx", False)),
        stored_upscale=int(getattr(augment, "upscale", 1)),
        stored_interpolate=int(getattr(augment, "interpolate", 1)),
        stored_interp_backend=str(getattr(augment, "interp_backend", "rife")),
        stored_sfx_dual_pan=bool(getattr(sfx_section, "dual_pan", True)),
        stored_mastering=bool(getattr(audio_section, "mastering", True)),
    )


def _finalize_run_dir(
    run_dir: Path,
    manifest: dict[str, object],
    args: argparse.Namespace,
    console: VoyageConsole | None = None,
) -> int:
    """Run the finalize step with manifest policy + generate skip flags.

    Generate-only skips are non-persistent: --no-upscale/--no-interpolate
    force multiplier 1 via explicit overrides (stored manifest untouched),
    --no-sfx ORs with the manifest policy, --no-music/--no-master ride
    the synthetic namespace for `cmd_finalize` to consume.
    """
    from voyage.cli_finalize import cmd_finalize

    skips = resolve_generate_skips(args)
    final_value = manifest.get("final_video")
    final = Path(final_value).resolve() if isinstance(final_value, str) else run_dir / "final.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    return cmd_finalize(
        argparse.Namespace(
            run=str(run_dir),
            output=str(final),
            skip_bad=bool(manifest.get("skip_bad", False)),
            no_sfx=bool(manifest.get("no_sfx", False) or skips["skip_sfx"]),
            no_music=bool(skips["skip_music"]),
            no_master=bool(skips.get("skip_mastering", False)),
            sfx_backend=None,
            sfx_caption=None,
            sfx_device=None,
            sfx_model_size=None,
            # None = stored run config rules (`cmd_finalize` falls back to
            # config.sfx.* — same sentinel pattern as the augment
            # multipliers above and sfx_dual_pan below).
            sfx_workers=None,
            sfx_dual_pan=None,
            upscale=1 if skips["force_upscale_1"] else None,
            interpolate=1 if skips["force_interpolate_1"] else None,
            interp_backend=None,
            verbose=bool(getattr(args, "verbose", False)),
            no_color=bool(getattr(args, "no_color", False)),
            quiet=bool(getattr(args, "quiet", False)),
            progress_sink=getattr(args, "progress_sink", None),
            invoker="generate",
        )
    )


def _heal_safe_transients(run_dir: Path) -> int:
    """Remove orphan transient files/dirs generate may safely delete (DESIGN §56).

    Mirrors `validate_run`'s orphan scan exactly: `*.partial` /
    `*.partial.*` / `*.tmp.npy` / `*.tmp*` files under `segments/`,
    `novelty/`, `audio/`, `augment/` (recursive), `*.partial` files at the
    run root, plus `voyage-final-*` staging dirs (via
    `media.prune_stale_finalize_tmpdirs`). All are crash-torn staging the
    next pass re-creates (atomic-write partials, poller/SFX/ACE staging,
    finalize window wavs re-rendered from the takes ledger) — never
    DONE/video.mp4/manifests/state/takes. Best-effort, files-only, never
    follows or removes symlinks. Stale augment chunk records whose
    outputs are gone or short are likewise stripped from every
    plan-dir ledger via `augment_sidecar.heal_augment_ledgers`, so
    the pre-finalize validator sees the healed view and the pollers
    re-render the gaps on the next finalize instead of aborting
    INVALID. Returns the removed count.
    """
    from voyage import paths
    from voyage.cli_validate import _ORPHAN_PATTERNS
    from voyage.media import prune_stale_finalize_tmpdirs

    healed = prune_stale_finalize_tmpdirs(run_dir)
    roots = (
        run_dir / paths.SEGMENTS_DIRNAME,
        run_dir / "novelty",
        run_dir / "audio",
        run_dir / "augment",
    )
    seen: set[Path] = set()
    for root in roots:
        try:
            if root.is_symlink() or not root.is_dir():
                continue
        except OSError:
            continue
        for pattern in _ORPHAN_PATTERNS:
            try:
                candidates = sorted(root.rglob(pattern))
            except OSError:
                continue
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    candidate.unlink()
                    healed += 1
                except OSError:
                    continue
    try:
        strays = sorted(run_dir.glob("*.partial"))
    except OSError:
        strays = []
    for stray in strays:
        try:
            if stray.is_symlink() or not stray.is_file():
                continue
            stray.unlink()
            healed += 1
        except OSError:
            continue
    from voyage.augment_sidecar import heal_augment_ledgers

    healed += heal_augment_ledgers(run_dir)
    return healed


def _finalize_feature_text(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    effective: Any,
) -> str:
    """One-line summary of what this finalize will (and will not) render."""
    skips = resolve_generate_skips(args)
    parts = ["silent" if skips["skip_music"] else "music"]
    sfx_off = (
        bool(manifest.get("no_sfx", False))
        or skips["skip_sfx"]
        or getattr(getattr(effective, "sfx", None), "backend", "fake") == "fake"
    )
    parts.append("no sfx" if sfx_off else "sfx")
    dual_off = not bool(getattr(getattr(effective, "sfx", None), "dual_pan", True))
    if not sfx_off and dual_off:
        parts.append("single bed")
    mastering_off = bool(
        skips.get("skip_mastering", False)
        or not bool(getattr(getattr(effective, "audio", None), "mastering", True))
    )
    parts.append("no master" if mastering_off else "mastered")
    upscale = 1 if skips["force_upscale_1"] else effective.augment.upscale
    interpolate = 1 if skips["force_interpolate_1"] else effective.augment.interpolate
    parts.append(f"upscale x{upscale}" if upscale > 1 else "native size")
    parts.append(f"interp x{interpolate}" if interpolate > 1 else "no interp")
    return " + ".join(parts)


def _announce_finalize(
    console: Any,
    *,
    backend: str,
    width: int,
    height: int,
    fps: float,
    committed: int,
    frames: int,
    final_name: str,
    feature_text: str,
) -> None:
    """Print the finalize-branch header (generate verb only)."""
    console.rule(f"voyage finalize · {backend} {width}x{height} @{fps}fps")
    seconds = frames / fps if fps else 0.0
    console.info(
        f"finalizing {committed} segment(s) "
        f"({frames} frames, ~{seconds:.1f}s) -> {final_name} · {feature_text}"
    )


def _heal_and_report(run_dir: Path, console: Any | None = None) -> int:
    """Heal safe transients, reporting the count (console when present).

    The bare-`print` fallback keeps the exact historical stdout words for
    direct callers with no console (pinned by tests).
    """
    healed = _heal_safe_transients(run_dir)
    if healed:
        message = f"healed: removed {healed} transient file(s)/dir(s)"
        if console is not None:
            console.ok(message)
        else:
            print(message)
    return healed


def _pre_finalize_errors(
    run_dir: Path,
    manifest: dict[str, Any],
    effective: Any,
    args: argparse.Namespace | None = None,
) -> list[str]:
    """Validate errors minus the healable SFX tail shortfall.

    SFX stems render only at finalize, so a resume-generate on a run whose
    ledger predates the new segments always trips the pure-shortfall line
    — which finalize heals via `render_sfx_bed` (cache-hits + renders).
    The filter applies only when the SFX pass will actually run; with
    `no_sfx` (manifest policy or the generate-only --no-sfx/--no-audio
    skip) or a fake sfx backend nothing heals it, so it stays hard.

    Safe transients heal first: `validate_run` is read-only, so a crashed
    finalize's `voyage-final-*` staging (or any `*.partial` / `*.tmp*`
    staging) would otherwise abort generate before finalize's own prune
    ever runs. `_heal_safe_transients` removes exactly that set here, so
    both pre-finalize gates are self-healing without `--skip-bad`.
    """
    from voyage.sfx_finalize import is_healable_sfx_shortfall

    _heal_safe_transients(run_dir)
    errors = validate_run(run_dir)
    generate_skip_sfx = bool(args is not None and resolve_generate_skips(args)["skip_sfx"])
    sfx_enabled = (
        not bool(manifest.get("no_sfx", False))
        and not generate_skip_sfx
        and (getattr(getattr(effective, "sfx", None), "backend", "fake") != "fake")
    )
    if not sfx_enabled:
        return errors
    return [error for error in errors if not is_healable_sfx_shortfall(error)]


def _extend_plan(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, object],
    effective: Any,
    console: Any | None = None,
) -> int | None:
    """Additive plan extension from --segments XOR --duration (0 if neither).

    The new plan is `manifest segments + added`, written back in place
    (every other key — config, finalize policy, final_coverage — is
    preserved; the coverage stamp goes stale on its own once new
    segments commit, so the next finalize re-runs correctly). Returns
    the added count; bad values print an error and return None,
    leaving the manifest untouched (configure precedent).
    """
    segments = getattr(args, "segments", None)
    duration = getattr(args, "duration", None)
    if segments is None and duration is None:
        return 0
    if segments is not None and duration is not None:
        print("error: pass only one of --segments or --duration", file=sys.stderr)
        return None
    if segments is not None:
        if isinstance(segments, bool) or not isinstance(segments, int) or segments <= 0:
            print(f"error: --segments must be positive, got {segments}", file=sys.stderr)
            return None
        added = segments
    elif duration is not None:
        try:
            added = segments_for_duration(
                float(duration), effective.video.fps, _frames_per_segment(effective)
            )
        except (TypeError, ValueError) as exc:
            print(f"error: bad --duration: {exc}", file=sys.stderr)
            return None
    else:
        return 0  # Unreachable: both-None returned above; keeps the chain exhaustive.
    old = manifest.get("segments")
    if isinstance(old, bool) or not isinstance(old, int):
        print(
            f"error: manifest in {run_dir} has no planned segment count "
            "— re-run `voyage configure`",
            file=sys.stderr,
        )
        return None
    manifest["segments"] = old + added
    write_manifest(run_dir, manifest)
    message = f"extended plan: {old} -> {old + added} segments"
    if console is not None:
        console.ok(message)
    else:
        print(message)
    return added


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate (or resume) the run NAME to its manifest plan.

    1. Resolve `output/<NAME>`; missing manifest → exit 2 (`configure` first).
    2. Read manifest plan (segment count) + state (committed truth).
    2b. With --segments/--duration (exclusive): extend the stored plan
        additively (manifest segments + added) before reconciling.
    3. Delete numeric segment folders beyond the committed count.
    4. Nothing left to do (planned covered + validate clean + final.mp4
       covers the timeline) → print + exit 0 without touching workers.
    5. Else render the remainder, validate, finalize.
    """
    # Test seams: Supervisor + validate_run resolve as module globals so
    # tests patch `voyage.cli_generate` directly; cmd_finalize resolves
    # through `voyage.cli_finalize` (see _finalize_run_dir) per the seam rule.
    name = getattr(args, "name", None)
    if not isinstance(name, str) or not name.strip():
        print(
            "error: generate needs a run NAME (run `voyage configure <name>` first)",
            file=sys.stderr,
        )
        return 2
    name = name.strip()
    if _check_run_id(name) != 0:
        return 2
    run_dir = (output_root() / name).resolve()
    try:
        manifest = read_manifest(run_dir)
    except StateError as exc:
        print(
            f"error: {exc} — inspect {run_dir / 'segments'} manually "
            "(DONE-less adoption or remove) — run `voyage configure {name}` first",
            file=sys.stderr,
        )
        return 2
    try:
        effective = read_effective_config(run_dir)
    except StateError as exc:
        print(
            f"error: {exc} — inspect {run_dir / 'segments'} manually "
            "(DONE-less adoption or remove)",
            file=sys.stderr,
        )
        return 1
    planned = manifest.get("segments")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned <= 0:
        print(
            f"error: manifest in {run_dir} has no planned segment count "
            f"— re-run `voyage configure {name}`",
            file=sys.stderr,
        )
        return 2
    console = get_console(args)
    added = _extend_plan(args, run_dir, manifest, effective, console)
    if added is None:
        return 2
    planned += added
    try:
        state = read_state(run_dir)
    except StateError as exc:
        print(
            f"error: {exc} — inspect {run_dir / 'segments'} manually "
            "(DONE-less adoption or remove)",
            file=sys.stderr,
        )
        return 1
    try:
        removed = _discard_uncommitted_segments(run_dir, state.committed_segments, effective)
    except (MediaError, VoyageError, StateError, OSError, ValueError) as exc:
        print(
            f"error: reconcile refused ({exc}) — inspect {run_dir / 'segments'} manually "
            "(DONE-less adoption or remove)",
            file=sys.stderr,
        )
        return 1
    if removed:
        plural = "y" if removed == 1 else "ies"
        console.ok(f"reconcile: removed {removed} uncommitted segment entr{plural}")
    _heal_and_report(run_dir, console)
    remaining = planned - state.committed_segments
    sink = getattr(args, "progress_sink", None)
    if remaining <= 0:
        errors = _pre_finalize_errors(run_dir, manifest, effective, args)
        if errors:
            print("INVALID:")
            for error in errors:
                print(f"  - {error}")
            if not bool(manifest.get("skip_bad", False)):
                print(
                    "aborting before finalize "
                    "(re-run `voyage configure` with --skip-bad to salvage)",
                    file=sys.stderr,
                )
                return 1
            print("continuing with --skip-bad ...", file=sys.stderr)
        final_value = manifest.get("final_video")
        final = (
            Path(final_value).resolve() if isinstance(final_value, str) else run_dir / "final.mp4"
        )
        if _final_is_fresh(
            run_dir,
            final,
            state.committed_segments,
            expected_skip_key=_expected_skip_key(args, manifest, effective),
        ):
            console.ok(
                f"nothing to do: {state.committed_segments} segment(s) committed, "
                f"{final.name} is current"
            )
            return 0
        if sink is None:
            _announce_finalize(
                console,
                backend=effective.video.backend,
                width=effective.video.width,
                height=effective.video.height,
                fps=effective.video.fps,
                committed=state.committed_segments,
                frames=state.timeline_frames,
                final_name=final.name,
                feature_text=_finalize_feature_text(args, manifest, effective),
            )
        return _finalize_run_dir(run_dir, manifest, args, console)
    if not _require_cuda_stack(effective):
        return 1
    from voyage.doctor import check_ffmpeg
    from voyage.media import check_free_space

    ffmpeg_ok, ffmpeg_message = check_ffmpeg()
    if not ffmpeg_ok:
        print(f"error: {ffmpeg_message}", file=sys.stderr)
        return 1
    try:
        check_free_space(run_dir, effective.min_free_space_gib)
        stack_dirs = {
            effective.video.models_dir,
            effective.audio.models_dir,
            effective.sfx.models_dir,
        }
        for stack_dir in sorted(stack_dirs):
            anchor = Path(stack_dir)
            while not anchor.exists():
                anchor = anchor.parent
            check_free_space(anchor, effective.min_free_space_gib)
    except DiskSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    from voyage.models_ensure import ensure_models

    skips = resolve_generate_skips(args)
    sfx_enabled = (
        not bool(manifest.get("no_sfx", False))
        and not skips["skip_sfx"]
        and effective.sfx.backend != "fake"
    )
    effective_upscale = 1 if skips["force_upscale_1"] else effective.augment.upscale
    effective_interpolate = 1 if skips["force_interpolate_1"] else effective.augment.interpolate
    mastering_enabled = bool(effective.audio.mastering and not skips.get("skip_mastering", False))
    if (
        ensure_models(
            effective,
            sfx_enabled,
            console,
            allow_download=True,
            augment_enabled=effective_upscale > 1 or effective_interpolate > 1,
            music_enabled=not skips["skip_music"],
            mastering_enabled=mastering_enabled,
        )
        != 0
    ):
        return 1
    if sink is None:
        console.rule(
            f"voyage generate · {effective.video.backend} "
            f"{effective.video.width}x{effective.video.height} @{effective.video.fps}fps"
        )
        console.info(
            f"generating {remaining} segment(s) "
            f"· segments {state.committed_segments}..{planned - 1} of {planned} "
            f"· {effective.video.backend} ..."
        )
    progress = sink if sink is not None else RichSegmentProgress(console)
    supervisor = Supervisor(run_dir, effective, progress=progress)
    from voyage.supervisor import install_stop_handlers

    restore_signals = install_stop_handlers(supervisor)
    try:
        # Batch mutual exclusion (Track A): hold one lock across the
        # reconcile tail + the whole `run_segments` batch (nested with
        # the supervisor's own batch/per-commit locks via the shared
        # registry) so no second writer interleaves between discard and
        # the first commit. Re-validate the committed count under that
        # lock before booting workers (fail fast on drift).
        with acquire_run_lock(run_dir):
            try:
                live = read_state(run_dir)
            except StateError as exc:
                print(
                    f"error: {exc} — inspect {run_dir / 'segments'} manually "
                    "(DONE-less adoption or remove)",
                    file=sys.stderr,
                )
                return 1
            if live.committed_segments != state.committed_segments:
                print(
                    f"error: run {run_dir} advanced under lock "
                    f"(was {state.committed_segments}, now {live.committed_segments}) — "
                    f"refusing to interleave; inspect {run_dir / 'segments'} manually",
                    file=sys.stderr,
                )
                return 1
            committed = supervisor.run_segments(remaining)
    except KeyboardInterrupt:
        print("interrupted — run rests at PAUSED (resume with `voyage generate`)", file=sys.stderr)
        return 130
    finally:
        try:
            restore_signals()
        except Exception:
            pass
    if sink is None:
        console.ok(f"run finished · {len(committed)} segment(s) committed")
    _heal_and_report(run_dir)
    errors = _pre_finalize_errors(run_dir, manifest, effective, args)
    if errors:
        print("INVALID:")
        for error in errors:
            print(f"  - {error}")
        if not bool(manifest.get("skip_bad", False)):
            print(
                "aborting before finalize (re-run `voyage configure` with --skip-bad to salvage)",
                file=sys.stderr,
            )
            return 1
        print("continuing with --skip-bad ...", file=sys.stderr)
    fresh = read_state(run_dir)
    if sink is None:
        final_value = manifest.get("final_video")
        final_name = Path(final_value).name if isinstance(final_value, str) else "final.mp4"
        _announce_finalize(
            console,
            backend=effective.video.backend,
            width=effective.video.width,
            height=effective.video.height,
            fps=effective.video.fps,
            committed=fresh.committed_segments,
            frames=fresh.timeline_frames,
            final_name=final_name,
            feature_text=_finalize_feature_text(args, manifest, effective),
        )
    final_code = _finalize_run_dir(run_dir, manifest, args, console)
    if final_code != 0:
        return final_code
    state = read_state(run_dir)
    actual_seconds = state.timeline_frames / effective.video.fps
    final_value = manifest.get("final_video")
    if sink is None:
        console.ok(
            f"generated {final_value if isinstance(final_value, str) else run_dir / 'final.mp4'} "
            f"({state.committed_segments} segments, "
            f"{state.timeline_frames} frames, ~{actual_seconds:.1f}s)"
        )
    return 0

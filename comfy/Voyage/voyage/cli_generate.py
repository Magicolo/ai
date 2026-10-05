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

from voyage.cli_core import get_console
from voyage.cli_paths import _check_run_id, output_root
from voyage.cli_planning import _frames_per_segment, _require_cuda_stack, segments_for_duration
from voyage.cli_validate import validate_run
from voyage.console import RichSegmentProgress
from voyage.errors import DiskSpaceError, StateError
from voyage.media import presented_frames
from voyage.persistence import read_effective_config, read_manifest, read_state, write_manifest
from voyage.supervisor import Supervisor


def _discard_uncommitted_segments(run_dir: Path, committed: int) -> int:
    """Delete segment folders at/after the committed count + transient partials.

    `state.json` rules: folders beyond it are incomplete/corrupt work by
    definition (e.g. a crash between DONE and the state advance). Only
    six-digit numeric segment dirs are touched, plus `*.partial`/`*.tmp*`
    transients; everything else is left for `validate_run` to report.
    Returns the number of removed entries.
    """
    from voyage import paths

    removed = 0
    segments_dir = run_dir / paths.SEGMENTS_DIRNAME
    if segments_dir.is_dir():
        for child in sorted(segments_dir.iterdir()):
            if len(child.name) == 6 and child.name.isdigit() and int(child.name) >= committed:
                if child.is_dir() and not child.is_symlink():
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


def _final_is_fresh(run_dir: Path, final: Path, committed: int) -> bool:
    """True when final.mp4 matches the recorded finalize coverage.

    Compares presented-against-presented: the manifest's `final_coverage`
    (segments + presented frames, stamped at finalize success) against the
    current commit count + a fresh ffprobe count. Any mismatch — new
    segments, a missing/unreadable final, a re-configure (which wipes the
    stamp), or a changed present — reads as stale and re-finalizes.
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
    presented = presented_frames(final)
    return presented is not None and presented == coverage.get("presented_frames")


def _finalize_run_dir(
    run_dir: Path,
    manifest: dict[str, object],
    args: argparse.Namespace,
    console: VoyageConsole | None = None,
) -> int:
    """Run the finalize step with manifest policy (final_video/skip_bad/no_sfx)."""
    from voyage.cli_finalize import cmd_finalize

    final_value = manifest.get("final_video")
    final = Path(final_value).resolve() if isinstance(final_value, str) else run_dir / "final.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    return cmd_finalize(
        argparse.Namespace(
            run=str(run_dir),
            output=str(final),
            skip_bad=bool(manifest.get("skip_bad", False)),
            no_sfx=bool(manifest.get("no_sfx", False)),
            sfx_backend=None,
            sfx_caption=None,
            sfx_device=None,
            sfx_model_size=None,
            sfx_workers=1,
            min_fps=None,
            min_resolution=None,
            no_augment=False,
            use_model_pass=None,
            verbose=bool(getattr(args, "verbose", False)),
            no_color=bool(getattr(args, "no_color", False)),
            quiet=bool(getattr(args, "quiet", False)),
            progress_sink=getattr(args, "progress_sink", None),
            invoker="generate",
        )
    )


def _pre_finalize_errors(run_dir: Path, manifest: dict[str, Any], effective: Any) -> list[str]:
    """Validate errors minus the healable SFX tail shortfall.

    SFX stems render only at finalize, so a resume-generate on a run whose
    ledger predates the new segments always trips the pure-shortfall line
    — which finalize heals via `render_sfx_bed` (cache-hits + renders).
    The filter applies only when the SFX pass will actually run; with
    `no_sfx` (or a fake sfx backend) nothing heals it, so it stays hard.
    """
    from voyage.sfx_finalize import is_healable_sfx_shortfall

    errors = validate_run(run_dir)
    sfx_enabled = not bool(manifest.get("no_sfx", False)) and (
        getattr(getattr(effective, "sfx", None), "backend", "fake") != "fake"
    )
    if not sfx_enabled:
        return errors
    return [error for error in errors if not is_healable_sfx_shortfall(error)]


def _extend_plan(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, object],
    effective: Any,
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
    print(f"extended plan: {old} -> {old + added} segments")
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
        print(f"error: {exc} — run `voyage configure {name}` first", file=sys.stderr)
        return 2
    try:
        effective = read_effective_config(run_dir)
    except StateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    planned = manifest.get("segments")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned <= 0:
        print(
            f"error: manifest in {run_dir} has no planned segment count "
            f"— re-run `voyage configure {name}`",
            file=sys.stderr,
        )
        return 2
    added = _extend_plan(args, run_dir, manifest, effective)
    if added is None:
        return 2
    planned += added
    try:
        state = read_state(run_dir)
    except StateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    removed = _discard_uncommitted_segments(run_dir, state.committed_segments)
    if removed:
        plural = "y" if removed == 1 else "ies"
        print(f"reconcile: removed {removed} uncommitted segment entr{plural}")
    remaining = planned - state.committed_segments
    console = get_console(args)
    sink = getattr(args, "progress_sink", None)
    if remaining <= 0:
        errors = _pre_finalize_errors(run_dir, manifest, effective)
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
        if _final_is_fresh(run_dir, final, state.committed_segments):
            print(
                f"nothing to do: {state.committed_segments} segment(s) committed, "
                f"{final.name} is current"
            )
            return 0
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

    sfx_enabled = not bool(manifest.get("no_sfx", False)) and effective.sfx.backend != "fake"
    if (
        ensure_models(
            effective,
            sfx_enabled,
            console,
            allow_download=True,
            augment_enabled=effective.augment.min_fps > 0 or effective.augment.min_width > 0,
        )
        != 0
    ):
        return 1
    if sink is None:
        console.rule(
            f"voyage generate · {effective.video.backend} "
            f"{effective.video.width}x{effective.video.height} @{effective.video.fps}fps"
        )
        print(
            f"generating {remaining} segment(s) "
            f"(segments {state.committed_segments}..{planned - 1} of {planned} planned) "
            f"with {effective.video.backend} (resuming at segment {state.committed_segments}) ..."
        )
    progress = sink if sink is not None else RichSegmentProgress(console)
    supervisor = Supervisor(run_dir, effective, progress=progress)
    committed = supervisor.run_segments(remaining)
    if sink is None:
        console.ok(f"run finished · {len(committed)} segment(s) committed")
    errors = _pre_finalize_errors(run_dir, manifest, effective)
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
        print(
            f"generated ({state.committed_segments} segments, "
            f"{state.timeline_frames} frames, ~{actual_seconds:.1f}s)"
        )
    return 0

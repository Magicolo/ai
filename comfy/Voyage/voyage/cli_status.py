"""`status` / `pause` / `resume` / `stop` verbs (DESIGN §59).

Verb module of the issue-080 split: read-only run monitoring plus
the status-file control plane. `cmd_stop --finalize` calls through
to `cli_finalize.cmd_finalize` via a function-level import (the
verb-module cycle rule: leaves at top level, verbs lazily).
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

from voyage import paths
from voyage.cli_paths import resolve_run_ref
from voyage.concepts import ConceptStore
from voyage.doctor import probe
from voyage.errors import StateError, VoyageError
from voyage.logrotate import iter_metric_files
from voyage.persistence import read_manifest, read_state, write_state


def _format_uptime(manifest: dict[str, object]) -> str:
    """HH:MM:SS of wall time since the manifest's created_at, else 'unknown'.

    This is the run's age, not active render time: a long-paused run
    reports days here. Callers qualify the label when not RUNNING (049).
    """
    created = manifest.get("created_at")
    if not isinstance(created, str):
        return "unknown"
    try:
        started = datetime.datetime.fromisoformat(created)
    except ValueError:
        return "unknown"
    now = datetime.datetime.now(tz=started.tzinfo)
    elapsed = now - started
    if elapsed.total_seconds() < 0:
        return "unknown"
    hours, rest = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _latest_novelty(run_dir: Path) -> str:
    """Newest concept-novelty verdict for `status` (issue 049, §59 Novelty).

    Read-only: the newest ConceptStore record decides (`accepted` /
    `hold`), `no concepts yet` before the first commit, `unknown` when
    the store cannot be read. Rotation-blindness of the metrics log is
    a separate track (scoreboard/logrotate readers) — this store is
    never rotated, so the verdict survives it.
    """
    try:
        store = ConceptStore(run_dir / "novelty", legacy_path=run_dir / paths.CONCEPTS_FILENAME)
        records = store.records()
    except Exception:
        return "unknown"
    if not records:
        return "no concepts yet"
    latest = records[-1]
    verdict = "accepted" if latest.accepted else "hold"
    return f"{verdict} (record {latest.id})"


def _slowest_stage(stages: dict[str, object]) -> str | None:
    """Slowest numeric stage as `name (Xs)` for `status` (issue 049).

    OPERATIONS promises slowest stages; the raw per-stage dump stays,
    this one line names the bottleneck. Non-numeric values are ignored.
    """
    numeric = {
        name: float(value) for name, value in stages.items() if isinstance(value, (int, float))
    }
    if not numeric:
        return None
    name = max(numeric, key=lambda key: numeric[key])
    return f"{name} ({numeric[name]}s)"


def _read_all_metric_events(run_dir: Path) -> list[dict[str, object]]:
    """All metric events across live + rotated siblings, oldest-first (049).

    Rotation splits history across dated siblings; readers needing full
    history (status last-commit, soak/benchmark averages) must use this
    instead of opening the live file directly. Torn lines are skipped;
    a missing/unreadable logs dir yields whatever subset exists.
    """
    events: list[dict[str, object]] = []
    for events_path in iter_metric_files(run_dir):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _last_commit_stages(run_dir: Path) -> tuple[str, dict[str, object]] | None:
    """Newest segment_committed event (id + stages) across rotated logs.

    Scans live + dated siblings newest-first via `iter_metric_files`
    (issue 049): after a daily rotation the live file alone would
    silently drop the Stages section the day after a run ends.
    Returns None when no commit is found — status must degrade, never fail.
    """
    for events_path in reversed(iter_metric_files(run_dir)):
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "segment_committed":
                stages = event.get("stages")
                segment_id = event.get("segment_id")
                if isinstance(stages, dict) and isinstance(segment_id, str):
                    return segment_id, stages
    return None


def _status_restart_counts(run_dir: Path) -> tuple[dict[str, int], int]:
    """Worker restarts per worker + circuit-breaker opens (issue 061).

    Counts `worker_restart` / `circuit_breaker_open` metric events so the
    §59 monitor shows failure history, not just the last error string.
    Torn lines never reach here (`_read_all_metric_events` skips them).
    """
    restarts: dict[str, int] = {}
    breakers = 0
    for event in _read_all_metric_events(run_dir):
        name = event.get("event")
        if name == "worker_restart":
            worker = event.get("worker")
            if isinstance(worker, str):
                restarts[worker] = restarts.get(worker, 0) + 1
        elif name == "circuit_breaker_open":
            breakers += 1
    return restarts, breakers


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = resolve_run_ref(run=getattr(args, "run", None), name=getattr(args, "name", None))
    if run_dir is None:
        return 2
    try:
        state = read_state(run_dir)
        manifest = read_manifest(run_dir)
    except StateError as exc:
        print(f"status: BROKEN ({exc})", file=sys.stderr)
        return 1
    try:
        from voyage.persistence import read_effective_config

        config, _ = read_effective_config(run_dir)
    except VoyageError:
        config = None
    seconds = state.timeline_frames / state.fps if state.fps else 0
    print(f"Voyage: {state.run_id}")
    print(f"Status: {state.status}")
    age = _format_uptime(manifest)
    if state.status == "RUNNING":
        print(f"Uptime: {age}")
    else:
        print(f"Uptime: {age} (age since init; not running)")
    print()
    print("Video")
    print(f"  Backend: {config.video.backend if config else 'unknown'}")
    if config:
        print(f"  Render: {config.video.width}×{config.video.height} @ {config.video.fps}fps")
    print(f"  Timeline: {seconds:.2f}s ({state.timeline_frames} frames @ {state.fps}fps)")
    if not isinstance(state.fps, int) or state.fps <= 0:
        print("  WARN: state fps is corrupt (expected a positive integer; see `voyage validate`)")
    print(f"  Segments: {state.committed_segments}")
    if config:
        print(f"  Blocks per segment: {config.video.blocks_per_segment}")
        print(f"  Quantization: {config.video.quantization}")
        print(f"  Device: {config.video.device}")
    live_probe = probe()
    gpus = live_probe.get("gpus")
    if isinstance(gpus, list) and gpus:
        print(f"  GPU (live probe): {gpus[0]}")
        for extra in gpus[1:]:
            print(f"       {extra}")
    else:
        print("  GPU (live probe): unavailable (no nvidia-smi)")
    hardware = manifest.get("hardware")
    if isinstance(hardware, dict) and hardware:
        print("  Hardware (recorded at init):")
        for key, value in hardware.items():
            print(f"    {key}: {value}")
    else:
        print("  Hardware (recorded at init): none recorded")
    print()
    print("World")
    print(f"  Current: {state.current_concept[:100]}")
    print(f"  Destination: {state.destination_concept[:100]}")
    print(f"  Phase: {state.phase}")
    print(f"  Novelty: {_latest_novelty(run_dir)}")
    print()
    print("Audio")
    print(f"  Backend: {config.audio.backend if config else 'unknown'}")
    if config:
        print(f"  Music: {config.audio.music_style}")
        print(f"  Energy: {config.audio.energy}")
    print(f"  Buffered audio: {state.audio_buffer_seconds:.2f}s")
    if config:
        print()
        print("Config")
        print(f"  Quantization: {config.video.quantization}")
        print(f"  SFX: {config.sfx.backend} on {config.sfx.device} ({config.sfx.model_size})")
        if config.augment.min_fps:
            print(f"  Augment fps floor: {config.augment.min_fps}")
        else:
            print("  Augment fps floor: disabled")
        if config.augment.min_width:
            print(
                f"  Augment resolution floor: {config.augment.min_width}"
                f"x{config.augment.min_height}"
            )
        else:
            print("  Augment resolution floor: disabled")
        inspector_state = "on" if config.experimental.visual_inspector else "off"
        print(f"  Inspector: {inspector_state} ({config.director.inspector_model_id})")
        print(
            f"  Beats: {config.audio.beats_per_segment}/segment "
            f"(take_seconds={config.audio.take_seconds}, "
            f"ahead_seconds={config.audio.ahead_seconds})"
        )
        if config.audio.take_seconds > config.audio.ahead_seconds:
            print("  Take/ahead invariant: OK (take_seconds > ahead_seconds)")
        else:
            print("  WARN: take_seconds <= ahead_seconds (audio re-renders every segment)")
        print(f"  Drift: every {config.voyage.drift_every_n_segments} segment(s)")
        print(f"  Director: {config.director.backend} on {config.director.device}")
    print()
    print("Workers")
    for name in ("video", "audio", "director"):
        backend = "unknown"
        if config:
            backend = {"video": config.video.backend, "audio": config.audio.backend}.get(
                name, config.director.backend
            )
        print(f"  {name}: idle ({backend} backend; workers run during `voyage run`)")
    restarts, breakers = _status_restart_counts(run_dir)
    if restarts or breakers:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(restarts.items()))
        print(f"  Restarts: {summary or 'none'}; circuit-breakers open: {breakers}")
    else:
        print("  Restarts: none; circuit-breakers open: 0")
    last_commit = _last_commit_stages(run_dir)
    if last_commit is not None:
        segment_id, stages = last_commit
        print()
        print(f"Stages (last commit {segment_id})")
        for stage, stage_seconds in stages.items():
            print(f"  {stage}: {stage_seconds}s")
        slowest = _slowest_stage(stages)
        if slowest is not None:
            print(f"  Slowest stage: {slowest}")
    from voyage.bench import summarize_gauges

    gauge_events = [
        event
        for event in _read_all_metric_events(run_dir)
        if event.get("event") == "resource_gauges"
    ]
    if gauge_events:
        trend = summarize_gauges(gauge_events)
        print()
        print(f"Gauges (last {trend['segments']} segment(s))")
        print(f"  RSS: {trend['rss_first_mb']} -> {trend['rss_last_mb']} MB")
        print(f"  Disk: {trend['disk_first_gib']} -> {trend['disk_last_gib']} GiB")
    print()
    print("Storage")
    try:
        free_gib = shutil.disk_usage(run_dir).free / (1024**3)
        print(f"  Free: {free_gib:.1f} GiB")
        if config is not None:
            from voyage.doctor import meets_reserve

            reserve = config.min_free_space_gib
            verdict = meets_reserve(free_gib, reserve)
            if verdict is True:
                print(f"  Reserve: {reserve:.1f} GiB — OK")
            elif verdict is False:
                print(f"  WARN: free {free_gib:.1f} GiB below reserve {reserve:.1f} GiB")
    except OSError:
        print("  Free: unknown")
    if state.last_error:
        print(f"Last error: {state.last_error}")
    return 0


def _set_status(run: Path, status: str) -> int:
    try:
        state = read_state(run)
    except StateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    state.status = status  # type: ignore[assignment]
    write_state(run, state)
    print(f"status -> {status}")
    return 0


def _run_or_none(args: argparse.Namespace) -> Path | None:
    """Resolve `--run`/`--name` for the control-plane verbs (None on misuse)."""
    return resolve_run_ref(run=getattr(args, "run", None), name=getattr(args, "name", None))


def cmd_pause(args: argparse.Namespace) -> int:
    run_dir = _run_or_none(args)
    if run_dir is None:
        return 2
    return _set_status(run_dir, "PAUSE_REQUESTED")


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = _run_or_none(args)
    if run_dir is None:
        return 2
    code = _set_status(run_dir, "RUNNING")
    if code != 0:
        return code
    # Resume flips status only — it commits no segments, so the
    # refinalize gate (new commits required) never fires here. Wired
    # through the shared helper so the rule stays in one place.
    from voyage.cli_run_ops import maybe_refinalize

    return maybe_refinalize(args, run_dir, [])


def cmd_stop(args: argparse.Namespace) -> int:
    # Seam dispatch (issue 080): the finalize tail resolves through the
    # voyage.cli namespace at call time, so patching voyage.cli.cmd_finalize
    # (the pre-split interception point) keeps working.
    from voyage.cli import cmd_finalize

    run_dir = _run_or_none(args)
    if run_dir is None:
        return 2
    code = _set_status(run_dir, "STOP_REQUESTED")
    if code == 0 and args.finalize:
        args.output = str(run_dir / "final.mp4")
        return cmd_finalize(args)
    return code

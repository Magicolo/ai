"""`benchmark`, `soak`, and `inspect` verbs (DESIGN §§43-44, 104).

Verb module of the issue-080 split: performance reporting and the
visual-inspector entry point. `cmd_benchmark` reuses `cmd_init` and
`cmd_soak` reuses `validate_run` via seam-dispatch imports (from voyage.cli
at call time, so patching the seam keeps intercepting them).
"""

from __future__ import annotations

import argparse
import datetime
import sys
import tempfile
from pathlib import Path

from voyage import paths
from voyage.augment import DEFAULT_CHUNK_FRAMES
from voyage.cli_core import _load_run, get_console
from voyage.cli_paths import _run_dir_arg
from voyage.cli_planning import _require_cuda_stack
from voyage.cli_status import _read_all_metric_events
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig, SfxConfig
from voyage.console import RichSegmentProgress
from voyage.errors import MediaError
from voyage.media import plan_augmentation, presentation_setup_facts
from voyage.media import probe as media_probe
from voyage.supervisor import Supervisor


def _benchmark_env() -> dict[str, object]:
    """§104 setup fields: GPU/driver/runtime/torch/revisions (issue 060).

    Honest unknowns off-GPU: absent facts read "unknown"/None, never raise.
    Run-geometry fields (resolution/blocks/steps/quantization) ride the
    per-target setup dicts in `cmd_benchmark`/`cmd_soak` — this helper is
    machine facts only, so every target shares one schema.
    """
    import voyage as voyage_package
    from voyage.doctor import probe as doctor_probe

    facts = doctor_probe()
    gpus = facts.get("gpus")
    gpu = gpus[0] if isinstance(gpus, list) and gpus else "unknown"
    driver = facts.get("driver")
    compute_cap = facts.get("compute_cap")
    cuda_runtime = facts.get("cuda_runtime")
    try:
        import torch

        torch_version: object = torch.__version__
        cuda_available: object = torch.cuda.is_available()
    except ImportError:
        torch_version = "unknown"
        cuda_available = "unknown"
    return {
        "gpu": gpu,
        "driver": driver if isinstance(driver, str) else "unknown",
        "compute_cap": compute_cap if compute_cap is not None else "unknown",
        "cuda_runtime": cuda_runtime if isinstance(cuda_runtime, str) else "unknown",
        "torch": torch_version,
        "cuda_available": cuda_available,
        "voyage_version": voyage_package.__version__,
        "revisions": _benchmark_revisions(),
    }


def _benchmark_revisions() -> dict[str, object]:
    """Pinned model/code revisions behind the benchmark (issue 060).

    Quoted from `model_registry`, not measured — the image IDs in the
    report tables come from the same constants. `None` marks the one
    floating pin (Wan2.2 base, issue 070); a missing registry degrades to
    "unknown" instead of breaking the benchmark path.
    """
    try:
        from voyage import model_registry
    except ImportError:
        return {"registry": "unknown"}
    names = (
        "WAN_HF_REVISION",
        "LTXV_HF_REVISION",
        "LTXV_TE_REVISION",
        "CAUSVID_HF_REVISION",
        "WAN21_HF_REVISION",
        "QWEN_HF_REVISION",
        "QWEN4B_AWQ_HF_REVISION",
        "QWEN35_HF_REVISION",
        "ACE_MAIN_REVISION",
        "MINILM_HF_REVISION",
        "MMAUDIO_HF_REVISION",
        "FILM_HF_REVISION",
        "REALESRGAN_HF_REVISION",
    )
    revisions: dict[str, object] = {}
    for name in names:
        value = getattr(model_registry, name, "unknown")
        revisions[name] = value if value is None or isinstance(value, str) else "unknown"
    return revisions


def _video_geometry_setup(config: ProjectConfig) -> dict[str, object]:
    """Run-geometry half of a benchmark setup (issue 060).

    Resolution/blocks/attention/quantization decide the 16 GB fit, so a
    report without them is not reproducible. Read from the stored config —
    never re-derive — so the artifact matches the run that produced it.
    """
    return {
        "width": config.video.width,
        "height": config.video.height,
        "fps": config.video.fps,
        "blocks_per_segment": config.video.blocks_per_segment,
        "local_attn_size": config.video.local_attn_size,
        "quantization": config.video.quantization,
        "device": config.video.device,
    }


def _presentation_setup(config: ProjectConfig) -> dict[str, object]:
    """Floor triple + resolved presentation plan for §104 setup blocks (163/194).

    The run's own geometry doubles as finalize source and target (segments
    finalize at native geometry), so the plan is pure math over stored
    config — no probing, no renders. Every benchmark/soak setup spreads
    this so re-encode and stream-copy reports are never silently compared
    (the 091 BENCHMARKING floors note names the cost; this records it).
    """
    plan = plan_augmentation(
        config.video.width,
        config.video.height,
        float(config.video.fps),
        config.video.width,
        config.video.height,
        int(config.video.fps),
        config.augment.min_fps,
        config.augment.min_width,
        config.augment.min_height,
    )
    return presentation_setup_facts(
        plan,
        min_fps=config.augment.min_fps,
        min_width=config.augment.min_width,
        min_height=config.augment.min_height,
    )


def _persist_benchmark_report(
    run_dir: Path, stem: str, title: str, setup: dict[str, object], metrics: dict[str, object]
) -> Path | None:
    """Tee a benchmark/soak report JSON into the run logs (issue 060).

    Stdout stays the human view; this sidecar is the machine artifact
    (`logs/<stem>-<utc-ts>.json`) so reruns stay comparable without
    hand-copying terminal output. Never raises: a failed persist warns on
    stderr and the command still exits on its benchmark verdict.
    """
    from voyage.atomic import atomic_write_json
    from voyage.bench import report_document

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = run_dir / paths.LOGS_DIRNAME / f"{stem}-{stamp}.json"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, report_document(title, setup, metrics))
    except OSError as exc:
        print(f"warning: benchmark artifact not persisted ({exc})", file=sys.stderr)
        return None
    return destination


def _check_benchmark_counts(warmup: int, measured: int) -> int:
    """CLI-side mirror of the worker `validate_benchmark_counts` (079/060).

    Fail at parse-adjacent time (stderr + 2) instead of a worker-side
    ZeroDivisionError: warmup >= 0 and measured >= 1.
    """
    if warmup < 0 or measured <= 0:
        print(
            "error: benchmark needs warmup >= 0 and measured >= 1 "
            f"(got warmup={warmup}, measured={measured})",
            file=sys.stderr,
        )
        return 2
    return 0


def _sfx_probe(
    *,
    workdir: Path,
    backend: str,
    device: str,
    model_size: str,
    models_dir: str,
    extra_setup: dict[str, object],
    persist_dir: Path | None,
    warmup: int,
    measured: int,
) -> int:
    """Spawn one SFX worker and run its `benchmark` op (154/163 shared core).

    Direct spawn via `SFX_WORKER_MODULES` — the supervisor exposes no SFX
    handle, and the finalize path (`sfx_finalize.render_sfx_bed`) spawns
    the same way. `persist_dir=None` means stdout is the record (the
    no-run fake probe, mirroring the end-to-end throwaway note).
    """
    from voyage.bench import format_report, sfx_benchmark_setup
    from voyage.rpc import SubprocessWorker
    from voyage.sfx_finalize import SFX_WORKER_MODULES

    try:
        module = SFX_WORKER_MODULES[backend]
    except KeyError:
        known = ", ".join(sorted(SFX_WORKER_MODULES))
        print(f"error: unknown sfx backend {backend!r} (known: {known})", file=sys.stderr)
        return 2
    (workdir / paths.LOGS_DIRNAME).mkdir(parents=True, exist_ok=True)
    setup: dict[str, object] = {
        **sfx_benchmark_setup(
            model_size=model_size,
            sfx_workers=1,
            device=device,
            warmup=warmup,
            measured=measured,
        ),
        "backend": backend,
        **_benchmark_env(),
        **extra_setup,
    }
    worker = SubprocessWorker(
        module,
        workdir,
        workdir / paths.LOGS_DIRNAME / "sfx-benchmark.log",
        init_op="init",
        init_payload={"models_dir": models_dir, "device": device, "model_size": model_size},
    )
    worker.start()
    try:
        metrics = worker.call("benchmark", {"warmup": warmup, "measured": measured})
    finally:
        worker.stop()
    print(format_report("sfx", setup, metrics))
    if persist_dir is not None:
        artifact = _persist_benchmark_report(persist_dir, "benchmark-sfx", "sfx", setup, metrics)
        if artifact is not None:
            print(f"report: {artifact}")
    return 0


def _benchmark_sfx(args: argparse.Namespace, warmup: int, measured: int) -> int:
    """`benchmark sfx` dispatch (154/163): the run's `[sfx]` backend when
    `--run` is given, else the torch-free fake probe with no run dir."""
    if _check_benchmark_counts(warmup, measured) != 0:
        return 2
    if args.run:
        run_dir = _run_dir_arg(args.run)
        config, _digest = _load_run(run_dir)
        if not _require_cuda_stack(config):
            return 1
        return _sfx_probe(
            workdir=run_dir,
            backend=config.sfx.backend,
            device=config.sfx.device,
            model_size=config.sfx.model_size,
            models_dir=config.sfx.models_dir,
            extra_setup=_presentation_setup(config),
            persist_dir=run_dir,
            warmup=warmup,
            measured=measured,
        )
    with tempfile.TemporaryDirectory(prefix="voyage-bench-sfx-") as tmp:
        workdir = Path(tmp)
        return _sfx_probe(
            workdir=workdir,
            backend="fake",
            device="cpu",
            model_size=SfxConfig().model_size,
            models_dir="/models",
            extra_setup={
                **_presentation_setup(ProjectConfig()),
                "note": "no --run: torch-free fake probe, stdout is the record",
            },
            persist_dir=None,
            warmup=warmup,
            measured=measured,
        )


def _augment_probe(
    *,
    chunk_frames: int,
    crf: int,
    preset: str,
    device: str,
    warmup: int,
    measured: int,
) -> dict[str, object]:
    """Stage + time the ffmpeg chunk-encode orchestration (154 CPU-safe core).

    Staging (testsrc PNGs) sits outside the measured region; each measured
    iteration runs the real `augment_plan` + `run_augment_chunks` +
    `ffmpeg_encode_chunk` path and records its wall. The GPU
    upscale/interpolate leg is a recorded skip until the FILM port lands
    (the worker is a quarantined stand-in that cannot load official
    weights) — with torch presence noted so GPU runs stay comparable.
    A missing ffmpeg binary degrades to a skip note, never a traceback.
    """
    import importlib.util
    import os
    import subprocess
    import time

    from voyage.augment import AugmentChunk, augment_plan, ffmpeg_encode_chunk, run_augment_chunks

    torch_present = importlib.util.find_spec("torch") is not None
    model_note = (
        "skipped (spike stand-in quarantine: augment_worker cannot load official "
        "weights until the FILM port lands; torch "
        f"{'present' if torch_present else 'absent'} on this box)"
    )
    skip_metrics = {"model_upscale": model_note}
    total_frames = chunk_frames * 2
    fps = 8
    with tempfile.TemporaryDirectory(prefix="voyage-bench-augment-") as tmp:
        staging = Path(tmp)
        source = staging / "src"
        source.mkdir(parents=True, exist_ok=True)
        try:
            staged = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc=size=320x180:rate={fps}:duration={total_frames / fps}",
                    "-vsync",
                    "0",
                    "-start_number",
                    "0",
                    str(source / "frame_%06d.png"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return {"chunk_encode": f"skipped (ffmpeg unavailable: {exc})", **skip_metrics}
        if staged.returncode != 0:
            return {
                "chunk_encode": f"skipped (testsrc staging failed: {staged.stderr[-500:]})",
                **skip_metrics,
            }
        plan = augment_plan(total_frames, chunk=chunk_frames, devices=(device,))
        chunk_dirs: dict[int, Path] = {}
        for chunk in plan:
            chunk_dir = staging / f"chunk_{chunk.index:02d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            for offset in range(chunk.source_frames):
                os.replace(
                    source / f"frame_{chunk.start_frame + offset:06d}.png",
                    chunk_dir / f"frame_{offset:06d}.png",
                )
            chunk_dirs[chunk.index] = chunk_dir
        outputs = staging / "out"
        outputs.mkdir(parents=True, exist_ok=True)
        current: dict[str, Path] = {}

        def _encode(chunk: AugmentChunk, chunk_device: str) -> str:
            del chunk_device
            return str(
                ffmpeg_encode_chunk(
                    chunk_dirs[chunk.index] / "frame_%06d.png",
                    current["dir"] / f"chunk_{chunk.index:02d}.mp4",
                    fps,
                    crf=crf,
                    preset=preset,
                )
            )

        walls: list[float] = []
        for iteration in range(warmup + measured):
            current["dir"] = outputs / f"iter_{iteration:02d}"
            current["dir"].mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            run_augment_chunks(plan, _encode)
            elapsed = time.monotonic() - started
            if iteration >= warmup:
                walls.append(elapsed)
    mean = sum(walls) / len(walls)
    return {
        "chunks": len(plan),
        "chunk_encode_wall_seconds": [round(wall, 3) for wall in walls],
        "chunk_encode_mean_seconds": round(mean, 3),
        **skip_metrics,
    }


def _benchmark_augment(args: argparse.Namespace, warmup: int, measured: int) -> int:
    """`benchmark augment` dispatch (154): ffmpeg orchestration probe with
    the run's floors + geometry when `--run` is given, defaults otherwise."""
    from voyage.augment import (
        CHUNK_CRF_DEFAULT,
        CHUNK_PRESET_DEFAULT,
        DEFAULT_UPSCALE_FACTOR,
        augment_devices,
    )
    from voyage.bench import augment_benchmark_setup, format_report

    if _check_benchmark_counts(warmup, measured) != 0:
        return 2
    if args.run:
        run_dir = _run_dir_arg(args.run)
        config, _digest = _load_run(run_dir)
        persist_dir: Path | None = run_dir
    else:
        config = ProjectConfig()
        persist_dir = None
    devices = augment_devices()
    device = devices[0] if devices else "cpu"
    chunk_frames = DEFAULT_CHUNK_FRAMES
    setup: dict[str, object] = {
        **augment_benchmark_setup(
            chunk_frames=chunk_frames,
            upscale_factor=DEFAULT_UPSCALE_FACTOR,
            crf=CHUNK_CRF_DEFAULT,
            preset=CHUNK_PRESET_DEFAULT,
            device=device,
            warmup=warmup,
            measured=measured,
        ),
        **_benchmark_env(),
        **_presentation_setup(config),
    }
    if persist_dir is None:
        setup["note"] = "no --run: default-config ffmpeg probe, stdout is the record"
    metrics = _augment_probe(
        chunk_frames=chunk_frames,
        crf=CHUNK_CRF_DEFAULT,
        preset=CHUNK_PRESET_DEFAULT,
        device=device,
        warmup=warmup,
        measured=measured,
    )
    print(format_report("augment", setup, metrics))
    if persist_dir is not None:
        artifact = _persist_benchmark_report(
            persist_dir, "benchmark-augment", "augment", setup, metrics
        )
        if artifact is not None:
            print(f"report: {artifact}")
    return 0


def _soak_sfx_timeline(run_dir: Path) -> float | None:
    """Sum of committed segment video durations; None when unprobable.

    The tiling verdict needs a timeline; segment videos are the SFX
    pass's own source of truth (same walk the window planner uses), so
    any probe gap degrades to a skipped verdict, never a crash.
    """
    videos = sorted((run_dir / paths.SEGMENTS_DIRNAME).glob("*/video.mp4"))
    if not videos:
        return None
    total = 0.0
    for video in videos:
        try:
            info = media_probe(video)
            duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        except (MediaError, TypeError, ValueError):
            return None
        if duration <= 0.0:
            return None
        total += duration
    return total


def _soak_sfx_section(run_dir: Path) -> dict[str, object]:
    """Post-run SFX rollup over stems + ledger (issue 163): no extra renders.

    Ledger durations feed the shared `summarize_sfx_windows` aggregation
    (walls are unrecorded post-run, so the mean rests at zero while counts
    and audio totals stay genuine); stems are counted excluding crashed
    `*.partial.wav` leftovers; the tiling verdict runs only when a ledger
    exists and the timeline probes cleanly — an SFX-less soak reports
    zeros with no errors.
    """
    from voyage.bench import summarize_sfx_windows
    from voyage.sfx_finalize import (
        SFX_LEDGER_NAME,
        SFX_STEMS_DIRNAME,
        load_sfx_ledger,
        validate_sfx_ledger,
    )

    sfx_dir = run_dir / "audio" / SFX_STEMS_DIRNAME
    try:
        records = load_sfx_ledger(sfx_dir / SFX_LEDGER_NAME)
    except (OSError, ValueError) as exc:
        return {
            **summarize_sfx_windows([]),
            "stems": 0,
            "ledger_errors": [f"sfx ledger unreadable: {exc}"],
        }
    windows: list[dict[str, object]] = [
        {"audio_seconds": record.get("duration")} for record in records
    ]
    section = summarize_sfx_windows(windows)
    stems = sum(
        1
        for stem in sfx_dir.glob("*.wav")
        if stem.is_file() and not stem.name.endswith(".partial.wav")
    )
    if not records:
        errors: list[str] = []
    else:
        timeline = _soak_sfx_timeline(run_dir)
        errors = (
            validate_sfx_ledger(run_dir, timeline)
            if timeline is not None
            else ["sfx timeline unprobable — tiling verdict skipped"]
        )
    return {**section, "stems": stems, "ledger_errors": errors}


def cmd_benchmark(args: argparse.Namespace) -> int:
    from voyage.bench import format_report, summarize_gauges
    from voyage.cli import cmd_init  # seam dispatch (issue 080)

    target = args.benchmark_target
    warmup = int(args.warmup)
    measured = int(args.measured)
    if target == "sfx":
        return _benchmark_sfx(args, warmup, measured)
    if target == "augment":
        return _benchmark_augment(args, warmup, measured)
    if target in ("video", "audio"):
        if _check_benchmark_counts(warmup, measured) != 0:
            return 2
        if not args.run:
            print("benchmark video/audio requires --run <dir>", file=sys.stderr)
            return 2
        run_dir = _run_dir_arg(args.run)
        config, _digest = _load_run(run_dir)
        # Same fast-fail run/generate gate (issue 115): a CUDA backend
        # without torch dies late at worker init otherwise — for soak,
        # after the rule header, looking healthy until the first crash.
        if not _require_cuda_stack(config):
            return 1
        worker_name = target
        setup: dict[str, object] = {
            "backend": (config.video.backend if target == "video" else config.audio.backend),
            "device": (config.video.device if target == "video" else config.audio.device),
            "warmup": warmup,
            "measured": measured,
            **_video_geometry_setup(config),
            **_presentation_setup(config),
            **_benchmark_env(),
        }
        supervisor = Supervisor(run_dir, config)
        supervisor.start_workers()
        try:
            worker = supervisor._video if target == "video" else supervisor._audio
            metrics = worker.call("benchmark", {"warmup": warmup, "measured": measured})
        finally:
            supervisor.stop_workers()
        print(format_report(worker_name, setup, metrics))
        artifact = _persist_benchmark_report(
            run_dir, f"benchmark-{worker_name}", worker_name, setup, metrics
        )
        if artifact is not None:
            print(f"report: {artifact}")
        return 0
    # end-to-end: a throwaway run (never mutates the user's data) whose
    # per-stage means + gauge deltas are the steady-state report.
    segments = int(args.segments)
    if segments <= 0:
        print(
            f"error: --segments must be positive, got {segments}",
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory(prefix="voyage-bench-") as tmp:
        init_args = argparse.Namespace(
            output=str(Path(tmp) / "run"),
            run_id="benchmark",
            style="pastel neon line-art, peaceful",
            seed=11,
            force=True,
            backend="fake",
        )
        if cmd_init(init_args) != 0:
            return 1
        run_dir = Path(init_args.output)
        config, _digest = _load_run(run_dir)
        committed = Supervisor(run_dir, config).run_segments(segments)
        events = _read_all_metric_events(run_dir)
        setup = {
            "backend": "fake",
            "warmup": warmup,
            "measured": segments,
            **_video_geometry_setup(config),
            **_presentation_setup(config),
            **_benchmark_env(),
            "note": "throwaway run (TemporaryDirectory): no logs/ artifact, stdout is the record",
        }
        metrics = {
            "segments": committed,
            "stages": _stage_means(events),
            "gauges": summarize_gauges(
                [event for event in events if event.get("event") == "resource_gauges"]
            ),
        }
        print(format_report("end-to-end", setup, metrics))
        return 0


def _stage_means(events: list[dict[str, object]]) -> dict[str, float]:
    """Mean per-stage seconds across `segment_committed` events."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for event in events:
        if event.get("event") != "segment_committed":
            continue
        stages = event.get("stages")
        if not isinstance(stages, dict):
            continue
        for name, seconds in stages.items():
            if isinstance(seconds, (int, float)):
                totals[str(name)] = totals.get(str(name), 0.0) + float(seconds)
                counts[str(name)] = counts.get(str(name), 0) + 1
    return {name: round(totals[name] / counts[name], 3) for name in totals}


def cmd_soak(args: argparse.Namespace) -> int:
    """Run N segments on a real run, then report the stability trend (§68)."""
    from voyage.bench import format_report, summarize_gauges
    from voyage.cli import validate_run  # seam dispatch (issue 080)
    from voyage.supervisor import summarize_prefetch_outcome

    run_dir = _run_dir_arg(args.run)
    segments = int(args.segments)
    if segments <= 0:
        print(
            f"error: --segments must be positive, got {segments}",
            file=sys.stderr,
        )
        return 2
    config, _digest = _load_run(run_dir)
    # Same fast-fail run/generate gate (issue 115): checked before the
    # rule line so a doomed run never prints a healthy-looking header.
    if not _require_cuda_stack(config):
        return 1
    console = get_console(args)
    console.rule(f"voyage soak · {segments} segments")
    committed = Supervisor(run_dir, config, progress=RichSegmentProgress(console)).run_segments(
        segments
    )
    events = _read_all_metric_events(run_dir)
    setup: dict[str, object] = {
        "run": str(run_dir),
        "segments_requested": segments,
        "backend": config.video.backend,
        **_video_geometry_setup(config),
        **_presentation_setup(config),
        "sfx_backend": config.sfx.backend,
        "sfx_device": config.sfx.device,
        "sfx_model_size": config.sfx.model_size,
        "augment_chunk_frames": DEFAULT_CHUNK_FRAMES,
        **_benchmark_env(),
    }
    metrics: dict[str, object] = {
        "segments_committed": committed,
        "stages": _stage_means(events),
        **summarize_gauges([event for event in events if event.get("event") == "resource_gauges"]),
        "prefetch": summarize_prefetch_outcome(events),
        "sfx": _soak_sfx_section(run_dir),
        "validate_errors": validate_run(run_dir),
    }
    print(format_report("soak", setup, metrics))
    artifact = _persist_benchmark_report(run_dir, "soak", "soak", setup, metrics)
    if artifact is not None:
        print(f"report: {artifact}")
    return 0 if not metrics["validate_errors"] else 1


def _probe_media_line(name: Path) -> str:
    """One `inspect media` line for a segment mp4 (issue 031 PERF203).

    The try/except lives here — not in the calling loop — so the loop
    stays a straight call+print while per-file probe failures still
    report per file instead of failing loud (same messages, same order
    as the inline version).
    """
    try:
        info = media_probe(name)
    except MediaError as exc:
        return f"{name}: PROBE FAILED ({exc})"
    duration = info.get("format", {}).get("duration")
    return f"{name.parent.name}/video.mp4: duration={duration}"


def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    if args.inspect_target == "scoreboard":
        from voyage.cli_scoreboard import render_scoreboard

        return render_scoreboard(run_dir)
    if args.inspect_target == "concepts":
        try:
            store = ConceptStore(run_dir / "novelty", legacy_path=run_dir / paths.CONCEPTS_FILENAME)
            records = store.records()
        except (OSError, ValueError) as exc:
            # Lock-free best-effort view (issue 142): a concurrent commit
            # can leave a torn tail that fails validation — degrade to
            # `unknown` (exit 0, like `_latest_novelty`) instead of
            # tracing on a read-only verb.
            print(f"concepts: unknown ({exc})", file=sys.stderr)
            print("unknown")
            return 0
        for record in records:
            flag = "+" if record.accepted else "-"
            print(f"[{flag}] #{record.id}: {record.canonical_name[:120]}")
        return 0
    if args.inspect_target == "segments":
        root = run_dir / paths.SEGMENTS_DIRNAME
        if root.exists():
            for segment in sorted(p for p in root.iterdir() if p.is_dir()):
                done = (segment / paths.DONE_MARKER).exists()
                print(f"{segment.name}: {'DONE' if done else 'partial'}")
        return 0
    if args.inspect_target == "media":
        for name in sorted((run_dir / paths.SEGMENTS_DIRNAME).glob("*/*.mp4")):
            print(_probe_media_line(name))
        return 0
    if args.inspect_target == "metrics":
        from voyage.cli_inspect_metrics import render_inspect_metrics

        return render_inspect_metrics(run_dir)
    return 2

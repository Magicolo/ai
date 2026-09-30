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
from voyage.cli_core import _load_run, get_console
from voyage.cli_paths import _run_dir_arg
from voyage.cli_status import _read_all_metric_events
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig
from voyage.console import RichSegmentProgress
from voyage.errors import MediaError
from voyage.logrotate import iter_metric_files
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
        "LONGLIVE_COMMIT",
        "LONGLIVE_HF_REVISION",
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


def cmd_benchmark(args: argparse.Namespace) -> int:
    from voyage.bench import format_report, summarize_gauges
    from voyage.cli import cmd_init  # seam dispatch (issue 080)

    target = args.benchmark_target
    warmup = int(args.warmup)
    measured = int(args.measured)
    if target in ("video", "audio"):
        if _check_benchmark_counts(warmup, measured) != 0:
            return 2
        if not args.run:
            print("benchmark video/audio requires --run <dir>", file=sys.stderr)
            return 2
        run_dir = _run_dir_arg(args.run)
        config, _digest = _load_run(run_dir)
        worker_name = target
        setup: dict[str, object] = {
            "backend": (config.video.backend if target == "video" else config.audio.backend),
            "device": (config.video.device if target == "video" else config.audio.device),
            "warmup": warmup,
            "measured": measured,
            **_video_geometry_setup(config),
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
        **_benchmark_env(),
    }
    metrics: dict[str, object] = {
        "segments_committed": committed,
        "stages": _stage_means(events),
        **summarize_gauges([event for event in events if event.get("event") == "resource_gauges"]),
        "prefetch": summarize_prefetch_outcome(events),
        "validate_errors": validate_run(run_dir),
    }
    print(format_report("soak", setup, metrics))
    artifact = _persist_benchmark_report(run_dir, "soak", "soak", setup, metrics)
    if artifact is not None:
        print(f"report: {artifact}")
    return 0 if not metrics["validate_errors"] else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    if args.inspect_target == "scoreboard":
        from voyage.scoreboard import METRIC_KEYS, scoreboard_rows

        rows = scoreboard_rows(run_dir)
        if not rows:
            print("no committed segments")
            return 0
        header = ["seg", "frames", "stages", *[f"{key[:12]}" for key in METRIC_KEYS]]
        print("  ".join(header))
        for row in rows:
            stages = row["stages"]
            stage_cells = (
                ",".join(f"{name}={seconds:.1f}" for name, seconds in stages.items())
                if isinstance(stages, dict) and stages
                else "-"
            )
            metrics = row["metrics"]
            deltas = row["deltas"]
            if isinstance(metrics, dict) and isinstance(deltas, dict):
                cells = [f"{metrics[key]:.3f}({deltas[key]:+.3f})" for key in METRIC_KEYS]
            else:
                cells = ["no-visual"] * len(METRIC_KEYS)
            print("  ".join([str(row["segment_id"]), str(row["frames"]), stage_cells, *cells]))
            print(f"  -> {row['destination']} [{row['phase']}] takes={row['take_ids']}")
            print(f"  view: {row['video_path']} + {row['audio_path']}")
        final = run_dir / "final.mp4"
        if final.exists():
            print(f"final: {final}")
        return 0
    if args.inspect_target == "concepts":
        store = ConceptStore(run_dir / "novelty", legacy_path=run_dir / paths.CONCEPTS_FILENAME)
        records = store.records()
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
            try:
                info = media_probe(name)
                duration = info.get("format", {}).get("duration")
                print(f"{name.parent.name}/video.mp4: duration={duration}")
            except MediaError as exc:
                print(f"{name}: PROBE FAILED ({exc})")
        return 0
    if args.inspect_target == "metrics":
        events = _read_all_metric_events(run_dir)
        if not events:
            print("no metrics yet")
            return 0
        files = iter_metric_files(run_dir)
        print(f"{len(events)} metric events across {len(files)} files")
        for event in events[-5:]:
            print(f"  {event.get('event')}: {event.get('segment_id', event.get('worker', ''))}")
        return 0
    return 2

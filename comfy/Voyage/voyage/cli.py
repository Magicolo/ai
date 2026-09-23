"""`voyage` CLI (DESIGN §§58, task group J).

init / doctor / models / benchmark / run / status / pause / resume /
stop / validate / finalize / inspect
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from types import FrameType

from voyage import paths
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig, apply_draft_overrides, default_config_toml, load_config
from voyage.doctor import check_ffmpeg, probe
from voyage.errors import MediaError, StateError, VoyageError
from voyage.media import finalize_run
from voyage.media import probe as media_probe
from voyage.model_registry import (
    download_audio_models,
    download_director_models,
    download_inspector_models,
    download_longlive2_bf16,
    verify_audio_models,
    verify_director_models,
    verify_inspector_models,
    verify_longlive2_bf16,
)
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_manifest,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor


def _run_dir_arg(value: str) -> Path:
    return Path(value)


def cmd_init(args: argparse.Namespace) -> int:
    run_dir = Path(args.output)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        print(f"refusing to init non-empty directory {run_dir} (use --force)", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    config_text = default_config_toml(args.run_id, args.style, args.seed)
    (run_dir / paths.CONFIG_FILENAME).write_text(config_text, encoding="utf-8")
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    hardware = {"note": "recorded at init; see `voyage doctor` for live facts"}
    software = {"python": sys.version.split()[0]}
    write_manifest(run_dir, build_manifest(config, digest, hardware, software))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")
    print(f"initialized voyage run at {run_dir}")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    facts = probe()
    ok, message = check_ffmpeg()
    print(f"python: {facts['python']}")
    print(f"ffmpeg: {message}")
    if facts["nvidia_smi"]:
        for line in facts["gpus"]:
            print(f"gpu: {line}")
    else:
        print("gpu: no nvidia-smi data (CPU-only environment)")
    return 0 if ok else 1


def _models_dir(args: argparse.Namespace) -> Path:
    raw = getattr(args, "models_dir", None) or os.environ.get("VOYAGE_MODELS_DIR", "/models")
    return Path(raw)


def _download_director(models_dir: Path) -> int:
    """Download the Phase 3 director stack (DESIGN §§8-9, 85)."""
    print(f"downloading director-qwen8b into {models_dir} ...")
    try:
        record = download_director_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    director = record["director"]
    assert isinstance(director, dict)
    print(f"qwen3-8b: {director.get('checkpoint_bytes')} bytes")
    print(f"minilm: {director.get('embedding_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_audio(models_dir: Path) -> int:
    """Download the Phase 4 music stack (DESIGN §§6, 37, 85)."""
    print(f"downloading audio-acestep into {models_dir} ...")
    try:
        record = download_audio_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    audio = record["audio"]
    assert isinstance(audio, dict)
    print(f"turbo dit: {audio.get('turbo_bytes')} bytes")
    print(f"planner lm: {audio.get('planner_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_inspector(models_dir: Path) -> int:
    """Download the Phase 5 inspector VLM (DESIGN §§43-44, 85)."""
    print(f"downloading inspector-qwen35 into {models_dir} ...")
    try:
        record = download_inspector_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    inspector = record["inspector"]
    assert isinstance(inspector, dict)
    print(f"qwen3.5-9b: {inspector.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    action = args.models_action
    if action == "list":
        print("video: fake (built-in) | longlive2-bf16 (LongLive 2.0 BF16 + FP8 PTQ)")
        print("audio: fake (built-in) | acestep (ACE-Step 1.5 turbo + 0.6B planner)")
        print("director: deterministic (built-in) | qwen3-8b (Qwen3-8B + MiniLM)")
        print("inspector: skipped (built-in) | qwen3.5-9b (Qwen3.5-9B VLM, experimental)")
        return 0
    if action == "verify":
        ok, message = verify_longlive2_bf16(_models_dir(args))
        print(message)
        dok, dmessage = verify_director_models(_models_dir(args))
        print(dmessage)
        aok, amessage = verify_audio_models(_models_dir(args))
        print(amessage)
        iok, imessage = verify_inspector_models(_models_dir(args))
        print(imessage)
        print("fake backends need no model files: OK")
        return 0 if (ok and dok and aok and iok) else 1
    if action == "download":
        target = getattr(args, "models_target", "longlive2-bf16")
        if target == "director-qwen8b":
            return _download_director(_models_dir(args))
        if target == "audio-acestep":
            return _download_audio(_models_dir(args))
        if target == "inspector-qwen35":
            return _download_inspector(_models_dir(args))
        if target != "longlive2-bf16":
            print(f"unknown models target {target!r}")
            print("known: longlive2-bf16, director-qwen8b, audio-acestep, inspector-qwen35")
            return 2
        models_dir = _models_dir(args)
        print(f"downloading longlive2-bf16 into {models_dir} ...")
        try:
            record = download_longlive2_bf16(models_dir)
        except Exception as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            return 1
        video = record["video"]
        assert isinstance(video, dict)
        print(f"generator: {video.get('checkpoint_bytes')} bytes")
        print(f"sha256: {video.get('checkpoint_sha256')}")
        print(f"manifest: {models_dir / 'manifest.json'}")
        return 0
    if action == "info":
        print("Use `voyage models list` for backends, `verify` for file checks.")
        return 0
    return 2


def _load_run(run: Path) -> tuple[ProjectConfig, str]:
    return load_config(run / paths.CONFIG_FILENAME)


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    config, _digest = _load_run(run_dir)
    if (
        args.draft
        or args.director
        or args.blocks is not None
        or args.take_seconds is not None
        or args.quantization is not None
    ):
        config = apply_draft_overrides(
            config,
            draft=args.draft,
            director=args.director,
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
        )
        print(
            "effective settings: "
            f"director={config.director.backend} "
            f"blocks={config.video.blocks_per_segment} "
            f"{config.video.width}x{config.video.height} "
            f"latent={list(config.video.latent_shape)} "
            f"take_seconds={config.audio.take_seconds} "
            f"quantization={config.video.quantization}"
        )
    supervisor = Supervisor(run_dir, config)

    def _on_signal(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        supervisor.request_stop()

    previous = signal.signal(signal.SIGINT, _on_signal)
    try:
        committed = supervisor.run_segments(args.segments)
    finally:
        signal.signal(signal.SIGINT, previous)
    for segment_id in committed:
        print(f"committed segment {segment_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    try:
        state = read_state(run_dir)
        manifest = read_manifest(run_dir)
    except StateError as exc:
        print(f"status: BROKEN ({exc})", file=sys.stderr)
        return 1
    seconds = state.timeline_frames / state.fps if state.fps else 0
    print(f"Voyage: {state.run_id}")
    print(f"Status: {state.status}")
    print(f"Segments: {state.committed_segments}")
    print(f"Timeline: {seconds:.2f}s ({state.timeline_frames} frames @ {state.fps}fps)")
    print(f"Concept: {state.current_concept[:100]}")
    print(f"Destination: {state.destination_concept[:100]}")
    print(f"Phase: {state.phase}")
    print(f"Audio buffered: {state.audio_buffer_seconds:.2f}s")
    if state.last_error:
        print(f"Last error: {state.last_error}")
    _ = manifest
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


def cmd_pause(args: argparse.Namespace) -> int:
    return _set_status(_run_dir_arg(args.run), "PAUSE_REQUESTED")


def cmd_resume(args: argparse.Namespace) -> int:
    return _set_status(_run_dir_arg(args.run), "RUNNING")


def cmd_stop(args: argparse.Namespace) -> int:
    code = _set_status(_run_dir_arg(args.run), "STOP_REQUESTED")
    if code == 0 and args.finalize:
        args.output = str(Path(args.run) / "final.mp4")
        return cmd_finalize(args)
    return code


def cmd_validate(args: argparse.Namespace) -> int:
    """Read-only consistency check (DESIGN §70). Never mutates the run."""
    run_dir = _run_dir_arg(args.run)
    errors: list[str] = []
    try:
        state = read_state(run_dir)
        read_manifest(run_dir)
    except StateError as exc:
        print(f"INVALID: {exc}")
        return 1
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
    expected_frames = 0
    for segment in committed:
        for name in ("video.mp4", "audio.wav", "world_state.json", "sha256.json"):
            if not (segment / name).exists():
                errors.append(f"{segment.name} DONE but missing {name}")
        metrics_path = segment / "metrics.json"
        if metrics_path.exists():
            try:
                import json

                frames = int(json.loads(metrics_path.read_text()).get("frames", 0))
                expected_frames += frames
            except (ValueError, KeyError):
                errors.append(f"{segment.name} has unreadable metrics.json")
    if expected_frames != state.timeline_frames:
        errors.append(
            f"timeline frames {state.timeline_frames} != sum of segment frames {expected_frames}"
        )
    orphans = (
        [p.name for p in segments_root.iterdir() if p.suffix == ".partial"]
        if segments_root.exists()
        else []
    )
    if orphans:
        errors.append(f"orphan partial files: {orphans}")
    if errors:
        print("INVALID:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"VALID: {len(committed)} committed segments, {state.timeline_frames} frames")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    config, _digest = _load_run(run_dir)
    output = Path(args.output)
    try:
        finalize_run(run_dir, output, fps=config.video.fps)
    except (MediaError, StateError) as exc:
        print(f"finalize failed: {exc}", file=sys.stderr)
        return 1
    print(f"finalized -> {output}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    target = args.benchmark_target
    print(
        f"benchmark {target}: fake backends only in Phase 0 "
        f"(real timing harness lands with GPU workers, Phase 6)."
    )
    if target in ("video", "end-to-end"):
        import time

        from voyage.fake_backends import FakeVideoBackend

        backend = FakeVideoBackend()
        tmp = (
            Path(args.run) / "logs" / "bench_video.mp4"
            if args.run
            else Path("/tmp/bench_video.mp4")
        )
        tmp.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        backend.generate_segment(tmp, "bench", 0, 320, 180, 24, 24)
        elapsed = time.monotonic() - started
        print(f"fake 24-frame 320x180 segment: {elapsed:.2f}s")
    return 0


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
        import json

        metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
        if not metrics_path.exists():
            print("no metrics yet")
            return 0
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        print(f"{len(lines)} metric events")
        for line in lines[-5:]:
            event = json.loads(line)
            print(f"  {event.get('event')}: {event.get('segment_id', event.get('worker', ''))}")
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voyage", description="Autonomous infinite audiovisual voyage"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a new run directory")
    init.add_argument("--output", required=True)
    init.add_argument("--run-id", default="voyage")
    init.add_argument("--style", required=True)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    doctor = sub.add_parser("doctor", help="Probe hardware and environment")
    doctor.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", help="Model management")
    models.add_argument("models_action", choices=["list", "download", "verify", "info"])
    models.add_argument("models_target", nargs="?", default="longlive2-bf16")
    models.add_argument("--models-dir", default=None)
    models.set_defaults(func=cmd_models)

    run = sub.add_parser("run", help="Generate segments (infinite unless --segments)")
    run.add_argument("--run", required=True)
    run.add_argument(
        "--segments",
        type=int,
        default=None,
        help="segments to generate; omit to run until pause/stop/SIGINT",
    )
    run.add_argument(
        "--draft",
        action="store_true",
        help="apply the [draft] profile (fast low-res iteration settings)",
    )
    run.add_argument(
        "--director",
        default=None,
        help="override the director backend (e.g. deterministic, qwen)",
    )
    run.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="override video blocks per segment",
    )
    run.add_argument(
        "--take-seconds",
        type=float,
        default=None,
        help="override audio take length in seconds",
    )
    run.add_argument(
        "--quantization",
        default=None,
        choices=("fp8", "bf16"),
        help="override DiT quantization (fp8 default, bf16 for clean highlights)",
    )
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Show run status")
    status.add_argument("--run", required=True)
    status.set_defaults(func=cmd_status)

    pause = sub.add_parser("pause", help="Request a safe pause")
    pause.add_argument("--run", required=True)
    pause.set_defaults(func=cmd_pause)

    resume = sub.add_parser("resume", help="Resume from last commit")
    resume.add_argument("--run", required=True)
    resume.set_defaults(func=cmd_resume)

    stop = sub.add_parser("stop", help="Safely stop generation")
    stop.add_argument("--run", required=True)
    stop.add_argument("--finalize", action="store_true")
    stop.set_defaults(func=cmd_stop)

    validate = sub.add_parser("validate", help="Offline consistency check (read-only)")
    validate.add_argument("--run", required=True)
    validate.set_defaults(func=cmd_validate)

    finalize = sub.add_parser("finalize", help="Assemble the final MP4")
    finalize.add_argument("--run", required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(func=cmd_finalize)

    benchmark = sub.add_parser("benchmark", help="Performance probes")
    benchmark.add_argument("benchmark_target", choices=["video", "audio", "end-to-end"])
    benchmark.add_argument("--run", default="")
    benchmark.set_defaults(func=cmd_benchmark)

    inspect = sub.add_parser("inspect", help="Inspect run artifacts")
    inspect.add_argument(
        "inspect_target", choices=["concepts", "segments", "media", "metrics", "scoreboard"]
    )
    inspect.add_argument("--run", required=True)
    inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except VoyageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

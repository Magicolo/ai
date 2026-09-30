"""`voyage` CLI (DESIGN §§58, task group J).

init / doctor / models / benchmark / run / generate / status / pause / resume /
stop / validate / finalize / inspect
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from types import FrameType
from typing import Any

from pydantic import ValidationError

from voyage import paths
from voyage.concepts import ConceptStore, validate_concepts
from voyage.config import (
    ProjectConfig,
    VideoBackendName,
    apply_draft_overrides,
    default_config_toml,
    is_provided,
    load_config,
    resolve_config,
)
from voyage.console import RichSegmentProgress, VoyageConsole
from voyage.doctor import check_ffmpeg, probe
from voyage.errors import DiskSpaceError, MediaError, StateError, VoyageError
from voyage.logrotate import iter_metric_files
from voyage.media import AV_ALIGNMENT_TOLERANCE_SECONDS, check_free_space, finalize_run
from voyage.media import probe as media_probe
from voyage.model_registry import (
    download_audio_models,
    download_causvid_models,
    download_director_awq_models,
    download_director_models,
    download_film_models,
    download_inspector_models,
    download_longlive2_bf16,
    download_ltxv_models,
    download_realesrgan_models,
    download_sfx_models,
    verify_audio_models,
    verify_causvid_models,
    verify_director_awq_models,
    verify_director_models,
    verify_film_models,
    verify_inspector_models,
    verify_longlive2_bf16,
    verify_ltxv_models,
    verify_realesrgan_models,
    verify_sfx_models,
)
from voyage.persistence import (
    build_manifest,
    initial_state,
    read_manifest,
    read_state,
    write_manifest,
    write_state,
)
from voyage.supervisor import Supervisor, sha256_file


def _run_dir_arg(value: str) -> Path:
    # Absolute: workers spawn with CWD=run_dir, so a relative dir doubles up
    # inside payload paths (qual-longlive2 2026-09-24: generate_blocks
    # circuit-breaker on `output/.../segments/...` missing). Single funnel
    # for every subcommand; mirrors cmd_generate's resolve-once rule.
    return Path(value).resolve()


def resolve_run_dir(value: str) -> Path:
    """Canonical run-dir resolution (issue 008/057: one helper, every verb).

    Absolute + normalized so worker CWD-relative payloads never double up.
    Kept as a named alias of `_run_dir_arg` (which predates it and stays
    for backward-compatible imports) — new code should call this one.
    """
    return _run_dir_arg(value)


_RESERVED_FOLDER_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
"""Windows-reserved basenames, mirrored from the TUI (issue 080)."""


def is_flat_folder_name(value: str) -> bool:
    """Whether the value is usable as a single output folder name (008).

    Mirrors the TUI `_flat_folder_name` check (`tui_state.py`): rejects
    path separators and parent-dotdot so a crafted `--run-id` cannot
    escape `output/` (imported locally here, not from the TUI, because
    the TUI imports this module's `parse_duration` — reverse import
    would be circular). Also rejects `.` and reserved basenames (080).
    """
    text = value.strip()
    if not text or "/" in text or "\\" in text or ".." in text:
        return False
    if text in (".",):
        return False
    return text.split(".")[0].lower() not in _RESERVED_FOLDER_NAMES


def _check_run_id(run_id: str) -> int:
    """Reject traversal run names at the CLI layer (issue 008)."""
    if not is_flat_folder_name(run_id):
        print(
            f"error: --name/--run-id must be a flat folder name (no slashes), got {run_id!r}",
            file=sys.stderr,
        )
        return 2
    return 0


def _effective_run_id(args: argparse.Namespace) -> str:
    """Run name for init/generate: --name wins, --run-id is the legacy alias."""
    named = getattr(args, "name", None)
    if isinstance(named, str) and named != "":
        return named
    return str(args.run_id)


def get_console(args: argparse.Namespace) -> VoyageConsole:
    """Console for a subcommand (flags default off for test Namespaces)."""
    return VoyageConsole(
        verbose=bool(getattr(args, "verbose", False)),
        no_color=bool(getattr(args, "no_color", False)),
    )


def _add_console_args(parser: argparse.ArgumentParser) -> None:
    """Two verbosity levels + color kill-switch (console output only)."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="detailed console output (retry feedback, beat math, take decisions, seeds)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="plain console output (no colors or animation; also honors NO_COLOR)",
    )


def launch_tui() -> int:
    """Bare-command launcher (indirection so tests can monkeypatch).

    The Textual import stays lazy so every CLI verb works without the
    display extra installed.
    """
    from voyage.tui import run_tui

    return run_tui()


def cmd_init(args: argparse.Namespace) -> int:
    run_id = _effective_run_id(args)
    if _check_run_id(run_id) != 0:
        return 2
    run_dir = resolve_run_dir(args.output)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        print(f"refusing to init non-empty directory {run_dir} (use --force)", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    backend: VideoBackendName = getattr(args, "backend", None) or "ltxv"
    director_backend: str = getattr(args, "director", None) or "qwen"
    director_device: str = getattr(args, "director_device", None) or "cuda:1"
    config_text = default_config_toml(
        run_id,
        args.style,
        args.seed,
        video_backend=backend,
        director_backend=director_backend,
        director_device=director_device,
    )
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
    director_python = facts.get("director_python")
    if director_python and facts.get("director_python_exists"):
        print(f"director python: {director_python}")
    elif director_python:
        print(f"director python: {director_python} (missing — director falls back to CPU)")
    else:
        print("director python: unset (director runs in-process on CPU)")
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


def _download_director_awq(models_dir: Path) -> int:
    """Download the GPU decider stack (Qwen3-4B-AWQ + MiniLM)."""
    print(f"downloading director-qwen4b-awq into {models_dir} ...")
    try:
        record = download_director_awq_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    director = record["director-awq"]
    assert isinstance(director, dict)
    print(f"qwen3-4b-awq: {director.get('checkpoint_bytes')} bytes")
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


def _download_sfx(models_dir: Path) -> int:
    """Download the SFX effects stack (SFX slice 2, three-caption doctrine)."""
    print(f"downloading sfx-mmaudio into {models_dir} ...")
    try:
        record = download_sfx_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    sfx = record["sfx"]
    assert isinstance(sfx, dict)
    print(f"variants: {sfx.get('variants')}")
    print(f"large: {sfx.get('large_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_film(models_dir: Path) -> int:
    """Download the FILM interpolation weights (Track C: augment floors)."""
    print(f"downloading film into {models_dir} ...")
    try:
        record = download_film_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    film = record["film"]
    assert isinstance(film, dict)
    print(f"checkpoint: {film.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_realesrgan(models_dir: Path) -> int:
    """Download the Real-ESRGAN anime upscaler (Track C: augment floors)."""
    print(f"downloading realesrgan-anime into {models_dir} ...")
    try:
        record = download_realesrgan_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    realesrgan = record["realesrgan"]
    assert isinstance(realesrgan, dict)
    print(f"checkpoint: {realesrgan.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    action = args.models_action
    if action == "list":
        print("video: fake (built-in) | longlive2-bf16 (LongLive 2.0 BF16 + FP8 PTQ)")
        print("video: ltxv-2b (LTXV 2B distilled, Phase 7 alternative)")
        print("video: causvid (CausVid DMD causal generator + Wan2.1-1.3B base)")
        print("audio: fake (built-in) | acestep (ACE-Step 1.5 turbo + 0.6B planner)")
        print("sfx: fake (built-in) | mmaudio (MMAudio 44k effects, CC-BY-NC-4.0)")
        print("director: deterministic (built-in) | qwen3-8b (Qwen3-8B + MiniLM)")
        print("inspector: skipped (built-in) | qwen3.5-9b (Qwen3.5-9B VLM, experimental)")
        return 0
    if action == "verify":
        ok, message = verify_longlive2_bf16(_models_dir(args))
        print(message)
        lok, lmessage = verify_ltxv_models(_models_dir(args))
        print(lmessage)
        cok, cmessage = verify_causvid_models(_models_dir(args))
        print(cmessage)
        dok, dmessage = verify_director_models(_models_dir(args))
        print(dmessage)
        dawq_ok, dawq_message = verify_director_awq_models(_models_dir(args))
        print(dawq_message)
        aok, amessage = verify_audio_models(_models_dir(args))
        print(amessage)
        sok, smessage = verify_sfx_models(_models_dir(args))
        print(smessage)
        fok, fmessage = verify_film_models(_models_dir(args))
        print(fmessage)
        rok, rmessage = verify_realesrgan_models(_models_dir(args))
        print(rmessage)
        iok, imessage = verify_inspector_models(_models_dir(args))
        print(imessage)
        print("fake backends need no model files: OK")
        all_ok = ok and lok and cok and dok and dawq_ok and aok and sok and fok and rok and iok
        return 0 if all_ok else 1
    if action == "download":
        target = getattr(args, "models_target", "longlive2-bf16")
        if target == "ltxv-2b":
            models_dir = _models_dir(args)
            print(f"downloading ltxv-2b into {models_dir} ...")
            try:
                record = download_ltxv_models(models_dir)
            except Exception as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 1
            video = record["ltxv"]
            assert isinstance(video, dict)
            print(f"DiT: {video.get('checkpoint_bytes')} bytes")
            print(f"manifest: {models_dir / 'manifest.json'}")
            return 0
        if target == "causvid":
            models_dir = _models_dir(args)
            print(f"downloading causvid into {models_dir} ...")
            try:
                record = download_causvid_models(models_dir)
            except Exception as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 1
            video = record["causvid"]
            assert isinstance(video, dict)
            print(f"DMD checkpoint: {video.get('checkpoint_bytes')} bytes")
            print(f"manifest: {models_dir / 'manifest.json'}")
            return 0
        if target == "director-qwen8b":
            return _download_director(_models_dir(args))
        if target == "director-qwen4b-awq":
            return _download_director_awq(_models_dir(args))
        if target == "audio-acestep":
            return _download_audio(_models_dir(args))
        if target == "sfx-mmaudio":
            return _download_sfx(_models_dir(args))
        if target == "film":
            return _download_film(_models_dir(args))
        if target == "realesrgan-anime":
            return _download_realesrgan(_models_dir(args))
        if target == "inspector-qwen35":
            return _download_inspector(_models_dir(args))
        if target != "longlive2-bf16":
            print(f"unknown models target {target!r}", file=sys.stderr)
            known_targets = ", ".join(
                [
                    "longlive2-bf16",
                    "ltxv-2b",
                    "causvid",
                    "director-qwen8b",
                    "director-qwen4b-awq",
                    "audio-acestep",
                    "sfx-mmaudio",
                    "film",
                    "realesrgan-anime",
                    "inspector-qwen35",
                ]
            )
            print(f"known: {known_targets}", file=sys.stderr)
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
        print("backends: `voyage models list` (video/audio/director/inspector)")
        print("weights: longlive2-bf16 (~48 GB) | ltxv-2b (~7 GB) | causvid (~28 GB)")
        print("weights: director-qwen8b (~16 GB) | audio-acestep | inspector-qwen35 (~19 GB)")
        print("weights: sfx-mmaudio (~8 GB: 3 variants + VAE/sync/CLIP/vocoder)")
        print("weights: film (~66 MB interpolation) | realesrgan-anime (~18 MB upscaler)")
        print("note: MMAudio weights are CC-BY-NC-4.0 (non-commercial)")
        print("pins: voyage/model_registry.py (single source); human mirror docs/MODELS.md")
        print("note: CausVid DMD checkpoint is CC BY-NC-SA 4.0 (non-commercial)")
        print("check: `voyage models verify` for presence + size sanity")
        return 0
    return 2


def _load_run(run: Path) -> tuple[ProjectConfig, str]:
    return load_config(run / paths.CONFIG_FILENAME)


def cmd_run(args: argparse.Namespace) -> int:
    if args.segments is not None and args.segments <= 0:
        print(
            f"error: --segments must be positive, got {args.segments}",
            file=sys.stderr,
        )
        return 2
    run_dir = _run_dir_arg(args.run)
    config, _digest = _load_run(run_dir)
    # Absent-encoding (issue 045): TUI namespaces carry Unset, argparse
    # carries None — branch on the predicate so an unset TUI field is
    # never mistaken for an explicit override.
    if (
        args.draft
        or is_provided(args.director)
        or is_provided(getattr(args, "director_device", None))
        or is_provided(args.blocks)
        or is_provided(args.take_seconds)
        or is_provided(args.quantization)
        or is_provided(getattr(args, "beats_per_segment", None))
        or is_provided(getattr(args, "drift_every_n", None))
        or is_provided(getattr(args, "music_caption", None))
        or is_provided(getattr(args, "video_caption", None))
        or bool(_augment_overrides(args))
    ):
        try:
            config = apply_draft_overrides(
                config,
                draft=args.draft,
                director=args.director,
                director_device=getattr(args, "director_device", None),
                blocks=args.blocks,
                take_seconds=args.take_seconds,
                quantization=args.quantization,
                beats_per_segment=getattr(args, "beats_per_segment", None),
                drift_every_n_segments=getattr(args, "drift_every_n", None),
                music_caption=getattr(args, "music_caption", None),
                video_caption=getattr(args, "video_caption", None),
                **_augment_overrides(args),
            )
        except (ValidationError, ValueError) as exc:
            print(f"error: invalid numeric override: {exc}", file=sys.stderr)
            return 2
        print(
            "effective settings: "
            f"director={config.director.backend} "
            f"director_device={config.director.device} "
            f"blocks={config.video.blocks_per_segment} "
            f"{config.video.width}x{config.video.height} "
            f"latent={list(config.video.latent_shape)} "
            f"take_seconds={config.audio.take_seconds} "
            f"beats_per_segment={config.audio.beats_per_segment} "
            f"drift_every_n={config.voyage.drift_every_n_segments} "
            f"quantization={config.video.quantization} "
            f"min_fps={config.augment.min_fps} "
            f"min_resolution={config.augment.min_width}x{config.augment.min_height}"
        )
    if not _require_cuda_stack(config):
        return 1
    console = get_console(args)
    # The TUI swaps the progress sink from stdout to its RichLog: when a
    # sink rides the namespace, the console stays silent (the TUI owns
    # the display) and only the sink reports.
    sink = getattr(args, "progress_sink", None)
    progress = sink if sink is not None else RichSegmentProgress(console)
    if sink is None:
        console.rule(
            f"voyage run · {config.video.backend} {config.video.width}x{config.video.height} "
            f"@{config.video.fps}fps · director {config.director.backend} · "
            f"{config.audio.beats_per_segment} beats/segment · "
            f"drift every {config.voyage.drift_every_n_segments}"
        )
    supervisor = Supervisor(run_dir, config, progress=progress)

    def _on_signal(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        supervisor.request_stop()

    previous = None
    try:
        previous = signal.signal(signal.SIGINT, _on_signal)
    except ValueError:
        # Non-main thread (e.g. the TUI worker): no SIGINT handler here —
        # the TUI Stop button drives the file-status control plane instead.
        pass
    try:
        committed = supervisor.run_segments(args.segments)
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if sink is None:
        console.ok(f"run finished · {len(committed)} segment(s) committed")
    return 0


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


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    try:
        state = read_state(run_dir)
        manifest = read_manifest(run_dir)
    except StateError as exc:
        print(f"status: BROKEN ({exc})", file=sys.stderr)
        return 1
    try:
        config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
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
    print(f"  Segments: {state.committed_segments}")
    if config:
        print(f"  Blocks per segment: {config.video.blocks_per_segment}")
    gpus = probe().get("gpus")
    if isinstance(gpus, list) and gpus:
        print(f"  GPU: {gpus[0]}")
        for extra in gpus[1:]:
            print(f"       {extra}")
    else:
        print("  GPU: unavailable (no nvidia-smi)")
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
    print()
    print("Workers")
    for name in ("video", "audio", "director"):
        backend = "unknown"
        if config:
            backend = {"video": config.video.backend, "audio": config.audio.backend}.get(
                name, config.director.backend
            )
        print(f"  {name}: idle ({backend} backend; workers run during `voyage run`)")
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
    print()
    print("Storage")
    try:
        free_gib = shutil.disk_usage(run_dir).free / (1024**3)
        print(f"  Free: {free_gib:.1f} GiB")
    except OSError:
        print("  Free: unknown")
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
        args.output = str(resolve_run_dir(args.run) / "final.mp4")
        return cmd_finalize(args)
    return code


_SEGMENT_ID_PATTERN = re.compile(r"^\d{6}$")


def _check_segment_checksums(segment: Path) -> list[str]:
    """Recompute sha256.json entries (DESIGN §70: checksum mismatches)."""
    errors: list[str] = []
    checksums_path = segment / "sha256.json"
    if not checksums_path.exists():
        return errors  # missing file already reported by the caller
    try:
        expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    except ValueError:
        return [f"{segment.name} has unreadable sha256.json"]
    if not isinstance(expected, dict):
        return [f"{segment.name} has malformed sha256.json"]
    for artifact in ("video.mp4", "audio.wav"):
        recorded = expected.get(artifact)
        target = segment / artifact
        if not target.exists():
            continue  # missing artifact already reported by the caller
        if not isinstance(recorded, str) or not recorded:
            errors.append(f"{segment.name} sha256.json missing entry for {artifact}")
        elif sha256_file(target) != recorded:
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
    except (ValueError, KeyError, AttributeError, TypeError):
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
        drift = abs(video_duration - audio_duration)
        if drift > AV_ALIGNMENT_TOLERANCE_SECONDS:
            errors.append(
                f"{segment.name} A/V alignment drift {drift:.3f}s "
                f"exceeds {AV_ALIGNMENT_TOLERANCE_SECONDS:.1f}s"
            )
    tape = metrics.get("recovery_tape")
    if isinstance(tape, str) and tape:
        tape_path = paths.resolve_stored_path(run_dir, tape) if run_dir is not None else Path(tape)
        if not tape_path.exists():
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


def _augment_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI augment flags → resolve_config kwargs (Track A).

    `--no-augment` wins over explicit floors (both to 0). Every read
    goes through getattr + `is_provided` so TUI/hand-built namespaces
    (Unset blanks, missing attrs) resolve to absent, never to a value.
    """
    if bool(getattr(args, "no_augment", False)):
        return {"min_fps": 0, "min_resolution": "0"}
    overrides: dict[str, Any] = {}
    min_fps = getattr(args, "min_fps", None)
    if is_provided(min_fps):
        overrides["min_fps"] = min_fps
    min_resolution = getattr(args, "min_resolution", None)
    if is_provided(min_resolution):
        overrides["min_resolution"] = min_resolution
    return overrides


def cmd_finalize(args: argparse.Namespace) -> int:
    run_dir = _run_dir_arg(args.run)
    config, _digest = _load_run(run_dir)
    try:
        # CLI floors ride the stored [augment] section: explicit flags win,
        # otherwise the run TOML rules. Invalid floors fail here (exit 2),
        # before any media work. The resolved floors ride `config.augment`
        # for the finalize consumer (a later slice reads them).
        config = resolve_config(config, **_augment_overrides(args))
    except (ValidationError, ValueError) as exc:
        print(f"error: invalid augment override: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    try:
        finalize_run(
            run_dir,
            output,
            # Finalize keeps the generation resolution (no downscale): the
            # run config snapshot carries what the segments rendered at,
            # so old 768x512 runs refinalize natively too.
            width=config.video.width,
            height=config.video.height,
            fps=config.video.fps,
            skip_bad=args.skip_bad,
            min_free_space_gib=config.min_free_space_gib,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            overlap_fraction=config.audio.final_overlap_fraction,
            overlap_cap_seconds=config.audio.final_overlap_cap_seconds,
            min_fps=config.augment.min_fps,
            min_width=config.augment.min_width,
            min_height=config.augment.min_height,
        )
    except (MediaError, StateError, DiskSpaceError) as exc:
        print(f"finalize failed: {exc}", file=sys.stderr)
        return 1
    sfx_backend = getattr(args, "sfx_backend", None) or config.sfx.backend
    if not getattr(args, "no_sfx", False) and sfx_backend != "fake":
        from voyage.sfx_finalize import finalize_sfx_pass

        try:
            finalize_sfx_pass(
                run_dir,
                output,
                backend=sfx_backend,
                models_dir=config.sfx.models_dir,
                device=getattr(args, "sfx_device", None) or config.sfx.device,
                model_size=getattr(args, "sfx_model_size", None) or config.sfx.model_size,
                seed=config.seed,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
                num_workers=getattr(args, "sfx_workers", 1),
                fps=config.video.fps,
                caption_override=getattr(args, "sfx_caption", None),
            )
        except (MediaError, StateError) as exc:
            print(f"sfx pass failed (music-only final kept at {output}): {exc}", file=sys.stderr)
            return 1
    console = get_console(args)
    try:
        info = media_probe(output)
        duration = info.get("format", {}).get("duration", "?") if isinstance(info, dict) else "?"
    except MediaError:
        duration = "?"
    try:
        size_mb = output.stat().st_size / (1024**2)
        size_text = f"{size_mb:.1f} MiB"
    except OSError:
        size_text = "unknown size"
    console.ok(f"finalized -> {output} ({duration}s, {size_text})")
    print(f"finalized -> {output}")
    return 0


def cmd_sfx(args: argparse.Namespace) -> int:
    """Dub SFX onto an existing video (standalone post-pass, no re-finalize).

    Conditions on `--video` (default: the run's `final.mp4`), mixes the
    bed under its audio, and publishes `--output` (default:
    `final-sfx.mp4` beside the input — the input is never modified).
    Caption source is per-segment director captions unless
    `--sfx-caption` overrides (required for runs committed before SFX
    captions existed, e.g. poulah).
    """
    import shutil

    from voyage.sfx_finalize import finalize_sfx_pass

    run_dir = _run_dir_arg(args.run)
    config, _digest = _load_run(run_dir)
    video = Path(args.video).resolve() if args.video else run_dir / "final.mp4"
    if not video.exists():
        print(f"sfx failed: no such video {video}", file=sys.stderr)
        return 1
    output = Path(args.output).resolve() if args.output else video.parent / "final-sfx.mp4"
    backend = args.sfx_backend or config.sfx.backend
    try:
        if output != video:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, output)
        finalize_sfx_pass(
            run_dir,
            output,
            backend=backend,
            models_dir=config.sfx.models_dir,
            device=args.sfx_device or config.sfx.device,
            model_size=args.sfx_model_size or config.sfx.model_size,
            seed=config.seed,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            num_workers=args.sfx_workers,
            fps=config.video.fps,
            caption_override=args.sfx_caption,
        )
    except (MediaError, StateError, OSError) as exc:
        print(f"sfx dub failed: {exc}", file=sys.stderr)
        return 1
    console = get_console(args)
    try:
        info = media_probe(output)
        duration = info.get("format", {}).get("duration", "?") if isinstance(info, dict) else "?"
    except MediaError:
        duration = "?"
    console.ok(f"sfx dubbed -> {output} ({duration}s, backend {backend})")
    print(f"sfx dubbed -> {output}")
    return 0


_DURATION_EXAMPLES = "'5s', '90', '1m30s', '2m', '1h', '1h2m3.5s'"

_DURATION_PATTERN = re.compile(
    r"(?:(?P<hours>-?\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>-?\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>-?\d+(?:\.\d+)?)s?)?"
)


def parse_duration(raw: str) -> float:
    """Human-readable duration -> seconds (e.g. '5s', '90', '1m30s', '2m').

    Accepts hours (`1h`), fractional (`2.5m`, `1.5h`), combined
    (`1h2m3.5s`), bare (`90`) and whitespace-padded values. Signed
    numbers parse so negatives reach the positivity error below
    instead of the regex error. Rounds up to whole segments downstream
    (`segments_for_duration`), so the video never runs short.
    """
    match = _DURATION_PATTERN.fullmatch(raw.strip())
    if match is None or not any(match.groupdict().values()):
        raise ValueError(f"invalid duration {raw!r} (examples: {_DURATION_EXAMPLES})")
    total = 0.0
    for name, scale in (("hours", 3600.0), ("minutes", 60.0), ("seconds", 1.0)):
        value = match.group(name)
        if value is not None:
            total += float(value) * scale
    if total <= 0:
        raise ValueError(f"duration must be positive, got {raw!r}")
    return total


# Stream-A accounting (DESIGN §5.3, measured from the real tensors in
# voyage/workers/video_ltxv.py): every clip renders 121 frames; fresh blocks
# commit all 121, conditioned blocks drop the 25-frame prefix and commit 96
# novel. Duration planning uses the steady-state minimum (96 per block) so
# `generate --duration` never runs short however the fresh/extension mix
# lands (worker-reported frames remain the timeline truth).
_LTXV_NOVEL_BLOCK_FRAMES = 96
# LongLive Wan temporal VAE: L latents decode to (L-1)*4+1 frames, one block
# appends 8 latents (measured: 29f per 1-block segment, 93f per 3-block
# segment — qual-longlive2 + Phase-2 E2E, 2026-09-24/22).
_LONGLIVE_LATENTS_PER_BLOCK = 8
_LONGLIVE_DECODE_EXPANSION = 4
# CausVid DMD rollout: 81 decoded frames per rollout, the last
# 4*(overlap-1)+1 are the conditioning tail (9 at overlap 3) — 72 novel
# committed per rollout, uniform including rollout 0 (upstream long-video
# script parity). Duration planning uses the steady-state 72 so `generate
# --duration` never runs short (worker-reported frames stay the truth).
_CAUSVID_NOVEL_PER_ROLLOUT = 72


def _frames_per_segment(config: ProjectConfig) -> int:
    """Committed frames per segment for duration math (backend-specific)."""
    if config.video.backend == "ltxv":
        return _LTXV_NOVEL_BLOCK_FRAMES * config.video.blocks_per_segment
    if config.video.backend == "longlive2":
        latents = _LONGLIVE_LATENTS_PER_BLOCK * config.video.blocks_per_segment
        return (latents - 1) * _LONGLIVE_DECODE_EXPANSION + 1
    if config.video.backend == "causvid":
        return _CAUSVID_NOVEL_PER_ROLLOUT * config.video.blocks_per_segment
    return config.video.segment_frames


def segments_for_duration(duration_seconds: float, fps: int, frames_per_segment: int) -> int:
    """Segments needed to reach at least duration_seconds (rounds up, min 1)."""
    return max(1, math.ceil(duration_seconds * fps / frames_per_segment - 1e-9))


_CUDA_BACKENDS = frozenset({"ltxv", "longlive2", "causvid", "acestep"})


def _torch_available() -> bool:
    """Whether torch is importable (find_spec locates without importing)."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _cuda_stack_error(backend: str) -> str:
    return (
        f"error: backend {backend!r} needs the CUDA worker stack (torch), "
        "but torch is not importable in this container; re-run with "
        "VOYAGE_IMAGE=voyage-video:latest and VOYAGE_GPUS=1 (run.sh selects "
        "both automatically for CUDA backends and GPU-box bare launches)"
    )


def _cuda_offenders(config: ProjectConfig) -> list[str]:
    """CUDA backends configured on this run, video/audio qualified (051).

    The old message always blamed the video backend even when only the
    audio stack needed CUDA (e.g. acestep-audio + fake-video on CPU).
    """
    offenders: list[str] = []
    if config.video.backend in _CUDA_BACKENDS:
        offenders.append(f"video {config.video.backend!r}")
    if config.audio.backend in _CUDA_BACKENDS:
        offenders.append(f"audio {config.audio.backend!r}")
    return offenders


def _require_cuda_stack(config: ProjectConfig) -> bool:
    """Fast-fail when a CUDA backend is configured but torch is unavailable.

    Workers are in-container subprocesses, so the container image must carry
    the worker stack (run.sh selects voyage-video automatically for CUDA
    backends and GPU-box bare launches; direct `docker run` users must
    pass the image + --gpus all themselves).
    find_spec locates torch without importing it — the supervisor never
    imports GPU libraries (§83).
    """
    needs_cuda = config.video.backend in _CUDA_BACKENDS or config.audio.backend in _CUDA_BACKENDS
    if not needs_cuda or _torch_available():
        return True
    offenders = _cuda_offenders(config)
    label = " + ".join(offenders) if offenders else config.video.backend
    print(_cuda_stack_error(label), file=sys.stderr)
    return False


def _warn_if_no_cuda(config: ProjectConfig) -> None:
    """Preflight: a CUDA device with no visible GPU fails late at worker init."""
    if not config.video.device.startswith("cuda"):
        return
    from voyage.doctor import probe as doctor_probe

    if not doctor_probe().get("nvidia_smi"):
        print(
            "warning: video device is CUDA but no GPU is visible; "
            "the worker will fail at init (use VOYAGE_GPUS=1 with run.sh)",
            file=sys.stderr,
        )


def cmd_generate(args: argparse.Namespace) -> int:
    """One-shot fixed-duration video: init -> run -> validate -> finalize."""
    # Fail before init: generate always writes a fresh config from the
    # backend preset (CUDA video backends pair with ACE-Step audio), so the
    # preset alone decides the stack.
    if args.backend in _CUDA_BACKENDS and not _torch_available():
        print(_cuda_stack_error(args.backend), file=sys.stderr)
        return 1
    run_id = _effective_run_id(args)
    if _check_run_id(run_id) != 0:
        return 2
    # Resolve once: workers spawn with CWD=run_dir, so every downstream path
    # (payloads, takes, slices) must be absolute or they double up.
    output = Path(args.output) if args.output else Path("output") / run_id
    run_dir = resolve_run_dir(str(output))
    init_args = argparse.Namespace(
        output=str(run_dir),
        run_id=run_id,
        name=run_id,
        style=args.style,
        seed=args.seed,
        force=args.force,
        backend=args.backend,
        director=args.director,
        director_device=getattr(args, "director_device", None),
    )
    code = cmd_init(init_args)
    if code != 0:
        return code
    config, _digest = _load_run(run_dir)
    if not _require_cuda_stack(config):
        return 1
    _warn_if_no_cuda(config)
    # `generate` enables the full stack by default: qwen director drift,
    # beat-grid audio and overlap-blend finalize all ride the run config
    # written at init; explicit flags still win. The parser default for
    # --director is qwen, so args.director carries it directly.
    director = args.director
    try:
        effective = apply_draft_overrides(
            config,
            draft=args.draft,
            director=director,
            director_device=getattr(args, "director_device", None),
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n_segments=args.drift_every_n,
            music_caption=getattr(args, "music_caption", None),
            video_caption=getattr(args, "video_caption", None),
            **_augment_overrides(args),
        )
    except (ValidationError, ValueError) as exc:
        print(f"error: invalid numeric override: {exc}", file=sys.stderr)
        return 2
    frames_per_segment = _frames_per_segment(effective)
    segments = segments_for_duration(args.duration, effective.video.fps, frames_per_segment)
    planned_frames = segments * frames_per_segment
    console = get_console(args)
    sink = getattr(args, "progress_sink", None)
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
            # The mount may not exist yet (downloads create it): check the
            # nearest existing ancestor so a missing dir never crashes.
            anchor = Path(stack_dir)
            while not anchor.exists():
                anchor = anchor.parent
            check_free_space(anchor, effective.min_free_space_gib)
    except DiskSpaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    from voyage.models_ensure import ensure_models

    sfx_enabled = (
        not bool(getattr(args, "no_sfx", False))
        and (getattr(args, "sfx_backend", None) or effective.sfx.backend) != "fake"
    )
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
    if sink is None:
        console.rule(
            f"voyage generate · {effective.video.backend} "
            f"{effective.video.width}x{effective.video.height} @{effective.video.fps}fps · "
            f"director {effective.director.backend}"
        )
        console.info(
            f"plan: ~{planned_frames / effective.video.fps:.1f}s "
            f"({segments} segments, {planned_frames} frames) "
            f"· {effective.audio.beats_per_segment} beats/segment · "
            f"drift every {effective.voyage.drift_every_n_segments}"
        )
        print(
            f"generating ~{planned_frames / effective.video.fps:.1f}s "
            f"({segments} segments, {planned_frames} frames) "
            f"with {effective.video.backend} ..."
        )
    cmd_run(
        argparse.Namespace(
            run=str(run_dir),
            segments=segments,
            draft=args.draft,
            director=director,
            director_device=getattr(args, "director_device", None),
            blocks=args.blocks,
            take_seconds=args.take_seconds,
            quantization=args.quantization,
            beats_per_segment=args.beats_per_segment,
            drift_every_n=args.drift_every_n,
            min_fps=getattr(args, "min_fps", None),
            min_resolution=getattr(args, "min_resolution", None),
            no_augment=bool(getattr(args, "no_augment", False)),
            verbose=console.verbose,
            no_color=getattr(args, "no_color", False),
            progress_sink=sink,
        )
    )
    errors = validate_run(run_dir)
    if errors:
        print("INVALID:")
        for error in errors:
            print(f"  - {error}")
        if not getattr(args, "skip_bad", False):
            print(
                "aborting before finalize (re-run with --skip-bad to salvage)",
                file=sys.stderr,
            )
            return 1
        print("continuing with --skip-bad ...", file=sys.stderr)
    final_video = getattr(args, "final_video", None)
    final = Path(final_video).resolve() if final_video else run_dir / "final.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    final_code = cmd_finalize(
        argparse.Namespace(
            run=str(run_dir),
            output=str(final),
            skip_bad=getattr(args, "skip_bad", False),
            no_sfx=getattr(args, "no_sfx", False),
            sfx_backend=getattr(args, "sfx_backend", None),
            sfx_caption=getattr(args, "sfx_caption", None),
            sfx_device=getattr(args, "sfx_device", None),
            sfx_model_size=getattr(args, "sfx_model_size", None),
            sfx_workers=getattr(args, "sfx_workers", 1),
            min_fps=getattr(args, "min_fps", None),
            min_resolution=getattr(args, "min_resolution", None),
            no_augment=bool(getattr(args, "no_augment", False)),
        )
    )
    if final_code != 0:
        return final_code
    state = read_state(run_dir)
    actual_seconds = state.timeline_frames / effective.video.fps
    if sink is None:
        console.ok(
            f"generated {final} ({state.committed_segments} segments, "
            f"{state.timeline_frames} frames, ~{actual_seconds:.1f}s)"
        )
        print(
            f"generated {final} ({state.committed_segments} segments, "
            f"{state.timeline_frames} frames, ~{actual_seconds:.1f}s)"
        )
    return 0


def _benchmark_env() -> dict[str, object]:
    """§104 setup fields: GPU/driver/torch, honest unknowns off-GPU."""
    from voyage.doctor import probe as doctor_probe

    facts = doctor_probe()
    gpus = facts.get("gpus")
    gpu = gpus[0] if isinstance(gpus, list) and gpus else "unknown"
    try:
        import torch

        torch_version: object = torch.__version__
        cuda_available: object = torch.cuda.is_available()
    except ImportError:
        torch_version = "unknown"
        cuda_available = "unknown"
    return {"gpu": gpu, "torch": torch_version, "cuda_available": cuda_available}


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
            "warmup": warmup,
            "measured": measured,
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
            **_benchmark_env(),
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
    setup: dict[str, object] = {"run": str(run_dir), "segments_requested": segments}
    metrics: dict[str, object] = {
        "segments_committed": committed,
        "stages": _stage_means(events),
        **summarize_gauges([event for event in events if event.get("event") == "resource_gauges"]),
        "prefetch": summarize_prefetch_outcome(events),
        "validate_errors": validate_run(run_dir),
    }
    print(format_report("soak", setup, metrics))
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


def _add_init_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`init` verb: create a new run directory."""
    init = sub.add_parser("init", help="Create a new run directory")
    init.add_argument("--output", required=True, help="run directory to create")
    init.add_argument("--run-id", default="voyage", help="run name (flat folder name, no slashes)")
    init.add_argument(
        "--name",
        default=None,
        help="run name (primary spelling; wins over --run-id, same flat folder rule)",
    )
    init.add_argument("--style", required=True, help="permanent style charter for the run")
    init.add_argument("--seed", type=int, default=0, help="master seed for the run")
    init.add_argument("--force", action="store_true", help="allow init into a non-empty directory")
    init.add_argument(
        "--backend",
        choices=("fake", "longlive2", "ltxv", "causvid"),
        default="ltxv",
        help="video backend preset written into the run config",
    )
    init.add_argument(
        "--director",
        choices=("qwen", "deterministic"),
        default="qwen",
        help="director backend written into the run config "
        "(default qwen; deterministic disables the LLM)",
    )
    init.add_argument(
        "--director-device",
        default="cuda:1",
        help="director decider placement written into the run config "
        "(default cuda:1; cpu = legacy bf16 CPU path)",
    )
    init.set_defaults(func=cmd_init)


def _add_doctor_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`doctor` verb: probe hardware and environment."""
    doctor = sub.add_parser("doctor", help="Probe hardware and environment")
    doctor.set_defaults(func=cmd_doctor)


def _add_models_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`models` verb: model management."""
    models = sub.add_parser("models", help="Model management")
    models.add_argument(
        "models_action",
        choices=["list", "download", "verify", "info"],
        help="list backends, download weights, verify files, or show pointers",
    )
    models.add_argument(
        "models_target",
        nargs="?",
        default="ltxv-2b",
        choices=[
            "longlive2-bf16",
            "ltxv-2b",
            "causvid",
            "director-qwen8b",
            "audio-acestep",
            "sfx-mmaudio",
            "film",
            "realesrgan-anime",
            "inspector-qwen35",
        ],
        help="weight bundle for download (default ltxv-2b)",
    )
    models.add_argument("--models-dir", default=None, help="model root (default /models)")
    models.set_defaults(func=cmd_models)


def _add_generation_overrides(parser: argparse.ArgumentParser) -> None:
    """In-memory run overrides shared by `run` and `generate` (issue 020).

    One helper so the flag set cannot drift between the two verbs; order
    matches the historical layout (help output unchanged).
    """
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="override video blocks per segment (must be positive)",
    )
    parser.add_argument(
        "--take-seconds",
        type=float,
        default=None,
        help="override audio take length in seconds (must be positive)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=("fp8", "bf16"),
        help="override DiT quantization (fp8 default, bf16 for clean highlights)",
    )
    parser.add_argument(
        "--beats-per-segment",
        type=int,
        default=None,
        help="override beats per segment (must be positive; default 4, doubles to hold >=60 BPM)",
    )
    parser.add_argument(
        "--drift-every-n",
        type=int,
        default=None,
        help="director drifts every Nth segment (must be positive; default 1); other segments hold",
    )
    parser.add_argument(
        "--music-caption",
        default=None,
        help="pin the music caption family (default: director drives + evolves it)",
    )
    parser.add_argument(
        "--video-caption",
        default=None,
        help="pin the video caption family (default: director drives + evolves it)",
    )
    parser.add_argument(
        "--director-device",
        default=None,
        help="director decider placement (default cuda:1 = second GPU via 4-bit AWQ; "
        "cpu = legacy bf16 CPU path, explicit opt-out for single-GPU/CI boxes)",
    )


def _add_sfx_args(parser: argparse.ArgumentParser, *, include_no_sfx: bool = True) -> None:
    """Finalize-time SFX flags, shared by every verb that finalizes (092).

    `finalize` owns the pass; `generate`/`stop --finalize` forward into
    it; the standalone `sfx` verb dubs an existing video (no --no-sfx
    there — the verb IS the pass). One helper so the flags (and their
    defaults) cannot drift apart across verbs — a missing flag on any
    finalizing verb is an AttributeError at finalize time.
    """
    if include_no_sfx:
        parser.add_argument(
            "--no-sfx",
            action="store_true",
            help="skip the finalize-time SFX pass even when [sfx] is configured",
        )
    parser.add_argument(
        "--sfx-backend",
        default=None,
        choices=["fake", "mmaudio"],
        help="SFX backend override (default: [sfx] backend)",
    )
    parser.add_argument(
        "--sfx-caption",
        default=None,
        help="single SFX caption for the whole timeline (default: per-segment "
        "director captions; required for runs committed before SFX captions existed)",
    )
    parser.add_argument(
        "--sfx-device",
        default=None,
        help="SFX worker device override (default: [sfx] device)",
    )
    parser.add_argument(
        "--sfx-model-size",
        default=None,
        choices=["small_44k", "medium_44k", "large_44k_v2"],
        help="MMAudio variant override (default: [sfx] model_size)",
    )
    parser.add_argument(
        "--sfx-workers",
        type=int,
        default=1,
        choices=[1, 2],
        help="1 = one worker (default); 2 = shard small_44k across cuda:0+cuda:1",
    )


def _add_augment_args(parser: argparse.ArgumentParser) -> None:
    """Finalize-time augmentation floors, shared by verbs carrying them (Track A).

    `finalize` owns the floors (defaults ride the run's [augment] TOML
    section); `generate`/`run` forward overrides into resolve_config so
    the effective config carries them. One helper so the flags (and their
    defaults) cannot drift apart across verbs — mirrors `_add_sfx_args`.
    """
    parser.add_argument(
        "--min-fps",
        type=int,
        default=None,
        help="floor output fps at finalize (default: [augment] min_fps 32; 0 disables)",
    )
    parser.add_argument(
        "--min-resolution",
        default=None,
        help='floor output resolution at finalize, WxH e.g. "1280x720" '
        '(default: [augment] 1280x720; "0" disables)',
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="disable all finalize augmentation floors (fps + resolution floors to 0)",
    )


def _add_run_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`run` verb: generate segments (infinite unless --segments)."""
    run = sub.add_parser("run", help="Generate segments (infinite unless --segments)")
    run.add_argument("--run", required=True, help="run directory (absolute or relative)")
    run.add_argument(
        "--segments",
        type=int,
        default=None,
        help="segments to generate (must be positive; omit to run until pause/stop/SIGINT)",
    )
    run.add_argument(
        "--draft",
        action="store_true",
        help="apply the [draft] profile (fast low-res iteration settings)",
    )
    run.add_argument(
        "--director",
        default=None,
        choices=("qwen", "deterministic"),
        help="override the director backend",
    )
    _add_generation_overrides(run)
    _add_augment_args(run)
    _add_console_args(run)
    run.set_defaults(func=cmd_run)


def _add_generate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`generate` verb: one-shot fixed-duration video."""
    gen = sub.add_parser(
        "generate",
        help="One-shot fixed-duration video (init + run + validate + finalize)",
    )
    gen.add_argument(
        "--backend",
        choices=("fake", "longlive2", "ltxv", "causvid"),
        default="ltxv",
        help="video backend preset written into the run config",
    )
    gen.add_argument(
        "--duration",
        type=parse_duration,
        required=True,
        help=f"target length, e.g. {_DURATION_EXAMPLES} (rounds up to whole segments)",
    )
    gen.add_argument("--style", required=True, help="permanent style charter for the run")
    gen.add_argument("--run-id", default="voyage", help="run name (flat folder name, no slashes)")
    gen.add_argument(
        "--name",
        default=None,
        help="run name (primary spelling; wins over --run-id, same flat folder rule)",
    )
    gen.add_argument("--output", default=None, help="run directory (default output/<run-id>)")
    gen.add_argument("--seed", type=int, default=0, help="master seed for the run")
    gen.add_argument("--force", action="store_true", help="allow init into a non-empty directory")
    gen.add_argument(
        "--final-video",
        default=None,
        help="final mp4 path (default <run>/final.mp4)",
    )
    gen.add_argument(
        "--skip-bad",
        action="store_true",
        help="finalize past corrupt segments instead of aborting",
    )
    gen.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading missing models (verify only)",
    )
    _add_sfx_args(gen)
    _add_augment_args(gen)
    gen.add_argument(
        "--draft",
        action="store_true",
        help="apply the [draft] profile (fast low-res iteration settings)",
    )
    gen.add_argument(
        "--director",
        default="qwen",
        choices=("qwen", "deterministic"),
        help="director backend (default qwen)",
    )
    _add_generation_overrides(gen)
    _add_console_args(gen)
    gen.set_defaults(func=cmd_generate)


def _add_status_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`status` verb: show run status."""
    status = sub.add_parser("status", help="Show run status")
    status.add_argument("--run", required=True, help="run directory to report on")
    status.set_defaults(func=cmd_status)


def _add_pause_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`pause` verb: request a safe pause."""
    pause = sub.add_parser("pause", help="Request a safe pause")
    pause.add_argument("--run", required=True, help="run directory to pause")
    pause.set_defaults(func=cmd_pause)


def _add_resume_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`resume` verb: resume from last commit."""
    resume = sub.add_parser("resume", help="Resume from last commit")
    resume.add_argument("--run", required=True, help="run directory to resume")
    resume.set_defaults(func=cmd_resume)


def _add_stop_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`stop` verb: safely stop generation."""
    stop = sub.add_parser("stop", help="Safely stop generation")
    stop.add_argument("--run", required=True, help="run directory to stop")
    stop.add_argument(
        "--finalize",
        action="store_true",
        help="run the finalizer inline after requesting stop",
    )
    _add_sfx_args(stop)
    stop.set_defaults(func=cmd_stop)


def _add_validate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`validate` verb: offline consistency check (read-only)."""
    validate = sub.add_parser("validate", help="Offline consistency check (read-only)")
    validate.add_argument("--run", required=True, help="run directory to check")
    validate.set_defaults(func=cmd_validate)


def _add_finalize_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`finalize` verb: assemble the final MP4."""
    finalize = sub.add_parser("finalize", help="Assemble the final MP4")
    finalize.add_argument("--run", required=True, help="run directory to finalize")
    finalize.add_argument("--output", required=True, help="final mp4 path to write")
    finalize.add_argument(
        "--skip-bad",
        action="store_true",
        help="skip corrupt segments with a warning instead of aborting",
    )
    _add_sfx_args(finalize)
    _add_augment_args(finalize)
    _add_console_args(finalize)
    finalize.set_defaults(func=cmd_finalize)


def _add_sfx_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`sfx` verb: dub SFX onto an existing video (no re-finalize)."""
    sfx = sub.add_parser("sfx", help="Dub SFX onto an existing video")
    sfx.add_argument("--run", required=True, help="run directory (captions + ledger + seeds)")
    sfx.add_argument(
        "--video",
        default=None,
        help="existing mp4 to condition on (default: <run>/final.mp4)",
    )
    sfx.add_argument(
        "--output",
        default=None,
        help="dubbed mp4 to write (default: final-sfx.mp4 beside the input)",
    )
    _add_sfx_args(sfx, include_no_sfx=False)
    _add_console_args(sfx)
    sfx.set_defaults(func=cmd_sfx)


def _add_benchmark_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`benchmark` verb: performance probes."""
    benchmark = sub.add_parser("benchmark", help="Performance probes")
    benchmark.add_argument(
        "benchmark_target",
        choices=["video", "audio", "end-to-end"],
        help="which probe to run",
    )
    benchmark.add_argument(
        "--run", default="", help="run directory (required for video/audio targets)"
    )
    benchmark.add_argument("--warmup", type=int, default=1, help="warmup iterations (must be >= 0)")
    benchmark.add_argument(
        "--measured", type=int, default=3, help="measured iterations (must be >= 1)"
    )
    benchmark.add_argument(
        "--segments",
        type=int,
        default=2,
        help="segments for the end-to-end target (must be positive)",
    )
    benchmark.set_defaults(func=cmd_benchmark)


def _add_soak_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`soak` verb: stability run with a resource-trend report."""
    soak = sub.add_parser("soak", help="Stability run with a resource-trend report")
    soak.add_argument("--run", required=True, help="run directory to soak-test")
    soak.add_argument(
        "--segments", type=int, required=True, help="segments to run (must be positive)"
    )
    _add_console_args(soak)
    soak.set_defaults(func=cmd_soak)


def _add_inspect_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """`inspect` verb: inspect run artifacts."""
    inspect = sub.add_parser("inspect", help="Inspect run artifacts")
    inspect.add_argument(
        "inspect_target",
        choices=["concepts", "segments", "media", "metrics", "scoreboard"],
        help="which artifact view to print",
    )
    inspect.add_argument("--run", required=True, help="run directory to inspect")
    inspect.set_defaults(func=cmd_inspect)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voyage", description="Autonomous infinite audiovisual voyage"
    )
    # No `required=True`: the bare command (no verb) launches the
    # interactive launcher TUI (see main), which configures `generate`.
    sub = parser.add_subparsers(dest="command", required=False)

    _add_init_parser(sub)
    _add_doctor_parser(sub)
    _add_models_parser(sub)
    _add_run_parser(sub)
    _add_generate_parser(sub)

    _add_status_parser(sub)
    _add_pause_parser(sub)
    _add_resume_parser(sub)
    _add_stop_parser(sub)
    _add_validate_parser(sub)
    _add_finalize_parser(sub)
    _add_sfx_parser(sub)
    _add_benchmark_parser(sub)
    _add_soak_parser(sub)
    _add_inspect_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        # Bare `voyage`: interactive launcher TUI (TTY + Textual required;
        # pipes and missing extras get guidance, exit 2). CLI verbs below
        # stay fully usable non-interactively.
        try:
            return int(launch_tui())
        except VoyageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        return int(args.func(args))
    except VoyageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

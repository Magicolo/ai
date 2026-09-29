"""Inline model ensure for `voyage generate` (selective + parallel).

Why this module exists: `generate` used to fail late inside workers when
weight files were missing (a `FileNotFoundError` surfacing as a worker
retry/circuit-breaker after minutes of GPU work). This module derives the
exact stacks the effective run config needs — the video backend's own spec
plus its paired ACE-Step audio, the qwen director unless deterministic,
MMAudio SFX only when the finalize pass will run, the VLM inspector only
when enabled — verifies each via `model_registry`, and downloads the
missing ones in parallel with per-model console progress. Fake backends
need no weight files at all (the director falls back to deterministic),
so a fake run ensures the empty set and stays offline-friendly.

Manifest race note: `model_registry.download_model` bundles the hub fetch
with a read-modify-write of `manifest.json`, so parallel calls can drop
each other's manifest entries (last write wins). Fetches run unlocked for
speed; afterwards `_repair_manifest` re-merges any missing entries under
`_MANIFEST_LOCK` (record building only stats files, so it is cheap). The
repair is best-effort — `verify_model` stays authoritative.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voyage.config import ProjectConfig

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

_MANIFEST_LOCK = threading.Lock()
"""Serializes the manifest repair pass (never the hub fetches)."""

_MAX_PARALLEL_DOWNLOADS = 4
"""Thread-pool cap: hub fetches are network-bound, four keeps the pulse
lively without hammering the registry (at most three weight specs plus
the inspector ever download together)."""

_VIDEO_SPEC_FOR_BACKEND = {
    "longlive2": "longlive2-bf16",
    "ltxv": "ltxv-2b",
    "causvid": "causvid",
}
"""CUDA video backend → its registry spec (fake needs no files)."""


@dataclass(frozen=True)
class RequiredModel:
    """One registry spec `generate` must verify/download, with its dir."""

    spec: str
    models_dir: Path


def _resolve_dir(models_root: str | Path | None, configured: str) -> Path:
    """Single-mount override (tests/dev) or the stack's configured dir."""
    return Path(models_root) if models_root is not None else Path(configured)


def required_specs(
    config: ProjectConfig,
    sfx_enabled: bool = False,
    models_root: str | Path | None = None,
) -> list[RequiredModel]:
    """Specs the effective generate config needs — nothing else.

    `sfx_enabled` mirrors the finalize gate (`not no_sfx and backend is
    mmaudio`): the SFX stack downloads only when the pass will run. A
    fake video backend needs the empty set (weight-free offline runs).
    `VideoBackendName` is a closed Literal, so past the fake early-return
    the `_VIDEO_SPEC_FOR_BACKEND` index below is total (no KeyError).
    """
    if config.video.backend == "fake":
        return []
    required = [
        RequiredModel(
            spec=_VIDEO_SPEC_FOR_BACKEND[config.video.backend],
            models_dir=_resolve_dir(models_root, config.video.models_dir),
        )
    ]
    if config.audio.backend == "acestep":
        required.append(
            RequiredModel(
                spec="audio-acestep",
                models_dir=_resolve_dir(models_root, config.audio.models_dir),
            )
        )
    if config.director.backend == "qwen":
        required.append(
            RequiredModel(
                spec="director-qwen8b",
                models_dir=_resolve_dir(models_root, config.video.models_dir),
            )
        )
    if sfx_enabled and config.sfx.backend == "mmaudio":
        required.append(
            RequiredModel(
                spec="sfx-mmaudio",
                models_dir=_resolve_dir(models_root, config.sfx.models_dir),
            )
        )
    if config.experimental.visual_inspector:
        required.append(
            RequiredModel(
                spec="inspector-qwen35",
                models_dir=_resolve_dir(models_root, config.video.models_dir),
            )
        )
    return required


def _repair_manifest(entries: list[RequiredModel]) -> None:
    """Re-merge manifest entries lost to parallel-download races (best-effort).

    Skips silently on any filesystem miss: `verify_model` already passed
    for these entries, so the manifest entry is bookkeeping, never a gate.
    """
    from voyage import model_registry

    with _MANIFEST_LOCK:
        for entry in entries:
            try:
                spec = model_registry.MODEL_SPECS[entry.spec]
                manifest_path = entry.models_dir / "manifest.json"
                present: dict[str, Any] = {}
                if manifest_path.is_file():
                    raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        present = raw
                if spec.manifest_key in present:
                    continue
                model_registry._merge_manifest_record(
                    entry.models_dir,
                    spec.manifest_key,
                    spec.record_builder(entry.models_dir),
                )
            except OSError:
                continue


def ensure_models(
    config: ProjectConfig,
    sfx_enabled: bool,
    console: VoyageConsole,
    models_root: str | Path | None = None,
    *,
    allow_download: bool = True,
) -> int:
    """Verify (+ download when allowed) every spec `generate` needs.

    Returns 0 when all verify, 1 with console feedback otherwise.
    `allow_download=False` (`--no-download`) never touches the network:
    missing stacks fail fast with their verify message instead.
    """
    from voyage import model_registry

    required = required_specs(config, sfx_enabled, models_root)
    if not required:
        return 0
    checked = [
        (entry, model_registry.verify_model(entry.models_dir, entry.spec)) for entry in required
    ]
    missing = [(entry, message) for entry, (ok, message) in checked if not ok]
    if not missing:
        return 0
    if not allow_download:
        for entry, message in missing:
            console.warn(f"missing model {entry.spec}: {message} (re-run to download)")
        return 1
    failures: dict[str, str] = {}
    with (
        console.parallel_downloads([entry.spec for entry, _ in missing]) as tracker,
        ThreadPoolExecutor(max_workers=min(len(missing), _MAX_PARALLEL_DOWNLOADS)) as pool,
    ):
        future_to_entry: dict[Future[dict[str, Any]], RequiredModel] = {
            pool.submit(model_registry.download_model, entry.models_dir, entry.spec): entry
            for entry, _ in missing
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                future.result()
            except Exception as exc:
                failures[entry.spec] = str(exc)
                tracker.fail(entry.spec, str(exc))
                continue
            ok, message = model_registry.verify_model(entry.models_dir, entry.spec)
            if ok:
                tracker.succeed(entry.spec)
            else:
                failures[entry.spec] = message
                tracker.fail(entry.spec, message)
    if failures:
        _repair_manifest([entry for entry, _ in missing if entry.spec not in failures])
        for spec, detail in failures.items():
            console.error(f"model {spec} failed: {detail}")
        return 1
    _repair_manifest([entry for entry, _ in missing])
    console.ok(f"models ready: {', '.join(entry.spec for entry, _ in missing)}")
    return 0

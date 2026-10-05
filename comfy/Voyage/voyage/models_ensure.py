"""Inline model ensure for `voyage generate` (selective + parallel).

DESIGN §§84-85, §140 generate-ensure as-built.

Why this module exists: `generate` used to fail late inside workers when
weight files were missing (a `FileNotFoundError` surfacing as a worker
retry/circuit-breaker after minutes of GPU work). This module derives the
exact stacks the effective run config needs — the video backend's own spec
plus the FILM/Real-ESRGAN finalize augmentation on CUDA (unless
`augment_enabled=False`), its paired ACE-Step audio, the qwen director
unless deterministic (or the GGUF sidecar file for the llama backend),
MMAudio SFX only when the finalize pass will run,
the VLM inspector only when enabled — verifies each via `model_registry`,
and downloads the missing ones in parallel with per-model console
progress. Joint-audio video backends (`ltx25`, `ltx23`) render their own
soundtrack, so ACE-Step and MMAudio are never required for them.
Fake backends need no weight files at all (the director falls
back to deterministic), so a fake run ensures the empty set and stays
offline-friendly.

Manifest race note: `model_registry.download_model` bundles the hub fetch
with a read-modify-write of `manifest.json`, so parallel calls can drop
each other's manifest entries (last write wins). Fetches run unlocked for
speed; afterwards `_repair_manifest` re-merges any missing entries under
`_MANIFEST_LOCK` (record building only stats files, so it is cheap) with
one atomic write per models dir, retried on transient I/O. Repair is
fail-loud, never best-effort: entries still missing afterwards fail the
ensure (no hash-less `"models ready"`). Residual: the fetch+merge inside
`download_model` itself is still unlocked, so a concurrent downloader can
clobber the repair write — serializing that merge needs a registry-side
lock around fetch+merge (model_registry.py, proposed in issue 077).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from voyage.atomic import JsonValue, atomic_write_json, fsync_dir
from voyage.config import ProjectConfig

if TYPE_CHECKING:
    from voyage.console import VoyageConsole

_MANIFEST_LOCK = threading.Lock()
"""Serializes the manifest repair pass (never the hub fetches)."""

_REPAIR_ATTEMPTS = 3
"""Repair tries per models dir: transient I/O gets two retries; a torn
manifest is never retried (it cannot heal by re-reading)."""

_MAX_PARALLEL_DOWNLOADS = 4
"""Thread-pool cap: hub fetches are network-bound, four keeps the pulse
lively without hammering the registry (at most three weight specs plus
the inspector ever download together)."""

_VIDEO_SPEC_FOR_BACKEND = {
    "ltxv": "ltxv-2b",
    "causvid": "causvid",
    "ltx25": "ltx25",
    "ltx23": "ltx23",
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
    augment_enabled: bool = True,
    music_enabled: bool = True,
) -> list[RequiredModel]:
    """Specs the effective generate config needs — nothing else.

    `sfx_enabled` mirrors the finalize gate (`not no_sfx and backend is
    mmaudio`): the SFX stack downloads only when the pass will run. A
    fake video backend needs the empty set (weight-free offline runs).
    `augment_enabled` gates the finalize-stage model pass (FILM
    interpolation + Real-ESRGAN anime upscaler): CUDA backends include
    them when the multipliers demand work, `False` restores the
    pre-augmentation set (tests, weight-free probes). ACE-Step is required whenever the effective
    config pairs it (`audio.backend == "acestep"`): ltx25/ltx23 take
    their continuous music from the ACE planner's long caption-driven
    takes (DESIGN §140 audio continuity), not from the worker's joint
    track. SFX is pulled when enabled: MMAudio dubs effects under the
    soundtrack at finalize on cuda:0, after the video worker has
    stopped (DESIGN §140 GPU defaults). `music_enabled=False` (the
    generate-only --no-music/--no-audio skip) drops the ACE-Step stack:
    the finalize ships silent AAC, so no takes render.
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
    if augment_enabled:
        required.extend(
            [
                RequiredModel(
                    spec="film",
                    models_dir=_resolve_dir(models_root, config.video.models_dir),
                ),
                RequiredModel(
                    spec="realesrgan-anime",
                    models_dir=_resolve_dir(models_root, config.video.models_dir),
                ),
            ]
        )
    if music_enabled and config.audio.backend == "acestep":
        required.append(
            RequiredModel(
                spec="audio-acestep",
                models_dir=_resolve_dir(models_root, config.audio.models_dir),
            )
        )
    if config.director.backend == "qwen":
        # CUDA placements serve the 4-bit AWQ decider; the CPU opt-out
        # keeps the bf16 8B stack.
        director_spec = (
            "director-qwen8b" if config.director.device == "cpu" else "director-qwen4b-awq"
        )
        required.append(
            RequiredModel(
                spec=director_spec,
                models_dir=_resolve_dir(models_root, config.video.models_dir),
            )
        )
    elif config.director.backend == "llama":
        # llama-server sidecar (DESIGN §140): the single Q4_K_M GGUF the
        # supervisor serves on loopback — never the AWQ stack alongside
        # it (two decider stacks would double the ensure for no reason).
        required.append(
            RequiredModel(
                spec="director-qwen35-gguf",
                models_dir=_resolve_dir(models_root, config.video.models_dir),
            )
        )
    if sfx_enabled and config.sfx.backend == "mmaudio":
        # SFX dubs over the ACE-planner soundtrack at finalize (DESIGN
        # §140 GPU defaults).
        required.append(
            RequiredModel(
                spec="sfx-mmaudio",
                models_dir=_resolve_dir(models_root, config.sfx.models_dir),
            )
        )
    return required


def _read_manifest_keys(models_dir: Path) -> dict[str, JsonValue] | None:
    """Manifest mapping, `{}` when absent, `None` when torn (never overwrite).

    A torn file is `validate_run`'s territory (it recomputes checksums and
    reports); the repair must not blindly replace bytes it cannot parse.
    Values are `JsonValue`-typed (issue 035): the manifest is JSON-shaped
    by construction, so the repair's merge targets stay inside the typed
    boundary instead of bare `Any`.
    """
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        raw: JsonValue = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): value for key, value in raw.items()}


def _repair_manifest(entries: list[RequiredModel]) -> list[RequiredModel]:
    """Re-merge manifest entries lost to parallel-download races (fail-loud).

    One atomic write per models dir (`atomic_write_json` + `fsync_dir`,
    the §12 durability rule) under `_MANIFEST_LOCK`, retried
    `_REPAIR_ATTEMPTS` times on transient I/O. Returns entries still
    missing afterwards — `ensure_models` fails on them instead of
    reporting hash-less success.

    Only manifests that EXIST but lack keys count as race evidence (a
    parallel merge demonstrably dropped a record). An absent manifest is
    skipped, not repaired: real `download_model` merges on every success
    (a merge error fails the download outright), so absent-after-success
    cannot happen outside custom downloaders — and last-writer-wins
    always leaves the final writer's record behind, never an empty file.
    """
    from voyage import model_registry

    still_missing: list[RequiredModel] = []
    with _MANIFEST_LOCK:
        by_dir: dict[Path, list[RequiredModel]] = {}
        for entry in entries:
            by_dir.setdefault(entry.models_dir, []).append(entry)
        for models_dir, dir_entries in by_dir.items():
            if not (models_dir / "manifest.json").is_file():
                continue
            for _attempt in range(_REPAIR_ATTEMPTS):
                present = _read_manifest_keys(models_dir)
                if present is None:
                    break
                pending = [
                    entry
                    for entry in dir_entries
                    if model_registry.MODEL_SPECS[entry.spec].manifest_key not in present
                ]
                if not pending:
                    break
                try:
                    for entry in pending:
                        spec = model_registry.MODEL_SPECS[entry.spec]
                        present[spec.manifest_key] = spec.record_builder(models_dir)
                    atomic_write_json(models_dir / "manifest.json", present)
                    fsync_dir(models_dir)
                except OSError:
                    continue
                break
            verified = _read_manifest_keys(models_dir)
            if verified is None:
                still_missing.extend(dir_entries)
                continue
            for entry in dir_entries:
                if model_registry.MODEL_SPECS[entry.spec].manifest_key not in verified:
                    still_missing.append(entry)
    return still_missing


def ensure_models(
    config: ProjectConfig,
    sfx_enabled: bool,
    console: VoyageConsole,
    models_root: str | Path | None = None,
    *,
    allow_download: bool = True,
    augment_enabled: bool = True,
    music_enabled: bool = True,
) -> int:
    """Verify (+ download when allowed) every spec `generate` needs.

    Returns 0 when all verify, 1 with console feedback otherwise.
    `allow_download=False` (`--no-download`) never touches the network:
    missing stacks fail fast with their verify message instead.
    `augment_enabled=False` skips the FILM/Real-ESRGAN floors (weight-free
    probes); the CLI never passes it today, so CUDA runs ensure them.
    `music_enabled=False` (generate-only --no-music/--no-audio) skips
    the ACE-Step stack.
    """
    from voyage import model_registry

    required = required_specs(config, sfx_enabled, models_root, augment_enabled, music_enabled)
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
        future_to_entry: dict[Future[dict[str, JsonValue]], RequiredModel] = {
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
        unrepaired = _repair_manifest([entry for entry, _ in missing if entry.spec not in failures])
        for spec, detail in failures.items():
            console.error(f"model {spec} failed: {detail}")
        for entry in unrepaired:
            console.error(f"model {entry.spec} manifest record missing after repair")
        return 1
    unrepaired = _repair_manifest([entry for entry, _ in missing])
    if unrepaired:
        for entry in unrepaired:
            console.error(f"model {entry.spec} manifest record missing after repair")
        return 1
    console.ok(f"models ready: {', '.join(entry.spec for entry, _ in missing)}")
    return 0

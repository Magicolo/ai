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
    from voyage.model_registry import ModelSpec

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
    mastering_enabled: bool = True,
) -> list[RequiredModel]:
    """Specs the effective generate config needs — nothing else.

    `sfx_enabled` mirrors the finalize gate (`not no_sfx and backend is
    mmaudio`): the SFX stack downloads only when the pass will run. A
    fake video backend needs the empty set (weight-free offline runs).
    `augment_enabled` gates the finalize-stage model pass (configured-backend
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
    `mastering_enabled=False` (the generate-only --no-master/--no-audio
    skip) drops the SonicMaster stack: the finalize ships the unmastered
    mix. The stack is also dropped when `config.audio.mastering` is False
    (persistent opt-out). Fake video stays empty even when mastering is
    enabled (weight-free offline runs).
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
        interp_spec = "rife" if config.augment.interp_backend == "rife" else "film"
        required.extend(
            [
                RequiredModel(
                    spec=interp_spec,
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
    if mastering_enabled and config.audio.mastering:
        # SonicMaster mastering stage (Track A): public model + gated
        # Stable Audio Open VAE. Fake video already returned empty above,
        # so this branch only fires on CUDA backends.
        required.append(
            RequiredModel(
                spec="audio-sonicmaster",
                models_dir=_resolve_dir(models_root, config.audio.models_dir),
            )
        )
    return required


def validate_llama_placement(config: ProjectConfig) -> None:
    """Refuse cpu+llama at config time (Track D), or mask explicitly.

    The llama sidecar is GPU-only by design (`-ngl 99`); a `cpu`
    director device carries no CUDA index to pin, so the server inherits
    full visibility and can straddle the video card (or fail late with
    `cannot spawn llama-server`). Refuse it here with `ConfigurationError`
    so `configure` fails fast instead of dying mid-run. Explicit opt-out:
    set `VOYAGE_LLAMA_ALLOW_CPU=1` to keep the inherited-visibility
    behavior (documented, loud — the caller logs the bypass). Track A
    calls this at config time; `required_specs` stays total (it still
    selects the GGUF row for any llama backend — selection is not
    validation).
    """
    import os as _os

    from voyage.errors import ConfigurationError

    if config.director.backend != "llama":
        return
    if config.director.device != "cpu":
        return
    if _os.environ.get("VOYAGE_LLAMA_ALLOW_CPU", "").strip().lower() in {"1", "true", "yes"}:
        return
    raise ConfigurationError(
        "director backend 'llama' requires a CUDA device (got 'cpu') — "
        "the sidecar is GPU-only (`-ngl 99`); pass --director-device cuda:1 "
        "or set VOYAGE_LLAMA_ALLOW_CPU=1 to keep inherited visibility explicitly"
    )


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


def _manifest_entry_complete(models_dir: Path, spec_name: str) -> bool:
    """True when the manifest carries a usable hash baseline for `spec_name`.

    Fail-closed workers (`ltx25`/`ltx23`/`ltxv` via `verify_recorded_shas`,
    `causvid` via `verify_checkpoint_against_manifest`) refuse to load
    without a recorded sha, while `verify_model` only checks file presence
    when no baseline exists — so a present-files/missing-entry volume
    passes ensure and then dies in the worker (asporgue 2026-10-07:
    `no manifest entry for 'ltx25'`). A missing key always reads
    incomplete (race repair still restores unpinned rows); specs with
    neither `expected_hashes` nor `manifest_checkpoint` need no hash
    baseline beyond key presence.
    Stale single-sha rows without any sha (pre-hash `film`/`realesrgan`
    entries) read incomplete so they refresh to the recorded-sha shape.
    """

    from voyage import model_registry

    spec = model_registry.MODEL_SPECS[spec_name]
    present = _read_manifest_keys(models_dir)
    if not present:
        return False
    return _manifest_key_complete(present, spec)


def _manifest_key_complete(present: dict[str, JsonValue], spec: ModelSpec) -> bool:
    """Completeness of one spec against an already-loaded manifest map.

    Pure helper so the repair write-loop checks the `fresh` map it is
    about to write instead of re-reading the file per entry (the file
    read in `_manifest_entry_complete` is only the entry point).
    """
    entry = present.get(spec.manifest_key)
    if not isinstance(entry, dict):
        return False
    if not spec.expected_hashes and spec.manifest_checkpoint is None:
        return True
    if spec.expected_hashes:
        # Multi-file stacks gate on the full per-file dict: a lone
        # `checkpoint_sha256` cannot attest every pinned file, so legacy
        # single-sha rows read incomplete and refresh to the dict shape.
        shas = entry.get("checkpoint_shas")
        if not isinstance(shas, dict):
            return False
        for expected in spec.expected_hashes:
            recorded = shas.get(expected.relative_path)
            if not isinstance(recorded, str) or not recorded:
                return False
        return True
    recorded = entry.get("checkpoint_sha256")
    return isinstance(recorded, str) and bool(recorded)


def _expected_hash_mismatches(models_dir: Path, spec_name: str) -> list[str]:
    """Files disagreeing with the registry ingest pins (repair gate).

    Mirrors `download_model`'s pre-merge check: whatever is on disk must
    match `expected_hashes` before it becomes the attested-good manifest
    record. A mismatch means tampered/truncated bytes — the caller must
    not record the measured sha (that would attest bad bytes good) and
    must fail loud so the download path re-fetches.
    """

    from voyage import model_registry

    spec = model_registry.MODEL_SPECS[spec_name]
    mismatched: list[str] = []
    for expected in spec.expected_hashes:
        candidate = models_dir / expected.relative_path
        if not candidate.is_file():
            mismatched.append(str(candidate))
            continue
        try:
            model_registry.verify_checkpoint_sha256(candidate, expected.expected_sha256)
        except ValueError:
            mismatched.append(str(candidate))
    return mismatched


def _repair_manifest(entries: list[RequiredModel]) -> list[RequiredModel]:
    """Re-merge manifest entries lost to parallel-download races (fail-loud).

    One atomic write per models dir (`atomic_write_json` + `fsync_dir`,
    the §12 durability rule) under `_MANIFEST_LOCK`, retried
    `_REPAIR_ATTEMPTS` times on transient I/O. Returns entries still
    missing afterwards — `ensure_models` fails on them instead of
    reporting hash-less success.

    Repairs both race-dropped records and stale-volume gaps (present
    files, missing/stale manifest entry — asporgue 2026-10-07): an
    absent manifest starts from `{}` and is created by the write. A
    torn manifest is never overwritten (it stays missing so the caller
    fails loud).

    Entries whose files disagree with the registry ingest pins are never
    recorded (that would attest tampered bytes good) — they stay in the
    returned still-missing list so the caller fails loud.
    """
    from voyage import model_registry

    still_missing: list[RequiredModel] = []
    with _MANIFEST_LOCK:
        by_dir: dict[Path, list[RequiredModel]] = {}
        for entry in entries:
            by_dir.setdefault(entry.models_dir, []).append(entry)
        for models_dir, dir_entries in by_dir.items():
            for _attempt in range(_REPAIR_ATTEMPTS):
                present = _read_manifest_keys(models_dir)
                if present is None:
                    break
                pending = [
                    entry
                    for entry in dir_entries
                    if not _manifest_entry_complete(models_dir, entry.spec)
                ]
                if not pending:
                    break
                # Fail loud on tampered bytes: never attest mismatched files.
                tainted = [
                    entry for entry in pending if _expected_hash_mismatches(models_dir, entry.spec)
                ]
                pending = [entry for entry in pending if entry not in tainted]
                if not pending:
                    break
                try:
                    # Re-read under lock before writing (present may be stale
                    # after the hashing above); completeness re-checked via
                    # the fresh map so concurrent repairs do not clobber.
                    fresh = _read_manifest_keys(models_dir)
                    if fresh is None:
                        break
                    rewritten = False
                    for entry in pending:
                        spec = model_registry.MODEL_SPECS[entry.spec]
                        if _manifest_key_complete(fresh, spec):
                            continue
                        fresh[spec.manifest_key] = spec.record_builder(models_dir)
                        rewritten = True
                    if rewritten:
                        atomic_write_json(models_dir / "manifest.json", fresh)
                        fsync_dir(models_dir)
                except OSError:
                    continue
                break
            verified = _read_manifest_keys(models_dir)
            if verified is None:
                still_missing.extend(dir_entries)
                continue
            for entry in dir_entries:
                if (
                    not _manifest_entry_complete(models_dir, entry.spec)
                    and entry not in still_missing
                ):
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
    mastering_enabled: bool = True,
) -> int:
    """Verify (+ download when allowed) every spec `generate` needs.

    Returns 0 when all verify, 1 with console feedback otherwise.
    `allow_download=False` (`--no-download`) never touches the network:
    missing stacks fail fast with their verify message instead.
    `augment_enabled=False` skips the FILM/Real-ESRGAN floors (weight-free
    probes); the CLI never passes it today, so CUDA runs ensure them.
    `music_enabled=False` (generate-only --no-music/--no-audio) skips
    the ACE-Step stack. `mastering_enabled=False` (generate-only
    --no-master/--no-audio) skips the SonicMaster stack.
    """
    from voyage import model_registry

    required = required_specs(
        config, sfx_enabled, models_root, augment_enabled, music_enabled, mastering_enabled
    )
    if not required:
        return 0
    checked = [
        (entry, model_registry.verify_model(entry.models_dir, entry.spec)) for entry in required
    ]
    missing = [(entry, message) for entry, (ok, message) in checked if not ok]
    if not missing:
        # Files present but the fail-closed workers still need a recorded
        # sha baseline (asporgue 2026-10-07: `verify_model` passed on the
        # ltx25 files while the worker refused `no manifest entry`). Repair
        # incomplete entries without any download — files already match the
        # checklist, so only the manifest record is missing/stale.
        incomplete = [
            entry
            for entry, (_ok, _message) in checked
            if not _manifest_entry_complete(entry.models_dir, entry.spec)
        ]
        if not incomplete:
            return 0
        unrepaired = _repair_manifest(incomplete)
        if unrepaired:
            for entry in unrepaired:
                mismatched = _expected_hash_mismatches(entry.models_dir, entry.spec)[:3]
                console.error(
                    f"model {entry.spec} files present but hash baseline missing "
                    f"or mismatched ({mismatched}) — re-run to download"
                )
            return 1
        repaired = ", ".join(entry.spec for entry in incomplete)
        console.ok(f"models ready: {repaired} (manifest repaired)")
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

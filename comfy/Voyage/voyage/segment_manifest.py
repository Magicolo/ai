"""Per-segment manifest (DESIGN §§56, 70, §31 atomic rule).

One `manifest.json` per committed segment replaces the six legacy JSONs
(`transition.json`, `prompt_plan.json`, `audio_state.json`,
`world_state.json`, `metrics.json`, `sha256.json`). Top-level keys are
`format` (int 1), `transition`, `prompt_plan`, `audio_state`,
`world_state`, `metrics`, and `checksums` (relative filename -> sha256
for `video.mp4`, `audio.wav`, and `recovery.pt` when present).

Tradeoff (no self-hash): `checksums` covers the three binary artifacts
only. The old `sha256.json` also hashed the five metadata JSONs, which
required a self-hash dance (the manifest cannot hash itself without a
second write). Metadata integrity now rides on the single atomic
`manifest.json` write (temp + fsync + rename + fsync_dir): a torn write
leaves no manifest at all, and a committed manifest is whole. Readers
verify binaries from `checksums`; metadata is trusted once the manifest
parses and carries `format == 1`.

Legacy fallback: `load_segment_manifest` reads `manifest.json` when
present and otherwise assembles the same normalized dict from the
individual files, so old runs (e.g. pre-prune checkouts) still
validate, scoreboard, and resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import atomic_write_json
from voyage.errors import MediaError
from voyage.hashing import sha256_file

SEGMENT_MANIFEST_FORMAT = 1
"""Version of the per-segment manifest schema (int, top-level `format`)."""

REQUIRED_CHECKSUM_ARTIFACTS = ("video.mp4", "audio.wav")
"""Binary artifacts every manifest must checksum (DESIGN §56 step 4)."""

LEGACY_METADATA_FILES = (
    "transition.json",
    "prompt_plan.json",
    "audio_state.json",
    "world_state.json",
    "metrics.json",
)
"""Legacy per-section files the fallback reader assembles (old layout)."""

LEGACY_CHECKSUMS_FILENAME = "sha256.json"
"""Legacy checksum file the fallback reader uses for `checksums`."""


def segment_manifest_path(segment_dir: Path) -> Path:
    """Absolute path of the per-segment manifest file."""
    return segment_dir / paths.SEGMENT_MANIFEST_FILENAME


def build_segment_manifest(
    transition: dict[str, Any],
    prompt_plan: dict[str, Any],
    audio_state: dict[str, Any],
    world_state: dict[str, Any],
    metrics: dict[str, Any],
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Assemble a manifest payload (pure; caller writes it atomically)."""
    return {
        "format": SEGMENT_MANIFEST_FORMAT,
        "transition": transition,
        "prompt_plan": prompt_plan,
        "audio_state": audio_state,
        "world_state": world_state,
        "metrics": metrics,
        "checksums": checksums,
    }


def write_segment_manifest(segment_dir: Path, manifest: dict[str, Any]) -> Path:
    """Atomically write `manifest.json` into `segment_dir` (DESIGN §31)."""
    dest = segment_manifest_path(segment_dir)
    payload = dict(manifest)
    payload["format"] = SEGMENT_MANIFEST_FORMAT
    atomic_write_json(dest, payload)
    return dest


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Best-effort JSON object read; None on missing/unreadable/non-object."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return raw if isinstance(raw, dict) else None


def load_segment_manifest(segment_dir: Path) -> dict[str, Any]:
    """Load the normalized manifest dict for `segment_dir`.

    Returns `{"transition": ..., "prompt_plan": ..., "audio_state": ...,
    "world_state": ..., "metrics": ..., "checksums": ...}` where each
    section is a dict (empty when absent in the legacy fallback).

    Reads `manifest.json` when present; otherwise falls back to the
    legacy individual files (`transition.json` etc. plus `sha256.json`
    for `checksums`). A present-but-unreadable/malformed manifest raises
    `MediaError` (torn atomic write or hand-edit corruption must fail
    loud); missing legacy files degrade to empty sections so old partial
    runs still report precisely which section is missing downstream.
    """
    manifest_file = segment_manifest_path(segment_dir)
    if manifest_file.exists():
        try:
            raw: Any = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as exc:
            raise MediaError(
                f"segment {segment_dir.name} has unreadable manifest.json: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise MediaError(f"segment {segment_dir.name} has malformed manifest.json")
        if raw.get("format") != SEGMENT_MANIFEST_FORMAT:
            raise MediaError(
                f"segment {segment_dir.name} has unsupported manifest format {raw.get('format')!r}"
            )
        normalized: dict[str, Any] = {}
        for key in ("transition", "prompt_plan", "audio_state", "world_state", "metrics"):
            section = raw.get(key)
            normalized[key] = dict(section) if isinstance(section, dict) else {}
        checksums = raw.get("checksums")
        normalized["checksums"] = dict(checksums) if isinstance(checksums, dict) else {}
        return normalized
    sections: dict[str, Any] = {}
    legacy_names = {
        "transition": "transition.json",
        "prompt_plan": "prompt_plan.json",
        "audio_state": "audio_state.json",
        "world_state": "world_state.json",
        "metrics": "metrics.json",
    }
    for key, filename in legacy_names.items():
        loaded = _read_json_object(segment_dir / filename)
        sections[key] = loaded if loaded is not None else {}
    checksums_raw = _read_json_object(segment_dir / LEGACY_CHECKSUMS_FILENAME)
    sections["checksums"] = checksums_raw if checksums_raw is not None else {}
    return sections


def load_manifest_section(segment_dir: Path, key: str) -> dict[str, Any]:
    """Best-effort single section (`transition`, `metrics`, ...); {} on failure."""
    try:
        manifest = load_segment_manifest(segment_dir)
    except (MediaError, OSError, ValueError):
        return {}
    section = manifest.get(key)
    return dict(section) if isinstance(section, dict) else {}


def load_transition(segment_dir: Path) -> dict[str, Any]:
    """Best-effort transition section; {} when missing/torn/legacy-absent."""
    return load_manifest_section(segment_dir, "transition")


def load_metrics(segment_dir: Path) -> dict[str, Any]:
    """Best-effort metrics section; {} when missing/torn/legacy-absent."""
    return load_manifest_section(segment_dir, "metrics")


def load_audio_state(segment_dir: Path) -> dict[str, Any]:
    """Best-effort audio_state section; {} when missing/torn/legacy-absent."""
    return load_manifest_section(segment_dir, "audio_state")


def load_world_state(segment_dir: Path) -> dict[str, Any]:
    """Best-effort world_state section; {} when missing/torn/legacy-absent."""
    return load_manifest_section(segment_dir, "world_state")


def load_checksums(segment_dir: Path) -> dict[str, Any]:
    """Best-effort checksums section; {} when missing/torn/legacy-absent."""
    return load_manifest_section(segment_dir, "checksums")


def update_manifest_metrics(segment_dir: Path, metrics: dict[str, Any]) -> None:
    """Rewrite the manifest's metrics section atomically (inspect merge).

    Manifest path: reload the present manifest, replace `metrics`, and
    atomically rewrite (checksums cover binaries only, so they are
    untouched). Legacy path (no `manifest.json`): rewrite `metrics.json`
    atomically and refresh its `sha256.json` entry when recorded, so
    validate never false-positives after an inspect merge. Raises
    `MediaError` when no metrics source is writable.
    """
    manifest_file = segment_manifest_path(segment_dir)
    if manifest_file.exists():
        current = load_segment_manifest(segment_dir)
        current["metrics"] = dict(metrics)
        atomic_write_json(manifest_file, {"format": SEGMENT_MANIFEST_FORMAT, **current})
        return
    atomic_write_json(segment_dir / "metrics.json", metrics)
    legacy_checksums = segment_dir / LEGACY_CHECKSUMS_FILENAME
    recorded = _read_json_object(legacy_checksums)
    if recorded is not None and "metrics.json" in recorded:
        recorded["metrics.json"] = sha256_file(segment_dir / "metrics.json")
        atomic_write_json(legacy_checksums, recorded)

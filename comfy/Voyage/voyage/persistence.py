"""Run manifest + run state persistence (DESIGN §§32-33, task group C).

All writes are atomic (DESIGN §31). state.json is owned by the
supervisor only — the director proposes, the supervisor commits.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import voyage
from voyage import paths
from voyage.atomic import atomic_write_json, read_json
from voyage.config import ProjectConfig
from voyage.errors import StateError
from voyage.models import RunState

#: Timeline geometry recorded in the run manifest. Informational only —
#: no reader scales media from it (the supervisor's frame counts are the
#: timeline truth); kept as named constants so a future geometry change
#: updates the manifest in exactly one place. This is the *source* hint,
#: not the shipped presentation: finalize lifts backend-native segments to
#: the `[augment]` floors by default, so `presentation` (floors, recorded
#: at init) + `final_geometry` (the validated output box, filled by the
#: first finalize — issue 141) carry the shipping contract instead.
FINAL_VIDEO_WIDTH = 768
FINAL_VIDEO_HEIGHT = 432


def effective_config_digest(config: ProjectConfig) -> str:
    """Traceability digest over the effective config (clean-break rule).

    CLI-is-config: the digest covers the canonical JSON of the in-memory
    effective config, not file bytes — there is no TOML file to hash.
    `sort_keys` + compact separators keep it stable across writers.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    config: ProjectConfig,
    config_sha256: str,
    hardware: dict[str, str],
    software: dict[str, str],
    argv: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": paths.SCHEMA_VERSION,
        "run_id": config.run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 — worker image is py3.10, datetime.UTC needs 3.11+
        "voyage_version": voyage.__version__,
        "config_sha256": config_sha256,
        "style": config.style,
        "argv": list(argv) if argv is not None else [],
        "effective_config": config.model_dump(mode="json"),
        "hardware": hardware,
        "software": software,
        "models": {
            "video": {"backend": config.video.backend, "profile": config.video.profile},
            "audio": {"backend": config.audio.backend},
            "director": {
                "backend": config.director.backend,
                "model_id": config.director.model_id,
                "device": config.director.device,
            },
            "embedding": {"backend": "token-set-fallback"},
        },
        "seed": config.seed,
        "timeline": {
            "fps": config.video.fps,
            "final_width": FINAL_VIDEO_WIDTH,
            "final_height": FINAL_VIDEO_HEIGHT,
        },
        "presentation": {
            "min_fps": config.augment.min_fps,
            "min_width": config.augment.min_width,
            "min_height": config.augment.min_height,
        },
        "final_geometry": None,
        "committed_segments": 0,
    }


def write_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    atomic_write_json(run_dir / paths.MANIFEST_FILENAME, manifest)


def read_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / paths.MANIFEST_FILENAME
    if not path.exists():
        raise StateError(f"missing {paths.MANIFEST_FILENAME} in {run_dir}")
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        # Torn manifest (SIGKILL mid-write) or hand-edit corruption must
        # read as StateError (issue 002) — never a bare JSONDecodeError
        # that escapes the commit boundary and strands the run at RUNNING.
        raise StateError(f"invalid {paths.MANIFEST_FILENAME} in {run_dir}: {exc}") from exc
    if not isinstance(data, dict):
        raise StateError(f"{paths.MANIFEST_FILENAME} is not a JSON object")
    return data


def read_effective_config(run_dir: Path) -> tuple[ProjectConfig, str]:
    """Load the run's effective config from its manifest (clean break).

    No TOML fallback: runs created before the CLI-is-config migration
    carry no `effective_config` and fail loud with the re-generate hint.
    Returns (config, digest) mirroring the old file-loader shape; the
    digest is recomputed over the stored config, never trusted blind.
    """
    manifest = read_manifest(run_dir)
    raw = manifest.get("effective_config")
    if not isinstance(raw, dict):
        raise StateError(
            f"{paths.MANIFEST_FILENAME} in {run_dir} carries no effective config "
            "(run created before CLI-is-config — re-generate; no legacy format is read)"
        )
    try:
        config = ProjectConfig.model_validate(raw)
    except Exception as exc:
        raise StateError(
            f"invalid effective config in {paths.MANIFEST_FILENAME} ({run_dir}): {exc}"
        ) from exc
    return config, effective_config_digest(config)


def write_state(run_dir: Path, state: RunState) -> None:
    atomic_write_json(run_dir / paths.STATE_FILENAME, state.model_dump())


def read_state(run_dir: Path) -> RunState:
    path = run_dir / paths.STATE_FILENAME
    if not path.exists():
        raise StateError(f"missing {paths.STATE_FILENAME} in {run_dir}")
    try:
        return RunState.model_validate(read_json(path))
    except (OSError, ValueError) as exc:
        # Same taxonomy as read_manifest above (issue 002): unreadable
        # files (OSError) and torn/hand-edited JSON or schema violations
        # (ValueError covers JSONDecodeError, UnicodeDecodeError, and
        # pydantic ValidationError) read as StateError. Anything else
        # (MemoryError and friends) propagates raw — masking resource
        # exhaustion as corrupt state would send recovery down the
        # wrong path.
        raise StateError(f"invalid state file {path}: {exc}") from exc


def initial_state(config: ProjectConfig) -> RunState:
    return RunState(
        run_id=config.run_id,
        status="CREATED",
        fps=config.video.fps,
        current_concept=config.style,
        destination_concept=config.style,
    )


def create_run_dir(run_dir: Path, config: ProjectConfig, argv: list[str] | None = None) -> str:
    """Scaffold a fresh run directory (sole creator: `generate` + tests).

    CLI-is-config: mkdirs segments/logs, then writes the manifest
    carrying the full effective config plus the invoking argv, and the
    initial state. Returns the traceability digest. The caller owns all
    validation (non-empty guard, style, seed, backend) before calling —
    this function only writes.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    digest = effective_config_digest(config)
    hardware = {"note": "recorded at creation; see `voyage doctor` for live facts"}
    software = {"python": sys.version.split()[0]}
    write_manifest(run_dir, build_manifest(config, digest, hardware, software, argv=argv))
    write_state(run_dir, initial_state(config))
    return digest

"""Run manifest + run state persistence (DESIGN §§32-33, task group C).

All writes are atomic (DESIGN §31). state.json is owned by the
supervisor only — the director proposes, the supervisor commits.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from voyage import paths
from voyage.atomic import atomic_write_json, read_json
from voyage.config import ProjectConfig
from voyage.errors import StateError
from voyage.models import RunState


def build_manifest(
    config: ProjectConfig,
    config_sha256: str,
    hardware: dict[str, str],
    software: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": paths.SCHEMA_VERSION,
        "run_id": config.run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 — worker image is py3.10, datetime.UTC needs 3.11+
        "voyage_version": __import__("voyage").__version__,
        "config_sha256": config_sha256,
        "style": config.style,
        "hardware": hardware,
        "software": software,
        "models": {
            "video": {"backend": config.video.backend, "profile": config.video.profile},
            "audio": {"backend": config.audio.backend},
            "director": {
                "backend": config.director.backend,
                "model_id": config.director.model_id,
            },
            "embedding": {"backend": "token-set-fallback"},
        },
        "seed": config.seed,
        "timeline": {
            "fps": config.video.fps,
            "final_width": 768,
            "final_height": 432,
        },
        "committed_segments": 0,
    }


def write_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    atomic_write_json(run_dir / paths.MANIFEST_FILENAME, manifest)


def read_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / paths.MANIFEST_FILENAME
    if not path.exists():
        raise StateError(f"missing {paths.MANIFEST_FILENAME} in {run_dir}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise StateError(f"{paths.MANIFEST_FILENAME} is not a JSON object")
    return data


def write_state(run_dir: Path, state: RunState) -> None:
    atomic_write_json(run_dir / paths.STATE_FILENAME, state.model_dump())


def read_state(run_dir: Path) -> RunState:
    path = run_dir / paths.STATE_FILENAME
    if not path.exists():
        raise StateError(f"missing {paths.STATE_FILENAME} in {run_dir}")
    try:
        return RunState.model_validate(read_json(path))
    except Exception as exc:
        raise StateError(f"invalid state file {path}: {exc}") from exc


def initial_state(config: ProjectConfig) -> RunState:
    return RunState(
        run_id=config.run_id,
        status="CREATED",
        fps=config.video.fps,
        current_concept=config.style,
        destination_concept=config.style,
    )

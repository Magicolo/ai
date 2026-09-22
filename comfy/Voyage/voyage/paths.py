"""Run-directory layout (DESIGN §21 persistent files, §29 segment dir).

run/
  voyage.toml
  run_manifest.json
  state.json
  concepts.jsonl
  segments/
  logs/
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 1

CONFIG_FILENAME = "voyage.toml"
MANIFEST_FILENAME = "run_manifest.json"
STATE_FILENAME = "state.json"
CONCEPTS_FILENAME = "concepts.jsonl"
SEGMENTS_DIRNAME = "segments"
LOGS_DIRNAME = "logs"
DONE_MARKER = "DONE"


def segment_dir(run_dir: Path, segment_id: str) -> Path:
    return run_dir / SEGMENTS_DIRNAME / segment_id


def format_segment_id(number: int) -> str:
    return f"{number:06d}"

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
    """Format a segment number as a zero-padded six-digit id.

    Why the range check: negative or >999999 inputs break the %06d
    lexicographic/contiguous invariant validate_run relies on (issue 091)
    — -1 previously rendered as '-00001' and huge ids grew past six
    digits into directory names and ordering checks.
    """
    if number < 0 or number > 999999:
        raise ValueError(f"segment number must be within [0, 999999] (got {number})")
    return f"{number:06d}"

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

# Layout dirnames usable as re-anchor points for legacy absolute entries
# (issue 016): a moved run's stale absolute path still names the layout
# position (`audio/take_0000.wav`), so consumers can find it again.
_LAYOUT_ANCHORS = frozenset({"segments", "audio", "novelty", "logs"})


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


def resolve_stored_path(run_dir: Path, stored: str | Path) -> Path:
    """Resolve a persisted run artifact path (issue 016 consumer side).

    Stored form is run-relative POSIX (e.g. `audio/take_0000.wav`,
    `segments/000000/recovery.pt`); legacy entries are absolute. An
    existing path is used as-is (so a non-relocated absolute entry
    keeps working); relative entries resolve against `run_dir`, so
    `cp -r`/`mv` of a run keeps every consumer working. A missing
    absolute entry is re-anchored on the first known layout dirname
    (`segments/`, `audio/`, `novelty/`, `logs/`) when that target
    exists — this heals pre-fix runs after a move; otherwise the
    stale path is returned so the caller still fails loudly.
    """
    candidate = Path(stored)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        return run_dir / candidate
    parts = candidate.parts
    for index, part in enumerate(parts):
        if part in _LAYOUT_ANCHORS:
            reanchored = run_dir.joinpath(*parts[index:])
            if reanchored.exists():
                return reanchored
    return candidate

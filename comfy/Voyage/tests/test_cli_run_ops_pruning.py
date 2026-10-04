"""Run-file pruning: `create_run_dir` no longer creates the root legacy concepts dup (DESIGN §21).

The live concept store lives at `novelty/concepts.jsonl`; the 0-byte root
`concepts.jsonl` the old `init` used to write was a legacy duplicate that only
fed the deprecated `ConceptStore(legacy_path=...)` migration path. New runs
start without it — every legacy reader already tolerates a missing file.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import DEFAULT_RUN_SEED, DEFAULT_STYLE
from voyage import paths
from voyage.config import preset_config
from voyage.persistence import create_run_dir


def test_create_run_dir_creates_no_root_concepts_dup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_run_dir(run_dir, preset_config("run", DEFAULT_STYLE, DEFAULT_RUN_SEED))
    assert not (run_dir / paths.CONCEPTS_FILENAME).exists()


def test_create_run_dir_still_scaffolds_the_live_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_run_dir(run_dir, preset_config("run", DEFAULT_STYLE, DEFAULT_RUN_SEED))
    assert (run_dir / paths.MANIFEST_FILENAME).is_file()
    assert (run_dir / paths.SEGMENTS_DIRNAME).is_dir()
    assert (run_dir / paths.LOGS_DIRNAME).is_dir()
    assert (run_dir / "state.json").is_file()

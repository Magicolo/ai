"""Run-file pruning: `init` no longer creates the root legacy concepts dup (DESIGN §21).

The live concept store lives at `novelty/concepts.jsonl`; the 0-byte root
`concepts.jsonl` `cmd_init` used to write was a legacy duplicate that only
fed the deprecated `ConceptStore(legacy_path=...)` migration path. New runs
start without it — every legacy reader already tolerates a missing file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voyage import paths
from voyage.cli import _add_init_parser
from voyage.cli_run_ops import cmd_init


def _init_args(output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    _add_init_parser(sub)
    return parser.parse_args(
        ["init", "--output", str(output), "--style", "pastel neon line-art, peaceful"]
    )


def test_cmd_init_creates_no_root_concepts_dup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert cmd_init(_init_args(run_dir)) == 0
    assert not (run_dir / paths.CONCEPTS_FILENAME).exists()


def test_cmd_init_still_scaffolds_the_live_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert cmd_init(_init_args(run_dir)) == 0
    assert (run_dir / paths.CONFIG_FILENAME).is_file()
    assert (run_dir / paths.SEGMENTS_DIRNAME).is_dir()
    assert (run_dir / paths.LOGS_DIRNAME).is_dir()
    assert (run_dir / "state.json").is_file()
    assert (run_dir / paths.MANIFEST_FILENAME).is_file()

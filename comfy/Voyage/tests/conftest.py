"""Shared pytest fixtures: canonical run-directory scaffold (DESIGN §21).

Why this file exists: over a dozen test modules each carry a private
`_init_run` copy (mkdir segments/logs, write voyage.toml, manifest,
state, empty concepts). The bodies drifted apart cosmetically
(top-level versus function-level persistence imports, seed 7 versus
11) while staying semantically identical, so a layout change would
need thirteen matching edits. New tests should use
`initialize_run_directory` (or the `run_directory_factory` fixture);
the legacy `_init_run` copies are being converted incrementally —
two converted as proof (test_console, test_av_alignment_consumer) —
and stay valid because the helper keeps their exact semantics.

The property-test module needs Hypothesis, which is not in the gate
image's dev extras yet (see the ALL-gap report): it guards itself with
`pytest.importorskip`, so it reports a skip there and runs once
Hypothesis is installed — no conftest hook needed (per-file ignore
hooks do not fire in this package-layout suite).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from voyage import paths
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    write_manifest,
    write_state,
)

# Every legacy `_init_run` copy uses this exact style prompt; the seed
# default 11 is the modal value across the copies (integration-style
# tests pass 7 explicitly, so nothing is silently renumbered).
DEFAULT_STYLE = "pastel neon line-art, peaceful"
DEFAULT_RUN_ID = "shared"
DEFAULT_RUN_SEED = 11


def initialize_run_directory(
    run_dir: Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    style: str = DEFAULT_STYLE,
    seed: int = DEFAULT_RUN_SEED,
    visual_inspector: bool = False,
) -> None:
    """Create a minimal valid run directory: config, manifest, state.

    Mirrors the legacy `_init_run` bodies exactly (same files, same
    order, same empty concepts store) so converted call sites keep
    passing unchanged. The `visual_inspector` flag applies the same
    TOML string replacement the inspector test modules use.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / paths.SEGMENTS_DIRNAME).mkdir(exist_ok=True)
    (run_dir / paths.LOGS_DIRNAME).mkdir(exist_ok=True)
    toml_text = default_config_toml(run_id, style, seed)
    if visual_inspector:
        toml_text = toml_text.replace("visual_inspector = false", "visual_inspector = true")
    (run_dir / paths.CONFIG_FILENAME).write_text(toml_text, encoding="utf-8")
    config, digest = load_config(run_dir / paths.CONFIG_FILENAME)
    write_manifest(run_dir, build_manifest(config, digest, {}, {}))
    write_state(run_dir, initial_state(config))
    (run_dir / paths.CONCEPTS_FILENAME).write_text("", encoding="utf-8")


@pytest.fixture
def run_directory_factory(tmp_path: Path) -> Callable[..., Path]:
    """Build a fresh initialized run directory per call (never shared).

    Each call scaffolds under the test's own `tmp_path`, so parallel
    or repeated runs can never observe each other's segments or state.
    """

    def _create_run_directory(
        run_id: str = DEFAULT_RUN_ID,
        style: str = DEFAULT_STYLE,
        seed: int = DEFAULT_RUN_SEED,
    ) -> Path:
        run_dir = tmp_path / run_id
        initialize_run_directory(run_dir, run_id=run_id, style=style, seed=seed)
        return run_dir

    return _create_run_directory

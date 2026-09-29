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

Hypothesis runs with a container-tuned profile below (issue 039): no
example database, so failing examples are reported verbosely but never
replayed from — or written to — the bind-mounted tree, which is the
correct trade-off for ephemeral container runs. The whole block is
guarded by `find_spec` (not a bare import): images that have not picked
up the pinned dev extra yet still collect and run the suite, with the
property modules skipping themselves via `pytest.importorskip` before
they reach the shared strategies.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

from voyage import paths
from voyage.config import default_config_toml, load_config
from voyage.persistence import (
    build_manifest,
    initial_state,
    write_manifest,
    write_state,
)

if importlib.util.find_spec("hypothesis") is not None:
    from hypothesis import settings
    from hypothesis import strategies as hypothesis_strategies

    settings.register_profile("container", database=None)
    settings.load_profile("container")

    # Shared domain-constrained generators (issue 039): property modules draw
    # from these instead of inventing overlapping alphabets. `st.data()`
    # draws keep `@given` signatures narrow; deterministic pins for the
    # singular cases generators hit only probabilistically (nan/inf/empty)
    # live in the property modules themselves, next to the properties they
    # anchor.
    bounded_counts = hypothesis_strategies.integers(min_value=0, max_value=999999)
    """Small non-negative counts — seeds derive from run/segment/block numbers."""

    # Verified live: `characters()` with only `blacklist_characters` still
    # draws lone surrogates (probed `\ud800`), so the surrogate category
    # stays blacklisted explicitly alongside NUL. The tuple needs the Literal
    # type — hypothesis types the parameter as a collection of category
    # literals.
    surrogate_category: tuple[Literal["Cs"], ...] = ("Cs",)
    short_texts = hypothesis_strategies.text(
        alphabet=hypothesis_strategies.characters(
            blacklist_categories=surrogate_category, blacklist_characters="\x00"
        ),
        max_size=24,
    )
    """Short NUL-free texts — safe for JSON round-trips and tokenizers."""

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

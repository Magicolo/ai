"""Shared pytest fixtures: canonical run-directory scaffold (DESIGN §21).

Why this file exists: over a dozen test modules each carry a private
`_init_run` copy (mkdir segments/logs, write manifest,
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
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

from voyage.config import VideoBackendName, preset_config
from voyage.persistence import create_run_dir

if importlib.util.find_spec("hypothesis") is not None:
    from hypothesis import settings
    from hypothesis import strategies as hypothesis_strategies

    # Example-database policy (issue 039): `None` keeps failing examples
    # out of the bind-mounted tree (correct for ephemeral container runs —
    # failures are reported verbosely, just never replayed). Set
    # `VOYAGE_HYPOTHESIS_DATABASE=1` for a local replay run; the default
    # database then persists under `.hypothesis/` (gitignored, never baked
    # into images — see `.dockerignore`).
    if os.environ.get("VOYAGE_HYPOTHESIS_DATABASE"):
        settings.register_profile("container")
    else:
        settings.register_profile("container", database=None)
    settings.load_profile("container")

    # Health-check / deadline policy (issue 039): no property module
    # suppresses a health check today (verified: zero
    # `suppress_health_check` hits) — suppress narrowly as encountered,
    # never suite-wide. No module sets a custom `deadline` either: the
    # properties cover CPU-only pure cores, so the default deadline never
    # fires. Any future ffmpeg/subprocess-adjacent property must set an
    # explicit `@settings(deadline=...)` with a reason, not rely on this
    # profile.

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
    video_backend: VideoBackendName = "fake",
) -> None:
    """Create a minimal valid run directory: manifest + state.

    Mirrors the legacy `_init_run` bodies exactly (same files, same
    order) so converted call sites keep passing unchanged — except the
    config now resolves CLI-style (preset + overrides, no TOML file).
    No root concepts file is scaffolded (2026-09-30 pruning: new runs
    start without the legacy dup; readers tolerate its absence).
    `video_backend` pins the CPU fake pipeline
    (2026-09-29 ltxv decision: product defaults are ltxv/CUDA, but the
    suite runs CPU-only on fake workers — no GPU, no model weights).
    """
    config = preset_config(run_id, style, seed, video_backend=video_backend)
    create_run_dir(run_dir, config)


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


@pytest.fixture(autouse=True)
def _never_spawn_llama_sidecar(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never spawn the real llama-server sidecar in the suite.

    The default director backend is llama, but the slim test image
    carries no server binary and no GGUF weights — spawning would fail
    loud (correct production behavior, wrong for unit tests). Suite
    tests exercise the supervisor/worker contract, not process spawn:
    the sidecar stays unstarted (`None` handle; `stop(None)` is a safe
    no-op) and decide payloads degrade through the standard worker
    fallback path, exactly like the AWQ path without torch.
    `test_llama_sidecar.py` owns sidecar behavior (real `start` with
    stubbed Popen) and is exempt: it monkeypatches `llama_server.start`
    itself.
    """
    if getattr(request.node, "module", None) is not None and request.node.module.__name__ == (
        "tests.test_llama_sidecar"
    ):
        return
    from voyage import llama_server

    monkeypatch.setattr(llama_server, "start", lambda *args, **kwargs: None)

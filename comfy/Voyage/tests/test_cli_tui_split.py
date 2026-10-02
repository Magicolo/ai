"""Parser-helper + TUI-seam tests (issue 020).

`build_parser` is now an assembler over per-verb helpers: every verb must
still parse with its historical defaults, and the `run`/`generate` override
flags must stay identical (one shared helper). The TUI form/run halves must
expose the same widget ids through Pilot as `compose` mounts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from voyage import cli


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep remembered TUI settings out of the real home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))


def _parse(verb_args: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(verb_args)


def test_init_parser_defaults() -> None:
    args = _parse(["init", "--output", "out", "--style", "calm"])
    assert args.func is cli.cmd_init
    assert args.backend == "ltxv"
    assert args.seed == 0


def test_doctor_parser() -> None:
    assert _parse(["doctor"]).func is cli.cmd_doctor


def test_models_parser_defaults() -> None:
    args = _parse(["models", "list"])
    assert args.func is cli.cmd_models
    assert args.models_target == "ltxv-2b"


def test_run_parser_defaults() -> None:
    args = _parse(["run", "--run", "rundir"])
    assert args.func is cli.cmd_run
    assert args.segments is None
    assert args.director is None


def test_generate_parser_defaults() -> None:
    args = _parse(["generate", "--duration", "5s", "--style", "calm"])
    assert args.func is cli.cmd_generate
    assert args.backend == "ltxv"
    assert args.duration == 5.0
    assert args.director == "llama"


def test_status_pause_resume_stop_validate_parsers() -> None:
    assert _parse(["status", "--run", "r"]).func is cli.cmd_status
    assert _parse(["pause", "--run", "r"]).func is cli.cmd_pause
    assert _parse(["resume", "--run", "r"]).func is cli.cmd_resume
    assert _parse(["stop", "--run", "r"]).func is cli.cmd_stop
    assert _parse(["validate", "--run", "r"]).func is cli.cmd_validate


def test_finalize_benchmark_soak_inspect_parsers() -> None:
    args = _parse(["finalize", "--run", "r", "--output", "final.mp4"])
    assert args.func is cli.cmd_finalize
    assert _parse(["benchmark", "video", "--run", "r"]).func is cli.cmd_benchmark
    assert _parse(["soak", "--run", "r", "--segments", "2"]).func is cli.cmd_soak
    assert _parse(["inspect", "scoreboard", "--run", "r"]).func is cli.cmd_inspect


def test_run_and_generate_share_override_flags() -> None:
    """The shared override helper keeps both verbs in lockstep (020)."""
    flags = [
        "--blocks",
        "3",
        "--take-seconds",
        "45",
        "--quantization",
        "bf16",
        "--beats-per-segment",
        "8",
        "--drift-every-n",
        "2",
    ]
    run_args = _parse(["run", "--run", "r", *flags])
    gen_args = _parse(["generate", "--duration", "5s", "--style", "calm", *flags])
    for field in ("blocks", "take_seconds", "quantization", "beats_per_segment", "drift_every_n"):
        assert getattr(run_args, field) == getattr(gen_args, field)


def _require_app() -> Any:
    """Import the Textual app (skip when the display extra is missing)."""
    pytest.importorskip("textual")
    from voyage.tui import VoyageApp

    return VoyageApp


_FORM_IDS = (
    "field-style",
    "field-name",
    "field-duration",
    "field-backend",
    "gpu-warning",
    "field-director",
    "field-quantization",
    "field-blocks",
    "field-take-seconds",
    "field-beats",
    "field-drift",
    "field-min-fps",
    "field-min-resolution",
    "field-seed",
    "flag-draft",
    "flag-force",
    "flag-skip-bad",
    "flag-no-download",
    "flag-no-sfx",
    "flag-use-model-pass",
    "flag-verbose",
    "flag-no-color",
    "plan-line",
    "errors-line",
    "button-row",
    "button-generate",
    "button-quit",
    "key-hints",
)

_RUN_IDS = (
    "run-head",
    "run-bar",
    "run-log",
    "run-result",
    "run-buttons",
    "button-stop",
    "button-back",
    "button-quit-run",
    "run-keys",
)


def test_form_fields_expose_expected_ids() -> None:
    """`compose` mounts the form seam's widgets (020, Pilot-pinned).

    The seam generators keep `with`-container compose semantics (an
    explicit-children rewrite mounts a click-dead copy — verified live),
    so the pin is the mounted `#form-col` region, not a standalone call.
    """
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.containers import Vertical

        app = VoyageApp()
        async with app.run_test(size=(120, 40)):
            await asyncio.sleep(0)
            form_col = app.query_one("#form-col", Vertical)
            # 13 field rows + gpu-warning + 8 flags + plan/errors lines +
            # button row + key hints: the seam owns exactly this region.
            assert len(form_col.children) == 26
            for expected in _FORM_IDS:
                app.query_one(f"#{expected}")

    asyncio.run(_run())


def test_run_view_widgets_expose_expected_ids() -> None:
    """`compose` mounts the run seam's widgets in order (020, Pilot-pinned)."""
    import asyncio

    VoyageApp = _require_app()

    async def _run() -> None:
        from textual.containers import Horizontal, Vertical

        app = VoyageApp()
        async with app.run_test(size=(120, 40)):
            await asyncio.sleep(0)
            run_view = app.query_one("#run-view", Vertical)
            assert [child.id for child in run_view.children] == [
                "run-head",
                "run-bar",
                "run-log",
                "run-result",
                "run-buttons",
                "run-keys",
            ]
            buttons = app.query_one("#run-buttons", Horizontal)
            assert [child.id for child in buttons.children] == [
                "button-stop",
                "button-back",
                "button-quit-run",
            ]
            for expected in _RUN_IDS:
                app.query_one(f"#{expected}")

    asyncio.run(_run())

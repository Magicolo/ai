"""Qwen-by-default: every generation path uses the qwen director unless
explicitly disabled (user request 2026-09-29; the poulah run froze for
31 segments on the deterministic default with zero prompt evolution).

Deterministic stays available as an explicit opt-out
(`--director deterministic` / `backend = \"deterministic\"`), never as
the silent default.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from voyage.cli import _add_generate_parser, _add_init_parser, _add_run_parser
from voyage.config import DirectorConfig, default_config_toml, load_config
from voyage.workers import director as director_worker


def test_director_config_defaults_to_qwen() -> None:
    assert DirectorConfig().backend == "qwen"


def test_default_toml_writes_qwen_director(tmp_path: Path) -> None:
    path = tmp_path / "voyage.toml"
    path.write_text(default_config_toml("qwen-default", "pastel neon", 7), encoding="utf-8")
    config, _ = load_config(path)
    assert config.director.backend == "qwen"


def test_default_toml_allows_explicit_deterministic_opt_out(tmp_path: Path) -> None:
    path = tmp_path / "voyage.toml"
    path.write_text(
        default_config_toml("qwen-default", "pastel neon", 7, director_backend="deterministic"),
        encoding="utf-8",
    )
    config, _ = load_config(path)
    assert config.director.backend == "deterministic"


def test_director_worker_defaults_to_qwen() -> None:
    assert director_worker._CONFIG["backend"] == "qwen"


def _parse(parser_adder: Callable[..., None], args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    parser_adder(sub)
    return parser.parse_args(args)


def test_init_parser_defaults_to_qwen() -> None:
    args = _parse(_add_init_parser, ["init", "--output", "out", "--style", "calm"])
    assert args.director == "qwen"


def test_generate_parser_defaults_to_qwen() -> None:
    args = _parse(
        _add_generate_parser,
        ["generate", "--duration", "2s", "--style", "calm"],
    )
    assert args.director == "qwen"


def test_run_parser_leaves_stored_config_alone() -> None:
    args = _parse(_add_run_parser, ["run", "--run", "out"])
    assert args.director is None

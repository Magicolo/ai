"""Llama-by-default: every generation path uses the llama-server sidecar
director unless explicitly disabled (A/B-proven 2026-10-01: 10.7s/directive
at 67-70 tok/s vs 23.4s AWQ).

The in-process AWQ path stays available as an explicit opt-in
(`--director qwen` / `backend = "qwen"`), deterministic stays the
LLM-off opt-out (`--director deterministic` / `backend =
"deterministic"`), never as the silent default.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from voyage.cli import _add_generate_parser, _add_run_parser
from voyage.config import DirectorConfig, preset_config
from voyage.workers import director as director_worker


def test_director_config_defaults_to_llama() -> None:
    assert DirectorConfig().backend == "llama"


def test_default_toml_writes_llama_director(tmp_path: Path) -> None:
    config = preset_config("llama-default", "pastel neon", 7)
    assert config.director.backend == "llama"


def test_default_toml_allows_explicit_qwen_opt_in(tmp_path: Path) -> None:
    config = preset_config("llama-default", "pastel neon", 7, director_backend="qwen")
    assert config.director.backend == "qwen"


def test_default_toml_allows_explicit_deterministic_opt_out(tmp_path: Path) -> None:
    config = preset_config("qwen-default", "pastel neon", 7, director_backend="deterministic")
    assert config.director.backend == "deterministic"


def test_director_worker_defaults_to_qwen() -> None:
    assert director_worker._CONFIG["backend"] == "qwen"


def _parse(parser_adder: Callable[..., None], args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    parser_adder(sub)
    return parser.parse_args(args)


def test_generate_parser_defaults_to_llama() -> None:
    args = _parse(
        _add_generate_parser,
        ["generate", "--duration", "2s", "--style", "calm"],
    )
    assert args.director == "llama"


def test_run_parser_leaves_stored_config_alone() -> None:
    args = _parse(_add_run_parser, ["run", "--run", "out"])
    assert args.director is None

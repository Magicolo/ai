"""Parser tests for the two-verb CLI (issue 020).

`build_parser` assembles `configure` (all run options) + `generate`
(NAME only): both verbs must parse with their defaults, and the
shared override/augment/sfx helpers must expose their flags on
`configure`.
"""

from __future__ import annotations

import argparse

from voyage import cli


def _parse(verb_args: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(verb_args)


def test_configure_parser_defaults() -> None:
    args = _parse(["configure", "calm"])
    assert args.func is cli.cmd_configure
    assert args.name == "calm"
    assert args.backend is None
    assert args.duration is None
    assert args.segments is None
    assert args.style is None
    assert args.seed is None
    assert args.director is None


def test_configure_parser_accepts_explicit_duration() -> None:
    args = _parse(["configure", "calm", "--duration", "5s"])
    assert args.duration == 5.0


def test_generate_parser_takes_name_only() -> None:
    args = _parse(["generate", "calm"])
    assert args.func is cli.cmd_generate
    assert args.name == "calm"


def test_configure_carries_override_flags() -> None:
    """The shared override helper exposes every flag on `configure` (020)."""
    args = _parse(
        [
            "configure",
            "calm",
            "--segments",
            "1",
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
    )
    assert args.blocks == 3
    assert args.take_seconds == 45
    assert args.quantization == "bf16"
    assert args.beats_per_segment == 8
    assert args.drift_every_n == 2

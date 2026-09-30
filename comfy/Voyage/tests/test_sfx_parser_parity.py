"""SFX flag parity across finalizing verbs (issue 092).

`finalize` owns the SFX pass; `generate` and `stop --finalize` forward
into `cmd_finalize`. Every finalizing parser must carry the same six
flags — a missing flag is an AttributeError at finalize time (the
`generate` E2E failures that motivated the shared `_add_sfx_args`
helper). CPU-only.
"""

from __future__ import annotations

import argparse

from voyage.cli import build_parser


def _sfx_defaults(args: argparse.Namespace) -> dict[str, object]:
    return {
        "no_sfx": args.no_sfx,
        "sfx_backend": args.sfx_backend,
        "sfx_caption": args.sfx_caption,
        "sfx_device": args.sfx_device,
        "sfx_model_size": args.sfx_model_size,
        "sfx_workers": args.sfx_workers,
    }


def test_all_finalizing_verbs_carry_sfx_flags() -> None:
    parser = build_parser()
    generate = parser.parse_args(["generate", "--style", "x", "--duration", "5s"])
    finalize = parser.parse_args(["finalize", "--run", "r", "--output", "o.mp4"])
    stop = parser.parse_args(["stop", "--run", "r"])
    assert _sfx_defaults(generate) == _sfx_defaults(finalize) == _sfx_defaults(stop)


def test_generate_sfx_overrides_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--style",
            "x",
            "--duration",
            "5s",
            "--no-sfx",
            "--sfx-device",
            "cuda:1",
            "--sfx-model-size",
            "small_44k",
            "--sfx-workers",
            "2",
        ]
    )
    assert _sfx_defaults(args) == {
        "no_sfx": True,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": "cuda:1",
        "sfx_model_size": "small_44k",
        "sfx_workers": 2,
    }

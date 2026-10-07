"""Dual-pan SFX pair: two seeds, ±75% constant-power pan, stereo mix (DESIGN §140).

Covers the pure dsp helpers (constant-power gains, pair filter graph),
the track separation (offset seeds, independent ledgers/stems), the
spatialization proof (opposite-polarity tracks land on opposite
channels), the legacy resume (a single-bed run renders only the right
track), and the wiring (validate both ledgers, skip-key segment,
bed fingerprint, config default + overrides).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path
from typing import Any

import pytest

from voyage.sfx_finalize import (
    SFX_DUAL_SEED_OFFSET,
    SFX_PAN_POSITION,
    SFX_RIGHT_LEDGER_NAME,
    SFX_RIGHT_STEM_SUFFIX,
    dual_pan_gains,
    is_healable_sfx_shortfall,
    plan_sfx_windows,
    sfx_pair_filter_graph,
    validate_sfx_ledger,
)


def test_dual_pan_gains_are_constant_power() -> None:
    """Equal-power law: gains square-sum to 1 at every position."""
    for pan in (-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0):
        left, right = dual_pan_gains(pan)
        assert left * left + right * right == pytest.approx(1.0)


def test_dual_pan_positions() -> None:
    """Hard channels, center, and the ±75% product positions."""
    assert dual_pan_gains(-1.0) == pytest.approx((1.0, 0.0))
    assert dual_pan_gains(1.0) == pytest.approx((0.0, 1.0))
    center = dual_pan_gains(0.0)
    assert center == pytest.approx((1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)))
    left = dual_pan_gains(-SFX_PAN_POSITION)
    right = dual_pan_gains(SFX_PAN_POSITION)
    assert left == pytest.approx((0.980785, 0.195090), abs=1e-5)
    assert right == pytest.approx((0.195090, 0.980785), abs=1e-5)
    assert left == pytest.approx((right[1], right[0]))


def test_dual_pan_gains_reject_bogus() -> None:
    from voyage.errors import MediaError

    for bogus in (-1.5, 1.5, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(MediaError):
            dual_pan_gains(bogus)


def test_seed_offset_separates_tracks() -> None:
    """The right track tiles identically with offset seeds, same captions."""
    bounds = [(0.0, 20.0, "rain on glass")]
    left = plan_sfx_windows(20.0, bounds, seed_base=7)
    right = plan_sfx_windows(20.0, bounds, seed_base=7 + SFX_DUAL_SEED_OFFSET)
    assert [w.window_id for w in left] == [w.window_id for w in right]
    assert [(w.start, w.duration) for w in left] == [(w.start, w.duration) for w in right]
    assert [w.caption for w in left] == [w.caption for w in right]
    assert [w.seed for w in right] == [w.seed + SFX_DUAL_SEED_OFFSET for w in left]


def test_pair_filter_graph_pans_downmixed_beds() -> None:
    """The graph downmixes each bed to mono, pans, and sums without gain."""
    graph = sfx_pair_filter_graph(dual_pan_gains(-0.75), dual_pan_gains(0.75))
    assert graph.count("pan=mono|c0=0.5*c0+0.5*c1") == 2
    assert "pan=stereo|c0=0.980785*c0|c1=0.195090*c0" in graph
    assert "pan=stereo|c0=0.195090*c0|c1=0.980785*c0" in graph
    assert "amix=inputs=2" in graph
    assert "normalize=0" in graph


class _PolaritySfxWorker:
    """SubprocessWorker double rendering seed-keyed DC polarity.

    Left-track seeds (below the offset) render a positive constant,
    right-track seeds a negative one — through the linear join and pan
    gains the pair bed must read positive on the left channel and
    negative on the right, proving opposite spatialization.
    """

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        del op
        type(self).calls.append(dict(payload))
        out = Path(str(payload["output_path"]))
        rate = int(payload["sample_rate"])
        channels = int(payload["channels"])
        frames = max(1, int(float(payload["duration_seconds"]) * rate))
        seed = int(payload["seed"])
        level = 10000 if seed < SFX_DUAL_SEED_OFFSET else -10000
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(
                struct.pack(f"<{frames * channels}h", *([level] * frames * channels))
            )
        return {"sfx": {"duration_seconds": float(payload["duration_seconds"])}}


def _channel_means(bed: Path) -> tuple[float, float]:
    """Mean sample value per channel (polarity probe, no numpy needed)."""
    with wave.open(str(bed), "rb") as handle:
        channels = handle.getnchannels()
        assert channels == 2
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    values = struct.unpack(f"<{frames * channels}h", raw)
    left = sum(values[0::2]) / frames
    right = sum(values[1::2]) / frames
    return (left, right)


def test_dual_render_spatializes_opposite_polarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opposite-polarity tracks land on opposite stereo channels."""
    import json

    from voyage.sfx_finalize import load_sfx_ledger, render_sfx_bed

    _PolaritySfxWorker.calls.clear()
    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _PolaritySfxWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    bed = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        [(0.0, 12.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        11,
        48000,
        2,
        1,
        dual_pan=True,
    )
    assert bed.exists()
    left_mean, right_mean = _channel_means(bed)
    assert left_mean > 1000.0
    assert right_mean < -1000.0
    left_records = load_sfx_ledger(run_dir / "audio" / "sfx" / "sfx.jsonl")
    right_records = load_sfx_ledger(run_dir / "audio" / "sfx" / SFX_RIGHT_LEDGER_NAME)
    assert len(left_records) == len(right_records) == 2
    assert [r["seed"] for r in right_records] == [
        r["seed"] + SFX_DUAL_SEED_OFFSET for r in left_records
    ]
    assert (run_dir / "audio" / "sfx" / "w0000.wav").exists()
    assert (run_dir / "audio" / "sfx" / f"w0000{SFX_RIGHT_STEM_SUFFIX}.wav").exists()
    assert validate_sfx_ledger(run_dir, 12.0, expect_right=True) == []
    events = [
        json.loads(line)
        for line in (run_dir / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    timed = [entry for entry in events if entry.get("event") == "sfx_pass_completed"]
    assert len(timed) == 1
    assert timed[0]["dual_pan"] is True
    assert timed[0]["windows"] == 2


def test_resume_completes_only_the_right_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy single-bed run resumes by rendering only the second channel."""
    from voyage.sfx_finalize import render_sfx_bed

    _PolaritySfxWorker.calls.clear()
    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _PolaritySfxWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    legacy = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        [(0.0, 12.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        11,
        48000,
        2,
        1,
    )
    assert legacy.exists()
    left_stem = run_dir / "audio" / "sfx" / "w0000.wav"
    left_bytes = left_stem.read_bytes()
    _PolaritySfxWorker.calls.clear()
    bed = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        [(0.0, 12.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        11,
        48000,
        2,
        1,
        dual_pan=True,
    )
    rendered = [str(call["output_path"]) for call in _PolaritySfxWorker.calls]
    assert rendered
    assert all(SFX_RIGHT_STEM_SUFFIX in path for path in rendered)
    assert left_stem.read_bytes() == left_bytes
    assert bed.exists()
    assert validate_sfx_ledger(run_dir, 12.0, expect_right=True) == []


def test_validate_right_shortfall_is_healable(tmp_path: Path) -> None:
    """A single-bed run under dual-pan reports exactly the healable shortfall."""
    from voyage.sfx_finalize import SfxWindow, append_sfx_window

    run_dir = tmp_path / "run"
    ledger = run_dir / "audio" / "sfx" / "sfx.jsonl"
    append_sfx_window(
        ledger,
        SfxWindow("w0000", 0.0, 8.0, "rain", 7),
        path="audio/sfx/w0000.wav",
        model_size="small_44k",
    )
    (run_dir / "audio" / "sfx" / "w0000.wav").write_bytes(b"RIFF" + b"\0" * 100)
    assert validate_sfx_ledger(run_dir, 8.0) == []
    errors = validate_sfx_ledger(run_dir, 8.0, expect_right=True)
    assert len(errors) == 1
    assert errors[0].startswith("sfx-right coverage ")
    assert is_healable_sfx_shortfall(errors[0])


def test_validate_right_gap_stays_fatal(tmp_path: Path) -> None:
    """A gapped right ledger is fatal (only the pure shortfall heals)."""
    from voyage.sfx_finalize import SfxWindow, append_sfx_window

    run_dir = tmp_path / "run"
    ledger = run_dir / "audio" / "sfx" / SFX_RIGHT_LEDGER_NAME
    append_sfx_window(
        ledger,
        SfxWindow("w0001", 7.0, 8.0, "rain", 7 + SFX_DUAL_SEED_OFFSET),
        path=f"audio/sfx/w0001{SFX_RIGHT_STEM_SUFFIX}.wav",
        model_size="small_44k",
    )
    (run_dir / "audio" / "sfx" / f"w0001{SFX_RIGHT_STEM_SUFFIX}.wav").write_bytes(
        b"RIFF" + b"\0" * 100
    )
    errors = validate_sfx_ledger(run_dir, 15.0, expect_right=True)
    assert any(error.startswith("sfx-right coverage gap:") for error in errors)
    assert not all(is_healable_sfx_shortfall(error) for error in errors)


def test_skip_key_carries_dual_pan() -> None:
    from voyage.cli_core import generate_skip_key

    base = {"skip_music": False, "skip_sfx": False, "force_upscale_1": False}
    on = generate_skip_key(
        {**base, "force_interpolate_1": False},
        manifest_no_sfx=False,
        stored_upscale=1,
        stored_interpolate=1,
        stored_sfx_dual_pan=True,
    )
    off = generate_skip_key(
        {**base, "force_interpolate_1": False},
        manifest_no_sfx=False,
        stored_upscale=1,
        stored_interpolate=1,
        stored_sfx_dual_pan=False,
    )
    assert on != off
    assert on.endswith(",dual=1")
    assert off.endswith(",dual=0")
    muted = generate_skip_key(
        {**base, "skip_sfx": True, "force_interpolate_1": False},
        manifest_no_sfx=False,
        stored_upscale=1,
        stored_interpolate=1,
        stored_sfx_dual_pan=True,
    )
    assert muted.endswith(",dual=0")


def test_bed_fingerprint_misses_across_dual_flag(tmp_path: Path) -> None:
    from voyage.final_mix_cache import bed_fingerprint

    usable: list[Path] = []
    stem_dir = tmp_path / "audio" / "sfx"
    stem_dir.mkdir(parents=True, exist_ok=True)
    (stem_dir / "w0000.wav").write_bytes(b"\x04" * 64)
    (stem_dir / "sfx.jsonl").write_text(
        json.dumps({"window_id": "w0000", "path": "audio/sfx/w0000.wav"}) + "\n",
        encoding="utf-8",
    )
    kwargs: dict[str, Any] = {
        "sample_rate": 48000,
        "channels": 2,
        "backend": "fake",
        "model_size": "small",
        "seed": 11,
        "caption_override": None,
        "music_digest": "music-abc",
    }
    assert bed_fingerprint(tmp_path, usable, dual_pan=False, **kwargs) != bed_fingerprint(
        tmp_path, usable, dual_pan=True, **kwargs
    )
    first = bed_fingerprint(tmp_path, usable, dual_pan=True, **kwargs)
    (stem_dir / SFX_RIGHT_LEDGER_NAME).write_text(
        json.dumps({"window_id": "w0001", "path": "audio/sfx/w0001_right.wav"}) + "\n",
        encoding="utf-8",
    )
    assert bed_fingerprint(tmp_path, usable, dual_pan=True, **kwargs) != first


def test_config_dual_pan_default_and_overrides() -> None:
    from voyage.cli_core import _sfx_dual_pan_overrides
    from voyage.config import SfxConfig, preset_config, resolve_config

    assert SfxConfig().dual_pan is True
    assert _sfx_dual_pan_overrides(argparse.Namespace()) == {}
    assert _sfx_dual_pan_overrides(argparse.Namespace(sfx_dual_pan=None)) == {}
    assert _sfx_dual_pan_overrides(argparse.Namespace(sfx_dual_pan=True)) == {"sfx_dual_pan": True}
    assert _sfx_dual_pan_overrides(argparse.Namespace(no_sfx_dual_pan=True)) == {
        "sfx_dual_pan": False
    }
    with pytest.raises(ValueError, match="only one"):
        _sfx_dual_pan_overrides(argparse.Namespace(sfx_dual_pan=True, no_sfx_dual_pan=True))
    base = preset_config("dualpantest", "pastel neon line-art, peaceful", 7)
    assert base.sfx.dual_pan is True
    assert resolve_config(base, sfx_dual_pan=False).sfx.dual_pan is False


def test_configure_parser_accepts_dual_pan_flags() -> None:
    from voyage.cli import build_parser

    args = build_parser().parse_args(["configure", "x", "--segments", "1", "--sfx-dual-pan"])
    assert args.sfx_dual_pan is True
    args = build_parser().parse_args(["configure", "x", "--segments", "1", "--no-sfx-dual-pan"])
    assert args.no_sfx_dual_pan is True

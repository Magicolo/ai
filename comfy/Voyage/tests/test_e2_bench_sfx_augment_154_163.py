"""Benchmark setup/aggregation shapes for the SFX + augment targets (154/163).

CPU-only pure tests over `voyage.bench` helpers. The CLI verbs that
call them (`benchmark sfx`, `benchmark augment`, soak SFX section)
live in `cli_observe.py` / `cli.py` — out of this group's scope, logged
as residuals — so these tests pin the shared shapes both future
branches must use: knob-recording setups and the soak SFX aggregation.
"""

from __future__ import annotations

from voyage.bench import (
    augment_benchmark_setup,
    sfx_benchmark_setup,
    summarize_sfx_windows,
)


def test_sfx_setup_records_the_discriminating_knobs() -> None:
    """163: the numbers must discriminate model_size × sfx_workers."""
    setup = sfx_benchmark_setup(
        model_size="small_44k", sfx_workers=2, device="cuda:1", warmup=1, measured=3
    )
    assert setup["model_size"] == "small_44k"
    assert setup["sfx_workers"] == 2
    assert setup["device"] == "cuda:1"
    assert setup["warmup"] == 1
    assert setup["measured"] == 3


def test_augment_setup_records_chunk_and_quality_knobs() -> None:
    setup = augment_benchmark_setup(
        chunk_frames=32,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
        device="cuda:0",
        warmup=1,
        measured=3,
    )
    assert setup["chunk_frames"] == 32
    assert setup["upscale_factor"] == 2
    assert setup["crf"] == 15
    assert setup["preset"] == "veryfast"
    assert setup["device"] == "cuda:0"


def test_summarize_sfx_windows_aggregates_plain_records() -> None:
    windows = [
        {"window_id": "w0", "wall_seconds": 4.0, "audio_seconds": 8.0},
        {"window_id": "w1", "wall_seconds": 6.0, "audio_seconds": 8.0},
    ]
    summary = summarize_sfx_windows(windows)
    assert summary["windows"] == 2
    assert summary["mean_wall_seconds"] == 5.0
    assert summary["total_audio_seconds"] == 16.0
    assert summary["audio_seconds_per_wall_second"] == 16.0 / 10.0


def test_summarize_sfx_windows_empty_is_zero_not_crash() -> None:
    summary = summarize_sfx_windows([])
    assert summary["windows"] == 0
    assert summary["mean_wall_seconds"] == 0.0

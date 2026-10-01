"""Benchmark count validation across all four workers (issue 060).

CPU-only: invalid counts must raise `ValueError` before any session/stack
check, so these tests need no GPU, no models, and no resident session.
"""

from __future__ import annotations

import pytest

from voyage.workers import audio_acestep, video_causvid, video_ltxv
from voyage.workers import director as director_worker
from voyage.workers.loop import validate_benchmark_counts


def test_validate_benchmark_counts_rejects_empty_measurements() -> None:
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        validate_benchmark_counts(0, 0)
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        validate_benchmark_counts(1, 0)
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        validate_benchmark_counts(-2, 1)


def test_validate_benchmark_counts_accepts_zero_warmup() -> None:
    # warmup=0 is legitimate (existing stage-shape tests rely on it).
    validate_benchmark_counts(0, 2)
    validate_benchmark_counts(1, 3)


def test_director_benchmark_rejects_zero_measured() -> None:
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        director_worker.handle_benchmark({"warmup": 0, "measured": 0})
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        director_worker.handle_benchmark({"warmup": -2, "measured": 1})


def test_audio_benchmark_validates_before_loading_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise AssertionError("stack must not load for invalid counts")

    monkeypatch.setattr(audio_acestep, "_require_stack", _boom)
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        audio_acestep.handle_benchmark({"warmup": 0, "measured": 0})


def test_ltxv_benchmark_validates_before_session_check() -> None:
    assert video_ltxv._SESSION is None
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        video_ltxv.handle_benchmark({"warmup": 1, "measured": 0})


def test_causvid_benchmark_validates_before_session_check() -> None:
    assert video_causvid._SESSION is None
    with pytest.raises(ValueError, match="warmup >= 0 and measured >= 1"):
        video_causvid.handle_benchmark({"warmup": -1, "measured": 2})

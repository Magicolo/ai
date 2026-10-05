"""SFX coverage is a per-source union; `conditioning_timeline` is provenance.

Regression for the kaolin `generate` abort: a timeline extension reuses
head stems (the hit-test ignores `conditioning_timeline` by design —
append-only proxy means head pixels are unchanged) and appends only the
new tail, so the ledger holds two timeline keys for one source. Walking
each timeline group from zero demanded the tail tile from zero and
aborted a complete bed (`w0117 starts at 819.00s, expected ~0.00s`).
The union per source must tile from zero instead.
"""

from __future__ import annotations

from pathlib import Path

from voyage.sfx_finalize import (
    SFX_CONDITIONING_PROXY,
    SFX_CONDITIONING_SHIPPED,
    SfxWindow,
    append_sfx_window,
    is_healable_sfx_shortfall,
    validate_sfx_ledger,
)


def _stem(run_dir: Path, window_id: str) -> None:
    stem = run_dir / "audio" / "sfx" / f"{window_id}.wav"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.write_bytes(b"RIFF" + b"\0" * 100)


def _ledger_append(
    run_dir: Path,
    window_id: str,
    start: float,
    duration: float,
    timeline: float,
    source: str = SFX_CONDITIONING_PROXY,
) -> None:
    ledger = run_dir / "audio" / "sfx" / "sfx.jsonl"
    append_sfx_window(
        ledger,
        SfxWindow(window_id, start, duration, "rumble", 7),
        path=f"audio/sfx/{window_id}.wav",
        model_size="small_44k",
        conditioning_source=source,
        conditioning_timeline=timeline,
    )
    _stem(run_dir, window_id)


def test_mixed_timeline_head_plus_tail_validates_clean(tmp_path: Path) -> None:
    """Kaolin shape: old-timeline head + new-timeline tail tile the union."""
    run_dir = tmp_path / "run"
    # Head rendered against the 22s proxy; the tail after extension to 29s.
    # 8s windows overlap 1s: starts 0/7/14 cover to 22, tail 21/28 to 29.
    _ledger_append(run_dir, "w0000", 0.0, 8.0, 22.0)
    _ledger_append(run_dir, "w0001", 7.0, 8.0, 22.0)
    _ledger_append(run_dir, "w0002", 14.0, 8.0, 22.0)
    _ledger_append(run_dir, "w0003", 21.0, 8.0, 29.0)
    assert validate_sfx_ledger(run_dir, 29.0) == []


def test_head_only_after_extension_is_healable_shortfall(tmp_path: Path) -> None:
    """Old bed alone against a grown timeline: shortfall, never a gap."""
    run_dir = tmp_path / "run"
    _ledger_append(run_dir, "w0000", 0.0, 8.0, 22.0)
    _ledger_append(run_dir, "w0001", 7.0, 8.0, 22.0)
    _ledger_append(run_dir, "w0002", 14.0, 8.0, 22.0)
    errors = validate_sfx_ledger(run_dir, 29.0)
    assert len(errors) == 1
    assert is_healable_sfx_shortfall(errors[0])


def test_real_gap_across_timelines_stays_fatal(tmp_path: Path) -> None:
    """A missing middle window is a gap even when timelines differ."""
    run_dir = tmp_path / "run"
    _ledger_append(run_dir, "w0000", 0.0, 8.0, 22.0)
    # w0001 (7..15) absent: w0002 starts at 14, expected ~7.
    _ledger_append(run_dir, "w0002", 14.0, 8.0, 29.0)
    errors = validate_sfx_ledger(run_dir, 29.0)
    # The gap is fatal (gate aborts); a shortfall line may coexist for the
    # uncovered tail — it is filtered, never masking the gap.
    assert any("coverage gap" in error for error in errors)


def test_tail_only_without_head_stays_fatal(tmp_path: Path) -> None:
    """The kaolin error shape with no head to union: gap, not clean."""
    run_dir = tmp_path / "run"
    _ledger_append(run_dir, "w0003", 21.0, 8.0, 29.0)
    errors = validate_sfx_ledger(run_dir, 29.0)
    assert any("coverage gap" in error for error in errors)


def test_sources_still_never_mix(tmp_path: Path) -> None:
    """Proxy head + shipped tail do not union: shipped group gaps fatally."""
    run_dir = tmp_path / "run"
    _ledger_append(run_dir, "w0000", 0.0, 8.0, 22.0, source=SFX_CONDITIONING_PROXY)
    _ledger_append(run_dir, "w0001", 7.0, 8.0, 22.0, source=SFX_CONDITIONING_SHIPPED)
    errors = validate_sfx_ledger(run_dir, 15.0)
    assert any("coverage gap" in error for error in errors)

"""Direct tests for the run-directory layout helpers (issue 038).

`paths` was imported widely but never asserted: the six-digit id format,
the segment-dir join, and the relocation-tolerant stored-path resolver
had zero pins, so a layout regression would surface only as mysterious
downstream failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage import paths
from voyage.paths import (
    MAX_SEGMENT_NUMBER,
    MIN_SEGMENT_NUMBER,
    format_segment_id,
    resolve_stored_path,
    segment_dir,
)


def test_format_segment_id_zero_pads() -> None:
    assert format_segment_id(0) == "000000"
    assert format_segment_id(1) == "000001"
    assert format_segment_id(42) == "000042"
    assert format_segment_id(MAX_SEGMENT_NUMBER) == "999999"


def test_format_segment_id_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="segment number"):
        format_segment_id(MIN_SEGMENT_NUMBER - 1)
    with pytest.raises(ValueError, match="segment number"):
        format_segment_id(MAX_SEGMENT_NUMBER + 1)


def test_format_segment_id_roundtrip_orders_lexicographically() -> None:
    rendered = [format_segment_id(number) for number in (0, 1, 9, 10, 999999)]
    assert rendered == sorted(rendered)
    assert all(len(segment_id) == 6 for segment_id in rendered)


def test_segment_dir_joins_layout(tmp_path: Path) -> None:
    segment_path = segment_dir(tmp_path, "000001")
    assert segment_path == tmp_path / paths.SEGMENTS_DIRNAME / "000001"


def test_resolve_stored_path_resolves_relative_against_run_dir(tmp_path: Path) -> None:
    resolved = resolve_stored_path(tmp_path, "audio/take_0000.wav")
    assert resolved == tmp_path / "audio" / "take_0000.wav"


def test_resolve_stored_path_keeps_existing_absolute(tmp_path: Path) -> None:
    existing = tmp_path / "video.mp4"
    existing.write_bytes(b"media")
    assert resolve_stored_path(tmp_path / "elsewhere", existing) == existing


def test_resolve_stored_path_reanchors_moved_run(tmp_path: Path) -> None:
    """A stale absolute entry heals when the layout anchor exists at the new home."""
    run_dir = tmp_path / "run"
    relocated = run_dir / paths.SEGMENTS_DIRNAME / "000000" / "recovery.pt"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(b"tape")
    stale = Path("/old/home/audio") / "x" / "segments" / "000000" / "recovery.pt"
    assert resolve_stored_path(run_dir, stale) == relocated


def test_resolve_stored_path_returns_stale_when_unhealable(tmp_path: Path) -> None:
    stale = tmp_path / "nowhere" / "recovery.pt"
    assert resolve_stored_path(tmp_path / "run", stale) == stale

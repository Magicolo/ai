"""Tape durability: `write_tape_atomic` persists content + directory entry (122).

CPU-only: `video_common` is numpy + stdlib, so these run in the slim
gates image. The fsync calls are observed by monkeypatching `os.fsync`
(no disk sync actually performed).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from voyage.atomic import fsync_dir
from voyage.workers import video_common


def test_write_tape_atomic_round_trips_sorted_json(tmp_path: Path) -> None:
    tape_path = tmp_path / "recovery.pt"
    tape = {"zeta": 1, "alpha": [1, 2], "backend": "causvid"}
    assert video_common.write_tape_atomic(tape_path, tape) == tape_path
    assert json.loads(tape_path.read_text(encoding="utf-8")) == tape
    assert tape_path.read_text(encoding="utf-8").endswith("\n")


def test_write_tape_atomic_fsyncs_file_and_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """122: the temp file is fsynced before rename, the dir after."""
    synced: list[int] = []
    real_fsync = os.fsync

    def _record_fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record_fsync)
    dir_synced: list[str] = []

    def _record_dir(directory: Path) -> None:
        dir_synced.append(str(directory))

    monkeypatch.setattr("voyage.workers.video_common.fsync_dir", _record_dir)
    tape_path = tmp_path / "recovery.pt"
    video_common.write_tape_atomic(tape_path, {"backend": "causvid"})
    assert synced, "expected at least one os.fsync call for the tape file"
    assert dir_synced == [str(tmp_path)], f"expected one fsync_dir({tmp_path}), got {dir_synced}"
    assert fsync_dir is not None  # the shared helper exists


def test_write_tape_atomic_preserves_previous_on_fsync_failure(tmp_path: Path) -> None:
    """122: atomicity is kept — a failed fsync never clobbers the old tape."""
    tape_path = tmp_path / "recovery.pt"
    video_common.write_tape_atomic(tape_path, {"generation": 1})
    with pytest.MonkeyPatch.context() as patch:

        def _fail(_fd: int) -> None:
            raise OSError("simulated power loss")

        patch.setattr(os, "fsync", _fail)
        with pytest.raises(OSError, match="simulated power loss"):
            video_common.write_tape_atomic(tape_path, {"generation": 2})
    assert json.loads(tape_path.read_text(encoding="utf-8")) == {"generation": 1}

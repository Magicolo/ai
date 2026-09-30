"""Issue 129: LongLive recovery.pt must be atomic + durable (CPU-only).

TDD pin for the bare ``torch.save`` write site
(``video_longlive.py`` ``generate_blocks``): the tape write must go
through ``save_recovery_tape_atomic`` (tmp + fsync + rename + fsync_dir),
so a crash mid-save never clobbers the previous good tape.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any

import pytest

from voyage.atomic import fsync_dir as real_fsync_dir
from voyage.workers import video_longlive


class _FakeTorchSave:
    """Minimal torch.save stand-in: pickle to the given path or handle."""

    def __init__(self) -> None:
        self.saves: list[str] = []

    def save(self, tape: dict[str, Any], path: Any) -> None:
        label = path if isinstance(path, str) else str(getattr(path, "name", "handle"))
        self.saves.append(label)
        if isinstance(path, str):
            with open(path, "wb") as handle:
                pickle.dump(tape, handle)
        else:
            pickle.dump(tape, path)


class _FailingTorchSave(_FakeTorchSave):
    """Writes a torn prefix then raises (simulated crash mid-save)."""

    def save(self, tape: dict[str, Any], path: Any) -> None:
        label = path if isinstance(path, str) else str(getattr(path, "name", "handle"))
        self.saves.append(label)
        if isinstance(path, str):
            with open(path, "wb") as handle:
                handle.write(b"\x80\x04torn-prefix")
                handle.flush()
        else:
            path.write(b"\x80\x04torn-prefix")
            path.flush()
        raise RuntimeError("simulated crash mid-save")


def _tape() -> dict[str, Any]:
    return {"profile": "longlive2", "next_start_frame": 8, "blocks_appended": 1}


def test_atomic_helper_writes_tape_and_fsyncs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_dirs: list[Path] = []

    def _recording_fsync_dir(path: Path) -> None:
        fsync_dirs.append(path)
        real_fsync_dir(path)

    monkeypatch.setattr("voyage.workers.video_longlive.fsync_dir", _recording_fsync_dir)
    recovery_path = tmp_path / "seg" / "recovery.pt"
    recovery_path.parent.mkdir(parents=True)
    result = video_longlive.save_recovery_tape_atomic(_FakeTorchSave(), _tape(), recovery_path)
    assert result == recovery_path
    with open(recovery_path, "rb") as handle:
        assert pickle.load(handle) == _tape()
    assert fsync_dirs == [recovery_path.parent]
    # No torn sibling left beside the live tape.
    assert not recovery_path.with_suffix(".tmp").exists()


def test_atomic_helper_preserves_previous_tape_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("voyage.workers.video_longlive.fsync_dir", lambda path: None)
    recovery_path = tmp_path / "seg" / "recovery.pt"
    recovery_path.parent.mkdir(parents=True)
    video_longlive.save_recovery_tape_atomic(_FakeTorchSave(), _tape(), recovery_path)
    with open(recovery_path, "rb") as handle:
        assert pickle.load(handle) == _tape()
    try:
        video_longlive.save_recovery_tape_atomic(
            _FailingTorchSave(), {"profile": "longlive2", "next_start_frame": 16}, recovery_path
        )
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)
    else:  # pragma: no cover - the fake always raises
        raise AssertionError("expected the simulated crash to propagate")
    with open(recovery_path, "rb") as handle:
        assert pickle.load(handle) == _tape()


def test_generate_blocks_routes_tape_through_atomic_helper() -> None:
    """Regression pin: the segment write site must not bare-save onto live."""
    source = inspect.getsource(video_longlive.LongLiveSession.generate_blocks)
    assert "save_recovery_tape_atomic" in source
    assert "torch.save(tape, str(recovery_path))" not in source

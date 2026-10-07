"""Segment-interleaved model pass: scoping, order, joints, per-segment bars.

Pins the pipeline fix: `_poll_to_completion` enumerates committed
segments once per pass and runs each segment's upscale immediately
followed by its interp (never a full upscale sweep then a full interp
sweep), scopes both pollers to the current segment, polls the
fix-stage joint units through the same legs after the segments, and
renders one leg bar per segment. CPU-only — pollers and the fix enter
via seams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyage import augment_finalize as finalize_module
from voyage.augment import AugmentWeights
from voyage.augment_finalize import _poll_to_completion
from voyage.augment_interp_poller import InterpPollResult
from voyage.augment_upscale_poller import UpscalePollResult


def _make_segment(
    run_dir: Path,
    segment_id: str,
    *,
    frames: int = 8,
    checksum: str = "ck",
) -> Path:
    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": frames},
        "checksums": {"video.mp4": checksum},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _make_weights(work: Path) -> AugmentWeights:
    work.mkdir(parents=True, exist_ok=True)
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    rife = work / "rife.safetensors"
    rife.write_bytes(b"rife-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    return AugmentWeights(film=film, rife=rife, realesrgan=esrgan)


def _zero_upscale(run_dir_arg: Path, **kwargs: Any) -> UpscalePollResult:
    return UpscalePollResult(
        segments_seen=1,
        segments_skipped=0,
        chunks_done=0,
        chunks_skipped=0,
        partials_pruned=0,
    )


def _zero_interp(run_dir_arg: Path, **kwargs: Any) -> InterpPollResult:
    return InterpPollResult(
        segments_seen=1,
        segments_skipped=0,
        chunks_done=0,
        chunks_skipped=0,
        chunks_waiting=0,
        partials_pruned=0,
    )


def _run_two_segments(
    tmp_path: Path,
    calls: list[tuple[str, Any]],
    *,
    multiplier: int = 1,
    joint_ids: list[str] | None = None,
    progress: Any = None,
    monkeypatch: Any = None,
    include_upscale: bool = True,
    include_interp: bool = True,
) -> None:
    run_dir = tmp_path / "run"
    _make_segment(run_dir, "000000", checksum="ck0")
    _make_segment(run_dir, "000001", checksum="ck1")
    weights = _make_weights(tmp_path / "weights")
    if monkeypatch is not None:
        from voyage.augment_joints import JointUnit

        units = [
            JointUnit(
                joint_dir=run_dir / joint_id,
                joint_video=run_dir / joint_id / "joint.mp4",
                joint_key=f"jk-{joint_id}",
                left_id="000000",
                right_id="000001",
                left_frames=8,
                right_frames=8,
            )
            for joint_id in (joint_ids or [])
        ]
        monkeypatch.setattr(
            finalize_module,
            "ensure_joint_units",
            lambda *args, **kwargs: units,
        )

    def _rec_upscale(run_dir_arg: Path, **kwargs: Any) -> UpscalePollResult:
        calls.append(("up", kwargs.get("segment_ids")))
        return _zero_upscale(run_dir_arg, **kwargs)

    def _rec_interp(run_dir_arg: Path, **kwargs: Any) -> InterpPollResult:
        calls.append(("ip", kwargs.get("segment_ids")))
        return _zero_interp(run_dir_arg, **kwargs)

    _poll_to_completion(
        run_dir,
        weights=weights,
        weights_key="k|k",
        out_width=768,
        out_height=432,
        source_fps=24.0,
        upscale_factor=1,
        multiplier=multiplier,
        chunk_frames=32,
        device="cpu",
        crf=15,
        preset="veryfast",
        upscale_poll_fn=_rec_upscale,
        interp_poll_fn=_rec_interp,
        progress=progress,
        include_upscale=include_upscale,
        include_interp=include_interp,
    )


def test_pollers_receive_scoped_segment_ids(tmp_path: Path, monkeypatch: Any) -> None:
    """Both pollers run scoped to the current segment (never whole-dir)."""
    calls: list[tuple[str, Any]] = []
    _run_two_segments(tmp_path, calls, monkeypatch=monkeypatch)
    assert calls == [
        ("up", ["000000"]),
        ("ip", ["000000"]),
        ("up", ["000001"]),
        ("ip", ["000001"]),
    ]


def test_interleave_order_up_then_ip_per_segment(tmp_path: Path, monkeypatch: Any) -> None:
    """Interp starts on committed frames: up,ip per segment, not sweep-then-sweep."""
    calls: list[tuple[str, Any]] = []
    _run_two_segments(tmp_path, calls, monkeypatch=monkeypatch)
    legs = [leg for leg, _scope in calls]
    assert legs == ["up", "ip", "up", "ip"]


def test_joints_poll_through_both_legs_after_segments(tmp_path: Path, monkeypatch: Any) -> None:
    """Fix-stage joint units poll through the same legs after the segments."""
    calls: list[tuple[str, Any]] = []
    _run_two_segments(
        tmp_path, calls, multiplier=2, joint_ids=["joint_000000_000001"], monkeypatch=monkeypatch
    )
    assert [leg for leg, _scope in calls] == ["up", "ip", "up", "ip", "up", "ip"]
    assert calls[4] == ("up", ["joint_000000_000001"])
    assert calls[5] == ("ip", ["joint_000000_000001"])


def test_joints_poll_without_segments_in_interp_only_mode(tmp_path: Path, monkeypatch: Any) -> None:
    """Interp-only polls still run the joint upscale (new units, no prior leg)."""
    calls: list[tuple[str, Any]] = []
    _run_two_segments(
        tmp_path,
        calls,
        multiplier=2,
        joint_ids=["joint_000000_000001"],
        monkeypatch=monkeypatch,
        include_upscale=False,
        include_interp=True,
    )
    assert ("up", ["joint_000000_000001"]) in calls
    assert ("ip", ["joint_000000_000001"]) in calls


class _Tracker:
    def __init__(self) -> None:
        self.total: int | None = None
        self.updates: list[int] = []

    def set_total(self, total: int) -> None:
        self.total = total

    def update(self, count: int) -> None:
        self.updates.append(count)

    def set_extra(self, _text: str) -> None:
        return None


class _BarContext:
    def __init__(self, tracker: _Tracker) -> None:
        self._tracker = tracker

    def __enter__(self) -> _Tracker:
        return self._tracker

    def __exit__(self, *args: Any) -> None:
        return None


class _Sink:
    verbose = False

    def __init__(self) -> None:
        self.bars: list[tuple[str, int | None, _Tracker]] = []

    def bar(self, label: str, total: int | None = None) -> _BarContext:
        tracker = _Tracker()
        self.bars.append((label, total, tracker))
        return _BarContext(tracker)


def test_per_segment_leg_bars(tmp_path: Path, monkeypatch: Any) -> None:
    """One leg bar per segment (sequential, never two concurrent displays)."""
    sink = _Sink()
    calls: list[tuple[str, Any]] = []
    _run_two_segments(tmp_path, calls, progress=sink, monkeypatch=monkeypatch)  # type: ignore[arg-type]
    assert [label for label, _total, _tracker in sink.bars] == [
        "upscale frames",
        "interp frames",
        "upscale frames",
        "interp frames",
    ]
    assert [total for _label, total, _tracker in sink.bars] == [8, 8, 8, 8]

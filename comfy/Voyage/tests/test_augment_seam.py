"""Seam interpolation between adjacent usable segments (CPU-only, stubbed GPU).

Pins the Phase 2 contract: the durable finalize path must not hard-cut
between segments — for each adjacent usable pair (A, B) it renders the
`multiplier - 1` FILM mids between A's last interpolated frame and B's
first into a dedicated joint sidecar dir (composite source key, same plan
shape as a segment dir so the drain reuses `drain_interpolated_plan`
unchanged), then interleaves the seam intermediates as [A, seam, B] in the
final concat. Endpoints are never dropped or duplicated: they stay in
their segments, the seam contributes mids only. No torch/GPU/ffmpeg —
seams enter through `seam_interp_fn`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from voyage import augment_sidecar as sidecar
from voyage.augment_finalize import run_durable_model_pass
from voyage.augment_seam import (
    render_seam_once,
    seam_endpoints,
    seam_plan_dir,
)
from voyage.augment_sidecar import ChunkKey, plan_dir_for_segment


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 8,
    checksum: str = "abc123",
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


def _make_interp_plan(
    plan_dir: Path,
    run_dir: Path,
    chunks: dict[int, int],
    *,
    source_key: str = "abc123",
    multiplier: int = 2,
) -> None:
    """Populate a segment plan dir with interpolated chunks + ledger records."""
    for index, count in chunks.items():
        png_dir = plan_dir / f"interpolated_{index:02d}"
        png_dir.mkdir(parents=True, exist_ok=True)
        for position in range(count):
            (png_dir / f"frame_{position:06d}.png").write_bytes(b"fake-png")
        key = ChunkKey(
            chunk_index=index,
            start_frame=0,
            source_frames=4,
            expected_frames=count,
            upscale_factor=2,
            multiplier=multiplier,
            crf=15,
            preset="veryfast",
            source_key=source_key,
            weights_key="wkey",
            out_width=1216,
            out_height=704,
            out_fps=48,
        )
        sidecar.append_chunk_record(
            plan_dir / sidecar.CHUNKS_LEDGER_FILENAME,
            key,
            stage=sidecar.STAGE_INTERPOLATED,
            path=f"augment/plan/interpolated_{index:02d}",
        )


def _seam_kwargs() -> dict[str, Any]:
    return {
        "weights_key": "wkey",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "upscale_factor": 2,
        "crf": 15,
        "preset": "veryfast",
    }


def test_seam_plan_dir_is_pair_sensitive(tmp_path: Path) -> None:
    forward = seam_plan_dir(tmp_path, key_a="aaa", key_b="bbb", **_seam_kwargs())
    assert seam_plan_dir(tmp_path, key_a="aaa", key_b="bbb", **_seam_kwargs()) == forward
    assert seam_plan_dir(tmp_path, key_a="bbb", key_b="aaa", **_seam_kwargs()) != forward
    assert seam_plan_dir(tmp_path, key_a="aaa", key_b="ccc", **_seam_kwargs()) != forward
    retuned = seam_plan_dir(tmp_path, key_a="aaa", key_b="bbb", **{**_seam_kwargs(), "crf": 10})
    assert retuned != forward  # recipe change never hits old seams


def test_seam_endpoints_pick_boundary_frames(tmp_path: Path) -> None:
    plan_a = tmp_path / "plan_a"
    plan_b = tmp_path / "plan_b"
    _make_interp_plan(plan_a, tmp_path, {0: 7, 1: 7})
    _make_interp_plan(plan_b, tmp_path, {0: 7})
    endpoints = seam_endpoints(plan_a, plan_b)
    assert endpoints is not None
    before, after = endpoints
    assert before == plan_a / "interpolated_01" / "frame_000006.png"
    assert after == plan_b / "interpolated_00" / "frame_000000.png"


def test_seam_endpoints_none_when_side_incomplete(tmp_path: Path) -> None:
    plan_a = tmp_path / "plan_a"
    plan_b = tmp_path / "plan_b"
    _make_interp_plan(plan_a, tmp_path, {0: 7})
    assert seam_endpoints(plan_a, plan_b) is None  # B has no interp yet: wait
    assert seam_endpoints(plan_b, plan_a) is None  # A-side missing likewise waits


def test_render_seam_writes_mids_only_and_ledgers(tmp_path: Path) -> None:
    seam_dir = tmp_path / "seam"
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"a")
    after.write_bytes(b"b")
    seen: dict[str, Any] = {}

    def stub_seam(  # type: ignore[no-untyped-def]
        first: Path, second: Path, dest: Path, multiplier: int, **kwargs
    ) -> list[Path]:
        seen["call"] = (first, second, multiplier, kwargs.get("device"))
        dest.mkdir(parents=True, exist_ok=True)
        written = []
        for position in range(multiplier - 1):
            frame = dest / f"frame_{position:06d}.png"
            frame.write_bytes(b"mid")
            written.append(frame)
        return written

    rendered = render_seam_once(
        tmp_path,
        seam_dir,
        before_png=before,
        after_png=after,
        multiplier=4,
        source_key="aaa|bbb",
        interp_fn=stub_seam,
        **_seam_kwargs(),
    )
    assert rendered is True
    assert seen["call"][:3] == (before, after, 4)
    png_dir = seam_dir / "interpolated_00"
    assert sorted(path.name for path in png_dir.glob("frame_*.png")) == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]
    records = sidecar.load_chunk_ledger(seam_dir / sidecar.CHUNKS_LEDGER_FILENAME)
    assert len(records) == 1
    assert records[0]["expected_frames"] == 3  # mids only: endpoints stay in segments
    assert records[0]["source_frames"] == 2
    assert records[0]["multiplier"] == 4
    assert records[0]["source_key"] == "aaa|bbb"

    # Second render is a ledger-truth no-op (crash resume never re-renders).
    rendered_again = render_seam_once(
        tmp_path,
        seam_dir,
        before_png=before,
        after_png=after,
        multiplier=4,
        source_key="aaa|bbb",
        interp_fn=stub_seam,
        **_seam_kwargs(),
    )
    assert rendered_again is False


def test_render_seam_rejects_degenerate_multiplier(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_seam_once(
            tmp_path,
            tmp_path / "seam",
            before_png=tmp_path / "a.png",
            after_png=tmp_path / "b.png",
            multiplier=1,
            source_key="aaa|bbb",
            interp_fn=None,
            **_seam_kwargs(),
        )


def test_durable_pass_interleaves_seam_between_segments(tmp_path: Path) -> None:
    first = _make_segment(tmp_path, "000000", frames=8, checksum="abc123")
    second = _make_segment(tmp_path, "000001", frames=8, checksum="def456")
    work = tmp_path / "work"
    work.mkdir()
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    weights = SimpleNamespace(film=film, realesrgan=esrgan)
    plan_a = plan_dir_for_segment(
        tmp_path,
        source_key="abc123",
        weights_key="wkey",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    plan_b = plan_dir_for_segment(
        tmp_path,
        source_key="def456",
        weights_key="wkey",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    _make_interp_plan(plan_a, tmp_path, {0: 7, 1: 7}, source_key="abc123")
    _make_interp_plan(plan_b, tmp_path, {0: 7}, source_key="def456")
    calls: dict[str, list[Any]] = {"drain": [], "concat": [], "seam": []}

    def stub_poll(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        if "multiplier" in kwargs:
            return SimpleNamespace(chunks_done=0, chunks_waiting=0)
        return SimpleNamespace(chunks_done=0)

    def stub_drain(plan_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        calls["drain"].append(plan_dir)
        intermediate = plan_dir / "model_intermediate.mp4"
        intermediate.write_bytes(b"fake-intermediate")
        return SimpleNamespace(intermediate_mp4=intermediate, chunks_drained=1)

    def stub_seam(  # type: ignore[no-untyped-def]
        before_png: Path, after_png: Path, dest: Path, multiplier: int, **kwargs
    ) -> list[Path]:
        calls["seam"].append((before_png, after_png, multiplier))
        dest.mkdir(parents=True, exist_ok=True)
        frame = dest / "frame_000000.png"
        frame.write_bytes(b"mid")
        return [frame]

    def stub_concat(chunks: list[Path], dest: Path) -> Path:
        calls["concat"].append(list(chunks))
        dest.write_bytes(b"".join(chunk.read_bytes() for chunk in chunks))
        return dest

    weights_key = f"{hashlib.sha256(b'film-weights').hexdigest()}|"
    weights_key += hashlib.sha256(b"esrgan-weights").hexdigest()
    with patch("voyage.augment_finalize.weights_key_for", return_value="wkey"):
        final, final_fps = run_durable_model_pass(
            tmp_path,
            [first, second],
            out_width=1216,
            out_height=704,
            source_fps=24.0,
            weights=weights,
            multiplier=2,
            chunk_frames=4,
            device="cuda:1",
            work_dir=work,
            upscale_poll_fn=stub_poll,
            interp_poll_fn=stub_poll,
            drain_fn=stub_drain,
            concat_fn=stub_concat,
            seam_interp_fn=stub_seam,
        )
    assert weights_key  # real key shape documented; wiring uses the patched one
    assert final_fps == round(24.0 * 2)
    # Seam rendered once between the boundary frames, mids only (m=2 → 1 mid).
    assert calls["seam"] == [
        (
            plan_a / "interpolated_01" / "frame_000006.png",
            plan_b / "interpolated_00" / "frame_000000.png",
            2,
        )
    ]
    # Concat interleaves [A, seam, B].
    (concat_order,) = calls["concat"]
    assert concat_order[0] == plan_a / "model_intermediate.mp4"
    assert concat_order[1].parent.name != plan_a.name
    assert concat_order[1].name == "model_intermediate.mp4"
    assert concat_order[2] == plan_b / "model_intermediate.mp4"
    assert len(concat_order) == 3

    # Re-running resumes: the seam ledger hit means no re-render.
    calls["seam"].clear()
    with patch("voyage.augment_finalize.weights_key_for", return_value="wkey"):
        run_durable_model_pass(
            tmp_path,
            [first, second],
            out_width=1216,
            out_height=704,
            source_fps=24.0,
            weights=weights,
            multiplier=2,
            chunk_frames=4,
            device="cuda:1",
            work_dir=work,
            upscale_poll_fn=stub_poll,
            interp_poll_fn=stub_poll,
            drain_fn=stub_drain,
            concat_fn=stub_concat,
            seam_interp_fn=stub_seam,
        )
    assert calls["seam"] == []


def test_durable_pass_skips_seam_at_multiplier_one(tmp_path: Path) -> None:
    first = _make_segment(tmp_path, "000000", frames=8)
    second = _make_segment(tmp_path, "000001", frames=8, checksum="def456")
    work = tmp_path / "work"
    work.mkdir()
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    weights = SimpleNamespace(film=film, realesrgan=esrgan)
    calls: dict[str, list[Any]] = {"concat": []}

    def stub_poll(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        if "multiplier" in kwargs:
            return SimpleNamespace(chunks_done=0, chunks_waiting=0)
        return SimpleNamespace(chunks_done=0)

    def stub_drain(plan_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        intermediate = plan_dir / "model_intermediate.mp4"
        intermediate.parent.mkdir(parents=True, exist_ok=True)
        intermediate.write_bytes(b"x")
        return SimpleNamespace(intermediate_mp4=intermediate, chunks_drained=1)

    def forbidden_seam(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("no seam at multiplier=1 (zero mids to render)")

    def stub_concat(chunks: list[Path], dest: Path) -> Path:
        calls["concat"].append(list(chunks))
        dest.write_bytes(b"".join(chunk.read_bytes() for chunk in chunks))
        return dest

    from unittest import mock

    with mock.patch("voyage.augment_finalize.weights_key_for", return_value="wkey"):
        run_durable_model_pass(
            tmp_path,
            [first, second],
            out_width=1216,
            out_height=704,
            source_fps=24.0,
            weights=weights,
            multiplier=1,
            chunk_frames=4,
            device="cuda:1",
            work_dir=work,
            upscale_poll_fn=stub_poll,
            interp_poll_fn=stub_poll,
            drain_fn=stub_drain,
            concat_fn=stub_concat,
            seam_interp_fn=forbidden_seam,
        )
    (concat_order,) = calls["concat"]
    assert len(concat_order) == 2  # hard concat of two segments, no seam


def test_rerendered_side_never_reuses_stale_seam(tmp_path: Path) -> None:
    """Characterization: seam dirs are re-derived from current source keys.

    When B is re-rendered (new video.mp4 checksum), the pollers produce a
    new plan dir for B and the next finalize must render the seam under the
    NEW joint dir H(A)|H(B'). The stale joint dir H(A)|H(B) still exists on
    disk with ledgered records, but nothing may reference it: the drain
    loop re-derives every seam dir from the current adjacent source keys,
    so the stale dir is an orphan, never consumed.
    """
    first = _make_segment(tmp_path, "000000", frames=8, checksum="abc123")
    second = _make_segment(tmp_path, "000001", frames=8, checksum="def456")
    work = tmp_path / "work"
    work.mkdir()
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    weights = SimpleNamespace(film=film, realesrgan=esrgan)
    plan_a = plan_dir_for_segment(
        tmp_path,
        source_key="abc123",
        weights_key="wkey",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    plan_b = plan_dir_for_segment(
        tmp_path,
        source_key="def456",
        weights_key="wkey",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    _make_interp_plan(plan_a, tmp_path, {0: 7, 1: 7}, source_key="abc123")
    _make_interp_plan(plan_b, tmp_path, {0: 7}, source_key="def456")
    calls: dict[str, list[Any]] = {"drain": [], "concat": [], "seam": []}

    def stub_poll(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        if "multiplier" in kwargs:
            return SimpleNamespace(chunks_done=0, chunks_waiting=0)
        return SimpleNamespace(chunks_done=0)

    def stub_drain(plan_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        calls["drain"].append(plan_dir)
        plan_dir.mkdir(parents=True, exist_ok=True)
        intermediate = plan_dir / "model_intermediate.mp4"
        intermediate.write_bytes(b"fake-intermediate")
        return SimpleNamespace(intermediate_mp4=intermediate, chunks_drained=1)

    def stub_seam(  # type: ignore[no-untyped-def]
        before_png: Path, after_png: Path, dest: Path, multiplier: int, **kwargs
    ) -> list[Path]:
        calls["seam"].append((before_png, after_png, multiplier))
        dest.mkdir(parents=True, exist_ok=True)
        frame = dest / "frame_000000.png"
        frame.write_bytes(b"mid")
        return [frame]

    def stub_concat(chunks: list[Path], dest: Path) -> Path:
        calls["concat"].append(list(chunks))
        dest.write_bytes(b"".join(chunk.read_bytes() for chunk in chunks))
        return dest

    def run_pass() -> list[Path]:
        with patch("voyage.augment_finalize.weights_key_for", return_value="wkey"):
            run_durable_model_pass(
                tmp_path,
                [first, second],
                out_width=1216,
                out_height=704,
                source_fps=24.0,
                weights=weights,
                multiplier=2,
                chunk_frames=4,
                device="cuda:1",
                work_dir=work,
                upscale_poll_fn=stub_poll,
                interp_poll_fn=stub_poll,
                drain_fn=stub_drain,
                concat_fn=stub_concat,
                seam_interp_fn=stub_seam,
            )
        (concat_order,) = calls["concat"]
        typed_order: list[Path] = list(concat_order)
        return typed_order

    stale_seam_dir = seam_plan_dir(tmp_path, key_a="abc123", key_b="def456", **_seam_kwargs())
    order_one = run_pass()
    assert calls["seam"] != []  # first pass renders the seam once
    assert stale_seam_dir / "model_intermediate.mp4" in order_one

    # Re-render B: new checksum, new plan dir, pollers re-ran interp there.
    _make_segment(tmp_path, "000001", frames=8, checksum="xyz789")
    plan_b_new = plan_dir_for_segment(
        tmp_path,
        source_key="xyz789",
        weights_key="wkey",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    assert plan_b_new != plan_b
    _make_interp_plan(plan_b_new, tmp_path, {0: 7}, source_key="xyz789")
    fresh_seam_dir = seam_plan_dir(tmp_path, key_a="abc123", key_b="xyz789", **_seam_kwargs())
    assert fresh_seam_dir != stale_seam_dir

    calls["drain"].clear()
    calls["concat"].clear()
    calls["seam"].clear()
    order_two = run_pass()
    # The new joint dir renders exactly once; the stale dir is never touched.
    assert len(calls["seam"]) == 1
    assert stale_seam_dir not in calls["drain"]
    assert fresh_seam_dir in calls["drain"]
    assert fresh_seam_dir / "model_intermediate.mp4" in order_two
    assert stale_seam_dir / "model_intermediate.mp4" not in order_two
    assert len(order_two) == 3  # [A, fresh seam, B'] — no duplication

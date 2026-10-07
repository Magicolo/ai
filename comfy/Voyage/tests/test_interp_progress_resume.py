"""Interp per-pair progress + resume hardening (CPU-only, stubbed GPU).

Pins the finalize-feedback contract: long FILM chunks advance the bar per
finished pair (fractional source-frame units summing to the chunk count),
three-arg injected fakes fall back to one chunk-end advance, corrupt
upscale inputs wait for the upscale healer instead of failing loud, a
re-tiled plan dir re-renders instead of skipping wrong outputs, and the
upfront total helper matches the committed segments. No torch/GPU/ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path

from voyage.augment_finalize import _expected_frame_totals
from voyage.augment_interp_poller import interp_poll_once
from voyage.augment_sidecar import STAGE_INTERPOLATED, load_chunk_ledger, plan_dir_for_segment
from voyage.augment_upscale_poller import upscale_poll_once


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


def _stub_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for position in range(count):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-frame")
        written.append(frame)
    return written


def _stub_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
    assert dest_dir.is_dir()
    return list(frame_paths)


def _stub_interp(frame_paths: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = (len(frame_paths) - 1) * multiplier + 1
    written = []
    for position in range(expected):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-interp")
        written.append(frame)
    return written


def _stub_encode(png_dir: Path, dest: Path, fps: float) -> Path:
    dest.write_bytes(b"fake-chunk")
    return dest


def _pair_aware_interp(
    frame_paths: list[Path],
    dest_dir: Path,
    multiplier: int,
    *,
    on_pair=None,  # type: ignore[no-untyped-def]
) -> list[Path]:
    if on_pair is not None:
        for pair_index in range(len(frame_paths) - 1):
            on_pair(pair_index, len(frame_paths) - 1)
    return _stub_interp(frame_paths, dest_dir, multiplier)


def _upscale_first(run_dir: Path, frames: int = 8, chunk_frames: int = 4) -> None:
    _make_segment(run_dir, frames=frames)
    upscale_poll_once(
        run_dir,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=chunk_frames,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )


def _interp_kwargs(**overrides):  # type: ignore[no-untyped-def]
    params = {
        "weights_path": Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "chunk_frames": 4,
        "multiplier": 4,
        "interp_fn": _stub_interp,
        "chunk_encode_fn": _stub_encode,
    }
    params.update(overrides)
    return params


def test_per_pair_live_updates_sum_to_chunk(tmp_path: Path) -> None:
    _upscale_first(tmp_path)
    fired: list[float] = []
    result = interp_poll_once(
        tmp_path,
        **_interp_kwargs(
            interp_fn=_pair_aware_interp,
            on_pair_frames=lambda _segment, frames: fired.append(frames),
        ),
    )
    assert result.chunks_done == 2
    # 2 chunks x 3 pairs each, fractional advances summing to the chunk count.
    assert len(fired) == 6
    assert abs(sum(fired) - 8.0) < 1e-9


def test_three_arg_interp_fn_falls_back_to_chunk_advance(tmp_path: Path) -> None:
    _upscale_first(tmp_path)
    fired: list[float] = []
    result = interp_poll_once(
        tmp_path,
        **_interp_kwargs(
            on_pair_frames=lambda _segment, frames: fired.append(frames),
        ),
    )
    assert result.chunks_done == 2
    assert fired == [4.0, 4.0]


def test_corrupt_upscale_input_waits_instead_of_failing(tmp_path: Path) -> None:
    _upscale_first(tmp_path)
    plan_dir = plan_dir_for_segment(
        tmp_path,
        source_key="abc123",
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    victim = sorted((plan_dir / "upscaled_00").glob("frame_*.png"))[0]
    victim.unlink()
    waiting = interp_poll_once(tmp_path, **_interp_kwargs())
    assert waiting.chunks_done == 1
    assert waiting.chunks_waiting == 1
    assert waiting.chunks_skipped == 0
    # The upscale poller heals the short dir; interp then completes it.
    upscale_poll_once(
        tmp_path,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )
    resumed = interp_poll_once(tmp_path, **_interp_kwargs())
    assert resumed.chunks_done == 1
    assert resumed.chunks_waiting == 0


def test_retiled_plan_dir_rerenders_and_keys_carry_chunk_frames(tmp_path: Path) -> None:
    _upscale_first(tmp_path, chunk_frames=4)
    first = interp_poll_once(tmp_path, **_interp_kwargs())
    assert first.chunks_done == 2
    # Re-tiling means re-upscaling too (windows no longer align); both
    # legs re-render into the same plan dir instead of skipping wrongly.
    upscale_poll_once(
        tmp_path,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=8,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )
    retiled = interp_poll_once(tmp_path, **_interp_kwargs(chunk_frames=8))
    assert retiled.chunks_done == 1
    plan_dir = plan_dir_for_segment(
        tmp_path,
        source_key="abc123",
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    records = [
        record
        for record in load_chunk_ledger(plan_dir / "chunks.jsonl")
        if record.get("stage") == STAGE_INTERPOLATED
    ]
    assert records and records[-1].get("chunk_frames") == 8
    assert records[-1].get("source_frames") == 8


def test_expected_frame_totals_match_committed_segments(tmp_path: Path) -> None:
    assert _expected_frame_totals(tmp_path, chunk_frames=4, multiplier=4) == (0, 0)
    _make_segment(tmp_path, frames=8)
    # 2 windows of 4 source frames; (4-1)*4+1 = 13 output frames each.
    assert _expected_frame_totals(tmp_path, chunk_frames=4, multiplier=4) == (8, 26)

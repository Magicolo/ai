"""Durable finalize model pass over sidecar plan dirs (CPU-only, stubbed GPU).

Pins the Phase 4b contract: `run_durable_model_pass` polls the upscale +
interp workers to completion (shared plan-dir derivation, both weight legs
in the key), drains one intermediate per usable segment, and concats them in
segment order. No torch/GPU/ffmpeg — poll/drain/concat enter via seams.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voyage import augment_sidecar as sidecar
from voyage.augment import AugmentWeights
from voyage.augment_finalize import (
    _poll_to_completion,
    plan_dir_for_segment,
    run_durable_model_pass,
    weights_key_for,
)
from voyage.console import VoyageConsole
from voyage.errors import MediaError


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 8,
    checksum: str = "abc123",
    with_done: bool = True,
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
    if with_done:
        (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _make_weights(work: Path, *, film_bytes: bytes = b"film-weights") -> AugmentWeights:
    work.mkdir(parents=True, exist_ok=True)
    film = work / "film.safetensors"
    film.write_bytes(film_bytes)
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    return AugmentWeights(film=film, realesrgan=esrgan)


def _durable_kwargs(weights: AugmentWeights, work_dir: Path, **overrides):  # type: ignore[no-untyped-def]
    params = {
        "out_width": 1216,
        "out_height": 704,
        "source_fps": 24.0,
        "weights": weights,
        "chunk_frames": 4,
        "device": "cuda:1",
        "work_dir": work_dir,
    }
    params.update(overrides)
    return params


def test_weights_key_needs_both_legs(tmp_path: Path) -> None:
    weights = _make_weights(tmp_path)
    key = weights_key_for(weights)
    assert isinstance(key, str) and key
    assert weights_key_for(weights) == key  # stable
    changed = _make_weights(tmp_path / "other", film_bytes=b"other-film")
    mixed = AugmentWeights(film=changed.film, realesrgan=weights.realesrgan)
    assert weights_key_for(mixed) != key  # any leg change misses
    with pytest.raises(ValueError):
        weights_key_for(AugmentWeights(film=None, realesrgan=weights.realesrgan))
    with pytest.raises(ValueError):
        weights_key_for(AugmentWeights(film=weights.film, realesrgan=None))


def test_plan_dir_matches_poller_derivation(tmp_path: Path) -> None:
    first = plan_dir_for_segment(
        tmp_path,
        source_key="seg",
        weights_key="w",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=15,
        preset="veryfast",
    )
    expected = sidecar.sidecar_dir(
        tmp_path,
        sidecar.plan_hash_for(
            source_key="seg",
            weights_key="w",
            out_width=1216,
            out_height=704,
            out_fps=24,
            upscale_factor=2,
            multiplier=1,
            crf=15,
            preset="veryfast",
        ),
    )
    assert first == expected
    retuned = plan_dir_for_segment(
        tmp_path,
        source_key="seg",
        weights_key="w",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=2,
        crf=10,
        preset="veryfast",
    )
    assert retuned != first  # recipe change never hits old outputs


def test_polls_then_drains_each_usable_segment(tmp_path: Path) -> None:
    first = _make_segment(tmp_path, "000000", frames=8)
    second = _make_segment(tmp_path, "000001", frames=8, checksum="def456")
    work = tmp_path / "work"
    work.mkdir()
    weights = _make_weights(work)
    calls: dict[str, list[Any]] = {"upscale": [], "interp": [], "drain": [], "concat": []}
    states = {"upscale": 4, "interp": 4}

    def stub_upscale(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        calls["upscale"].append(kwargs)
        done, states["upscale"] = states["upscale"], 0
        return SimpleNamespace(chunks_done=done)

    def stub_interp(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        calls["interp"].append(kwargs)
        done, states["interp"] = states["interp"], 0
        return SimpleNamespace(chunks_done=done, chunks_waiting=0)

    def stub_drain(plan_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        calls["drain"].append((plan_dir, kwargs))
        assert kwargs["out_fps"] == pytest.approx(24.0 * 1)
        chunk = plan_dir / "chunk_00.mp4"
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_bytes(b"fake-chunk")
        intermediate = plan_dir / "model_intermediate.mp4"
        intermediate.write_bytes(b"fake-intermediate")
        return SimpleNamespace(intermediate_mp4=intermediate)

    def stub_concat(chunks: list[Path], dest: Path) -> Path:
        calls["concat"].append(list(chunks))
        dest.write_bytes(b"".join(chunk.read_bytes() for chunk in chunks))
        return dest

    final, final_fps = run_durable_model_pass(
        tmp_path,
        [first, second],
        **_durable_kwargs(
            weights,
            work,
            multiplier=1,  # upscale-only: zero seam mids, hard concat
            upscale_poll_fn=stub_upscale,
            interp_poll_fn=stub_interp,
            drain_fn=stub_drain,
            concat_fn=stub_concat,
        ),
    )
    assert final_fps == round(24.0 * 1)
    assert final.read_bytes() == b"fake-intermediatefake-intermediate"
    # Both pollers share one key over both legs, the segment (source) fps,
    # and the finalize geometry; the upscale leg feeds upscale, FILM interp.
    upscale_kwargs = calls["upscale"][0]
    assert isinstance(upscale_kwargs, dict)
    assert upscale_kwargs["weights_path"] == weights.realesrgan
    assert upscale_kwargs["out_fps"] == pytest.approx(24.0)
    interp_kwargs = calls["interp"][0]
    assert isinstance(interp_kwargs, dict)
    assert interp_kwargs["weights_path"] == weights.film
    assert interp_kwargs["weights_key"] == upscale_kwargs["weights_key"]
    assert interp_kwargs["out_fps"] == pytest.approx(24.0)
    # One drain per usable segment, in segment order, at the lifted fps.
    assert [plan for plan, _ in calls["drain"]] == [
        plan_dir_for_segment(
            tmp_path,
            source_key=checksum,
            weights_key=str(upscale_kwargs["weights_key"]),
            out_width=1216,
            out_height=704,
            out_fps=24,
            upscale_factor=2,
            crf=15,
            preset="veryfast",
        )
        for checksum in ("abc123", "def456")
    ]
    assert calls["concat"] and len(calls["concat"][0]) == 2


def test_stuck_poll_fails_loud(tmp_path: Path) -> None:
    _make_segment(tmp_path, "000000", frames=8)
    work = tmp_path / "work"
    work.mkdir()
    weights = _make_weights(work)

    def stub_upscale(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(chunks_done=0)

    def stub_interp(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(chunks_done=0, chunks_waiting=2)

    with pytest.raises(MediaError):
        run_durable_model_pass(
            tmp_path,
            [tmp_path / "segments" / "000000"],
            **_durable_kwargs(
                weights,
                work,
                upscale_poll_fn=stub_upscale,
                interp_poll_fn=stub_interp,
            ),
        )


def test_usable_segment_without_source_fails_loud(tmp_path: Path) -> None:
    ghost = _make_segment(tmp_path, "000000", frames=8, with_done=False)
    work = tmp_path / "work"
    work.mkdir()
    weights = _make_weights(work)

    def stub_upscale(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(chunks_done=0)

    def stub_interp(run_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(chunks_done=0, chunks_waiting=0)

    with pytest.raises(MediaError):
        run_durable_model_pass(
            tmp_path,
            [ghost],
            **_durable_kwargs(
                weights,
                work,
                upscale_poll_fn=stub_upscale,
                interp_poll_fn=stub_interp,
            ),
        )


def _stub_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
    """Decode seam stub: stage `count` PNGs (contents never read)."""
    del source_video
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for position in range(count):
        frame = dest_dir / f"frame_{start + position:05d}.png"
        frame.write_bytes(b"png")
        out.append(frame)
    return out


def _stub_upscale_pngs(decoded: list[Path], dest_dir: Path) -> list[Path]:
    """Upscale seam stub: copy names over (one output per input)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for frame in decoded:
        dest = dest_dir / frame.name
        dest.write_bytes(b"up")
        out.append(dest)
    return out


def _stub_interp_pngs(upscaled: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
    """Interp seam stub: write exactly the recipe's expected frame count."""
    from voyage.augment import interpolated_frame_count

    expected = interpolated_frame_count(len(upscaled), multiplier)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for position in range(expected):
        frame = dest_dir / f"frame_{position:05d}.png"
        frame.write_bytes(b"ip")
        out.append(frame)
    return out


def _poll_kwargs() -> dict[str, Any]:
    return {
        "weights_key": "test-key",
        "out_width": 1216,
        "out_height": 704,
        "upscale_factor": 2,
        "chunk_frames": 4,
        "device": "cpu",
        "crf": 15,
        "preset": "veryfast",
    }


def test_upscale_on_chunk_fires_per_chunk(tmp_path: Path) -> None:
    """`on_chunk` reports (segment, index, window count) per rendered chunk."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _make_segment(tmp_path, "000000", frames=8)
    seen: list[tuple[str, int, int]] = []
    result = upscale_poll_once(
        tmp_path,
        weights_path=tmp_path / "esrgan.pth",
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale_pngs,
        on_chunk=lambda sid, idx, total: seen.append((sid, idx, total)),
        out_fps=24,
        **_poll_kwargs(),
    )
    assert result.chunks_done == 2
    assert seen == [("000000", 0, 2), ("000000", 1, 2)]


def test_interp_on_chunk_fires_per_chunk(tmp_path: Path) -> None:
    """Same contract on the interp leg (after a real upscale pass)."""
    from voyage.augment_interp_poller import interp_poll_once
    from voyage.augment_upscale_poller import upscale_poll_once

    _make_segment(tmp_path, "000000", frames=8)
    weights = _make_weights(tmp_path / "work")
    assert weights.realesrgan is not None
    upscale_poll_once(
        tmp_path,
        weights_path=weights.realesrgan,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale_pngs,
        out_fps=24,
        **_poll_kwargs(),
    )
    seen: list[tuple[str, int, int]] = []
    assert weights.film is not None
    result = interp_poll_once(
        tmp_path,
        weights_path=weights.film,
        interp_fn=_stub_interp_pngs,
        on_chunk=lambda sid, idx, total: seen.append((sid, idx, total)),
        out_fps=24,
        **_poll_kwargs(),
    )
    assert result.chunks_done == 2
    assert seen == [("000000", 0, 2), ("000000", 1, 2)]


def test_poll_to_completion_reports_per_leg_frame_bars(tmp_path: Path) -> None:
    """Separate `upscale frames` / `interp frames` bars count source frames."""
    from voyage.augment_interp_poller import InterpPollResult
    from voyage.augment_upscale_poller import UpscalePollResult

    weights = _make_weights(tmp_path / "work")
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)
    calls = {"upscale": 0, "interp": 0}

    def stub_upscale(run_dir: Path, **kwargs: Any) -> Any:
        del run_dir
        calls["upscale"] += 1
        on_frames = kwargs.get("on_chunk_frames")
        assert callable(on_frames)
        if calls["upscale"] == 1:
            on_frames("000000", 32)
            on_frames("000000", 32)
            return UpscalePollResult(1, 0, 2, 1, 0, frames_done=64, frames_skipped=32)
        return UpscalePollResult(1, 0, 0, 3, 0, frames_done=0, frames_skipped=96)

    def stub_interp(run_dir: Path, **kwargs: Any) -> Any:
        del run_dir
        calls["interp"] += 1
        on_pair = kwargs.get("on_pair_frames")
        assert callable(on_pair)
        if calls["interp"] == 1:
            on_pair("000000", 24.0)
            on_pair("000000", 24.0)
            return InterpPollResult(1, 0, 2, 1, 0, 0, frames_done=48, frames_skipped=16)
        return InterpPollResult(1, 0, 0, 3, 0, 0, frames_done=0, frames_skipped=64)

    timings: dict[str, float] = {}
    _poll_to_completion(
        tmp_path,
        weights=weights,
        upscale_poll_fn=stub_upscale,
        interp_poll_fn=stub_interp,
        progress=console,
        source_fps=24.0,
        multiplier=2,
        timings=timings,
        **_poll_kwargs(),
    )
    out = stream.getvalue()
    assert "▸ upscale frames ..." in out
    assert "▸ interp frames ..." in out
    # Live frame advance (64) + skipped catch-up (32) = 96-frame total.
    assert "✓ upscale frames (96/96," in out
    assert "✓ interp frames (64/64," in out
    # Rate extra rides the finish line; per-leg frame counts hit timings.
    assert "frames/s)" in out
    assert timings["upscale_frames_done"] == 64.0
    assert timings["interp_frames_done"] == 48.0


def test_poll_bar_tolerates_legacy_done_only_fakes(tmp_path: Path) -> None:
    """Done-only stub namespaces (no skipped/waiting/frames) never break polling."""
    weights = _make_weights(tmp_path / "work")
    stream = io.StringIO()
    console = VoyageConsole(stream=stream)

    def stub_upscale(run_dir: Path, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        del run_dir, kwargs
        return SimpleNamespace(chunks_done=0)

    def stub_interp(run_dir: Path, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        del run_dir, kwargs
        return SimpleNamespace(chunks_done=0, chunks_waiting=0)

    _poll_to_completion(
        tmp_path,
        weights=weights,
        upscale_poll_fn=stub_upscale,
        interp_poll_fn=stub_interp,
        progress=console,
        source_fps=24.0,
        multiplier=2,
        **_poll_kwargs(),
    )
    out = stream.getvalue()
    assert "upscale frames" in out
    assert "interp frames" in out

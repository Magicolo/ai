"""Background model-pass pre-warm during generation (CPU-only, stubbed GPU).

Pins the generation/finalize overlap contract: while video renders on
`cuda:0` and the llama director serves on `cuda:1`, committed segments
are upscaled + interpolated in the background on `cuda:1` whenever the
director is idle — so finalize's poll-to-completion wait shrinks to a
drain + concat + encode. The ledger is the resume truth for both a
resumed generation and a resumed finalization: ledgered chunks are
skipped, unledgered partials re-rendered. No torch/GPU/ffmpeg.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from voyage.config import AugmentConfig, preset_config


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


def _enabled_config(**overrides: Any) -> Any:
    config = preset_config("prewarm", "pastel neon line-art, peaceful", 11)
    augment = AugmentConfig(
        upscale=2,
        interpolate=4,
    )
    return config.model_copy(update={"augment": augment, **overrides})


def test_disabled_when_no_work_demanded(tmp_path: Path) -> None:
    from voyage import augment_background

    _make_segment(tmp_path)
    config = preset_config("prewarm", "pastel neon line-art, peaceful", 11)
    config = config.model_copy(
        update={"augment": config.augment.model_copy(update={"upscale": 1, "interpolate": 1})}
    )
    assert config.augment.upscale == 1
    assert config.augment.interpolate == 1
    calls: list[str] = []

    def _upscale_probe(**kwargs: Any) -> Any:
        calls.append("upscale")
        raise AssertionError("must not poll when the knob is off")

    assert (
        augment_background.prewarm_once(
            tmp_path,
            config,
            upscale_poll_fn=_upscale_probe,
        )
        is None
    )
    assert calls == []


def test_plan_matches_finalize_derivation(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights
    from voyage.media import plan_augmentation

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )
    plan = augment_background.resolve_background_plan(tmp_path, config)
    assert plan is not None
    expected = plan_augmentation(
        1216,
        704,
        24.0,
        upscale=config.augment.upscale,
        interpolate=config.augment.interpolate,
        presentation_fps=config.augment.presentation_fps,
    )
    assert (plan.out_width, plan.out_height, plan.out_fps) == (
        expected.out_w,
        expected.out_h,
        expected.out_fps,
    )
    assert plan.upscale_factor == config.augment.upscale
    assert plan.source_fps_key == 24
    assert plan.multiplier == 4
    assert plan.device == "cuda:1"


def test_prewarm_polls_with_finalize_params(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 5.0)
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )
    seen: dict[str, dict[str, Any]] = {}

    def _upscale_stub(run_dir: Path, **kwargs: Any) -> Any:
        from voyage.augment_upscale_poller import UpscalePollResult

        seen["upscale"] = dict(kwargs)
        return UpscalePollResult(1, 0, 2, 0, 0)

    def _interp_stub(run_dir: Path, **kwargs: Any) -> Any:
        from voyage.augment_interp_poller import InterpPollResult

        seen["interp"] = dict(kwargs)
        return InterpPollResult(1, 0, 2, 0, 0, 0)

    result = augment_background.prewarm_once(
        tmp_path,
        config,
        upscale_poll_fn=_upscale_stub,
        interp_poll_fn=_interp_stub,
    )
    assert result is not None
    assert result.upscale_chunks_done == 2
    assert result.interp_chunks_done == 2
    assert seen["upscale"]["device"] == "cuda:1"
    assert seen["interp"]["device"] == "cuda:1"
    assert seen["upscale"]["out_fps"] == 24
    assert seen["interp"]["out_fps"] == 24
    assert seen["interp"]["multiplier"] == 4
    assert seen["upscale"]["weights_key"] == seen["interp"]["weights_key"]


def test_prewarm_skipped_without_committed_segments(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
    )
    assert augment_background.prewarm_once(tmp_path, config) is None


def test_resume_skips_ledgered_chunks(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path, frames=8)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 5.0)
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )

    def _stub_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for position in range(count):
            frame = dest_dir / f"frame_{position + 1:06d}.png"
            frame.write_bytes(b"fake-frame")
            written.append(frame)
        return written

    def _stub_upscale(frame_paths: list[Path], dest_dir: Path, **kwargs: Any) -> list[Path]:
        assert dest_dir.is_dir()
        return list(frame_paths)

    def _stub_interp(
        frame_paths: list[Path],
        dest_dir: Path,
        multiplier: int,
        **kwargs: Any,
    ) -> list[Path]:
        # Pair-interp math: first frame + (multiplier - 1) mids per pair.
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        counter = 0
        for index, frame in enumerate(frame_paths):
            for _replica in range(multiplier if index else 1):
                counter += 1
                dest = dest_dir / f"frame_{counter:06d}.png"
                dest.write_bytes(frame.read_bytes())
                written.append(dest)
        return written

    def _stub_encode(png_dir: Path, dest: Path, fps: float) -> Path:
        dest.write_bytes(b"fake-chunk")
        return dest

    import voyage.augment_interp_poller as interp_module
    import voyage.augment_upscale_poller as upscale_module

    monkeypatch.setattr(upscale_module, "_default_decode_fn", _stub_decode)
    monkeypatch.setattr(upscale_module, "_default_upscale_pngs", _stub_upscale)
    monkeypatch.setattr(interp_module, "_default_interp_pngs", _stub_interp)

    first = augment_background.prewarm_once(tmp_path, config, chunk_encode_fn=_stub_encode)
    assert first is not None
    assert first.upscale_chunks_done > 0
    assert first.interp_chunks_done > 0

    calls: list[tuple[str, int]] = []

    def _counting_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
        calls.append(("decode", start))
        return _stub_decode(source_video, dest_dir, start, count)

    monkeypatch.setattr(upscale_module, "_default_decode_fn", _counting_decode)
    second = augment_background.prewarm_once(tmp_path, config, chunk_encode_fn=_stub_encode)
    assert second is not None
    assert second.upscale_chunks_done == 0
    assert second.upscale_chunks_skipped > 0
    assert second.interp_chunks_done == 0
    assert second.interp_chunks_skipped > 0
    assert calls == []


def test_background_thread_notifies_and_swallows_errors(tmp_path: Path) -> None:
    from voyage import augment_background
    from voyage.config import preset_config

    config = preset_config("prewarm", "pastel neon line-art, peaceful", 11)
    calls: list[str] = []

    def _okay(run_dir: Path, _config: Any) -> Any:
        calls.append("prewarm")
        return None

    driver = augment_background.BackgroundPrewarm(
        tmp_path, config, prewarm_fn=_okay, idle_fn=lambda: True
    )
    driver.start()
    try:
        assert driver.notify_committed() is True
        deadline = 5.0
        import time

        waited = 0.0
        while not calls and waited < deadline:
            time.sleep(0.05)
            waited += 0.05
        assert calls == ["prewarm"]
    finally:
        driver.stop()


def test_background_thread_skips_when_director_busy(tmp_path: Path) -> None:
    from voyage import augment_background
    from voyage.config import preset_config

    config = preset_config("prewarm", "pastel neon line-art, peaceful", 11)
    calls: list[str] = []

    def _okay(run_dir: Path, _config: Any) -> Any:
        calls.append("prewarm")
        return None

    driver = augment_background.BackgroundPrewarm(
        tmp_path, config, prewarm_fn=_okay, idle_fn=lambda: False
    )
    driver.start()
    try:
        assert driver.notify_committed() is True
        import time

        time.sleep(0.3)
        assert calls == []
    finally:
        driver.stop()


def test_background_thread_swallows_prewarm_errors(tmp_path: Path) -> None:
    from voyage import augment_background
    from voyage.config import preset_config

    config = preset_config("prewarm", "pastel neon line-art, peaceful", 11)

    def _boom(run_dir: Path, _config: Any) -> Any:
        raise RuntimeError("gpu busy")

    driver = augment_background.BackgroundPrewarm(
        tmp_path, config, prewarm_fn=_boom, idle_fn=lambda: True
    )
    driver.start()
    try:
        assert driver.notify_committed() is True
        import time

        time.sleep(0.3)
        assert driver.is_alive() is True
    finally:
        driver.stop()


def test_stop_without_start_is_safe(tmp_path: Path) -> None:
    from voyage import augment_background
    from voyage.config import preset_config

    config = preset_config("prewarm", "pastel neon line-art, peaceful", 11)
    driver = augment_background.BackgroundPrewarm(tmp_path, config)
    driver.stop()
    assert driver.notify_committed() is False


def test_plan_none_when_single_leg(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    film.write_bytes(b"f" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=None),
    )
    assert augment_background.resolve_background_plan(tmp_path, config) is None


def test_plan_none_when_no_devices(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ())
    assert augment_background.resolve_background_plan(tmp_path, config) is None


def test_plan_none_when_probe_fails(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))

    def _bad_probe(_video: Path) -> Any:
        raise ValueError("no ffprobe here")

    monkeypatch.setattr("voyage.augment_background.probe_segment_source", _bad_probe)
    assert augment_background.resolve_background_plan(tmp_path, config) is None


def test_prewarm_skips_sweeps_when_device_full(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )
    # The llama director sidecar holds ~5GB of the 6GB 2060: 13MB free.
    # Neither leg fits, so the sweep is held back as a zero-count result
    # (surfacing the reason) instead of a silent skip.
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 0.013)
    calls: list[str] = []

    def _must_not_run(**kwargs: Any) -> Any:
        calls.append("poll")
        raise AssertionError("must not sweep without VRAM headroom")

    result = augment_background.prewarm_once(
        tmp_path,
        config,
        upscale_poll_fn=_must_not_run,
        interp_poll_fn=_must_not_run,
    )
    assert result is not None
    assert result.upscale_frames_done == 0
    assert result.interp_frames_done == 0
    assert "upscale" in result.skip_reason
    assert calls == []


def test_prewarm_runs_when_headroom(tmp_path: Path, monkeypatch: Any) -> None:
    from voyage import augment_background
    from voyage.augment import AugmentWeights

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 5.0)

    def _upscale_stub(run_dir: Path, **kwargs: Any) -> Any:
        from voyage.augment_upscale_poller import UpscalePollResult

        return UpscalePollResult(1, 0, 2, 0, 0)

    def _interp_stub(run_dir: Path, **kwargs: Any) -> Any:
        from voyage.augment_interp_poller import InterpPollResult

        return InterpPollResult(1, 0, 2, 0, 0, 0)

    result = augment_background.prewarm_once(
        tmp_path,
        config,
        upscale_poll_fn=_upscale_stub,
        interp_poll_fn=_interp_stub,
    )
    assert result is not None
    assert result.upscale_chunks_done == 2


def test_device_free_gib_unknown_allows_prewarm(monkeypatch: Any) -> None:
    import subprocess

    from voyage import augment_background

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no nvidia-smi here")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert augment_background.device_free_gib("cuda:1") is None


def test_driver_accumulates_ledgered_totals(tmp_path: Path) -> None:
    """Finished passes accumulate (passes, upscale, interp) for the report."""
    from voyage.augment_background import BackgroundPrewarm, PrewarmResult

    def _fake_prewarm(run_dir: Path, config: Any) -> PrewarmResult | None:
        del run_dir, config
        return PrewarmResult(1, 2, 0, 3, 0, 0)

    driver = BackgroundPrewarm(tmp_path, object(), prewarm_fn=_fake_prewarm, idle_fn=lambda: True)
    assert driver.ledgered_totals() == (0, 0, 0)
    driver.start()
    try:
        deadline = time.monotonic() + 10.0
        while driver.ledgered_totals()[0] < 2 and time.monotonic() < deadline:
            driver.notify_committed()
            time.sleep(0.05)
    finally:
        driver.stop()
    passes, up, ip = driver.ledgered_totals()
    assert passes >= 2
    assert up == 2 * passes
    assert ip == 3 * passes


if __name__ == "__main__":
    pytest.main([__file__])


def test_prewarm_resident_nets_bypass_cold_floor(tmp_path: Path, monkeypatch: Any) -> None:
    """Resident nets gate on working set, not the cold-load floor (boba held-back).

    0.9 GiB free holds back a cold pass (1.0 GiB floor), but once the
    nets are resident in this process the H2D load already happened —
    the same 0.9 GiB must let the pass through.
    """
    from voyage import augment_background
    from voyage.augment import AugmentWeights
    from voyage.workers import augment_worker

    _make_segment(tmp_path)
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 0.9)
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )

    def _upscale_stub(run_dir: Path, **kwargs: Any) -> Any:
        from voyage.augment_upscale_poller import UpscalePollResult

        return UpscalePollResult(1, 0, 2, 0, 0)

    def _interp_stub(run_dir: Path, **kwargs: Any) -> Any:
        from voyage.augment_interp_poller import InterpPollResult

        return InterpPollResult(1, 0, 2, 0, 0, 0)

    cold = augment_background.prewarm_once(
        tmp_path,
        config,
        upscale_poll_fn=_upscale_stub,
        interp_poll_fn=_interp_stub,
    )
    assert cold is not None
    assert cold.skip_reason.startswith("upscale skipped")
    assert cold.upscale_chunks_done == 0
    monkeypatch.setattr(
        augment_worker,
        "_ESRGAN_CACHE",
        {augment_worker._model_cache_key(realesrgan, "cuda:1"): object()},
    )
    monkeypatch.setattr(
        augment_worker,
        "_RIFE_CACHE",
        {augment_worker._model_cache_key(rife, "cuda:1"): object()},
    )
    warm = augment_background.prewarm_once(
        tmp_path,
        config,
        upscale_poll_fn=_upscale_stub,
        interp_poll_fn=_interp_stub,
    )
    assert warm is not None
    assert warm.skip_reason == ""
    assert warm.upscale_chunks_done == 2
    assert warm.interp_chunks_done == 2


def _joint_ffmpeg(*argv: str) -> None:
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", *argv], check=True)


def _joint_clip(path: Path, frames: int, fps: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _joint_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x64:rate={fps}:duration={frames / fps}",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(path),
    )
    return path


def _joint_gray_png() -> bytes:
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        framed = struct.pack(">I", len(data)) + tag + data
        return framed + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * 64 for _ in range(64))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


_JOINT_PNG_BYTES = _joint_gray_png()


def _stub_joint_interp(
    png_a: Path, png_b: Path, weights: object, moment: float, device: str
) -> bytes:
    del weights, moment, device
    assert png_a.exists() and png_b.exists()
    return _JOINT_PNG_BYTES


def _make_real_segment(
    run_dir: Path, segment_id: str, *, frames: int = 12, checksum: str = "ck"
) -> Path:
    import json as _json

    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    _joint_clip(segment_dir / "video.mp4", frames)
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": frames},
        "checksums": {"video.mp4": checksum},
    }
    (segment_dir / "manifest.json").write_text(_json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _joint_prewarm_setup(tmp_path: Path, monkeypatch: Any, segments: int = 3) -> tuple[Any, Path]:
    from voyage.augment import AugmentWeights

    run_dir = tmp_path / "run"
    for index in range(segments):
        _make_real_segment(run_dir, f"{index:06d}", checksum=f"ck{index}")
    config = _enabled_config()
    film = tmp_path / "film.safetensors"
    realesrgan = tmp_path / "realesrgan.pth"
    film.write_bytes(b"f" * 64)
    realesrgan.write_bytes(b"r" * 64)
    rife = tmp_path / "rife.safetensors"
    rife.write_bytes(b"i" * 64)
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan, rife=rife),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cpu",))
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 5.0)
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (64, 64, 24.0),
    )
    return config, run_dir


def _recording_joint_stubs(
    calls: list[tuple[str, str, str]],
) -> tuple[Any, Any]:
    """Poll stubs recording (leg, scope, scope-id); render once, then settle."""
    from voyage.augment_interp_poller import InterpPollResult
    from voyage.augment_upscale_poller import UpscalePollResult

    seen: set[Any] = set()

    def _scope(kwargs: Any) -> tuple[str, str]:
        ids = tuple(kwargs.get("segment_ids") or ())
        unit_id = ids[0] if ids else "unknown"
        if kwargs.get("sources") is not None:
            return ("joint", unit_id)
        return ("segment", unit_id)

    def _upscale_stub(run_dir: Path, **kwargs: Any) -> Any:
        scope, unit_id = _scope(kwargs)
        calls.append(("up", scope, unit_id))
        key = ("up", scope, unit_id)
        if key in seen:
            return UpscalePollResult(
                segments_seen=1,
                segments_skipped=0,
                chunks_done=0,
                chunks_skipped=1,
                partials_pruned=0,
                frames_done=0,
                frames_skipped=4,
            )
        seen.add(key)
        if kwargs.get("on_chunk") is not None:
            kwargs["on_chunk"](unit_id, 0, 1)
        if kwargs.get("on_chunk_frames") is not None:
            kwargs["on_chunk_frames"](unit_id, 4)
        return UpscalePollResult(
            segments_seen=1,
            segments_skipped=0,
            chunks_done=1,
            chunks_skipped=0,
            partials_pruned=0,
            frames_done=4,
            frames_skipped=0,
        )

    def _interp_stub(run_dir: Path, **kwargs: Any) -> Any:
        scope, unit_id = _scope(kwargs)
        calls.append(("ip", scope, unit_id))
        joint_ids: tuple[str, ...] = (
            tuple(sorted(kwargs.get("segment_ids") or ())) if scope == "joint" else ()
        )
        key = ("ip", scope, unit_id, joint_ids)
        if key in seen:
            return InterpPollResult(
                segments_seen=1,
                segments_skipped=0,
                chunks_done=0,
                chunks_skipped=1,
                chunks_waiting=0,
                partials_pruned=0,
                frames_done=0,
                frames_skipped=4,
            )
        seen.add(key)
        if kwargs.get("on_chunk") is not None:
            kwargs["on_chunk"](unit_id, 0, 1)
        if kwargs.get("on_chunk_frames") is not None:
            kwargs["on_chunk_frames"](unit_id, 4)
        return InterpPollResult(
            segments_seen=1,
            segments_skipped=0,
            chunks_done=1,
            chunks_skipped=0,
            chunks_waiting=0,
            partials_pruned=0,
            frames_done=4,
            frames_skipped=0,
        )

    return _upscale_stub, _interp_stub


def test_prewarm_runs_joints_before_segments_with_joint_callbacks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Joints-first: joint uniform polls precede segment polls, routed apart."""
    from voyage import augment_background

    config, run_dir = _joint_prewarm_setup(tmp_path, monkeypatch, segments=3)
    calls: list[tuple[str, str, str]] = []
    upscale_stub, interp_stub = _recording_joint_stubs(calls)
    joint_events: list[tuple[str, str, int]] = []
    shared_segments: list[str] = []

    result = augment_background.prewarm_once(
        run_dir,
        config,
        upscale_poll_fn=upscale_stub,
        interp_poll_fn=interp_stub,
        joint_interp_fn=_stub_joint_interp,
        on_upscale_frames=lambda segment, frames: shared_segments.append(segment),
        on_interp_frames=lambda segment, frames: shared_segments.append(segment),
        on_joint_chunk=lambda unit, leg, index, total: joint_events.append((unit, leg, index)),
        on_joint_frames=lambda unit, leg, frames, position, total: joint_events.append(
            (unit, leg, frames)
        ),
    )
    assert result is not None
    assert result.joints_seen == 2
    assert result.joint_fix_done == 2
    assert result.joint_upscale_chunks_done == 2
    assert result.joint_interp_chunks_done == 2
    assert result.joint_frames_total == 2 * 3 * 4
    # Every joint uniform poll precedes every segment poll, per leg.
    joint_up = [call for call in calls if call[:2] == ("up", "joint")]
    seg_up = [call for call in calls if call[:2] == ("up", "segment")]
    joint_ip = [call for call in calls if call[:2] == ("ip", "joint")]
    seg_ip = [call for call in calls if call[:2] == ("ip", "segment")]
    assert len(joint_up) == 2 and len(seg_up) == 3
    assert len(joint_ip) == 2 and len(seg_ip) == 3
    assert max(calls.index(call) for call in joint_up) < min(calls.index(call) for call in seg_up)
    assert max(calls.index(call) for call in joint_ip) < min(calls.index(call) for call in seg_ip)
    # Routing: shared callbacks see segments only, joint callbacks see joints.
    assert shared_segments != []
    assert all(not segment.startswith("joint_") for segment in shared_segments)
    joint_units = {unit for unit, _leg, _n in joint_events}
    assert joint_units == {"joint_000000_000001", "joint_000001_000002"}
    joint_legs = {leg for _unit, leg, _n in joint_events}
    assert joint_legs == {"fix", "upscale", "interp"}


def test_prewarm_stop_between_joint_units_resumes_next_pass(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A mid-joints stop abandons cleanly; the next pass finishes the rest."""
    from voyage import augment_background

    config, run_dir = _joint_prewarm_setup(tmp_path, monkeypatch, segments=3)
    calls: list[tuple[str, str, str]] = []
    upscale_stub, interp_stub = _recording_joint_stubs(calls)
    joint_ip_calls = {"count": 0}

    def _counting_interp(run_dir: Path, **kwargs: Any) -> Any:
        scope = "joint" if kwargs.get("sources") is not None else "segment"
        if scope == "joint":
            joint_ip_calls["count"] += 1
        return interp_stub(run_dir, **kwargs)

    def _stop_after_first_joint_interp() -> bool:
        return joint_ip_calls["count"] >= 1

    stopped = augment_background.prewarm_once(
        run_dir,
        config,
        upscale_poll_fn=upscale_stub,
        interp_poll_fn=_counting_interp,
        joint_interp_fn=_stub_joint_interp,
        should_stop=_stop_after_first_joint_interp,
    )
    assert stopped is None
    resumed = augment_background.prewarm_once(
        run_dir,
        config,
        upscale_poll_fn=upscale_stub,
        interp_poll_fn=interp_stub,
        joint_interp_fn=_stub_joint_interp,
    )
    assert resumed is not None
    assert resumed.joints_seen == 2
    assert resumed.joint_fix_done == 0
    assert resumed.joint_upscale_chunks_done == 1
    assert resumed.joint_upscale_chunks_skipped == 1
    assert resumed.joint_interp_chunks_done == 1
    assert resumed.joint_interp_chunks_skipped == 1

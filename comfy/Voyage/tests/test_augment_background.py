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
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
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
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
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
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
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

    import voyage.augment_interp_poller as interp_module
    import voyage.augment_upscale_poller as upscale_module

    monkeypatch.setattr(upscale_module, "_default_decode_fn", _stub_decode)
    monkeypatch.setattr(upscale_module, "_default_upscale_pngs", _stub_upscale)
    monkeypatch.setattr(interp_module, "_default_interp_pngs", _stub_interp)

    first = augment_background.prewarm_once(tmp_path, config)
    assert first is not None
    assert first.upscale_chunks_done > 0
    assert first.interp_chunks_done > 0

    calls: list[tuple[str, int]] = []

    def _counting_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
        calls.append(("decode", start))
        return _stub_decode(source_video, dest_dir, start, count)

    monkeypatch.setattr(upscale_module, "_default_decode_fn", _counting_decode)
    second = augment_background.prewarm_once(tmp_path, config)
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
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
    )
    monkeypatch.setattr("voyage.augment_background.model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(
        "voyage.augment_background.probe_segment_source",
        lambda _video: (1216, 704, 24.0),
    )
    # The llama director sidecar holds ~5GB of the 6GB 2060: 13MB free.
    monkeypatch.setattr("voyage.augment_background.device_free_gib", lambda _device: 0.013)
    calls: list[str] = []

    def _must_not_run(**kwargs: Any) -> Any:
        calls.append("poll")
        raise AssertionError("must not sweep without VRAM headroom")

    assert (
        augment_background.prewarm_once(
            tmp_path,
            config,
            upscale_poll_fn=_must_not_run,
            interp_poll_fn=_must_not_run,
        )
        is None
    )
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
    monkeypatch.setattr(
        "voyage.augment.resolve_augment_weights",
        lambda _models_dir: AugmentWeights(film=film, realesrgan=realesrgan),
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

"""Finalize A/V streams: 2060 upscale -> interp, 4060 music -> SFX.

The user-approved pipeline runs two streams concurrently: Thread-A
renders the SRVGG upscale leg then the RIFE interp leg sequentially on
the 2060 (cuda:1, fitting beside the llama sidecar) while Thread-B
renders the deferred ACE music takes then the SFX bed sequentially on
the 4060 (cuda:0); then the mix; then publish. Interpolation (including
seam/morph work) uses the configured backend (RIFE by default) and also
runs during generation via the background pre-warm when the 2060 has
headroom.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from voyage.augment import AugmentWeights
from voyage.persistence import read_effective_config


def test_upscale_leg_pins_secondary_gpu() -> None:
    """2-GPU boxes upscale on cuda:1; fewer GPUs keep the visible selection."""
    from voyage.augment import upscale_pass_devices

    assert upscale_pass_devices(devices=("cuda:0", "cuda:1")) == ("cuda:1",)
    assert upscale_pass_devices(devices=("cuda:0",)) == ("cuda:0",)
    assert upscale_pass_devices(devices=()) == ()


def test_interp_leg_pins_secondary_gpu() -> None:
    """2-GPU boxes interpolate on cuda:1 (RIFE fits the llama share)."""
    from voyage.augment import interp_pass_devices

    assert interp_pass_devices(devices=("cuda:0", "cuda:1")) == ("cuda:1",)
    assert interp_pass_devices(devices=("cuda:0",)) == ("cuda:0",)
    assert interp_pass_devices(devices=()) == ()


def _commit_fake_run(tmp_path: Path, run_id: str) -> Path:
    """Commit one fake-backend segment; returns the run dir."""
    from tests.conftest import initialize_run_directory
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id=run_id, style="pastel neon line-art, peaceful")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    return run_dir


def _stub_weights(base: Path | str) -> AugmentWeights:
    """Legs that exist on paper only — never probed, only recorded."""
    return AugmentWeights(
        film=Path(str(base) + "/film"),
        realesrgan=Path(str(base) + "/esrgan"),
        rife=Path(str(base) + "/rife"),
    )


def test_phased_finalize_thread_a_runs_interleaved_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thread-A runs one interleaved model pass on the 2060.

    Under the 2-stream finalize Thread-A (interleaved upscale+interp)
    races Thread-B (music then bed), so interp no longer waits for
    music: the deterministic pins are a single durable call, both legs
    on cuda:1, and music completed by the join. Fake SFX (no bed) keeps
    the test to the phase boundary.
    """
    import voyage.audio_finalize as audio_finalize_module
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    from voyage.media import finalize_run

    run_dir = _commit_fake_run(tmp_path, "avorder")

    music_done = threading.Event()
    seen: dict[str, Any] = {}

    def _fake_durable_pass(
        run_dir_arg: Path,
        usable: list[Path],
        *,
        weights: AugmentWeights,
        out_width: int,
        out_height: int,
        source_fps: float,
        upscale_factor: int = 2,
        multiplier: int = 4,
        chunk_frames: int = 32,
        crf: int = 15,
        preset: str = "veryfast",
        device: str = "cuda:1",
        work_dir: Path,
        interp_backend: str = "rife",
        upscale_device: str | None = None,
        interp_device: str | None = None,
        timings: dict[str, float] | None = None,
        progress: Any = None,
    ) -> tuple[Path, int]:
        seen["durable_ran"] = True
        seen["upscale_device"] = upscale_device or device
        seen["interp_device"] = interp_device or device
        seen["segments"] = len(usable)
        work_dir.mkdir(parents=True, exist_ok=True)
        intermediate = work_dir / "model_intermediate.mp4"
        shutil.copy(usable[0] / "video.mp4", intermediate)
        return (intermediate, int(round(source_fps)))

    real_ensure = audio_finalize_module.ensure_deferred_for_finalize

    def _recording_ensure(**kwargs: Any) -> Any:
        result = real_ensure(**kwargs)
        music_done.set()
        return result

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(augment_module, "augment_devices", lambda: ("cuda:0", "cuda:1"))
    monkeypatch.setattr(augment_module, "model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _fake_durable_pass)
    import voyage.augment_parallel as parallel_module

    # Parallel disarmed: pin the pre-parallel path (model∥music fork /
    # single interleaved driver). The audio-first bidirectional branch
    # is pinned in tests/test_parallel_model_pass.py.
    monkeypatch.setattr(parallel_module, "parallel_model_pass_armed", lambda **kwargs: False)
    monkeypatch.setattr(audio_finalize_module, "ensure_deferred_for_finalize", _recording_ensure)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "avstream.mp4"
    finalize_run(run_dir, out, upscale=2, interpolate=1, models_dir=models_dir)
    assert out.exists() and out.stat().st_size > 0
    assert seen["durable_ran"] is True
    assert seen["upscale_device"] == "cuda:1"
    assert seen["interp_device"] == "cuda:1"
    assert seen["segments"] == 1
    assert music_done.is_set()


def test_phased_finalize_two_streams_join_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two finalize streams join before publish.

    Thread-B runs music then the SFX bed sequentially on the 4060 while
    Thread-A runs one interleaved model pass on the 2060, so bed-vs-model
    order is intentionally racy — the deterministic pins are bed ran,
    durable ran, music-before-bed (same thread), and the output existing
    (publish runs post-join, consuming both outcomes).
    """
    import voyage.audio_finalize as audio_finalize_module
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    import voyage.sfx_finalize as sfx_finalize_module
    from voyage.media import SfxParallelRequest, finalize_run

    run_dir = _commit_fake_run(tmp_path, "avbed")
    bed_done = threading.Event()
    music_done = threading.Event()
    seen: dict[str, Any] = {}

    real_ensure = audio_finalize_module.ensure_deferred_for_finalize

    def _recording_ensure(**kwargs: Any) -> Any:
        result = real_ensure(**kwargs)
        music_done.set()
        return result

    def _fake_durable_pass(
        run_dir_arg: Path,
        usable: list[Path],
        **kwargs: Any,
    ) -> tuple[Path, int]:
        seen["durable_ran"] = True
        work_dir = kwargs["work_dir"]
        work_dir.mkdir(parents=True, exist_ok=True)
        intermediate = work_dir / "model_intermediate.mp4"
        shutil.copy(usable[0] / "video.mp4", intermediate)
        return (intermediate, 24)

    def _fake_proxy(run_dir_arg: Path, usable_arg: list[Path], tmpdir: Path) -> tuple[Path, float]:
        return (usable_arg[0] / "video.mp4", 4.0)

    def _fake_bounds(
        run_dir_arg: Path, usable_arg: list[Path], fps: int, caption: str | None
    ) -> list[Any]:
        return []

    def _fake_bed(*args: Any, **kwargs: Any) -> Path:
        bed = tmp_path / "bed.wav"
        bed.write_bytes(b"\x00" * 64)
        seen["music_before_bed"] = music_done.is_set()
        bed_done.set()
        return bed

    def _fake_dub(*args: Any, **kwargs: Any) -> Path:
        staged, _bed, _audio, dubbed = args[0], args[1], args[2], args[3]
        shutil.copy(staged, dubbed)
        return dubbed

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(augment_module, "augment_devices", lambda: ("cuda:0", "cuda:1"))
    monkeypatch.setattr(augment_module, "model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _fake_durable_pass)
    import voyage.augment_parallel as parallel_module

    # Parallel disarmed: pin the pre-parallel path (model∥music fork /
    # single interleaved driver). The audio-first bidirectional branch
    # is pinned in tests/test_parallel_model_pass.py.
    monkeypatch.setattr(parallel_module, "parallel_model_pass_armed", lambda **kwargs: False)
    monkeypatch.setattr(sfx_finalize_module, "build_proxy_reference", _fake_proxy)
    monkeypatch.setattr(sfx_finalize_module, "segment_sfx_bounds", _fake_bounds)
    monkeypatch.setattr(sfx_finalize_module, "render_sfx_bed", _fake_bed)
    monkeypatch.setattr(sfx_finalize_module, "stretch_and_dub_sfx_bed", _fake_dub)
    monkeypatch.setattr(audio_finalize_module, "ensure_deferred_for_finalize", _recording_ensure)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "avbed.mp4"
    finalize_run(
        run_dir,
        out,
        upscale=2,
        interpolate=1,
        models_dir=models_dir,
        sfx_request=SfxParallelRequest(
            backend="mmaudio",
            models_dir=str(tmp_path / "sfx-models"),
            device="cpu",
            model_size="small",
            num_workers=1,
            fps=24,
        ),
    )
    assert out.exists() and out.stat().st_size > 0
    assert bed_done.is_set()
    assert seen["durable_ran"] is True
    assert seen["music_before_bed"] is True

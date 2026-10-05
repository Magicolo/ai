"""Finalize A/V stream: 2060 upscale-only, 4060 music -> SFX -> interp (DESIGN §140).

The user-approved pipeline: finalize Phase A runs the SRVGG upscale leg
on the 2060 (cuda:1) overlapping the deferred ACE music takes on the 4060
(cuda:0); then the mix; then the SFX bed renders synchronously on the
4060; then Phase C runs the FILM interp leg on the 4060, picking up the
Phase A upscale ledger; publish comes last. Interpolation (including
seam/morph FILM work) happens only at finalize time — never during
generation (the background pre-warm is upscale-only by construction).
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


def test_interp_leg_pins_primary_gpu() -> None:
    """2-GPU boxes interpolate on cuda:0 (the 4060 stream card)."""
    from voyage.augment import interp_pass_devices

    assert interp_pass_devices(devices=("cuda:0", "cuda:1")) == ("cuda:0",)
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
    return AugmentWeights(film=Path(str(base) + "/film"), realesrgan=Path(str(base) + "/esrgan"))


def test_phased_finalize_runs_upscale_and_music_before_interp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase C interp starts only after Phase A upscale + music finished.

    The Phase A fork still overlaps (upscale on cuda:1, music on cuda:0),
    but the interp phase is strictly post-join: when it starts, both the
    upscale ledger and the music takes exist. Fake SFX (no bed) keeps the
    test to the phase boundary.
    """
    import voyage.audio_finalize as audio_finalize_module
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    from voyage.media import finalize_run

    run_dir = _commit_fake_run(tmp_path, "avorder")

    events: dict[str, threading.Event] = {
        "upscale_done": threading.Event(),
        "music_done": threading.Event(),
    }
    seen: dict[str, Any] = {}

    def _fake_upscale_phase(
        run_dir_arg: Path,
        *,
        weights: AugmentWeights,
        out_width: int,
        out_height: int,
        source_fps: float,
        upscale_factor: int = 2,
        chunk_frames: int = 32,
        crf: int = 15,
        preset: str = "veryfast",
        device: str = "cuda:1",
        upscale_poll_fn: Any = None,
        timings: dict[str, float] | None = None,
        progress: Any = None,
    ) -> None:
        seen["upscale_device"] = device
        events["upscale_done"].set()

    def _fake_interp_phase(
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
        device: str = "cuda:0",
        work_dir: Path,
        interp_poll_fn: Any = None,
        drain_fn: Any = None,
        concat_fn: Any = None,
        seam_interp_fn: Any = None,
        morph_joints: bool = False,
        morph_interp_fn: Any = None,
        timings: dict[str, float] | None = None,
        progress: Any = None,
    ) -> tuple[Path, int]:
        # Post-join contract: both Phase A branches finished by now
        # (sequential path trivially holds; the parallel fork joins too).
        seen["upscale_done_first"] = events["upscale_done"].is_set()
        seen["music_done_first"] = events["music_done"].is_set()
        seen["interp_device"] = device
        intermediate = work_dir / "model_intermediate.mp4"
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(usable[0] / "video.mp4", intermediate)
        return (intermediate, int(round(source_fps)))

    real_ensure = audio_finalize_module.ensure_deferred_for_finalize

    def _recording_ensure(**kwargs: Any) -> Any:
        result = real_ensure(**kwargs)
        events["music_done"].set()
        return result

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(augment_module, "augment_devices", lambda: ("cuda:0", "cuda:1"))
    monkeypatch.setattr(augment_module, "model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(finalize_module, "run_upscale_phase", _fake_upscale_phase)
    monkeypatch.setattr(finalize_module, "run_interp_phase", _fake_interp_phase)
    monkeypatch.setattr(audio_finalize_module, "ensure_deferred_for_finalize", _recording_ensure)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = tmp_path / "avstream.mp4"
    finalize_run(run_dir, out, upscale=2, interpolate=1, models_dir=models_dir)
    assert out.exists() and out.stat().st_size > 0
    assert seen["upscale_device"] == "cuda:1"
    assert seen["interp_device"] == "cuda:0"
    assert seen["upscale_done_first"] is True
    assert seen["music_done_first"] is True


def test_phased_finalize_runs_sfx_bed_before_interp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 4060 stream is sequential: the SFX bed finishes before interp starts.

    A non-fake SFX backend with stubbed bed/dub fns arms the bed path;
    the bed records its completion, and the interp phase asserts it.
    """
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    import voyage.sfx_finalize as sfx_finalize_module
    from voyage.media import SfxParallelRequest, finalize_run

    run_dir = _commit_fake_run(tmp_path, "avbed")
    bed_done = threading.Event()
    seen: dict[str, Any] = {}

    def _fake_upscale_phase(*args: Any, **kwargs: Any) -> None:
        return None

    def _fake_interp_phase(
        run_dir_arg: Path,
        usable: list[Path],
        **kwargs: Any,
    ) -> tuple[Path, int]:
        seen["bed_before_interp"] = bed_done.is_set()
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
        bed_done.set()
        return bed

    def _fake_dub(*args: Any, **kwargs: Any) -> Path:
        staged, _bed, _audio, dubbed = args[0], args[1], args[2], args[3]
        shutil.copy(staged, dubbed)
        return dubbed

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(augment_module, "augment_devices", lambda: ("cuda:0", "cuda:1"))
    monkeypatch.setattr(augment_module, "model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(finalize_module, "run_upscale_phase", _fake_upscale_phase)
    monkeypatch.setattr(finalize_module, "run_interp_phase", _fake_interp_phase)
    monkeypatch.setattr(sfx_finalize_module, "build_proxy_reference", _fake_proxy)
    monkeypatch.setattr(sfx_finalize_module, "segment_sfx_bounds", _fake_bounds)
    monkeypatch.setattr(sfx_finalize_module, "render_sfx_bed", _fake_bed)
    monkeypatch.setattr(sfx_finalize_module, "stretch_and_dub_sfx_bed", _fake_dub)
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
    assert seen["bed_before_interp"] is True

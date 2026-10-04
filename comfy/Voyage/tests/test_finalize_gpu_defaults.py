"""Finalize GPU defaults: MMAudio SFX on cuda:0, model pass on cuda:1, parallel.

TDD contract for the user-approved defaults change (DESIGN §140): the
streaming backends defer ACE-Step music to finalize (timeline-exact
silent stubs at commit, takes rendered after the video worker stops —
no joint-audio GPU contention on the 4060), then dub MMAudio SFX over
the rendered music at finalize on the 4060 (cuda:0), while the
Real-ESRGAN + FILM model pass runs on the 2060 (cuda:1) — side by side
when two GPUs are visible, sequentially otherwise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest

from tests.conftest import initialize_run_directory
from voyage.config import ProjectConfig, VideoBackendName, with_video_backend
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor


def _config_with_style() -> ProjectConfig:
    return ProjectConfig(style="pastel neon line-art, peaceful")


def _commit_two(run_dir: Path) -> None:
    from voyage.config import resolve_config

    config = read_effective_config(run_dir)
    # Pin the deterministic director: finalize-path coverage must not
    # depend on whichever decider backend is the tree default today.
    config = resolve_config(config, director="deterministic")
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def test_ltx_backends_pair_acestep_music_and_sfx_mmaudio_on_cuda0() -> None:
    """ltx25/ltx23: ACE-Step music (deferred to finalize, planner long
    takes) and the finalize SFX dub runs MMAudio on cuda:0 by default."""
    for backend in ("ltx25", "ltx23"):
        config = with_video_backend(_config_with_style(), cast(VideoBackendName, backend))
        assert config.audio.backend == "acestep"
        assert config.audio.device == "cuda:0"
        assert config.sfx.backend == "mmaudio"
        assert config.sfx.device == "cuda:0"


@pytest.mark.parametrize("backend", ["ltxv", "causvid", "ltx25", "ltx23"])
def test_all_cuda_backends_route_to_parallel_finalize(backend: str) -> None:
    """Every CUDA backend's stock preset satisfies the parallel predicate:
    SFX on (mmaudio) + knob on + provisioned legs + one worker + two GPUs.
    `fake` never does (SFX off)."""
    from voyage.finalize_parallel import should_run_parallel

    config = with_video_backend(_config_with_style(), cast(VideoBackendName, backend))
    assert (
        should_run_parallel(
            sfx_backend=config.sfx.backend,
            use_model_pass=config.augment.use_model_pass,
            models_dir="/models",
            num_workers=1,
            devices=("cuda:0", "cuda:1"),
            weights_present=True,
        )
        is True
    )
    assert (
        should_run_parallel(
            sfx_backend=config.sfx.backend,
            use_model_pass=config.augment.use_model_pass,
            models_dir="/models",
            num_workers=1,
            devices=("cuda:0",),
            weights_present=True,
        )
        is False
    )


def test_fake_backend_never_routes_to_parallel() -> None:
    from voyage.finalize_parallel import should_run_parallel

    config = with_video_backend(_config_with_style(), "fake")
    assert config.sfx.backend == "fake"
    assert (
        should_run_parallel(
            sfx_backend=config.sfx.backend,
            use_model_pass=True,
            models_dir="/models",
            num_workers=1,
            devices=("cuda:0", "cuda:1"),
            weights_present=True,
        )
        is False
    )


def test_ltx_backends_require_sfx_stack_with_acestep() -> None:
    """`required_specs` for an ltx backend with SFX enabled pulls both the
    SFX stack and ACE-Step (continuous planner music, not the worker's
    joint track)."""
    from voyage.models_ensure import required_specs

    for backend, spec in (("ltx25", "ltx25"), ("ltx23", "ltx23")):
        config = _config_with_style()
        config.video.backend = cast(VideoBackendName, backend)
        config.audio.backend = "acestep"
        config.sfx.backend = "mmaudio"
        specs = {item.spec for item in required_specs(config, sfx_enabled=True)}
        assert "sfx-mmaudio" in specs
        assert "audio-acestep" in specs
        assert spec in specs


def test_model_pass_defaults_on_everywhere() -> None:
    """The model pass is default-on: config dataclass and stored config agree."""
    from voyage.config import AugmentConfig, preset_config

    assert AugmentConfig().use_model_pass is True
    assert preset_config("story", "style", 7).augment.use_model_pass is True


def test_model_pass_devices_pins_secondary_gpu() -> None:
    """Two visible GPUs pin the whole model pass to cuda:1 (the 2060);
    fewer GPUs keep the legacy single-device selection."""
    from voyage import augment as augment_module
    from voyage.augment import model_pass_devices

    assert model_pass_devices(devices=("cuda:0", "cuda:1")) == ("cuda:1",)
    assert model_pass_devices(devices=("cuda:0",)) == ("cuda:0",)
    assert model_pass_devices(devices=()) == ()
    # Default probes live visibility (patched here to two GPUs).
    monkeypatched = [("cuda:0", "cuda:1")]
    original = augment_module.augment_devices
    try:
        augment_module.augment_devices = lambda **_: monkeypatched[0]  # type: ignore[method-assign]
        assert model_pass_devices() == ("cuda:1",)
    finally:
        augment_module.augment_devices = original  # type: ignore[method-assign]


def test_should_run_parallel_gates_all_conditions() -> None:
    """The parallel predicate needs SFX-on + knob-on + provisioned legs +
    a single SFX worker + two visible GPUs — each missing leg vetoes."""
    from voyage.finalize_parallel import should_run_parallel

    full = {
        "sfx_backend": "mmaudio",
        "use_model_pass": True,
        "models_dir": "/models",
        "num_workers": 1,
        "devices": ("cuda:0", "cuda:1"),
        "weights_present": True,
    }
    assert should_run_parallel(**full) is True
    assert should_run_parallel(**{**full, "sfx_backend": "fake"}) is False
    assert should_run_parallel(**{**full, "use_model_pass": False}) is False
    assert should_run_parallel(**{**full, "models_dir": None}) is False
    assert should_run_parallel(**{**full, "num_workers": 2}) is False
    assert should_run_parallel(**{**full, "devices": ("cuda:0",)}) is False
    assert should_run_parallel(**{**full, "devices": ()}) is False
    assert should_run_parallel(**{**full, "weights_present": False}) is False


def _sine_wav(dest: Path, seconds: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    return dest


@pytest.mark.slow
def test_parallel_finalize_matches_sequential_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel (model pass + SFX bed side by side) ships byte-identical
    output to the legacy sequential finalize + SFX pass.

    GPU stages are stubbed CPU-only (model pass = stream-copy concat, SFX
    bed = sine of the timeline) so the test runs in the slim image; the
    orchestration — reference conditioning, demux/mix/remux, publish —
    is fully real ffmpeg.
    """
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    import voyage.finalize_parallel as parallel_module
    import voyage.sfx_finalize as sfx_module
    from voyage.augment import AugmentWeights
    from voyage.finalize_parallel import run_parallel_finalize
    from voyage.media import finalize_run
    from voyage.sfx_finalize import finalize_sfx_pass

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="parcmp", style="pastel neon line-art", seed=7)
    _commit_two(run_dir)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    legs = AugmentWeights(film=Path("film.safetensors"), realesrgan=Path("esrgan.pth"))
    two_gpus = ("cuda:0", "cuda:1")
    # `finalize_run` lazy-imports these from `voyage.augment` at call
    # time, so one patch point covers both the sequential and the
    # parallel (Thread A) video stages.
    monkeypatch.setattr(augment_module, "resolve_augment_weights", lambda _dir: legs)
    monkeypatch.setattr(augment_module, "augment_devices", lambda **_: two_gpus)

    def _stub_model_pass(
        run_dir_arg: Path, usable: list[Path], **kwargs: object
    ) -> tuple[Path, int]:
        work_dir = kwargs["work_dir"]
        assert isinstance(work_dir, Path)
        work_dir.mkdir(parents=True, exist_ok=True)
        segment_videos = [segment / "video.mp4" for segment in usable]
        intermediate = work_dir / "stub_intermediate.mp4"
        file_list = work_dir / "stub_list.txt"
        file_list.write_text(
            "".join(f"file '{video}'\n" for video in segment_videos), encoding="utf-8"
        )
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(file_list),
                "-c",
                "copy",
                str(intermediate),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-1000:]
        return (intermediate, int(cast(float, kwargs["source_fps"])))

    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _stub_model_pass)

    def _stub_bed(
        run_dir_arg: Path,
        final_video: Path,
        timeline_seconds: float,
        bounds: object,
        tmpdir: Path,
        backend: str,
        models_dir_arg: object,
        device: str,
        model_size: str,
        seed_base: int,
        sample_rate: int,
        channels: int,
        num_workers: int = 1,
        **kwargs: object,
    ) -> Path:
        assert backend == "mmaudio"
        return _sine_wav(tmpdir / "stub_bed.wav", timeline_seconds)

    monkeypatch.setattr(sfx_module, "render_sfx_bed", _stub_bed)
    monkeypatch.setattr(parallel_module, "render_sfx_bed", _stub_bed)

    common = {
        "width": 768,
        "height": 432,
        "fps": 24,
        "sample_rate": 48000,
        "channels": 2,
        "min_fps": 0,
        "min_width": 0,
        "min_height": 0,
        "use_model_pass": True,
        "models_dir": str(models_dir),
    }
    sequential = tmp_path / "final-sequential.mp4"
    assert finalize_run(run_dir, sequential, **common).exists()  # type: ignore[arg-type]
    finalize_sfx_pass(
        run_dir,
        sequential,
        backend="mmaudio",
        models_dir=str(models_dir),
        device="cuda:0",
        model_size="small_44k",
        seed=7,
        sample_rate=48000,
        channels=2,
        num_workers=1,
        fps=24,
    )
    parallel = tmp_path / "final-parallel.mp4"
    run_parallel_finalize(run_dir, parallel, seed=7, sfx_model_size="small_44k", **common)  # type: ignore[arg-type]
    assert parallel.read_bytes() == sequential.read_bytes()

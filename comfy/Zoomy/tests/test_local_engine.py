"""Tests for the in-process engine above its lazy heavy imports.

Every behavior here runs above the lazy heavy imports: constructor
validation, readiness, request validation, interrupt bookkeeping, and memory
figures. Anything that loads a model needs the GPU image and is covered by
live verification instead.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Self, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image as PillowImage

from zoomy.engine_protocol import FinalizeRequest, FrameRenderRequest, SegmentWindow
from zoomy.errors import (
    AssemblyError,
    EmptyFrameSequenceError,
    EngineConfigurationError,
    EngineExecutionError,
    RenderInterruptedError,
)
from zoomy.family_catalog import FAMILY_CATALOG, FamilyDefinition, find_family
from zoomy.frame_repository import FrameRepository
from zoomy.local_engine import (
    LocalEngine,
    _pad_frames_to_minimum,
    _read_system_memory_bytes,
    _write_silent_video,
    run_stage_with_retries,
)

if TYPE_CHECKING:
    from pathlib import Path


def _engine(
    tmp_path: Path,
    *,
    models_present: bool = True,
    seeds_present: bool = True,
) -> LocalEngine:
    """Build an engine over test directories, creating them on request."""
    models_directory = tmp_path / "models"
    seed_directory = tmp_path / "seed"
    music_directory = tmp_path / "music-project"
    if models_present:
        models_directory.mkdir(parents=True, exist_ok=True)
    if seeds_present:
        seed_directory.mkdir(parents=True, exist_ok=True)
    music_directory.mkdir(parents=True, exist_ok=True)
    return LocalEngine(
        models_directory=models_directory,
        seed_directory=seed_directory,
        repository=FrameRepository(tmp_path / "output"),
        device="cuda:0",
        music_project_directory=music_directory,
    )


def test_constructor_rejects_missing_directories(tmp_path: Path) -> None:
    """Each missing directory names itself in the configuration error."""
    with pytest.raises(EngineConfigurationError, match="models"):
        _engine(tmp_path / "first", models_present=False)
    with pytest.raises(EngineConfigurationError, match="seed"):
        _engine(tmp_path / "second", seeds_present=False)


def test_constructor_creates_missing_music_directory(tmp_path: Path) -> None:
    """The music project dir is engine scratch space, so it self-creates."""
    models_directory = tmp_path / "models"
    seed_directory = tmp_path / "seed"
    models_directory.mkdir(parents=True)
    seed_directory.mkdir(parents=True)
    music_directory = tmp_path / "music-project"
    engine = LocalEngine(
        models_directory=models_directory,
        seed_directory=seed_directory,
        repository=FrameRepository(tmp_path / "output"),
        device="cuda:0",
        music_project_directory=music_directory,
    )
    assert music_directory.is_dir()
    assert engine.is_ready() is True


def test_is_ready_reflects_directory_reachability(tmp_path: Path) -> None:
    """A fully present tree reads ready."""
    assert _engine(tmp_path).is_ready() is True


def test_render_frame_rejects_missing_seed(tmp_path: Path) -> None:
    """A cold start without its seed image fails before any model loads."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FrameRenderRequest(
        family=family,
        prompt="a prompt",
        negative_prompt="a negative",
        frame_count=0,
        lora_selections=(),
        seed=1,
    )
    with pytest.raises(EngineConfigurationError, match="seed image"):
        engine.render_frame(request)


def test_render_frame_rejects_unknown_family(tmp_path: Path) -> None:
    """A family without an engine recipe fails instead of guessing a model."""
    engine = _engine(tmp_path)
    seed_path = tmp_path / "seed" / "mystery_seed.png"
    PillowImage.new("RGB", (1376, 768)).save(seed_path)
    catalog_family = find_family(FAMILY_CATALOG, "z_fast")
    mystery_family = FamilyDefinition(
        key="mystery",
        display_name="Mystery",
        sequence_key="mystery",
        base_model_file=catalog_family.base_model_file,
        text_encoder_file=catalog_family.text_encoder_file,
        text_encoder_type=catalog_family.text_encoder_type,
        autoencoder_file=catalog_family.autoencoder_file,
        model_shift=catalog_family.model_shift,
        sampler_name=catalog_family.sampler_name,
        scheduler_name=catalog_family.scheduler_name,
        sampler_steps=catalog_family.sampler_steps,
        classifier_free_guidance=catalog_family.classifier_free_guidance,
        denoise_strength=catalog_family.denoise_strength,
        default_prompt=catalog_family.default_prompt,
        negative_prompt=catalog_family.negative_prompt,
        cold_start_image="mystery_seed.png",
        music_prompt=catalog_family.music_prompt,
        sound_effect_prompt=catalog_family.sound_effect_prompt,
        sound_effect_negative_prompt=catalog_family.sound_effect_negative_prompt,
        loras=(),
    )
    request = FrameRenderRequest(
        family=mystery_family,
        prompt="a prompt",
        negative_prompt=None,
        frame_count=0,
        lora_selections=(),
        seed=1,
    )
    with pytest.raises(EngineConfigurationError, match="No engine recipe"):
        engine.render_frame(request)


def test_finalize_sequence_rejects_empty_sequences(tmp_path: Path) -> None:
    """Finalizing nothing raises before any window runs."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    with pytest.raises(EmptyFrameSequenceError, match="no frames"):
        list(engine.finalize_sequence(FinalizeRequest(family=family, frame_count=0)))


def test_idle_interrupt_aborts_at_the_first_checkpoint(tmp_path: Path) -> None:
    """A press with no running job applies to the next job, never lost."""
    engine = _engine(tmp_path)
    engine.request_interrupt()
    engine._begin_job()  # noqa: SLF001
    with pytest.raises(RenderInterruptedError, match="interrupted"):
        engine._raise_if_interrupted()  # noqa: SLF001


def test_consumed_interrupt_never_kills_the_next_job(tmp_path: Path) -> None:
    """One press aborts exactly one job; the next job starts clean."""
    engine = _engine(tmp_path)
    engine.request_interrupt()
    with pytest.raises(RenderInterruptedError):
        engine._raise_if_interrupted()  # noqa: SLF001
    engine._begin_job()  # noqa: SLF001
    engine._raise_if_interrupted()  # noqa: SLF001


def test_idle_interrupt_aborts_a_finalize_sequence(tmp_path: Path) -> None:
    """The generator path observes an idle press before touching any window."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    engine.request_interrupt()
    with pytest.raises(RenderInterruptedError, match="interrupted"):
        list(engine.finalize_sequence(FinalizeRequest(family=family, frame_count=1)))


def test_engine_statistics_reports_memory_figures(tmp_path: Path) -> None:
    """Statistics carry system RAM and VRAM figures when a GPU is visible."""
    statistics = _engine(tmp_path).engine_statistics()
    assert statistics.system_memory_total_bytes is not None
    assert statistics.system_memory_free_bytes is not None
    assert statistics.system_memory_total_bytes >= 0
    assert statistics.system_memory_free_bytes >= 0
    if _cuda_visible():
        assert statistics.video_memory_free_bytes is not None
        assert statistics.video_memory_total_bytes is not None
        assert 0 <= statistics.video_memory_free_bytes <= statistics.video_memory_total_bytes
    else:
        assert statistics.video_memory_free_bytes is None
        assert statistics.video_memory_total_bytes is None


def test_system_memory_failure_reports_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable /proc/meminfo yields Nones, not a lying zero reading."""

    class _NoProcPath:
        def __init__(self, *_parts: object) -> None:
            pass

        def read_text(self, *_args: object, **_kwargs: object) -> str:
            raise OSError("no /proc here")

    monkeypatch.setattr("zoomy.local_engine.Path", _NoProcPath)
    assert _read_system_memory_bytes() == (None, None)


def _cuda_visible() -> bool:
    """Probe GPU visibility the same guarded way the engine does."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def test_audio_stacks_unload_after_render(tmp_path: Path) -> None:
    """Finalize must not hold ACE and MMAudio resident at once (16 GB card).

    Each audio stage evicts its own stack when its stem lands, so the next
    stage's ~5 GB of weights fit. Sequencing is covered by live verification;
    this pins the eviction itself above the heavy imports.
    """
    engine = _engine(tmp_path)
    engine._music_stack = {"diffusion_handler": object()}  # noqa: SLF001
    engine._effects_stack = {"model": object()}  # noqa: SLF001
    engine._unload_music_stack()  # noqa: SLF001
    engine._unload_effects_stack()  # noqa: SLF001
    assert engine._music_stack is None  # noqa: SLF001
    assert engine._effects_stack is None  # noqa: SLF001


def test_effects_frame_padding_tiles_short_batches() -> None:
    """MMAudio's synchformer needs >= 16 sync frames; short renders pad up.

    Whole-batch tiling mirrors the retired BatchPadToMin custom node: a
    2-frame sequence interpolates to 5 frames, which would crash
    encode_video_with_sync with an empty segment list.
    """
    frames = [object(), object(), object(), object(), object()]
    padded = _pad_frames_to_minimum(frames, 17)
    assert len(padded) == 20
    assert padded[:5] == frames
    assert padded[5:] == frames * 3


def test_effects_frame_padding_passes_long_batches_through() -> None:
    """Batches already at the minimum are returned untouched."""
    frames = [object() for _ in range(20)]
    assert _pad_frames_to_minimum(frames, 17) == frames


def test_effects_frame_padding_rejects_empty_batches() -> None:
    """Tiling from zero frames would spin forever; fail fast instead."""
    with pytest.raises(EngineConfigurationError, match="empty frame batch"):
        _pad_frames_to_minimum([], 17)


def test_interpolation_passes_single_frame_through(tmp_path: Path) -> None:
    """One frame has no pairs; it finalizes as-is (Comfy FILM behavior)."""
    engine = _engine(tmp_path)
    frame = PillowImage.new("RGB", (16, 16))
    assert engine._interpolate_frames([frame]) == [frame]  # noqa: SLF001


@given(
    batch_size=st.integers(min_value=1, max_value=40),
    minimum_frames=st.integers(min_value=1, max_value=64),
)
def test_effects_frame_padding_covers_the_minimum(batch_size: int, minimum_frames: int) -> None:
    """Tiling repeats whole batches: length lands in [minimum, minimum + batch)."""
    frames = [object() for _ in range(batch_size)]
    padded = _pad_frames_to_minimum(frames, minimum_frames)
    assert len(padded) >= max(batch_size, minimum_frames)
    assert len(padded) < max(batch_size, minimum_frames) + batch_size
    assert all(padded[index] is frames[index % batch_size] for index in range(len(padded)))


def test_single_frame_window_finalizes_with_stubbed_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One frame has no pairs to interpolate; the window still lands twins."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    repository = FrameRepository(tmp_path / "output")
    repository.save_next_frame(family.sequence_key, PillowImage.new("RGB", (16, 16)))

    def fake_write_silent_video(frames: list[object], destination: Path) -> None:
        assert len(frames) == 1
        destination.write_bytes(b"silent")

    def fake_render_music(
        _caption: str, _duration_seconds: float, stem_path: Path, *, seed: int
    ) -> None:
        del seed
        stem_path.write_bytes(b"music")

    def fake_render_sound_effects(
        _prompt: str,
        _negative_prompt: str | None,
        _interpolated_frames: list[object],
        _duration_seconds: float,
        stem_path: Path,
    ) -> None:
        stem_path.write_bytes(b"effects")

    def fake_mux_audio_twin(_silent_video: Path, _stem: Path, twin: Path) -> None:
        twin.write_bytes(b"twin")

    monkeypatch.setattr("zoomy.local_engine._write_silent_video", fake_write_silent_video)
    monkeypatch.setattr("zoomy.local_engine._mux_audio_twin", fake_mux_audio_twin)
    monkeypatch.setattr(engine, "_render_music", fake_render_music)
    monkeypatch.setattr(engine, "_render_sound_effects", fake_render_sound_effects)
    window = SegmentWindow(index=0, skip_first_images=0, frame_count=1)
    soundtrack = engine._finalize_window(  # noqa: SLF001
        family, family.sequence_key, window, extension=False
    )
    assert soundtrack.music_video_path.is_file()
    assert soundtrack.music_stem_path.is_file()
    assert soundtrack.sound_effect_stem_path.is_file()


def test_failed_window_removes_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing music stage leaves no seg000 files; the error is unchanged."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    output_directory = tmp_path / "output"
    repository = FrameRepository(output_directory)
    repository.save_next_frame(family.sequence_key, PillowImage.new("RGB", (16, 16)))

    def fake_write_silent_video(frames: list[object], destination: Path) -> None:
        del frames
        destination.write_bytes(b"silent")

    def broken_render_music(
        _caption: str, _duration_seconds: float, stem_path: Path, *, seed: int
    ) -> None:
        del seed
        stem_path.write_bytes(b"partial-music")
        raise EngineExecutionError("music", "boiler exploded")

    monkeypatch.setattr("zoomy.local_engine._write_silent_video", fake_write_silent_video)
    monkeypatch.setattr(engine, "_render_music", broken_render_music)
    window = SegmentWindow(index=0, skip_first_images=0, frame_count=1)
    with pytest.raises(EngineExecutionError, match="boiler exploded"):
        engine._finalize_window(  # noqa: SLF001
            family, family.sequence_key, window, extension=False
        )
    leftovers = [
        path for path in output_directory.rglob("*") if "seg000" in path.name and path.is_file()
    ]
    assert leftovers == []


def _stub_window_heavy_stages(monkeypatch: pytest.MonkeyPatch, engine: LocalEngine) -> None:
    """Replace ffmpeg/music/SFX/mux with file-touching fakes for window tests."""

    def fake_write_silent_video(frames: list[object], destination: Path) -> None:
        assert len(frames) == 1
        destination.write_bytes(b"silent")

    def fake_render_music(
        _caption: str, _duration_seconds: float, stem_path: Path, *, seed: int
    ) -> None:
        del seed
        stem_path.write_bytes(b"music")

    def fake_render_sound_effects(
        _prompt: str,
        _negative_prompt: str | None,
        _interpolated_frames: list[object],
        _duration_seconds: float,
        stem_path: Path,
    ) -> None:
        stem_path.write_bytes(b"effects")

    def fake_mux_audio_twin(_silent_video: Path, _stem: Path, twin: Path) -> None:
        twin.write_bytes(b"twin")

    monkeypatch.setattr("zoomy.local_engine._write_silent_video", fake_write_silent_video)
    monkeypatch.setattr("zoomy.local_engine._mux_audio_twin", fake_mux_audio_twin)
    monkeypatch.setattr(engine, "_render_music", fake_render_music)
    monkeypatch.setattr(engine, "_render_sound_effects", fake_render_sound_effects)


class _CloseTrackingImage:
    """Stand-in for an opened PNG: records close, converts to a blank frame."""

    def __init__(self) -> None:
        self.close_calls = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close_calls += 1

    def convert(self, _mode: str) -> PillowImage.Image:
        return PillowImage.new("RGB", (16, 16))


def _record_image_opens(monkeypatch: pytest.MonkeyPatch) -> list[_CloseTrackingImage]:
    """Replace ``PIL.Image.open`` with a close-recording fake; return opens."""
    instances: list[_CloseTrackingImage] = []

    def tracking_open(_path: object, *_args: object, **_kwargs: object) -> _CloseTrackingImage:
        instance = _CloseTrackingImage()
        instances.append(instance)
        return instance

    monkeypatch.setattr(PillowImage, "open", tracking_open)
    return instances


def test_window_render_closes_source_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every opened source PNG is closed after its pixels are converted."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    repository = FrameRepository(tmp_path / "output")
    repository.save_next_frame(family.sequence_key, PillowImage.new("RGB", (16, 16)))
    _stub_window_heavy_stages(monkeypatch, engine)
    opened = _record_image_opens(monkeypatch)
    window = SegmentWindow(index=0, skip_first_images=0, frame_count=1)
    engine._finalize_window(family, family.sequence_key, window, extension=False)  # noqa: SLF001
    assert len(opened) == 1
    assert [image.close_calls for image in opened] == [1]


def test_load_source_image_closes_seed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold- and warm-start loads close their PNGs after converting."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    seed_path = tmp_path / "seed" / family.cold_start_image
    PillowImage.new("RGB", (16, 16)).save(seed_path)
    FrameRepository(tmp_path / "output").save_next_frame(
        family.sequence_key, PillowImage.new("RGB", (16, 16))
    )
    opened = _record_image_opens(monkeypatch)
    for frame_count in (0, 1):
        request = FrameRenderRequest(
            family=family,
            prompt="a prompt",
            negative_prompt=None,
            frame_count=frame_count,
            lora_selections=(),
            seed=1,
        )
        image = engine._load_source_image(request)  # noqa: SLF001
        del image
    assert len(opened) == 2
    assert [image.close_calls for image in opened] == [1, 1]


class _BrokenTransformer:
    """Single-file DiT load fails (truncated weights, dtype mismatch)."""

    @staticmethod
    def from_single_file(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("truncated tensor file")


class _WorkingTransformer:
    """Single-file DiT load succeeds without touching any weights."""

    @staticmethod
    def from_single_file(*_args: object, **_kwargs: object) -> object:
        return object()


class _BrokenAutoencoder:
    """Local VAE load fails (the known 32ch-vs-8ch config gap)."""

    @staticmethod
    def from_single_file(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected channel count")


class _StubPipeline:
    """Stand-in for a loaded diffusers pipeline."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> _StubPipeline:
        return cls()

    def enable_sequential_cpu_offload(self) -> None:
        """Accept the offload call the loaders always make."""


def _stub_hub_download(*_args: object, **_kwargs: object) -> str:
    """Pretend the transformer config already sits in the hub cache."""
    return "stub-transformer-config.json"


def _stub_heavy_module(**attributes: object) -> ModuleType:
    """Build a fake heavy module (torch/diffusers/hub) for loader tests."""
    return cast("ModuleType", SimpleNamespace(**attributes))


def test_broken_local_transformer_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt local DiT errors naming the file; weights are never swapped."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    diffusion_path = tmp_path / "models" / "diffusion_models" / family.base_model_file
    diffusion_path.parent.mkdir(parents=True, exist_ok=True)
    diffusion_path.write_bytes(b"truncated")
    monkeypatch.setitem(sys.modules, "torch", _stub_heavy_module(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        _stub_heavy_module(
            ErnieImagePipeline=_StubPipeline,
            ErnieImageTransformer2DModel=_BrokenTransformer,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        _stub_heavy_module(hf_hub_download=_stub_hub_download),
    )
    with pytest.raises(EngineConfigurationError, match=r"ernie-image-turbo\.safetensors"):
        engine._load_ernie_pipeline(family)  # noqa: SLF001


def test_broken_local_autoencoder_warns_and_uses_official(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad local VAE warns naming the file, then falls back to official."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    diffusion_path = tmp_path / "models" / "diffusion_models" / family.base_model_file
    diffusion_path.parent.mkdir(parents=True, exist_ok=True)
    diffusion_path.write_bytes(b"weights")
    autoencoder_path = tmp_path / "models" / "vae" / family.autoencoder_file
    autoencoder_path.parent.mkdir(parents=True, exist_ok=True)
    autoencoder_path.write_bytes(b"wrong channels")
    monkeypatch.setitem(sys.modules, "torch", _stub_heavy_module(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        _stub_heavy_module(
            AutoencoderKL=_BrokenAutoencoder,
            ZImageImg2ImgPipeline=_StubPipeline,
            ZImageTransformer2DModel=_WorkingTransformer,
        ),
    )
    with pytest.warns(UserWarning, match=r"ae\.safetensors"):
        pipeline = engine._load_z_pipeline(family)  # noqa: SLF001
    assert isinstance(pipeline, _StubPipeline)


class _FailingEncodeStdin:
    """Pretend ffmpeg died mid-stream: writes fail, close records itself."""

    def __init__(self) -> None:
        self.closed = False

    def write(self, _data: bytes) -> int:
        raise OSError("broken pipe")

    def close(self) -> None:
        self.closed = True


class _StubEncodeProcess:
    """Record terminate/wait/kill so tests prove failed encodes are reaped."""

    def __init__(self, *, hang_on_wait: bool = False) -> None:
        self.stdin: _FailingEncodeStdin | None = _FailingEncodeStdin()
        self.returncode: int | None = None
        self.terminated = False
        self.waited = False
        self.killed = False
        self._hang_on_wait = hang_on_wait

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self._hang_on_wait:
            self._hang_on_wait = False
            raise subprocess.TimeoutExpired("ffmpeg", timeout or 0)
        self.returncode = -15
        return -15


def _stub_encode_subprocess(
    processes: list[_StubEncodeProcess], *, hang_on_wait: bool = False
) -> SimpleNamespace:
    """Return a subprocess stand-in recording every spawned encode child."""

    def fake_popen(command: object, *, stdin: object = None) -> _StubEncodeProcess:
        del command, stdin
        process = _StubEncodeProcess(hang_on_wait=hang_on_wait)
        processes.append(process)
        return process

    return SimpleNamespace(Popen=fake_popen, PIPE=-1, TimeoutExpired=subprocess.TimeoutExpired)


def test_failed_encode_terminates_the_child_and_closes_the_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-stream stdin failure must reap the ffmpeg child, then raise."""
    processes: list[_StubEncodeProcess] = []
    monkeypatch.setattr("zoomy.local_engine.subprocess", _stub_encode_subprocess(processes))
    frame = PillowImage.new("RGB", (16, 16))
    with pytest.raises(AssemblyError, match="Silent video encode failed"):
        _write_silent_video([frame], tmp_path / "silent.mp4")
    (process,) = processes
    assert process.stdin is not None
    assert process.stdin.closed
    assert process.terminated
    assert process.waited


def test_hung_encode_is_killed_after_terminate_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child ignoring terminate must be killed, never left behind."""
    processes: list[_StubEncodeProcess] = []
    monkeypatch.setattr(
        "zoomy.local_engine.subprocess",
        _stub_encode_subprocess(processes, hang_on_wait=True),
    )
    frame = PillowImage.new("RGB", (16, 16))
    with pytest.raises(AssemblyError, match="Silent video encode failed"):
        _write_silent_video([frame], tmp_path / "silent.mp4")
    (process,) = processes
    assert process.terminated
    assert process.killed
    assert process.waited


def _out_of_memory_error() -> RuntimeError:
    """Build a CUDA OOM failure without importing torch (slim-image safe)."""
    return RuntimeError("CUDA out of memory. Tried to allocate 1.2 GiB.")


def test_stage_runner_retries_out_of_memory_after_evicting(tmp_path: Path) -> None:
    """An OOM evicts resident stacks and retries the stage."""
    del tmp_path
    evictions: list[str] = []
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise _out_of_memory_error()
        return "rendered"

    result = run_stage_with_retries(
        "frame", flaky, evict_resident_stacks=lambda: evictions.append("evicted")
    )
    assert result == "rendered"
    assert len(attempts) == 3
    assert evictions == ["evicted", "evicted"]


def test_stage_runner_raises_non_memory_errors_immediately() -> None:
    """Ordinary stage failures never trigger an eviction retry."""
    evictions: list[str] = []

    def broken() -> str:
        raise EngineExecutionError("frame", "broken weights")

    with pytest.raises(EngineExecutionError, match="broken weights"):
        run_stage_with_retries(
            "frame", broken, evict_resident_stacks=lambda: evictions.append("evicted")
        )
    assert evictions == []


interrupt_or_config_failure = pytest.mark.parametrize(
    "failure",
    [RenderInterruptedError("stop now"), EngineConfigurationError("bad size")],
)


@interrupt_or_config_failure
def test_stage_runner_never_retries_interrupts_or_bad_config(failure: Exception) -> None:
    """Interrupts and configuration errors propagate on the first attempt."""
    attempts = 0

    def doomed() -> str:
        nonlocal attempts
        attempts += 1
        raise failure

    with pytest.raises(type(failure)):
        run_stage_with_retries("frame", doomed, evict_resident_stacks=lambda: None)
    assert attempts == 1


def test_stage_runner_exhausts_the_attempt_budget() -> None:
    """Persistent OOM raises the last failure after the final attempt."""
    attempts: list[int] = []

    def always_out_of_memory() -> str:
        attempts.append(len(attempts))
        raise _out_of_memory_error()

    with pytest.raises(RuntimeError, match="out of memory"):
        run_stage_with_retries(
            "music", always_out_of_memory, max_attempts=2, evict_resident_stacks=lambda: None
        )
    assert attempts == [0, 1]


def test_stage_runner_rejects_an_empty_attempt_budget() -> None:
    """Zero attempts would skip the stage silently, so it is refused."""
    with pytest.raises(ValueError, match="at least 1"):
        run_stage_with_retries(
            "frame", lambda: "never", max_attempts=0, evict_resident_stacks=lambda: None
        )


def test_render_frame_rejects_misaligned_sizes(tmp_path: Path) -> None:
    """Frame sizes must be positive multiples of 16 for the autoencoders."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    for width, height in ((512, 511), (0, 512), (-16, 512), (513, 512)):
        request = FrameRenderRequest(
            family=family,
            prompt="a prompt",
            negative_prompt="a negative",
            frame_count=0,
            lora_selections=(),
            seed=1,
            frame_width=width,
            frame_height=height,
        )
        with pytest.raises(EngineConfigurationError, match="multiple of 16"):
            engine.render_frame(request)


@pytest.mark.parametrize("bad_denoise", [0.0, 0.09, 0.96, 1.0, -0.5])
def test_render_frame_rejects_out_of_range_denoise(tmp_path: Path, bad_denoise: float) -> None:
    """Denoise outside 0.10..0.95 fails before any model loads."""
    engine = _engine(tmp_path)
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FrameRenderRequest(
        family=family,
        prompt="a prompt",
        negative_prompt="a negative",
        frame_count=0,
        lora_selections=(),
        seed=1,
        denoise_strength=bad_denoise,
    )
    with pytest.raises(EngineConfigurationError, match="Denoise strength"):
        engine.render_frame(request)


@given(denoise=st.floats(min_value=0.10, max_value=0.95))
def test_denoise_validation_accepts_the_full_span(denoise: float) -> None:
    """Every in-range denoise passes validation (render still needs a seed)."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    request = FrameRenderRequest(
        family=family,
        prompt="a prompt",
        negative_prompt="a negative",
        frame_count=0,
        lora_selections=(),
        seed=1,
        denoise_strength=denoise,
    )
    LocalEngine._validate_denoise_strength(request)  # noqa: SLF001

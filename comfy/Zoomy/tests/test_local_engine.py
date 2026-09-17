"""Tests for the in-process engine above its lazy heavy imports.

Every behavior here runs above the lazy heavy imports: constructor
validation, readiness, request validation, interrupt bookkeeping, and memory
figures. Anything that loads a model needs the GPU image and is covered by
live verification instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image as PillowImage

from zoomy.engine_protocol import FinalizeRequest, FrameRenderRequest
from zoomy.errors import (
    EmptyFrameSequenceError,
    EngineConfigurationError,
    EngineExecutionError,
    RenderInterruptedError,
)
from zoomy.family_catalog import FAMILY_CATALOG, FamilyDefinition, find_family
from zoomy.frame_repository import FrameRepository
from zoomy.local_engine import LocalEngine, _pad_frames_to_minimum, run_stage_with_retries

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


def test_request_interrupt_sets_the_abort_flag(tmp_path: Path) -> None:
    """The interrupt flag trips and clears around public operations."""
    engine = _engine(tmp_path)
    engine.request_interrupt()
    assert engine._interrupt.is_set()  # noqa: SLF001
    family = find_family(FAMILY_CATALOG, "z_fast")
    with pytest.raises(EmptyFrameSequenceError):
        list(engine.finalize_sequence(FinalizeRequest(family=family, frame_count=0)))
    assert not engine._interrupt.is_set()  # noqa: SLF001


def test_engine_statistics_reports_memory_figures(tmp_path: Path) -> None:
    """Statistics carry system RAM and VRAM figures when a GPU is visible."""
    statistics = _engine(tmp_path).engine_statistics()
    assert statistics.system_memory_total_bytes >= 0
    assert statistics.system_memory_free_bytes >= 0
    if _cuda_visible():
        assert statistics.video_memory_free_bytes is not None
        assert statistics.video_memory_total_bytes is not None
        assert 0 <= statistics.video_memory_free_bytes <= statistics.video_memory_total_bytes
    else:
        assert statistics.video_memory_free_bytes is None
        assert statistics.video_memory_total_bytes is None


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

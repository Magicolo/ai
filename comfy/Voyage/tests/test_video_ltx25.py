"""CPU-only tests for the ltx25 worker core (DESIGN §140; backend `ltx25`).

Everything here runs without torch/ComfyUI/GPU: validators, OOM
matching, prefix accounting, profile hashing, tape round-trips, and the
Mode A graph builder (pure function — its shape is the validated Spike
A/Spike B recipe, so any drift from the experiment graphs fails here
first, not on the GPU).
"""

from __future__ import annotations

import pytest

from voyage.workers import video_ltx25
from voyage.workers.video_ltx25 import (
    build_mode_a_graph,
    build_recovery_tape,
    generation_profile_hash,
    is_oom,
    parse_recovery_tape,
    split_prefix_novel,
)
from voyage.workers.video_ltx25_validators import (
    validate_conditioning_start,
    validate_fps,
    validate_frame_count,
    validate_spatial_size,
)


def test_spatial_accepts_mode_a_commit_geometry() -> None:
    """1216x704 clears /64 (two-stage contract)."""
    validate_spatial_size(1216, 704)


def test_spatial_rejects_non_64_sizes() -> None:
    """768x432 (the ltxv pad-case) and odd heights fail /64 loudly."""
    with pytest.raises(ValueError, match="not divisible by"):
        validate_spatial_size(768, 432)
    with pytest.raises(ValueError, match="not divisible by"):
        validate_spatial_size(1216, 700)
    with pytest.raises(ValueError, match="must be positive"):
        validate_spatial_size(0, 704)


def test_frame_count_accepts_121_and_25() -> None:
    """121f windows and 25f tails satisfy 8n+1."""
    validate_frame_count(121)
    validate_frame_count(25)


def test_frame_count_rejects_non_8n_plus_1() -> None:
    """120f/100f clips fail loudly instead of mis-sampling."""
    with pytest.raises(ValueError, match="8n\\+1"):
        validate_frame_count(120)
    with pytest.raises(ValueError, match="positive"):
        validate_frame_count(0)


def test_fps_and_conditioning_start_bounds() -> None:
    """Non-positive fps and off-grid starts fail fast (issue 064 idiom)."""
    validate_fps(24)
    with pytest.raises(ValueError, match="positive"):
        validate_fps(0)
    validate_conditioning_start(0, 121)
    with pytest.raises(ValueError, match="multiple of 8"):
        validate_conditioning_start(25, 121)


def test_is_oom_matches_allocator_text_and_class_name() -> None:
    """OOM matches by class name or allocator text (video_ltxv idiom)."""
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 3.75 GiB"))
    assert is_oom(MemoryError("out of memory"))
    assert not is_oom(ValueError("LTXV frame count must be positive"))
    assert not is_oom(RuntimeError("boom happened"))


def test_split_prefix_novel_fresh_and_conditioned() -> None:
    """Fresh commits all 121; conditioned drops 25 and commits 96."""
    assert split_prefix_novel(121, 0) == (0, 121)
    assert split_prefix_novel(121, 25) == (25, 96)
    with pytest.raises(ValueError, match="out of range"):
        split_prefix_novel(121, 122)


def test_profile_hash_deterministic_and_geometry_sensitive() -> None:
    """Same profile hashes equal; any geometry change flips the hash."""
    first = generation_profile_hash(1216, 704, 24, 121, 25)
    assert first == generation_profile_hash(1216, 704, 24, 121, 25)
    assert first != generation_profile_hash(768, 448, 24, 121, 25)
    assert first != generation_profile_hash(1216, 704, 24, 73, 25)


def test_tape_build_parse_roundtrip() -> None:
    """Build → parse keeps the tail path; hashes ride when provided."""
    tape = build_recovery_tape(
        source_segment_id="000007",
        conditioning_tail_path="/tmp/tail.mp4",
        conditioning_tail_sha256="abc123",
        prompts=["orchard", "orchard at dusk"],
        seeds=[202, 303],
        width=1216,
        height=704,
        fps=24,
        prompt_plan_digest="digest",
    )
    assert tape["backend"] == "ltx25"
    assert tape["state_mode"] == "reconstructable_prefix"
    assert tape["seeds"] == [202, 303]
    assert tape["last_prompt"] == "orchard at dusk"
    parsed = parse_recovery_tape(tape)
    assert parsed["conditioning_tail_path"] == "/tmp/tail.mp4"
    assert parsed["conditioning_tail_sha256"] == "abc123"


def test_tape_rejects_foreign_backend() -> None:
    """An ltxv tape never resumes an ltx25 session (numerics differ)."""
    with pytest.raises(ValueError, match="foreign format"):
        parse_recovery_tape({"backend": "ltxv", "state_mode": "reconstructable_prefix"})
    with pytest.raises(ValueError, match="no conditioning tail path"):
        parse_recovery_tape({"backend": "ltx25", "state_mode": "reconstructable_prefix"})


def test_fresh_graph_matches_spike_a_recipe() -> None:
    """Fresh graph: 29 nodes, stage wiring identical to s0_121_B.json."""
    graph = build_mode_a_graph(prompt="orchard", seed=202, save_prefix="seg000001-b0")
    assert len(graph) == 29
    dit = graph["1"]
    assert dit["class_type"] == "UnetLoaderGGUF"
    assert dit["inputs"]["unet_name"] == "LTX-2.5-Distilled-Q3_K_M.gguf"
    assert graph["2"]["inputs"]["clip_name"] == "gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf"
    assert graph["3"]["inputs"]["text"] == "orchard"
    assert graph["8"]["inputs"] == {"width": 608, "height": 352, "length": 121, "batch_size": 1}
    assert graph["10"]["inputs"]["video_latent"] == ["8", 0]
    assert graph["13"]["inputs"] == {"sampler_name": "euler_ancestral"}
    assert graph["14"]["inputs"]["sigmas"].startswith("1.0, 0.99375")
    assert graph["22"]["inputs"] == {"sampler_name": "euler"}
    assert graph["23"]["inputs"]["sigmas"] == "0.85, 0.7250, 0.4219, 0.0"
    assert graph["26"]["inputs"]["tile_size"] == 512
    assert graph["28"]["inputs"]["filename_prefix"] == "seg000001-b0/frames"
    assert graph["29"]["inputs"]["filename_prefix"] == "seg000001-b0/audio"


def test_chained_graph_adds_frozen_prefix() -> None:
    """Chained graph: 25 LoadImage + batch + strength-1.0 Inplace rewire."""
    prefix = [f"prefix_{index:02d}.png" for index in range(25)]
    graph = build_mode_a_graph(
        prompt="orchard", seed=303, save_prefix="seg000001-b1", prefix_filenames=prefix
    )
    assert len(graph) == 29 + 25 + 2
    assert graph["30"]["inputs"] == {"image": "prefix_00.png"}
    assert graph["54"]["inputs"] == {"image": "prefix_24.png"}
    batch_inputs = graph["55"]["inputs"]
    assert graph["55"]["class_type"] == "BatchImagesNode"
    assert batch_inputs["images.image0"] == ["30", 0]
    assert batch_inputs["images.image24"] == ["54", 0]
    inplace = graph["56"]["inputs"]
    assert inplace["strength"] == 1.0
    assert inplace["bypass"] is False
    assert graph["10"]["inputs"]["video_latent"] == ["56", 0]


def test_chained_graph_rejects_wrong_prefix_count() -> None:
    """24 or 26 prefix frames fail at build time, never mid-render."""
    with pytest.raises(ValueError, match="needs 25 frames"):
        build_mode_a_graph(
            prompt="orchard",
            seed=303,
            save_prefix="seg000001-b1",
            prefix_filenames=[f"prefix_{index:02d}.png" for index in range(24)],
        )


def test_worker_module_reexports_validator_facades() -> None:
    """`video_ltx25.<name>` importers hold (video_ltxv facade pattern)."""
    assert video_ltx25.SPATIAL_GRANULARITY == 32
    assert video_ltx25.TAPE_FILENAME == "recovery.pt"
    assert video_ltx25.RECOVERY_PROFILE == "ltx25"

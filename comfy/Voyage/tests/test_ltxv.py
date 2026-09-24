"""LTXV backend tests: registry verify, pad helper, worker op surface.

All run in the slim container (no torch/GPU): the worker module keeps
heavy imports inside handlers, so importing it here is safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voyage import model_registry
from voyage.config import VideoConfig, default_config_toml, load_config
from voyage.model_registry import verify_ltxv_models
from voyage.supervisor import Supervisor, video_worker_module
from voyage.workers import video_ltxv
from voyage.workers.loop import serve


def test_padded_size_rounds_up_to_32() -> None:
    assert video_ltxv.padded_size(512) == 512
    assert video_ltxv.padded_size(768) == 768
    assert video_ltxv.padded_size(500) == 512
    assert video_ltxv.padded_size(1) == 32
    assert video_ltxv.padded_size(24, 8) == 24
    assert video_ltxv.padded_size(25, 8) == 32


def test_probe_schedules_match_slice1() -> None:
    """Distilled schedules are baked constants, not drifted tunables."""
    assert video_ltxv.FIRST_PASS["guidance_scale"] == 1
    assert video_ltxv.FIRST_PASS["stg_scale"] == 0
    assert video_ltxv.FIRST_PASS["skip_block_list"] == [42]
    assert video_ltxv.SECOND_PASS["timesteps"] == [0.9094, 0.725, 0.4219]
    assert video_ltxv.SEGMENT_TARGET_FRAMES == 121
    assert video_ltxv.CONDITIONING_TAIL_FRAMES == 25
    assert video_ltxv.COMMITTED_NOVEL_FRAMES == 96
    assert video_ltxv.RECOVERY_PROFILE == "ltxv"


def test_worker_module_imports_without_heavy_deps() -> None:
    """Top-level worker import must stay torch-free (slim-safe)."""
    assert video_ltxv._SESSION is None
    assert video_ltxv.DIT_FILENAME == "ltxv-2b-0.9.8-distilled.safetensors"


def test_worker_serve_map_covers_protocol() -> None:
    """The LTXV worker speaks every op the supervisor may send."""
    import inspect

    source = inspect.getsource(video_ltxv.main)
    for op in (
        "init",
        "health",
        "generate_blocks",
        "benchmark",
        "evict_gpu",
        "rebuild",
        "checkpoint",
        "resume",
        "shutdown",
    ):
        assert f'"{op}"' in source
    assert callable(serve)


def test_verify_ltxv_missing_dir_fails(tmp_path: Path) -> None:
    ok, message = verify_ltxv_models(tmp_path / "models")
    assert not ok
    assert "missing" in message


def test_verify_ltxv_fixture_passes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    ltxv_dir = models / model_registry.LTXV_SUBDIR
    ltxv_dir.mkdir(parents=True)
    (ltxv_dir / model_registry.LTXV_DIT_FILE).write_bytes(
        b"x" * (model_registry.LTXV_DIT_MIN_BYTES + 8)
    )
    (ltxv_dir / model_registry.LTXV_UPSC_FILE).write_bytes(
        b"x" * (model_registry.LTXV_UPSC_MIN_BYTES + 8)
    )
    te_sub = models / model_registry.LTXV_TE_SUBDIR / "tokenizer"
    te_sub.mkdir(parents=True)
    (te_sub / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    te_enc = models / model_registry.LTXV_TE_SUBDIR / "text_encoder"
    te_enc.mkdir(parents=True)
    (te_enc / "config.json").write_text("{}", encoding="utf-8")
    ok, message = verify_ltxv_models(models)
    assert ok, message
    assert "ltxv-2b OK" in message


def test_layout_includes_ltxv_dir(tmp_path: Path) -> None:
    layout = model_registry.models_dir_layout(tmp_path)
    assert layout["ltxv_dir"] == str(tmp_path / model_registry.LTXV_SUBDIR)


def _ltxv_config(tmp_path: Path):  # type: ignore[no-untyped-def]
    from voyage import paths

    run_dir = tmp_path / "run"
    (run_dir / paths.LOGS_DIRNAME).mkdir(parents=True)
    (run_dir / paths.CONFIG_FILENAME).write_text(
        default_config_toml("ltxv-route", "probe", 11), encoding="utf-8"
    )
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    video = VideoConfig(
        **{
            **config.video.model_dump(),
            "backend": "ltxv",
            "device": "cuda:0",
            "width": 768,
            "height": 512,
        }
    )
    config.video = video
    return run_dir, config


def test_ltxv_module_routing() -> None:
    assert video_worker_module("ltxv") == "voyage.workers.video_ltxv"


def test_ltxv_supervisor_init_payload(tmp_path: Path) -> None:
    run_dir, config = _ltxv_config(tmp_path)
    supervisor = Supervisor(run_dir, config)
    assert supervisor._video._module == "voyage.workers.video_ltxv"
    assert supervisor._video._init_payload == {
        "models_dir": config.video.models_dir,
        "device": "cuda:0",
    }


def test_spatial_granularity_accepts_native_preset() -> None:
    video_ltxv.validate_spatial_size(768, 512)
    video_ltxv.validate_spatial_size(1216, 704)


def test_spatial_granularity_rejects_spec_text_size() -> None:
    """768x432 (§5.3 draft text) is not /32-aligned — it would pad to 448."""
    with pytest.raises(ValueError, match="not divisible by 32"):
        video_ltxv.validate_spatial_size(768, 432)


def test_frame_count_accepts_upstream_valid_counts() -> None:
    for valid in (9, 17, 25, 33, 49, 97, 121, 257):
        video_ltxv.validate_frame_count(valid)


def test_frame_count_rejects_non_conforming_counts() -> None:
    for invalid in (24, 48, 96, 100, 120):
        with pytest.raises(ValueError, match="8n\\+1"):
            video_ltxv.validate_frame_count(invalid)


def test_conditioning_start_must_be_multiple_of_eight() -> None:
    video_ltxv.validate_conditioning_start(0, 121)
    video_ltxv.validate_conditioning_start(8, 121)
    with pytest.raises(ValueError, match="multiple of 8"):
        video_ltxv.validate_conditioning_start(1, 121)
    with pytest.raises(ValueError, match="out of range"):
        video_ltxv.validate_conditioning_start(121, 121)


def test_prefix_discard_accounting() -> None:
    """121-frame clip minus the 25-frame prefix commits 96 novel frames."""
    discarded, novel = video_ltxv.split_prefix_novel(121, 25)
    assert (discarded, novel) == (25, 96)
    discarded_fresh, novel_fresh = video_ltxv.split_prefix_novel(121, 0)
    assert (discarded_fresh, novel_fresh) == (0, 121)
    with pytest.raises(ValueError, match="out of range"):
        video_ltxv.split_prefix_novel(121, 122)


def test_prompt_plan_hash_is_deterministic() -> None:
    first = video_ltxv.prompt_plan_hash(["amber dunes", "teal spires"])
    assert first == video_ltxv.prompt_plan_hash(["amber dunes", "teal spires"])
    assert first != video_ltxv.prompt_plan_hash(["amber dunes", "teal spires!"])


def test_recovery_tape_matches_spec_shape(tmp_path: Path) -> None:
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"tail-bytes")
    tape = video_ltxv.build_recovery_tape(
        source_segment_id="000123",
        conditioning_tail_path=str(tail),
        conditioning_tail_sha256=video_ltxv.sha256_file(tail),
        prompts=["amber dunes"],
        seeds=[123456],
        width=768,
        height=512,
        fps=24,
    )
    assert tape["backend"] == "ltxv"
    assert tape["state_mode"] == "reconstructable_prefix"
    assert tape["source_segment_id"] == "000123"
    assert tape["conditioning_tail_path"] == str(tail)
    assert tape["seed"] == 123456
    assert tape["model_revision"] == model_registry.LTXV_HF_REVISION
    assert tape["pipeline_revision"] == model_registry.LTXV_COMMIT
    assert len(tape["profile_hash"]) == 64
    # Round-trips through JSON (the on-disk format) and validates.
    loaded = json.loads(json.dumps(tape))
    assert video_ltxv.parse_recovery_tape(loaded) == loaded


def test_recovery_tape_rejects_legacy_torch_format(tmp_path: Path) -> None:
    legacy = {"profile": "ltxv", "tail_png": str(tmp_path / "video_tail.png")}
    with pytest.raises(ValueError, match="pre-Stream-A"):
        video_ltxv.parse_recovery_tape(legacy)


def test_recovery_tape_rejects_missing_tail() -> None:
    tape = video_ltxv.build_recovery_tape(
        source_segment_id="000124",
        conditioning_tail_path="/nonexistent/video_tail.mp4",
        conditioning_tail_sha256="0" * 64,
        prompts=["amber dunes"],
        seeds=[7],
        width=768,
        height=512,
        fps=24,
    )
    with pytest.raises(ValueError, match="conditioning tail missing"):
        video_ltxv.parse_recovery_tape(tape)


def test_load_tape_json_rejects_torch_pickle_bytes(tmp_path: Path) -> None:
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_bytes(b"\x80\x04torch-pickle-not-json")
    with pytest.raises(ValueError, match="pre-Stream-A"):
        video_ltxv._load_tape_json(str(tape_path))

"""LTXV backend tests: registry verify, pad helper, worker op surface.

All run in the slim container (no torch/GPU): the worker module keeps
heavy imports inside handlers, so importing it here is safe.
"""

from __future__ import annotations

from pathlib import Path

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
    assert video_ltxv.NATIVE_BLOCK_FRAMES == 25
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

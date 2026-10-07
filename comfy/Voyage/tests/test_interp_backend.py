"""RIFE default interp backend: registry, worker, key, config, prune.

Phase-1/2 RIFE work made RIFE v4.25-heavy (fp16) the default interpolation
engine: ~16.8x per pair over FILM at 2048x1152 on the 4060 Ti with a clean
line-art eyeball, 0.65 GiB peak (fits the 2060 beside the llama sidecar).
This module pins that decision end to end — registry single-sourcing plus
the heavy pin, torch-free worker validation, the legacy `sha|sha` weights
key shape (no backend prefix: the interp-leg sha already forks ledger
identity on switch), the rife config default, the backend-aware model
ensure, the old-manifest migration (augment section without a backend
rendered FILM, so it stays film), the stale-interp prune, and the
`enhance_frames` dispatch. Provisioned legs (torch + heavy weights)
prove the loader and a tiny CPU run; the slim gates image covers
everything else.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

import voyage.model_registry as model_registry
import voyage.registry_records as registry_records
import voyage.registry_rife as registry_rife
from tests.conftest import initialize_run_directory
from voyage.augment import AugmentWeights
from voyage.augment_finalize import weights_key_for
from voyage.augment_sidecar import (
    AUGMENT_DIRNAME,
    CHUNKS_LEDGER_FILENAME,
    ChunkKey,
    append_chunk_record,
    load_chunk_ledger,
    prune_stale_interp_plans,
)
from voyage.config import AugmentConfig, resolve_config
from voyage.models_ensure import required_specs
from voyage.persistence import read_effective_config
from voyage.workers import augment_worker


def _torch_available() -> bool:
    """Whether real torch imports (slim gates image: no — provisioned legs skip)."""
    return importlib.util.find_spec("torch") is not None


RIFE_PIN_NAMES = (
    "RIFE_HF_REPO",
    "RIFE_HF_REVISION",
    "RIFE_SUBDIR",
    "RIFE_FILE",
    "RIFE_REPO_PATH",
    "RIFE_MIN_BYTES",
    "RIFE_LICENSE",
    "RIFE_LICENSE_URL",
    "EXPECTED_RIFE_SHA256",
)


def test_rife_pins_are_single_sourced() -> None:
    """RIFE pins live once, in registry_rife (mirrors the FILM split contract)."""
    for name in RIFE_PIN_NAMES:
        assert getattr(registry_records, name) is getattr(registry_rife, name)
        assert getattr(model_registry, name) == getattr(registry_rife, name)


def test_rife_builders_are_single_sourced() -> None:
    """RIFE record/describe helpers live once, in registry_rife."""
    assert registry_records._record_rife is registry_rife._record_rife
    assert registry_records._describe_rife is registry_rife._describe_rife
    assert model_registry._record_rife is registry_rife._record_rife
    assert model_registry._describe_rife is registry_rife._describe_rife


def test_rife_spec_row_points_at_family_builders() -> None:
    """MODEL_SPECS rife row calls the single-sourced builders."""
    spec = model_registry.MODEL_SPECS["rife"]
    assert spec.name == "rife"
    assert spec.record_builder is registry_rife._record_rife
    assert spec.success_message is registry_rife._describe_rife


def test_rife_pin_is_best_quality_heavy() -> None:
    """The pin is v4.25-heavy: sharpest and closest to FILM at zero VRAM cost.

    Issue 207: the digest below is the sha256 of the pinned-revision
    bytes (downloaded + hashed 2026-10-07); the previous value was
    truncated (63 hex) and matched nothing at the pinned revision.
    """
    assert registry_rife.RIFE_FILE == "rife_v4.25_heavy.safetensors"
    assert registry_rife.RIFE_MIN_BYTES == 78_000_000
    assert (
        registry_rife.EXPECTED_RIFE_SHA256
        == "40aa1838b91531f829caaac026f40d9d2e2f1eb12b65d1d6029a58ae4c703191"
    )


def test_validate_interp_backend_passes_both_backends() -> None:
    """The closed backend pair validates through unchanged."""
    assert augment_worker.validate_interp_backend("film") == "film"
    assert augment_worker.validate_interp_backend("rife") == "rife"


def test_validate_interp_backend_rejects_non_string() -> None:
    """Non-string backends fail loud with the offending type."""
    with pytest.raises(TypeError, match="must be a string"):
        augment_worker.validate_interp_backend(1)  # type: ignore[arg-type]


def test_validate_interp_backend_rejects_unknown() -> None:
    """Unknown backends fail loud naming the closed pair."""
    with pytest.raises(ValueError, match="unknown interp backend"):
        augment_worker.validate_interp_backend("raft")


def test_interpolate_rife_mids_rejects_non_list_moments() -> None:
    """Moments must be a list or tuple of blend times (torch-free)."""
    with pytest.raises(TypeError, match="list or tuple"):
        augment_worker.interpolate_rife_mids([], "/models/rife.safetensors", moments="0.5")  # type: ignore[arg-type]


def test_interpolate_rife_mids_rejects_empty_moments() -> None:
    """At least one blend moment is required (torch-free)."""
    with pytest.raises(ValueError, match="at least one blend moment"):
        augment_worker.interpolate_rife_mids([], "/models/rife.safetensors", moments=[])


def test_interpolate_rife_mids_rejects_out_of_range_moment() -> None:
    """Blend moments inherit the [0, 1] range check (torch-free)."""
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        augment_worker.interpolate_rife_mids([], "/models/rife.safetensors", moments=[1.5])


def test_interpolate_rife_mids_rejects_non_callable_on_pair() -> None:
    """`on_pair` must be callable or None (torch-free)."""
    with pytest.raises(TypeError, match="callable or None"):
        augment_worker.interpolate_rife_mids(
            [],
            "/models/rife.safetensors",
            moments=[0.5],
            on_pair="fired",  # type: ignore[arg-type]
        )


def test_interpolate_rife_mids_rejects_fewer_than_two_frames() -> None:
    """One frame has no adjacent pair (torch-free bare length check)."""
    with pytest.raises(ValueError, match="at least two frames"):
        augment_worker.interpolate_rife_mids(["only"], "/models/rife.safetensors", moments=[0.5])


def test_interpolate_rife_mids_missing_weights_fails_torch_free(tmp_path: Path) -> None:
    """Absent weights raise before any torch import (provisioning error)."""
    missing = tmp_path / "rife_v4.25_heavy.safetensors"
    with pytest.raises(NotImplementedError, match="RIFE weights missing"):
        augment_worker.interpolate_rife_mids(["frame-a", "frame-b"], missing, moments=[0.5])


def test_interpolate_rife_mids_takes_no_pair_batch() -> None:
    """RIFE has no flow-once factorization, so there is no pair batch knob."""
    with pytest.raises(TypeError, match="pair_batch"):
        augment_worker.interpolate_rife_mids(
            ["frame-a", "frame-b"],
            "/models/rife.safetensors",
            moments=[0.5],
            pair_batch=2,  # type: ignore[call-arg]
        )


def _find_provisioned_weight(relative_path: str) -> Path | None:
    """First existing provisioned copy of `relative_path`, or None (skip signal)."""
    candidates = [
        os.environ.get("VOYAGE_MODELS", ""),
        "/models",
        str(Path.home() / ".cache" / "voyage-models"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        found = Path(candidate) / relative_path
        try:
            if found.is_file() and found.stat().st_size > 0:
                return found
        except OSError:
            continue
    return None


def _gradient_frames(count: int) -> list[Any]:
    """Deterministic 16x16 RGB gradient frames (CPU, no PIL/numpy needed)."""
    import torch

    frames = []
    for offset in range(count):
        rows = torch.linspace(0.0, 1.0, 16).unsqueeze(1).expand(16, 16)
        tilt = (rows + offset / max(count, 1)) % 1.0
        frames.append(torch.stack([tilt, rows, 1.0 - rows]))
    return frames


def test_provisioned_rife_loads_and_runs_sane_cpu() -> None:
    """Heavy weights strict-load and render one sane, deterministic CPU mid."""
    if not _torch_available():
        pytest.skip("needs torch (run in voyage-video, not the slim gates image)")
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("needs safetensors (run in voyage-video)")
    from voyage.registry_rife import RIFE_REPO_PATH

    rife = _find_provisioned_weight(RIFE_REPO_PATH)
    if rife is None:
        pytest.skip("needs provisioned rife weights (models volume absent)")
    assert rife is not None
    import torch

    augment_worker.evict_augment_models()
    frames = _gradient_frames(2)
    fired: list[tuple[int, int]] = []
    timings: dict[str, float] = {}
    first = augment_worker.interpolate_rife_mids(
        frames,
        rife,
        moments=[0.5],
        device="cpu",
        timings=timings,
        on_pair=lambda index, total: fired.append((index, total)),
    )
    assert len(first) == 1
    mid = first[0]
    assert mid.shape == (3, 16, 16)
    assert mid.dtype == torch.float32
    assert not torch.isnan(mid).any()
    assert float(mid.min()) >= -0.01 and float(mid.max()) <= 1.01
    assert fired == [(0, 1)]
    assert set(timings) == {"load_ms", "infer_ms"}
    second = augment_worker.interpolate_rife_mids(frames, rife, moments=[0.5], device="cpu")
    assert torch.equal(first[0], second[0])
    augment_worker.evict_augment_models()


def _tiny_weights(work: Path) -> AugmentWeights:
    """Three tiny real weight files (hashable legs for key/prune tests)."""
    work.mkdir(parents=True, exist_ok=True)
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    rife = work / "rife.safetensors"
    rife.write_bytes(b"rife-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    return AugmentWeights(film=film, rife=rife, realesrgan=esrgan)


def _sha(raw: bytes) -> str:
    """Hex sha256 of in-memory bytes (mirrors the file-hash key shape)."""
    return hashlib.sha256(raw).hexdigest()


def test_weights_key_keeps_legacy_shape() -> None:
    """The key stays `sha|sha` with no backend prefix (legacy ledgers keep hitting)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        weights = _tiny_weights(Path(tmp))
        film_key = weights_key_for(weights, "film")
        assert film_key == f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
        assert film_key.count("|") == 1
        rife_key = weights_key_for(weights, "rife")
        assert rife_key == f"{_sha(b'rife-weights')}|{_sha(b'esrgan-weights')}"
        assert rife_key.count("|") == 1
        assert weights_key_for(weights) == rife_key  # rife is the default backend
        assert weights_key_for(weights, "film") == film_key  # stable per backend


def test_weights_key_misses_on_interp_leg_change(tmp_path: Path) -> None:
    """Any interp-leg change misses (the sha difference forks ledger identity)."""
    weights = _tiny_weights(tmp_path)
    key = weights_key_for(weights, "rife")
    changed = tmp_path / "other-rife.safetensors"
    changed.write_bytes(b"other-rife")
    mixed = AugmentWeights(film=weights.film, rife=changed, realesrgan=weights.realesrgan)
    assert weights_key_for(mixed, "rife") != key


def test_weights_key_missing_active_leg_fails_loud(tmp_path: Path) -> None:
    """A missing active interp leg raises (never hashes the wrong leg)."""
    weights = _tiny_weights(tmp_path)
    with pytest.raises(ValueError):
        weights_key_for(AugmentWeights(film=weights.film, rife=None, realesrgan=weights.realesrgan))
    with pytest.raises(ValueError):
        weights_key_for(
            AugmentWeights(film=None, rife=weights.rife, realesrgan=weights.realesrgan),
            "film",
        )


def test_interp_backend_defaults_to_rife(tmp_path: Path) -> None:
    """Fresh runs configure the RIFE backend (best quality at 2060-friendly VRAM)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="backend-default", seed=7)
    effective = read_effective_config(run_dir)
    assert effective.augment.interp_backend == "rife"
    assert AugmentConfig().interp_backend == "rife"


def test_resolve_config_overrides_interp_backend(tmp_path: Path) -> None:
    """An explicit backend override sticks (FILM stays available for hero renders)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="backend-override", seed=7)
    base = read_effective_config(run_dir)
    assert resolve_config(base).augment.interp_backend == "rife"
    assert resolve_config(base, interp_backend="film").augment.interp_backend == "film"


def test_ensure_requires_rife_spec_by_default(tmp_path: Path) -> None:
    """The default config provisions RIFE (not FILM) for the interp leg."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="ensure-rife", seed=7, video_backend="ltx25")
    specs = required_specs(read_effective_config(run_dir))
    names = {spec.spec for spec in specs}
    assert "rife" in names
    assert "film" not in names


def test_ensure_requires_film_spec_when_configured(tmp_path: Path) -> None:
    """A film-configured run provisions FILM (opt-out stays honored)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="ensure-film", seed=7, video_backend="ltx25")
    configured = resolve_config(read_effective_config(run_dir), interp_backend="film")
    names = {spec.spec for spec in required_specs(configured)}
    assert "film" in names
    assert "rife" not in names


def test_manifest_without_backend_migrates_to_film(tmp_path: Path) -> None:
    """Augment sections written pre-knob rendered FILM, so they stay film."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="migrate-film", seed=7)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["augment"]["interp_backend"] == "rife"
    del manifest["augment"]["interp_backend"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert read_effective_config(run_dir).augment.interp_backend == "film"


def test_manifest_without_augment_keeps_rife_default(tmp_path: Path) -> None:
    """Runs that never configured a model pass keep the current default."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="migrate-none", seed=7)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["augment"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert read_effective_config(run_dir).augment.interp_backend == "rife"


def _chunk_key(*, weights_key: str, stage_frames: int, multiplier: int) -> Any:
    """One sidecar ChunkKey with production-shaped geometry (key content is free)."""
    return ChunkKey(
        chunk_index=0,
        start_frame=0,
        source_frames=32,
        expected_frames=stage_frames,
        upscale_factor=2,
        multiplier=multiplier,
        crf=15,
        preset="veryfast",
        source_key="seg",
        weights_key=weights_key,
        out_width=2048,
        out_height=1152,
        out_fps=48,
        chunk_frames=32,
    )


def _plan_with_records(
    run_dir: Path, plan: str, *, records: list[tuple[str, str, int, int]]
) -> Path:
    """One plan dir with ledger lines: (stage, weights_key, stage_frames, multiplier)."""
    plan_dir = run_dir / AUGMENT_DIRNAME / plan
    ledger = plan_dir / CHUNKS_LEDGER_FILENAME
    for stage, weights_key, stage_frames, multiplier in records:
        append_chunk_record(
            ledger,
            _chunk_key(weights_key=weights_key, stage_frames=stage_frames, multiplier=multiplier),
            stage=stage,
            path=f"{stage}_00",
        )
    return plan_dir


def test_prune_removes_stale_interp_subdirs_by_default(tmp_path: Path) -> None:
    """Stage 1 (default) deletes stale interp subdirs but keeps the dirs."""
    weights = _tiny_weights(tmp_path / "weights")
    film_key = f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
    rife_key = f"{_sha(b'rife-weights')}|{_sha(b'esrgan-weights')}"
    run_dir = tmp_path / "run"
    stale = _plan_with_records(
        run_dir,
        "a" * 16,
        records=[("upscaled", film_key, 32, 1), ("interpolated", film_key, 63, 2)],
    )
    current = _plan_with_records(
        run_dir,
        "b" * 16,
        records=[("upscaled", rife_key, 32, 1), ("interpolated", rife_key, 63, 2)],
    )
    unknown = _plan_with_records(
        run_dir,
        "c" * 16,
        records=[("interpolated", "notasha|alsonotasha", 63, 2)],
    )
    upscale_only = _plan_with_records(run_dir, "d" * 16, records=[("upscaled", film_key, 32, 1)])
    (stale / "interpolated_00").mkdir(parents=True)
    (stale / "interpolated_00" / "frame_00001.png").write_bytes(b"stale-png")
    (stale / "upscaled_00").mkdir(parents=True)
    (stale / "upscaled_00" / "frame_00001.png").write_bytes(b"kept-png")
    pruned = prune_stale_interp_plans(run_dir, interp_backend="rife", weights=weights)
    assert pruned == 1
    assert stale.exists()
    assert not (stale / "interpolated_00").exists()
    assert (stale / "upscaled_00").exists()
    assert (stale / CHUNKS_LEDGER_FILENAME).exists()
    assert current.exists()
    assert unknown.exists()
    assert upscale_only.exists()


def test_prune_strips_stale_interp_records_but_keeps_upscaled(tmp_path: Path) -> None:
    """Stage 1 strips orphaned interp ledger records; upscaled donors survive.

    Regression: the prune deleted `interpolated_*` outputs but kept the
    ledger, so the next restart's sidecar validator reported `ledgered
    but output missing` and generate aborted before the pollers that
    would heal the gap ever ran (kaolin, post-RIFE-switch restart).
    """
    weights = _tiny_weights(tmp_path / "weights")
    film_key = f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
    run_dir = tmp_path / "run"
    stale = _plan_with_records(
        run_dir,
        "a" * 16,
        records=[("upscaled", film_key, 32, 1), ("interpolated", film_key, 63, 2)],
    )
    (stale / "interpolated_00").mkdir(parents=True)
    (stale / "interpolated_00" / "frame_00001.png").write_bytes(b"stale-png")
    (stale / "upscaled_00").mkdir(parents=True)
    (stale / "upscaled_00" / "frame_00001.png").write_bytes(b"kept-png")
    assert prune_stale_interp_plans(run_dir, interp_backend="rife", weights=weights) == 1
    stages = [record.get("stage") for record in load_chunk_ledger(stale / CHUNKS_LEDGER_FILENAME)]
    assert stages == ["upscaled"]
    assert not (stale / "interpolated_00").exists()
    assert (stale / "upscaled_00" / "frame_00001.png").read_bytes() == b"kept-png"


def test_prune_heals_dangling_ledger_without_outputs(tmp_path: Path) -> None:
    """A previous prune's dangling ledger heals even with outputs already gone."""
    weights = _tiny_weights(tmp_path / "weights")
    film_key = f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
    run_dir = tmp_path / "run"
    stale = _plan_with_records(
        run_dir,
        "a" * 16,
        records=[("upscaled", film_key, 32, 1), ("interpolated", film_key, 63, 2)],
    )
    # No interpolated_00 subdir on disk: outputs already pruned, ledger dangling.
    assert prune_stale_interp_plans(run_dir, interp_backend="rife", weights=weights) == 1
    stages = [record.get("stage") for record in load_chunk_ledger(stale / CHUNKS_LEDGER_FILENAME)]
    assert stages == ["upscaled"]
    # Second pass is a clean no-op: no interpolated records left to match on.
    assert prune_stale_interp_plans(run_dir, interp_backend="rife", weights=weights) == 0


def test_healed_plan_passes_sidecar_consistency(tmp_path: Path) -> None:
    """After prune, the restart validator reports no findings on the healed dir."""
    from voyage.cli_validate import _check_sidecar_plan_consistency

    weights = _tiny_weights(tmp_path / "weights")
    film_key = f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
    run_dir = tmp_path / "run"
    stale = _plan_with_records(
        run_dir,
        "a" * 16,
        records=[("upscaled", film_key, 32, 1), ("interpolated", film_key, 63, 2)],
    )
    assert _check_sidecar_plan_consistency(run_dir) != []
    assert prune_stale_interp_plans(run_dir, interp_backend="rife", weights=weights) == 1
    # The surviving upscaled record still needs its pixels for full silence;
    # the validator's remaining note (if any) must never be an interpolated one.
    assert all(
        "stage interpolated" not in finding for finding in _check_sidecar_plan_consistency(run_dir)
    )
    _ = stale


def test_prune_whole_dir_removes_stale_dirs(tmp_path: Path) -> None:
    """Stage 2 (pre-publish) deletes whole known-other-backend plan dirs."""
    weights = _tiny_weights(tmp_path / "weights")
    film_key = f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
    rife_key = f"{_sha(b'rife-weights')}|{_sha(b'esrgan-weights')}"
    run_dir = tmp_path / "run"
    stale = _plan_with_records(
        run_dir,
        "a" * 16,
        records=[("upscaled", film_key, 32, 1), ("interpolated", film_key, 63, 2)],
    )
    current = _plan_with_records(
        run_dir,
        "b" * 16,
        records=[("upscaled", rife_key, 32, 1), ("interpolated", rife_key, 63, 2)],
    )
    pruned = prune_stale_interp_plans(
        run_dir, interp_backend="rife", weights=weights, whole_dir=True
    )
    assert pruned == 1
    assert not stale.exists()
    assert current.exists()


def test_prune_skips_when_other_leg_unresolvable(tmp_path: Path) -> None:
    """Without the other leg's file the prune cannot identify stale output: keep all."""
    weights = _tiny_weights(tmp_path / "weights")
    film_key = f"{_sha(b'film-weights')}|{_sha(b'esrgan-weights')}"
    run_dir = tmp_path / "run"
    stale = _plan_with_records(
        run_dir,
        "a" * 16,
        records=[("interpolated", film_key, 63, 2)],
    )
    broken = AugmentWeights(film=None, rife=weights.rife, realesrgan=weights.realesrgan)
    assert prune_stale_interp_plans(run_dir, interp_backend="rife", weights=broken) == 0
    assert stale.exists()


def test_prune_validates_inputs() -> None:
    """Bad backends and non-path run dirs fail loud (never prune blind)."""
    with pytest.raises(ValueError, match="must be 'film' or 'rife'"):
        prune_stale_interp_plans(Path("/tmp"), interp_backend="raft", weights=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="run_dir"):
        prune_stale_interp_plans("/tmp", interp_backend="rife", weights=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="whole_dir"):
        prune_stale_interp_plans(
            Path("/tmp"),
            interp_backend="rife",
            weights=None,
            whole_dir="yes",  # type: ignore[arg-type]
        )


def test_enhance_frames_dispatches_rife_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default backend routes interp through RIFE, never FILM."""
    from voyage.augment import enhance_frames

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _fake_upscale(*args: Any, **kwargs: Any) -> list[Any]:
        frames = args[0] if args else kwargs["frames"]
        return list(frames)

    def _fake_rife(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return ["m0", "m1"]

    def _fake_film(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("FILM must not run on the default backend")

    monkeypatch.setattr(augment_worker, "upscale_frames", _fake_upscale)
    monkeypatch.setattr(augment_worker, "interpolate_rife_mids", _fake_rife)
    monkeypatch.setattr(augment_worker, "interpolate_mids", _fake_film)
    weights = AugmentWeights(film=Path("film"), rife=Path("rife"), realesrgan=Path("esrgan"))
    assert enhance_frames(["f0", "f1", "f2"], weights, device="cpu", multiplier=2) == [
        "f0",
        "m0",
        "f1",
        "m1",
        "f2",
    ]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert weights.rife in list(args) + list(kwargs.values())


def test_enhance_frames_dispatches_film_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The film opt-out routes interp through FILM, never RIFE."""
    from voyage.augment import enhance_frames

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _fake_upscale(*args: Any, **kwargs: Any) -> list[Any]:
        frames = args[0] if args else kwargs["frames"]
        return list(frames)

    def _fake_film(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return ["m0", "m1"]

    def _fake_rife(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("RIFE must not run on the film backend")

    monkeypatch.setattr(augment_worker, "upscale_frames", _fake_upscale)
    monkeypatch.setattr(augment_worker, "interpolate_rife_mids", _fake_rife)
    monkeypatch.setattr(augment_worker, "interpolate_mids", _fake_film)
    weights = AugmentWeights(film=Path("film"), rife=Path("rife"), realesrgan=Path("esrgan"))
    assert enhance_frames(
        ["f0", "f1", "f2"], weights, device="cpu", interp_backend="film", multiplier=2
    ) == [
        "f0",
        "m0",
        "f1",
        "m1",
        "f2",
    ]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert weights.film in list(args) + list(kwargs.values())

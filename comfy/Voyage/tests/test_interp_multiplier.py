"""`interp_multiplier` knob: upscale-only finalize (multiplier 1) vs 4x FILM (4).

TDD for the 5s-ltxv run: the user wants the SRVGG upscale without the
FILM interpolation. `multiplier = 1` means each chunk keeps its frame
count (`(n-1)*1+1 = n`) and the interp poller passes frames through, so
the durable path yields an upscaled same-fps intermediate. The knob
rides `AugmentConfig` → TOML → CLI (`generate`/`finalize`) →
`finalize_run` → both model-pass entries, and the durable pass records
per-phase timings into an optional out-param (the 5s run mines these
for its elapsed-time report).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from voyage.augment import AugmentWeights
from voyage.persistence import read_effective_config


def test_augment_config_interp_multiplier_default_and_validation() -> None:
    """Default 4 preserves current behavior; <1 fails loud (DESIGN §56)."""
    from voyage.config import AugmentConfig

    assert AugmentConfig().interp_multiplier == 4
    assert AugmentConfig(interp_multiplier=1).interp_multiplier == 1
    with pytest.raises(ValidationError):
        AugmentConfig(interp_multiplier=0)
    with pytest.raises(ValidationError):
        AugmentConfig(interp_multiplier=-2)


def test_resolve_config_applies_interp_multiplier(tmp_path: Path) -> None:
    """CLI/TOML override lands on the effective config (DESIGN §56)."""
    from tests.conftest import initialize_run_directory
    from voyage.config import resolve_config

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="mult", style="s")
    config, _ = read_effective_config(run_dir)
    assert config.augment.interp_multiplier == 4
    resolved = resolve_config(config, interp_multiplier=1)
    assert resolved.augment.interp_multiplier == 1


def test_augment_overrides_passes_interp_multiplier() -> None:
    """Flag threading: provided value passes, absent stays unset (DESIGN §56)."""
    from voyage.cli_core import _augment_overrides

    args = argparse.Namespace(
        no_augment=False,
        min_fps=None,
        min_resolution=None,
        use_model_pass=None,
        interp_multiplier=1,
    )
    assert _augment_overrides(args)["interp_multiplier"] == 1
    args_blank = argparse.Namespace()
    assert "interp_multiplier" not in _augment_overrides(args_blank)


def test_finalize_options_carries_interp_multiplier() -> None:
    """FinalizeOptions default 4; 0 rejected before any media work (DESIGN §56)."""
    from voyage.media import FinalizeOptions

    assert FinalizeOptions().interp_multiplier == 4
    with pytest.raises(ValueError, match="interp_multiplier"):
        FinalizeOptions(interp_multiplier=0)


def test_finalize_run_forwards_multiplier_to_durable_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U upscale-only finalize reaches the sidecar with multiplier 1 (DESIGN §56).

    Same fake-commit harness as the issue-166 selection test: one real
    fake-backend segment, both legs stubbed present, timer on the
    durable entry asserting the forwarded multiplier.
    """
    import shutil

    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    from tests.conftest import initialize_run_directory
    from voyage.media import finalize_run
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="multfwd", style="pastel neon line-art, peaceful")
    config, _ = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()

    seen: dict[str, Any] = {}

    def _fake_resolve(base: Path | str) -> AugmentWeights:
        return AugmentWeights(
            film=Path(str(base) + "/film"), realesrgan=Path(str(base) + "/esrgan")
        )

    def _fake_durable(
        run_dir_arg: Path,
        usable: list[Path],
        *,
        out_width: int,
        out_height: int,
        source_fps: float,
        weights: AugmentWeights,
        upscale_factor: int = 2,
        multiplier: int = 4,
        chunk_frames: int = 32,
        crf: int = 15,
        preset: str = "veryfast",
        device: str = "cuda:1",
        work_dir: Path,
        timings: dict[str, float] | None = None,
    ) -> tuple[Path, int]:
        seen["multiplier"] = multiplier
        intermediate = work_dir / "model_intermediate.mp4"
        work_dir.mkdir(parents=True, exist_ok=True)
        first = usable[0] / "video.mp4"
        shutil.copy2(first, intermediate)
        return (intermediate, round(source_fps * multiplier))

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _fake_resolve)
    monkeypatch.setattr(finalize_module, "run_durable_model_pass", _fake_durable)
    # `media` imports these lazily from `voyage.augment`, so the source
    # module (not `voyage.media`) is the patch target.
    monkeypatch.setattr(augment_module, "augment_devices", lambda: ("cuda:0", "cuda:1"))
    monkeypatch.setattr(augment_module, "model_pass_devices", lambda: ("cuda:1",))
    output = tmp_path / "final.mp4"
    finalize_run(
        run_dir,
        output,
        width=768,
        height=512,
        fps=24,
        min_fps=24,
        min_width=1216,
        min_height=704,
        use_model_pass=True,
        interp_multiplier=1,
        models_dir=tmp_path,
    )
    assert seen["multiplier"] == 1
    assert output.exists()


def test_durable_pass_records_phase_timings(
    tmp_path: Path,
) -> None:
    """`timings` out-param carries per-phase seconds + counts (DESIGN §56).

    All seams stubbed: the test pins the schema (keys, non-negativity,
    chunk accounting), never the values.
    """
    from voyage.augment_drain import DrainResult
    from voyage.augment_finalize import run_durable_model_pass
    from voyage.augment_interp_poller import InterpPollResult
    from voyage.augment_upscale_poller import UpscalePollResult
    from voyage.errors import MediaError

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    segment = run_dir / "segments" / "000000"
    segment.mkdir(parents=True)

    def _fake_upscale(run_dir_arg: Path, **kwargs: Any) -> UpscalePollResult:
        return UpscalePollResult(
            segments_seen=1,
            segments_skipped=0,
            chunks_done=0,
            chunks_skipped=3,
            partials_pruned=0,
        )

    def _fake_interp(run_dir_arg: Path, **kwargs: Any) -> InterpPollResult:
        return InterpPollResult(
            segments_seen=1,
            segments_skipped=0,
            chunks_done=0,
            chunks_skipped=3,
            chunks_waiting=0,
            partials_pruned=0,
        )

    def _fake_drain(plan_dir: Path, out_fps: float = 24.0) -> DrainResult:
        intermediate = plan_dir / "model_intermediate.mp4"
        intermediate.write_bytes(b"\x00")
        return DrainResult(
            plan_dir=plan_dir,
            chunk_mp4s=(),
            intermediate_mp4=intermediate,
            chunks_drained=0,
        )

    def _fake_concat(parts: list[Path], dest: Path) -> Path:
        dest.write_bytes(b"\x00")
        return dest

    # `weights_key_for` hashes real files — stand in two tiny legs so the
    # key derives, then the source lookup (not the hash) is what fails.
    film = tmp_path / "film.safetensors"
    esrgan = tmp_path / "esrgan.pth"
    film.write_bytes(b"film")
    esrgan.write_bytes(b"esrgan")
    weights = AugmentWeights(film=film, realesrgan=esrgan)
    timings: dict[str, float] = {}
    with pytest.raises(MediaError, match="has no pollable source"):
        run_durable_model_pass(
            run_dir,
            [segment],
            out_width=1216,
            out_height=704,
            source_fps=24.0,
            weights=weights,
            multiplier=1,
            work_dir=tmp_path / "work",
            upscale_poll_fn=_fake_upscale,
            interp_poll_fn=_fake_interp,
            drain_fn=_fake_drain,
            concat_fn=_fake_concat,
            timings=timings,
        )
    # The source lookup fails before any phase runs — but the timings
    # schema keys must exist (zeroed) so miners never KeyError.
    assert timings["upscale_poll_s"] >= 0.0
    assert timings["interp_poll_s"] >= 0.0
    assert timings["upscale_chunks_done"] == 0
    assert timings["interp_chunks_done"] == 0

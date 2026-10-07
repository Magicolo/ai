"""SonicMaster Track A: registry + ensure + config/CLI/skip-key (DESIGN §§84-85, §140).

Scope: registry pins + MODEL_SPECS row + download/verify wrappers +
AudioConfig.mastering (default True) + resolve plumbing + generate-only
--no-master skip + skip-key freshness + ensure gating. No media/SFX/
console/Dockerfile/docs changes (Track A).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import voyage.model_registry as model_registry
import voyage.registry_mastering as registry_mastering
import voyage.registry_records as registry_records
from tests.conftest import initialize_run_directory


def test_spec_present() -> None:
    """MODEL_SPECS carries the audio-sonicmaster row (two FileSpecs)."""
    spec = model_registry.MODEL_SPECS["audio-sonicmaster"]
    assert spec.name == "audio-sonicmaster"
    assert spec.manifest_key == "sonicmaster"
    assert len(spec.files) == 2
    assert spec.files[0].repo_id == registry_mastering.SONICMASTER_HF_REPO
    assert spec.files[0].filename == registry_mastering.SONICMASTER_MODEL_FILE
    assert spec.files[1].repo_id == registry_mastering.SONICMASTER_VAE_REPO
    assert spec.files[1].filename == registry_mastering.SONICMASTER_VAE_FILE
    assert len(spec.checks) == 2
    assert len(spec.expected_hashes) == 2
    assert spec.record_builder is registry_mastering._record_mastering
    assert spec.success_message is registry_mastering._describe_mastering


def test_pins_are_single_sourced() -> None:
    """Mastering pins + builders live once, in registry_mastering."""
    pin_names = (
        "SONICMASTER_HF_REPO",
        "SONICMASTER_HF_REVISION",
        "SONICMASTER_SUBDIR",
        "SONICMASTER_MODEL_FILE",
        "SONICMASTER_MODEL_REPO_PATH",
        "SONICMASTER_MODEL_MIN_BYTES",
        "SONICMASTER_LICENSE",
        "SONICMASTER_LICENSE_URL",
        "SONICMASTER_VAE_REPO",
        "SONICMASTER_VAE_REVISION",
        "SONICMASTER_VAE_SUBFOLDER",
        "SONICMASTER_VAE_FILE",
        "SONICMASTER_VAE_REPO_PATH",
        "SONICMASTER_VAE_MIN_BYTES",
        "SONICMASTER_VAE_LICENSE",
        "SONICMASTER_VAE_LICENSE_URL",
        "EXPECTED_SONICMASTER_MODEL_SHA256",
        "EXPECTED_SONICMASTER_VAE_SHA256",
    )
    for name in pin_names:
        assert getattr(registry_records, name) == getattr(registry_mastering, name)
        assert getattr(model_registry, name) == getattr(registry_mastering, name)
    assert registry_records._record_mastering is registry_mastering._record_mastering
    assert registry_records._describe_mastering is registry_mastering._describe_mastering
    assert model_registry._record_mastering is registry_mastering._record_mastering
    assert model_registry._describe_mastering is registry_mastering._describe_mastering


def test_expected_shas_are_64_lower_hex() -> None:
    """Track A placeholder digests keep the ingest-gate shape (issue 207)."""
    import re

    for value in (
        registry_mastering.EXPECTED_SONICMASTER_MODEL_SHA256,
        registry_mastering.EXPECTED_SONICMASTER_VAE_SHA256,
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", value) is not None


def test_record_and_describe_builders(tmp_path: Path) -> None:
    """Record carries both shas; describe is byte-stable."""
    mastering_dir = tmp_path / registry_mastering.SONICMASTER_SUBDIR
    mastering_dir.mkdir(parents=True)
    (mastering_dir / registry_mastering.SONICMASTER_MODEL_FILE).write_bytes(b"model-bytes")
    (mastering_dir / registry_mastering.SONICMASTER_VAE_FILE).write_bytes(b"vae-bytes")
    record = registry_mastering._record_mastering(tmp_path)
    assert record["repo"] == registry_mastering.SONICMASTER_HF_REPO
    assert record["vae_repo"] == registry_mastering.SONICMASTER_VAE_REPO
    shas = record["checkpoint_shas"]
    assert isinstance(shas, dict)
    assert set(shas) == {
        registry_mastering.SONICMASTER_MODEL_REPO_PATH,
        registry_mastering.SONICMASTER_VAE_REPO_PATH,
    }
    message = registry_mastering._describe_mastering(tmp_path)
    assert message.startswith("sonicmaster OK (model ")


def test_download_verify_wrappers_wire_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download/verify wrappers target the audio-sonicmaster spec."""
    captured: list[str] = []

    def _fake_download(models_dir: Path, spec_name: str) -> dict[str, object]:
        captured.append(spec_name)
        return {}

    def _fake_verify(models_dir: Path, spec_name: str) -> tuple[bool, str]:
        captured.append(spec_name)
        return True, "ok"

    monkeypatch.setattr(model_registry, "download_model", _fake_download)
    monkeypatch.setattr(model_registry, "verify_model", _fake_verify)
    model_registry.download_mastering_models(tmp_path)
    model_registry.verify_mastering_models(tmp_path)
    assert captured == ["audio-sonicmaster", "audio-sonicmaster"]


def test_models_dir_layout_covers_mastering(tmp_path: Path) -> None:
    """Layout helper exposes the sonicmaster tree root (issue 086)."""
    layout = model_registry.models_dir_layout(tmp_path)
    assert layout["sonicmaster_dir"] == str(tmp_path / registry_mastering.SONICMASTER_SUBDIR)


def test_audio_config_mastering_default_true() -> None:
    """Persistent mastering stage defaults on."""
    from voyage.config import AudioConfig

    assert AudioConfig().mastering is True


def test_resolve_config_mastering_plumbing() -> None:
    """Resolve honors the mastering override; backend switch preserves it."""
    from voyage.config import ProjectConfig, preset_config, resolve_config

    base = ProjectConfig(style="mastering-probe")
    assert resolve_config(base).audio.mastering is True
    assert resolve_config(base, mastering=False).audio.mastering is False
    assert resolve_config(base, mastering=True).audio.mastering is True
    switched = resolve_config(base, backend="ltxv", mastering=False)
    assert switched.audio.mastering is False
    preserved = resolve_config(preset_config("probe", "pastel neon line-art, peaceful", 7))
    assert preserved.audio.mastering is True


def test_resolve_generate_skips_master() -> None:
    """Generate-only --no-master (+ --no-audio shorthand) maps to the skip."""
    from voyage.cli_core import resolve_generate_skips

    assert resolve_generate_skips(argparse.Namespace())["skip_mastering"] is False
    assert resolve_generate_skips(argparse.Namespace(no_master=True))["skip_mastering"] is True
    assert resolve_generate_skips(argparse.Namespace(no_audio=True))["skip_mastering"] is True
    plain = resolve_generate_skips(argparse.Namespace(no_music=True))
    assert plain["skip_mastering"] is False
    assert plain["skip_music"] is True


def test_generate_parser_accepts_no_master() -> None:
    """Generate accepts --no-master; configure rejects it (generate-only)."""
    from voyage.cli import build_parser

    args = build_parser().parse_args(["generate", "probe", "--no-master"])
    assert args.no_master is True
    plain = build_parser().parse_args(["generate", "probe"])
    assert plain.no_master is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["configure", "calm", "--segments", "1", "--no-master"])


def test_skip_key_carries_master() -> None:
    """Mastering rides the freshness key (stored + skip both flip it)."""
    from voyage.cli_core import generate_skip_key

    base = {"skip_music": False, "skip_sfx": False, "force_upscale_1": False}
    on = generate_skip_key(
        {**base, "force_interpolate_1": False, "skip_mastering": False},
        manifest_no_sfx=False,
        stored_upscale=1,
        stored_interpolate=1,
        stored_mastering=True,
    )
    assert on.endswith(",master=0")
    skipped = generate_skip_key(
        {**base, "force_interpolate_1": False, "skip_mastering": True},
        manifest_no_sfx=False,
        stored_upscale=1,
        stored_interpolate=1,
        stored_mastering=True,
    )
    assert skipped.endswith(",master=1")
    assert skipped != on
    stored_off = generate_skip_key(
        {**base, "force_interpolate_1": False, "skip_mastering": False},
        manifest_no_sfx=False,
        stored_upscale=1,
        stored_interpolate=1,
        stored_mastering=False,
    )
    assert stored_off.endswith(",master=1")
    assert stored_off != on


def test_required_specs_mastering_branch() -> None:
    """CUDA backends pull mastering by default; fake stays empty."""
    from voyage.config import ProjectConfig, with_video_backend
    from voyage.models_ensure import required_specs

    config = with_video_backend(ProjectConfig(style="pastel neon line-art, peaceful"), "ltxv")
    assert "audio-sonicmaster" in {entry.spec for entry in required_specs(config)}
    fake = with_video_backend(ProjectConfig(style="pastel neon line-art, peaceful"), "fake")
    assert required_specs(fake) == []
    assert required_specs(fake, mastering_enabled=True) == []
    assert "audio-sonicmaster" not in {
        entry.spec for entry in required_specs(config, mastering_enabled=False)
    }
    config.audio.mastering = False
    assert "audio-sonicmaster" not in {entry.spec for entry in required_specs(config)}
    mastering_dir = {
        entry.spec: entry.models_dir
        for entry in required_specs(
            with_video_backend(ProjectConfig(style="pastel neon line-art, peaceful"), "ltxv")
        )
    }["audio-sonicmaster"]
    assert mastering_dir == Path(config.audio.models_dir)


def test_gate_skip_key_mismatch_is_stale_for_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same segments + frames but mastering skipped must re-finalize."""
    import voyage.cli_generate as gen_ops
    from voyage.persistence import read_effective_config, record_final_coverage

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "mastered"
    initialize_run_directory(run_dir, run_id="mastered", seed=7)
    (run_dir / "final.mp4").write_bytes(b"fake-final")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    effective = read_effective_config(run_dir)
    record_final_coverage(
        run_dir,
        presented_frames=48,
        segments=1,
        skip_key=gen_ops._expected_skip_key(argparse.Namespace(), manifest, effective),
    )
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: 48)
    assert (
        gen_ops._final_is_fresh(
            run_dir,
            run_dir / "final.mp4",
            committed=1,
            expected_skip_key=gen_ops._expected_skip_key(argparse.Namespace(), manifest, effective),
        )
        is True
    )
    assert (
        gen_ops._final_is_fresh(
            run_dir,
            run_dir / "final.mp4",
            committed=1,
            expected_skip_key=gen_ops._expected_skip_key(
                argparse.Namespace(no_master=True), manifest, effective
            ),
        )
        is False
    )


def test_finalize_run_dir_threads_no_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic finalize namespace carries the mastering skip."""
    import voyage.cli_finalize as final_ops
    import voyage.cli_generate as gen_ops

    captured: dict[str, object] = {}

    def _fake_finalize(args: argparse.Namespace) -> int:
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(final_ops, "cmd_finalize", _fake_finalize)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "threadmaster"
    initialize_run_directory(run_dir, run_id="threadmaster", seed=7)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert gen_ops._finalize_run_dir(run_dir, manifest, argparse.Namespace(no_master=True)) == 0
    assert captured["no_master"] is True
    captured.clear()
    assert gen_ops._finalize_run_dir(run_dir, manifest, argparse.Namespace()) == 0
    assert captured["no_master"] is False

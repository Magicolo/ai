"""`voyage generate` inline model ensure: selective specs, parallel download.

TDD contract for `voyage.models_ensure` + `VoyageConsole.parallel_downloads`:
only the stacks the effective generate config needs are verified/downloaded
(video backend's own spec, ACE-Step audio when paired, qwen director unless
deterministic, MMAudio SFX only when the finalize pass runs). The VLM
inspector weights stay pinned in the registry but are never an ensure
gate (the supervisor piggyback was removed). Fake backends need nothing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from voyage.config import ProjectConfig, with_video_backend
from voyage.console import VoyageConsole

if TYPE_CHECKING:
    from voyage.model_registry import ModelSpec


def _config_with_style() -> ProjectConfig:
    return ProjectConfig(style="pastel neon line-art, peaceful")


def test_fake_backend_requires_nothing() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "fake")
    required = required_specs(config, sfx_enabled=False)
    assert required == []


def test_ltxv_requires_own_spec_plus_audio_and_director() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert specs == {
        "ltxv-2b",
        "audio-acestep",
        "audio-sonicmaster",
        "director-qwen35-gguf",
        "rife",
        "realesrgan-anime",
    }


def test_video_backends_map_to_their_own_spec_only() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert specs == {
        "ltxv-2b",
        "audio-acestep",
        "audio-sonicmaster",
        "director-qwen35-gguf",
        "rife",
        "realesrgan-anime",
    }

    config = with_video_backend(_config_with_style(), "causvid")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert specs == {
        "causvid",
        "audio-acestep",
        "audio-sonicmaster",
        "director-qwen35-gguf",
        "rife",
        "realesrgan-anime",
    }


def test_cpu_director_opt_out_requires_8b_stack() -> None:
    """`--director-device cpu` keeps the bf16 8B stack instead of the AWQ pin."""
    from voyage.config import resolve_config
    from voyage.models_ensure import required_specs

    config = resolve_config(
        with_video_backend(_config_with_style(), "ltxv"), director="qwen", director_device="cpu"
    )
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert "director-qwen8b" in specs
    assert "director-qwen4b-awq" not in specs
    assert "director-qwen35-gguf" not in specs


def test_deterministic_director_needs_no_director_models() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    config.director.backend = "deterministic"
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert specs == {
        "ltxv-2b",
        "audio-acestep",
        "audio-sonicmaster",
        "rife",
        "realesrgan-anime",
    }


def test_sfx_is_opt_in_and_inspector_is_never_required() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    assert "sfx-mmaudio" not in {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert "inspector-qwen35" not in {
        item.spec for item in required_specs(config, sfx_enabled=False)
    }

    sfx_config = with_video_backend(_config_with_style(), "ltxv")
    sfx_config.sfx.backend = "mmaudio"
    assert "sfx-mmaudio" in {item.spec for item in required_specs(sfx_config, sfx_enabled=True)}
    # The inspector weights stay provisionable but no config requests them:
    # the supervisor piggyback that consumed them was removed.
    assert "inspector-qwen35" not in {
        item.spec for item in required_specs(sfx_config, sfx_enabled=True)
    }


def test_cuda_backends_include_augmentation_by_default() -> None:
    from voyage.models_ensure import required_specs

    for backend in ("ltxv", "causvid"):
        config = with_video_backend(_config_with_style(), backend)
        specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
        assert {"rife", "realesrgan-anime"} <= specs


def test_ltx_backends_require_own_spec_with_acestep() -> None:
    """ltx25/ltx23 take continuous music from ACE-Step: own spec +
    director + augment + the ACE stack + SFX (the worker's joint
    audio.wav is ignored; DESIGN §140 audio continuity).

    Phase 4 extends `VideoBackendName` + presets; until then the cast pins
    the intended Literal membership (dataclasses never validate at runtime).
    """
    from typing import cast

    from voyage.config import VideoBackendName
    from voyage.models_ensure import required_specs

    for backend, spec in (("ltx25", "ltx25"), ("ltx23", "ltx23")):
        config = _config_with_style()
        config.video.backend = cast(VideoBackendName, backend)
        config.audio.backend = "acestep"
        config.sfx.backend = "mmaudio"
        # Explicit decider: this test pins the audio/SFX gates, not
        # whichever director backend is the tree default today.
        config.director.backend = "qwen"
        specs = {item.spec for item in required_specs(config, sfx_enabled=True)}
        assert specs == {
            spec,
            "director-qwen4b-awq",
            "rife",
            "realesrgan-anime",
            "sfx-mmaudio",
            "audio-acestep",
            "audio-sonicmaster",
        }


def test_augment_disabled_excludes_rife_and_realesrgan() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False, augment_enabled=False)}
    assert specs == {
        "ltxv-2b",
        "audio-acestep",
        "audio-sonicmaster",
        "director-qwen35-gguf",
    }


def test_fake_stays_empty_with_augment_enabled() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "fake")
    assert required_specs(config, sfx_enabled=False, augment_enabled=True) == []


def test_augment_specs_share_video_models_dir() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    entries = {item.spec: item.models_dir for item in required_specs(config, sfx_enabled=False)}
    assert entries["rife"] == entries["realesrgan-anime"] == Path(config.video.models_dir)


def test_ensure_no_download_flag_reports_augment_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (False, "missing"))

    def _fail_download(_dir: Path, _spec: str) -> dict[str, object]:
        raise AssertionError("must not download with allow_download=False")

    monkeypatch.setattr(registry, "download_model", _fail_download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    config = with_video_backend(_config_with_style(), "ltxv")
    assert ensure_models(config, False, console, str(tmp_path), allow_download=False) == 1
    output = stream.getvalue()
    assert "rife" in output
    assert "realesrgan-anime" in output


def test_ensure_augment_disabled_skips_augment_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    import voyage.models_ensure as ensure_module
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(ensure_module, "_manifest_entry_complete", lambda _dir, _spec: True)

    present = {"audio-acestep", "audio-sonicmaster", "director-qwen35-gguf"}

    def _verify(_dir: Path, spec: str) -> tuple[bool, str]:
        return (spec in present, "OK" if spec in present else "missing")

    downloaded: list[str] = []

    def _download(_dir: Path, spec: str) -> dict[str, object]:
        downloaded.append(spec)
        present.add(spec)
        return {}

    monkeypatch.setattr(registry, "verify_model", _verify)
    monkeypatch.setattr(registry, "download_model", _download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    config = with_video_backend(_config_with_style(), "ltxv")
    assert ensure_models(config, False, console, str(tmp_path), augment_enabled=False) == 0
    assert downloaded == ["ltxv-2b"]


def test_ensure_skips_download_when_everything_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    import voyage.models_ensure as ensure_module
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (True, "OK"))
    monkeypatch.setattr(ensure_module, "_manifest_entry_complete", lambda _dir, _spec: True)

    def _fail_download(_dir: Path, _spec: str) -> dict[str, object]:
        raise AssertionError("must not download when verified")

    monkeypatch.setattr(registry, "download_model", _fail_download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    config = with_video_backend(_config_with_style(), "ltxv")
    assert ensure_models(config, False, console, str(tmp_path)) == 0


def test_ensure_downloads_only_missing_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    import voyage.models_ensure as ensure_module
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(ensure_module, "_manifest_entry_complete", lambda _dir, _spec: True)
    present = {
        "audio-acestep",
        "audio-sonicmaster",
        "director-qwen35-gguf",
        "rife",
        "realesrgan-anime",
    }

    def _verify(_dir: Path, spec: str) -> tuple[bool, str]:
        return (spec in present, "OK" if spec in present else "missing")

    downloaded: list[str] = []

    def _download(_dir: Path, spec: str) -> dict[str, object]:
        downloaded.append(spec)
        present.add(spec)
        return {}

    monkeypatch.setattr(registry, "verify_model", _verify)
    monkeypatch.setattr(registry, "download_model", _download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    config = with_video_backend(_config_with_style(), "ltxv")
    assert ensure_models(config, False, console, str(tmp_path)) == 0
    assert downloaded == ["ltxv-2b"]


def test_ensure_no_download_flag_fails_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (False, "missing"))

    def _fail_download(_dir: Path, _spec: str) -> dict[str, object]:
        raise AssertionError("must not download with allow_download=False")

    monkeypatch.setattr(registry, "download_model", _fail_download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    config = with_video_backend(_config_with_style(), "ltxv")
    assert ensure_models(config, False, console, str(tmp_path), allow_download=False) == 1
    assert "missing" in stream.getvalue()


def test_ensure_download_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (False, "missing"))

    def _boom(_dir: Path, _spec: str) -> dict[str, object]:
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(registry, "download_model", _boom)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    config = with_video_backend(_config_with_style(), "ltxv")
    assert ensure_models(config, False, console, str(tmp_path)) == 1


def test_parallel_downloads_plain_fallback_prints_per_model() -> None:
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    with console.parallel_downloads(["ltxv-2b", "audio-acestep"]) as tracker:
        tracker.succeed("ltxv-2b")
        tracker.fail("audio-acestep", "hub exploded")
    output = stream.getvalue()
    assert "ltxv-2b" in output
    assert "audio-acestep" in output


def _configure_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    """Configure a fake-backend run (two-verb flow prerequisite)."""
    from voyage.cli import main

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "configure",
                "ensure",
                "--backend",
                "fake",
                "--duration",
                "2s",
                "--style",
                "pastel neon line-art, peaceful",
                "--seed",
                "11",
                *extra,
            ]
        )
        == 0
    )


def test_generate_no_download_flag_passes_for_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage.cli import main

    monkeypatch.chdir(tmp_path)
    _configure_fake(tmp_path, monkeypatch, "--no-download")
    assert main(["generate", "ensure"]) == 0
    assert (tmp_path / "output" / "ensure" / "final.mp4").exists()


def test_generate_fails_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli as cli
    import voyage.doctor as doctor_module

    _configure_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(doctor_module, "check_ffmpeg", lambda: (False, "ffmpeg not found on PATH"))
    assert cli.main(["generate", "ensure"]) == 1
    assert "ffmpeg" in capsys.readouterr().err


def test_generate_fails_on_low_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli as cli
    import voyage.media as media_module
    from voyage.errors import DiskSpaceError

    def _no_space(_path: Path, _reserve: float) -> float:
        raise DiskSpaceError("free space 0.0 GiB below reserve 5.0 GiB")

    _configure_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(media_module, "check_free_space", _no_space)
    assert cli.main(["generate", "ensure"]) == 1
    assert "free space" in capsys.readouterr().err


def test_generate_aborts_when_ensure_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli as cli
    import voyage.models_ensure as ensure

    _configure_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(ensure, "ensure_models", lambda *args, **kwargs: 1)
    assert cli.main(["generate", "ensure"]) == 1
    assert not (tmp_path / "output" / "ensure" / "final.mp4").exists()


def _stub_pinned_spec(manifest_key: str, relative_path: str, payload: bytes) -> ModelSpec:
    """Registry row with one ingest pin over a tiny tmp file."""
    import hashlib

    from voyage.atomic import JsonValue
    from voyage.model_registry import ExpectedHash, ModelSpec

    digest = hashlib.sha256(payload).hexdigest()

    def _record(_models_dir: Path) -> dict[str, JsonValue]:
        return {"checkpoint_shas": {relative_path: hashlib.sha256(payload).hexdigest()}}

    return ModelSpec(
        name="stub-pinned",
        manifest_key=manifest_key,
        snapshots=(),
        files=(),
        record_builder=_record,
        checks=(),
        success_message=lambda _models_dir: "stub-pinned OK",
        expected_hashes=(ExpectedHash(relative_path, digest),),
    )


def test_ensure_repairs_missing_manifest_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asporgue 2026-10-07: files present but no manifest entry.

    `verify_model` passes on presence alone while the fail-closed worker
    refuses `no manifest entry` — ensure must repair the record without
    any hub download (files already match the ingest pins).
    """
    import json

    import voyage.model_registry as registry
    import voyage.models_ensure as ensure_module
    from voyage.models_ensure import RequiredModel, _manifest_entry_complete, ensure_models

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = b"honest-bytes"
    candidate = models_dir / "stub/weights.bin"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(payload)
    (models_dir / "manifest.json").write_text(json.dumps({"other": {}}), encoding="utf-8")
    monkeypatch.setitem(
        registry.MODEL_SPECS, "stub-pinned", _stub_pinned_spec("stub", "stub/weights.bin", payload)
    )
    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (True, "stub-pinned OK"))
    assert not _manifest_entry_complete(models_dir, "stub-pinned")

    def _fail_download(_dir: Path, _spec: str) -> dict[str, object]:
        raise AssertionError("must not download when files already match pins")

    monkeypatch.setattr(registry, "download_model", _fail_download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    monkeypatch.setattr(
        ensure_module,
        "required_specs",
        lambda *args, **kwargs: [RequiredModel(spec="stub-pinned", models_dir=models_dir)],
    )
    assert ensure_models(_config_with_style(), False, console, str(tmp_path)) == 0
    assert _manifest_entry_complete(models_dir, "stub-pinned")
    assert "manifest repaired" in stream.getvalue()


def test_ensure_refuses_to_attest_tampered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair never records a sha over mismatched bytes (fail loud)."""
    import json

    import voyage.model_registry as registry
    import voyage.models_ensure as ensure_module
    from voyage.models_ensure import RequiredModel, ensure_models

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    candidate = models_dir / "stub/weights.bin"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"tampered-bytes")
    (models_dir / "manifest.json").write_text(json.dumps({"other": {}}), encoding="utf-8")
    monkeypatch.setitem(
        registry.MODEL_SPECS,
        "stub-pinned",
        _stub_pinned_spec("stub", "stub/weights.bin", b"honest-bytes"),
    )
    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (True, "stub-pinned OK"))

    def _fail_download(_dir: Path, _spec: str) -> dict[str, object]:
        raise AssertionError("tampered path must fail before any download")

    monkeypatch.setattr(registry, "download_model", _fail_download)
    stream = io.StringIO()
    console = VoyageConsole(no_color=True, stream=stream)
    monkeypatch.setattr(
        ensure_module,
        "required_specs",
        lambda *args, **kwargs: [RequiredModel(spec="stub-pinned", models_dir=models_dir)],
    )
    assert ensure_models(_config_with_style(), False, console, str(tmp_path)) == 1
    merged = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "stub" not in merged


def test_generate_ensure_receives_selective_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli as cli
    import voyage.models_ensure as ensure

    seen: dict[str, object] = {}
    real_ensure = ensure.ensure_models

    def _spy(
        config: ProjectConfig,
        sfx_enabled: bool,
        console: VoyageConsole,
        models_root: str | Path | None = None,
        *,
        allow_download: bool = True,
        augment_enabled: bool = True,
        music_enabled: bool = True,
        mastering_enabled: bool = True,
    ) -> int:
        seen["video"] = config.video.backend
        seen["sfx_enabled"] = sfx_enabled
        seen["allow_download"] = allow_download
        seen["augment_enabled"] = augment_enabled
        seen["music_enabled"] = music_enabled
        seen["mastering_enabled"] = mastering_enabled
        return real_ensure(config, sfx_enabled, console, models_root)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ensure, "ensure_models", _spy)
    _configure_fake(tmp_path, monkeypatch)
    assert cli.main(["generate", "ensure"]) == 0
    assert seen == {
        "video": "fake",
        "sfx_enabled": False,
        "allow_download": True,
        # Defaults are upscale=1/interpolate=1 (no augment work), so the
        # selective scope correctly carries augment_enabled=False.
        "augment_enabled": False,
        # No --no-music/--no-audio on this run, so the music stack stays ensured.
        "music_enabled": True,
        # No --no-master/--no-audio on this run, so mastering stays ensured
        # (fake stays empty downstream regardless).
        "mastering_enabled": True,
    }

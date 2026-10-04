"""`voyage generate` inline model ensure: selective specs, parallel download.

TDD contract for `voyage.models_ensure` + `VoyageConsole.parallel_downloads`:
only the stacks the effective generate config needs are verified/downloaded
(video backend's own spec, ACE-Step audio when paired, qwen director unless
deterministic, MMAudio SFX only when the finalize pass runs, the VLM
inspector only when enabled). Fake backends need nothing.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from voyage.config import ProjectConfig, with_video_backend
from voyage.console import VoyageConsole


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
    assert specs == {"ltxv-2b", "audio-acestep", "director-qwen35-gguf", "film", "realesrgan-anime"}


def test_video_backends_map_to_their_own_spec_only() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert specs == {
        "ltxv-2b",
        "audio-acestep",
        "director-qwen35-gguf",
        "film",
        "realesrgan-anime",
    }

    config = with_video_backend(_config_with_style(), "causvid")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert specs == {
        "causvid",
        "audio-acestep",
        "director-qwen35-gguf",
        "film",
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
    assert specs == {"ltxv-2b", "audio-acestep", "film", "realesrgan-anime"}


def test_sfx_and_inspector_are_opt_in_only() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    assert "sfx-mmaudio" not in {item.spec for item in required_specs(config, sfx_enabled=False)}
    assert "inspector-qwen35" not in {
        item.spec for item in required_specs(config, sfx_enabled=False)
    }

    sfx_config = with_video_backend(_config_with_style(), "ltxv")
    sfx_config.sfx.backend = "mmaudio"
    assert "sfx-mmaudio" in {item.spec for item in required_specs(sfx_config, sfx_enabled=True)}

    inspect_config = with_video_backend(_config_with_style(), "ltxv")
    inspect_config.experimental.visual_inspector = True
    assert "inspector-qwen35" in {
        item.spec for item in required_specs(inspect_config, sfx_enabled=False)
    }


def test_cuda_backends_include_augmentation_by_default() -> None:
    from voyage.models_ensure import required_specs

    for backend in ("ltxv", "causvid"):
        config = with_video_backend(_config_with_style(), backend)
        specs = {item.spec for item in required_specs(config, sfx_enabled=False)}
        assert {"film", "realesrgan-anime"} <= specs


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
            "film",
            "realesrgan-anime",
            "sfx-mmaudio",
            "audio-acestep",
        }


def test_augment_disabled_excludes_film_and_realesrgan() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    specs = {item.spec for item in required_specs(config, sfx_enabled=False, augment_enabled=False)}
    assert specs == {"ltxv-2b", "audio-acestep", "director-qwen35-gguf"}


def test_fake_stays_empty_with_augment_enabled() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "fake")
    assert required_specs(config, sfx_enabled=False, augment_enabled=True) == []


def test_augment_specs_share_video_models_dir() -> None:
    from voyage.models_ensure import required_specs

    config = with_video_backend(_config_with_style(), "ltxv")
    entries = {item.spec: item.models_dir for item in required_specs(config, sfx_enabled=False)}
    assert entries["film"] == entries["realesrgan-anime"] == Path(config.video.models_dir)


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
    assert "film" in output
    assert "realesrgan-anime" in output


def test_ensure_augment_disabled_skips_augment_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import voyage.model_registry as registry
    from voyage.models_ensure import ensure_models

    present = {"audio-acestep", "director-qwen35-gguf"}

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
    from voyage.models_ensure import ensure_models

    monkeypatch.setattr(registry, "verify_model", lambda _dir, _spec: (True, "OK"))

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
    from voyage.models_ensure import ensure_models

    present = {"audio-acestep", "director-qwen35-gguf", "film", "realesrgan-anime"}

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


def _fake_generate_argv(run_dir: Path, *extra: str) -> list[str]:
    return [
        "generate",
        "--backend",
        "fake",
        "--duration",
        "2s",
        "--style",
        "pastel neon line-art, peaceful",
        "--output",
        str(run_dir),
        "--run-id",
        "ensure",
        "--seed",
        "11",
        *extra,
    ]


def test_generate_no_download_flag_passes_for_fake(tmp_path: Path) -> None:
    from voyage.cli import main

    assert main(_fake_generate_argv(tmp_path / "run", "--no-download")) == 0
    assert (tmp_path / "run" / "final.mp4").exists()


def test_generate_fails_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli as cli

    monkeypatch.setattr(cli, "check_ffmpeg", lambda: (False, "ffmpeg not found on PATH"))
    assert cli.main(_fake_generate_argv(tmp_path / "run")) == 1
    assert "ffmpeg" in capsys.readouterr().err


def test_generate_fails_on_low_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli as cli
    from voyage.errors import DiskSpaceError

    def _no_space(_path: Path, _reserve: float) -> float:
        raise DiskSpaceError("free space 0.0 GiB below reserve 5.0 GiB")

    monkeypatch.setattr(cli, "check_free_space", _no_space)
    assert cli.main(_fake_generate_argv(tmp_path / "run")) == 1
    assert "free space" in capsys.readouterr().err


def test_generate_aborts_when_ensure_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli as cli
    import voyage.models_ensure as ensure

    monkeypatch.setattr(ensure, "ensure_models", lambda *args, **kwargs: 1)
    assert cli.main(_fake_generate_argv(tmp_path / "run")) == 1
    assert not (tmp_path / "run" / "final.mp4").exists()


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
    ) -> int:
        seen["video"] = config.video.backend
        seen["sfx_enabled"] = sfx_enabled
        seen["allow_download"] = allow_download
        seen["augment_enabled"] = augment_enabled
        return real_ensure(config, sfx_enabled, console, models_root)

    monkeypatch.setattr(ensure, "ensure_models", _spy)
    assert cli.main(_fake_generate_argv(tmp_path / "run")) == 0
    assert seen == {
        "video": "fake",
        "sfx_enabled": False,
        "allow_download": True,
        "augment_enabled": True,
    }

"""Generate-only post-processing skip flags (non-persistent).

`voyage generate` accepts --no-music/--no-sfx/--no-upscale/--no-interpolate
plus shorthands --no-audio (= music + sfx) and --no-augment (= upscale +
interpolate). The flags never reach the manifest: --no-upscale/--no-interpolate
force multiplier 1 via explicit overrides, --no-music ships silent AAC sized
to the timeline, and the freshness gate re-finalizes when the behavior key
differs from the stamped coverage (a music-only diff must never read fresh).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_resolve_generate_skips_defaults_off() -> None:
    from voyage.cli_core import resolve_generate_skips

    assert resolve_generate_skips(argparse.Namespace()) == {
        "skip_music": False,
        "skip_sfx": False,
        "force_upscale_1": False,
        "force_interpolate_1": False,
    }


def test_resolve_generate_skips_shorthands() -> None:
    from voyage.cli_core import resolve_generate_skips

    assert resolve_generate_skips(argparse.Namespace(no_audio=True)) == {
        "skip_music": True,
        "skip_sfx": True,
        "force_upscale_1": False,
        "force_interpolate_1": False,
    }
    assert resolve_generate_skips(argparse.Namespace(no_augment=True)) == {
        "skip_music": False,
        "skip_sfx": False,
        "force_upscale_1": True,
        "force_interpolate_1": True,
    }
    skips = resolve_generate_skips(
        argparse.Namespace(no_music=True, no_interpolate=True, no_audio=False, no_augment=False)
    )
    assert skips == {
        "skip_music": True,
        "skip_sfx": False,
        "force_upscale_1": False,
        "force_interpolate_1": True,
    }


def test_generate_parser_accepts_skip_flags() -> None:
    from voyage.cli import build_parser

    args = build_parser().parse_args(["generate", "probe", "--no-music", "--no-augment"])
    assert args.no_music is True
    assert args.no_augment is True
    assert args.no_sfx is False
    assert args.no_audio is False
    plain = build_parser().parse_args(["generate", "probe"])
    assert plain.no_music is False
    assert plain.no_sfx is False
    assert plain.no_upscale is False
    assert plain.no_interpolate is False
    assert plain.no_audio is False
    assert plain.no_augment is False


def test_generate_skip_flags_not_on_configure() -> None:
    """The skips are generate-only: configure must still reject them.

    Except --no-sfx, which already exists on configure as the persistent
    manifest policy — the generate flag is the one-shot counterpart.
    """
    from voyage.cli import build_parser

    for flag in ("--no-music", "--no-upscale", "--no-interpolate", "--no-audio", "--no-augment"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["configure", "calm", "--segments", "1", flag])
    args = build_parser().parse_args(["configure", "calm", "--segments", "1", "--no-sfx"])
    assert args.no_sfx is True


def test_generate_skip_key_combines_manifest_and_flags(tmp_path: Path) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.config import resolve_config
    from voyage.persistence import read_effective_config

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="skipkey", seed=7)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # Stored 1/1 (fake default) makes the forced-1 key vacuous, so pin a
    # lifted stored config for the multiplier assertions.
    effective = resolve_config(read_effective_config(run_dir), upscale=2, interpolate=4)
    base = gen_ops._expected_skip_key(argparse.Namespace(), manifest, effective)
    assert base == "music=0,sfx=0,up=2,interp=4,backend=rife"
    music = gen_ops._expected_skip_key(argparse.Namespace(no_music=True), manifest, effective)
    assert music.startswith("music=1,")
    assert music != base
    augment = gen_ops._expected_skip_key(argparse.Namespace(no_augment=True), manifest, effective)
    assert augment.endswith(",up=1,interp=1,backend=-")
    assert augment != base


def _seeded_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "skipped"
    initialize_run_directory(run_dir, run_id="skipped", seed=7)
    (run_dir / "final.mp4").write_bytes(b"fake-final")
    return run_dir


def test_gate_skip_key_mismatch_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.persistence import read_effective_config, record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
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
    # Same segments + same frames, but music skipped: must re-finalize.
    assert (
        gen_ops._final_is_fresh(
            run_dir,
            run_dir / "final.mp4",
            committed=1,
            expected_skip_key=gen_ops._expected_skip_key(
                argparse.Namespace(no_music=True), manifest, effective
            ),
        )
        is False
    )


def test_gate_legacy_coverage_without_key_is_stale_when_key_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverages stamped before the skip key existed re-finalize exactly once."""
    import voyage.cli_generate as gen_ops
    from voyage.persistence import read_effective_config, record_final_coverage

    run_dir = _seeded_run(tmp_path, monkeypatch)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    effective = read_effective_config(run_dir)
    record_final_coverage(run_dir, presented_frames=48, segments=1)
    monkeypatch.setattr(gen_ops, "presented_frames", lambda path: 48)
    # Legacy gate (no key passed) still reads fresh — old callers unaffected.
    assert gen_ops._final_is_fresh(run_dir, run_dir / "final.mp4", committed=1) is True
    assert (
        gen_ops._final_is_fresh(
            run_dir,
            run_dir / "final.mp4",
            committed=1,
            expected_skip_key=gen_ops._expected_skip_key(argparse.Namespace(), manifest, effective),
        )
        is False
    )


def test_finalize_run_dir_threads_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """1x overrides + no_sfx/no_music ride the synthetic finalize namespace."""
    import voyage.cli_finalize as final_ops
    import voyage.cli_generate as gen_ops

    captured: dict[str, object] = {}

    def _fake_finalize(args: argparse.Namespace) -> int:
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(final_ops, "cmd_finalize", _fake_finalize)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "thread"
    initialize_run_directory(run_dir, run_id="thread", seed=7)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(no_music=True, no_augment=True)
    assert gen_ops._finalize_run_dir(run_dir, manifest, args) == 0
    assert captured["no_music"] is True
    assert captured["upscale"] == 1
    assert captured["interpolate"] == 1
    # No skips: stored policy rides through (None = inherit).
    captured.clear()
    assert gen_ops._finalize_run_dir(run_dir, manifest, argparse.Namespace()) == 0
    assert captured["no_music"] is False
    assert captured["upscale"] is None
    assert captured["interpolate"] is None
    assert captured["no_sfx"] is False


def test_required_specs_music_gate() -> None:
    from voyage.config import preset_config
    from voyage.models_ensure import required_specs

    config = preset_config(
        "skippy",
        "pastel neon line-art, peaceful",
        7,
        video_backend="ltx25",
    )
    assert config.audio.backend == "acestep"
    with_music = {entry.spec for entry in required_specs(config)}
    assert "audio-acestep" in with_music
    without_music = {entry.spec for entry in required_specs(config, music_enabled=False)}
    assert "audio-acestep" not in without_music
    # The video stack itself is unaffected by the music skip.
    assert any(spec.startswith("ltx") for spec in without_music)

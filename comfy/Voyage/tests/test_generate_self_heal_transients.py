"""Generate self-heals safe transients before the pre-finalize gate (DESIGN §56).

`validate_run` is read-only and flags `voyage-final-*` staging dirs plus
`*.partial` / `*.partial.*` / `*.tmp.npy` / `*.tmp*` files under
segments/novelty/audio/augment and `*.partial` at the run root. Generate
must delete exactly that set (all crash-torn staging the next pass
re-creates) instead of aborting with --skip-bad; checksums, missing
artifacts, numbering, and hard SFX errors still need the flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_heal_removes_exactly_what_validate_flags(tmp_path: Path) -> None:
    import voyage.cli_generate as gen_ops
    from voyage.cli_validate import _collect_final_tmpdirs, _collect_transient_orphans

    run_dir = tmp_path / "run"
    (run_dir / "segments").mkdir(parents=True)
    victims = [
        _touch(run_dir / "segments" / "000000" / "take.wav.partial"),
        _touch(run_dir / "segments" / "nested" / "clip.tmp.npy"),
        _touch(run_dir / "novelty" / "vectors.npy.123.tmp.npy"),
        _touch(run_dir / "audio" / "sfx" / "w0000.partial.wav"),
        _touch(run_dir / "augment" / "plan" / "chunk.tmp0001"),
        _touch(run_dir / "state.json.7.partial"),
    ]
    final_dir = run_dir / "voyage-final-abc123"
    final_dir.mkdir(parents=True)
    _touch(final_dir / "000000_window.wav")
    keepers = [
        _touch(run_dir / "segments" / "000000" / "video.mp4", b"video"),
        _touch(run_dir / "segments" / "000000" / "DONE", b""),
        _touch(run_dir / "final.mp4", b"final"),
        _touch(run_dir / "notes.tmpness", b"legit-suffix"),
    ]
    assert gen_ops._heal_safe_transients(run_dir) == len(victims) + 1
    for victim in victims:
        assert not victim.exists()
    assert not final_dir.exists()
    for keeper in keepers:
        assert keeper.is_file()
    assert _collect_transient_orphans(run_dir / "segments", run_dir / "segments") == []
    assert _collect_final_tmpdirs(run_dir) == []
    assert list(run_dir.glob("*.partial")) == []


def test_heal_skips_symlinks_and_missing_dirs(tmp_path: Path) -> None:
    import voyage.cli_generate as gen_ops

    assert gen_ops._heal_safe_transients(tmp_path / "ghost") == 0
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = _touch(outside / "real.partial")
    link_dir = run_dir / "audio"
    link_dir.mkdir(parents=True)
    link = link_dir / "evil.partial"
    link.symlink_to(target)
    assert gen_ops._heal_safe_transients(run_dir) == 0
    assert link.is_symlink()
    assert target.is_file()


def test_heal_skips_symlinked_final_tmpdir(tmp_path: Path) -> None:
    import voyage.cli_generate as gen_ops

    run_dir = tmp_path / "run"
    real = tmp_path / "real-final"
    real.mkdir()
    _touch(real / "window.wav")
    run_dir.mkdir(parents=True)
    link = run_dir / "voyage-final-link"
    link.symlink_to(real, target_is_directory=True)
    assert gen_ops._heal_safe_transients(run_dir) == 0
    assert link.is_symlink()
    assert (real / "window.wav").is_file()


def test_cmd_generate_heals_before_pre_finalize_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voyage.cli_generate as gen_ops
    from tests.conftest import initialize_run_directory

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "output" / "healme"
    initialize_run_directory(run_dir, run_id="healme")
    _touch(run_dir / "voyage-final-stale" / "w.wav")
    seen: list[str] = []
    real_heal = gen_ops._heal_safe_transients

    def _spying_heal(path: Path) -> int:
        seen.append("heal")
        return real_heal(path)

    def _spying_validate(path: Path) -> list[str]:
        seen.append("validate")
        assert not (run_dir / "voyage-final-stale").exists()
        return []

    monkeypatch.setattr(gen_ops, "_heal_safe_transients", _spying_heal)
    monkeypatch.setattr(gen_ops, "validate_run", _spying_validate)
    from voyage.persistence import read_effective_config, read_manifest

    manifest = read_manifest(run_dir)
    errors = gen_ops._pre_finalize_errors(run_dir, manifest, read_effective_config(run_dir))
    assert errors == []
    # _pre_finalize_errors heals before validating: the spied validate
    # observes the tmpdir already gone, and heal ran exactly once first.
    assert seen == ["heal", "validate"]


def test_heal_and_report_prints_only_when_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.cli_generate as gen_ops

    run_dir = tmp_path / "run"
    assert gen_ops._heal_and_report(run_dir) == 0
    assert capsys.readouterr().out == ""
    _touch(run_dir / "audio" / "take.wav.partial")
    assert gen_ops._heal_and_report(run_dir) == 1
    assert capsys.readouterr().out == "healed: removed 1 transient file(s)/dir(s)\n"

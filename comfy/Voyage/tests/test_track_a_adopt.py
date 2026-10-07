"""Track A: DONE-less adoption + tail verification + fail-loud manifest (DESIGN §§58, 73).

Covers `verify_doneless_segment_for_adoption` (verifiable → True,
unverifiable → False, torn manifest → MediaError), the discard
adopt-vs-delete with checksums/size logging and DONE re-stat under lock,
and the adoption-path tail refusal wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.errors import MediaError
from voyage.persistence import read_effective_config, read_state, write_state
from voyage.supervisor import Supervisor, verify_doneless_segment_for_adoption


def _commit_one_fake_segment(run_dir: Path) -> Path:
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    committed = supervisor.run_segments(1)
    assert committed == ["000000"]
    return paths.segment_dir(run_dir, "000000")


def _rewind_state(run_dir: Path) -> None:
    state = read_state(run_dir)
    state.next_segment_number = 0
    state.committed_segments = 0
    state.timeline_frames = 0
    state.decision_index = 0
    write_state(run_dir, state)


def _geometry(run_dir: Path) -> dict[str, int]:
    effective = read_effective_config(run_dir)
    return {
        "width": int(effective.video.width),
        "height": int(effective.video.height),
        "fps": int(effective.video.fps),
        "segment_frames": int(effective.video.segment_frames),
    }


def test_verify_doneless_verifiable_returns_true(tmp_path: Path) -> None:
    """A real commit stripped of DONE still verifies (adoptable)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-adopt-true")
    segment = _commit_one_fake_segment(run_dir)
    (segment / paths.DONE_MARKER).unlink()
    _rewind_state(run_dir)
    assert verify_doneless_segment_for_adoption(run_dir, segment, **_geometry(run_dir)) is True


def test_verify_doneless_checksum_mismatch_returns_false(tmp_path: Path) -> None:
    """Corrupted media is unverifiable (delete after logging), not an error."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-adopt-false")
    segment = _commit_one_fake_segment(run_dir)
    (segment / paths.DONE_MARKER).unlink()
    _rewind_state(run_dir)
    with (segment / "video.mp4").open("ab") as handle:
        handle.write(b"\x00" * 64)
    assert verify_doneless_segment_for_adoption(run_dir, segment, **_geometry(run_dir)) is False


def test_verify_doneless_torn_manifest_raises(tmp_path: Path) -> None:
    """A torn manifest fails loud (never silent deletion)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-adopt-torn")
    segment = _commit_one_fake_segment(run_dir)
    (segment / paths.DONE_MARKER).unlink()
    _rewind_state(run_dir)
    (segment / "manifest.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(MediaError, match="inspect or remove"):
        verify_doneless_segment_for_adoption(run_dir, segment, **_geometry(run_dir))


def test_discard_adopts_verifiable_doneless(tmp_path: Path) -> None:
    """Discard marks a verifiable DONE-less dir DONE (removed == 0)."""
    from voyage.cli_generate import _discard_uncommitted_segments

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-discard-adopt")
    segment = _commit_one_fake_segment(run_dir)
    (segment / paths.DONE_MARKER).unlink()
    _rewind_state(run_dir)
    effective = read_effective_config(run_dir)
    removed = _discard_uncommitted_segments(run_dir, 0, effective)
    assert removed == 0
    assert (segment / paths.DONE_MARKER).is_file()


def test_discard_deletes_unverifiable_and_keeps_done(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unverifiable DONE-less dirs go (with checksums/size log); DONE stays."""
    from voyage.cli_generate import _discard_uncommitted_segments

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-discard-delete")
    good = _commit_one_fake_segment(run_dir)
    (good / paths.DONE_MARKER).unlink()
    _rewind_state(run_dir)
    # Corrupt the video so verification fails (unverifiable → delete).
    with (good / "video.mp4").open("ab") as handle:
        handle.write(b"\x00" * 64)
    effective = read_effective_config(run_dir)
    removed = _discard_uncommitted_segments(run_dir, 0, effective)
    assert removed == 1
    assert not good.exists()
    assert "discard 000000" in capsys.readouterr().err


def test_discard_torn_manifest_fails_loud_without_delete(tmp_path: Path) -> None:
    """A torn DONE-less manifest refuses deletion (fail loud, dir survives)."""
    from voyage.cli_generate import _discard_uncommitted_segments

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-discard-loud")
    segment = _commit_one_fake_segment(run_dir)
    (segment / paths.DONE_MARKER).unlink()
    _rewind_state(run_dir)
    (segment / "manifest.json").write_text("{torn", encoding="utf-8")
    effective = read_effective_config(run_dir)
    with pytest.raises(MediaError, match="refusing to delete"):
        _discard_uncommitted_segments(run_dir, 0, effective)
    assert segment.is_dir()


def test_discard_restats_done_under_lock(tmp_path: Path) -> None:
    """A DONE written between scan and lock is never deleted (TOCTOU-close)."""
    from voyage import cli_generate as gen_ops

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-discard-race")
    segment = run_dir / "segments" / "000000"
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "video.mp4").write_bytes(b"fake-video")
    effective = read_effective_config(run_dir)
    original_iterdir = Path.iterdir

    def _racing_iterdir(self: Path):  # type: ignore[no-untyped-def]
        yield from original_iterdir(self)
        if self == run_dir / "segments":
            (segment / "DONE").write_bytes(b"")

    import unittest.mock as mock

    with mock.patch.object(Path, "iterdir", _racing_iterdir):
        removed = gen_ops._discard_uncommitted_segments(run_dir, 0, effective)
    assert removed == 0
    assert (segment / "DONE").is_file()


def test_adopt_tape_tail_mismatch_refuses_like_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adoption calls the tail check; False refuses like a checksum failure."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-tail-refuse")
    segment = _commit_one_fake_segment(run_dir)
    _rewind_state(run_dir)
    # Plant a recovery.pt so the adoption path reaches the tail gate,
    # then force the gate to report mismatch.
    (segment / "recovery.pt").write_bytes(b"tape")
    monkeypatch.setattr(
        Supervisor, "_tape_tail_sha_matches", staticmethod(lambda _seg, _tape: False)
    )
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        with pytest.raises(MediaError, match="conditioning-tail"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    assert (segment / paths.DONE_MARKER).exists()
    assert read_state(run_dir).next_segment_number == 0


def test_adopt_catches_media_error_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A MediaError-raising manifest read maps to the refusal (not raw)."""
    import voyage.supervisor as supervisor_module

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="track-a-adopt-media")
    segment = _commit_one_fake_segment(run_dir)
    _rewind_state(run_dir)
    assert (segment / paths.DONE_MARKER).is_file()

    def _boom(_segment: Path) -> dict[str, object]:
        from voyage.errors import MediaError as _MediaError

        raise _MediaError("torn on purpose")

    monkeypatch.setattr(supervisor_module, "load_segment_manifest", _boom)
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        with pytest.raises(MediaError, match="refusing to re-render"):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()

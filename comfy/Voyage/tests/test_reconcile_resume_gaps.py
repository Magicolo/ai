"""Generate-reconcile + finalize-tmpdir + prewarm-nothing hardening (DESIGN §§56/58/59/140).

Why this module exists: three gaps mapped by read-only sweeps, each silent
data loss or silent idle in production. The reconcile discard deleted
numeric segment dirs at/after the committed count even when they already
carried DONE (colliding with the supervisor checksum-adopt path); finalize
staged everything in `run/voyage-final-*` with no prune on entry (a SIGKILL
orphans the dir forever, outside the orphan scan); a full pre-warm pass
doing zero work stayed invisible on a compact console. Each test below pins
one fixed behavior with a fake sink/dir — no GPU, no ffmpeg, no supervisor.
"""

from __future__ import annotations

from pathlib import Path

from voyage.augment_background import PrewarmResult


class FakeProgressSink:
    """Minimal `progress.note` sink for loud-line assertions (DESIGN §59)."""

    def __init__(self, *, verbose: bool = False) -> None:
        self.notes: list[str] = []
        self.verbose = verbose

    def note(self, message: str) -> None:
        self.notes.append(message)


def _make_segment_directory(
    run_dir: Path, segment_name: str, *, with_done: bool, done_text: str = ""
) -> Path:
    segment = run_dir / "segments" / segment_name
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "video.mp4").write_bytes(b"fake-video")
    if with_done:
        (segment / "DONE").write_text(done_text, encoding="utf-8")
    return segment


def test_discard_keeps_done_bearing_directory() -> None:
    """A DONE-bearing uncommitted dir survives discard for adopt-or-refuse (issue 013)."""
    import tempfile

    from voyage import paths
    from voyage.cli_generate import _discard_uncommitted_segments

    with tempfile.TemporaryDirectory() as raw_run:
        run_dir = Path(raw_run)
        (run_dir / paths.SEGMENTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        _make_segment_directory(run_dir, "000000", with_done=True, done_text="")
        _make_segment_directory(run_dir, "000001", with_done=True, done_text="")
        removed = _discard_uncommitted_segments(run_dir, 1)
        assert removed == 0
        assert (run_dir / "segments" / "000000" / "DONE").exists()
        assert (run_dir / "segments" / "000001" / "DONE").exists()


def test_discard_keeps_nonempty_done_marker() -> None:
    """A non-empty DONE file is also adoptable — existence, not size, is the signal."""
    import tempfile

    from voyage.cli_generate import _discard_uncommitted_segments

    with tempfile.TemporaryDirectory() as raw_run:
        run_dir = Path(raw_run)
        _make_segment_directory(run_dir, "000002", with_done=True, done_text="done\n")
        removed = _discard_uncommitted_segments(run_dir, 2)
        assert removed == 0
        assert (run_dir / "segments" / "000002").is_dir()
        assert (run_dir / "segments" / "000002" / "DONE").is_file()


def test_discard_removes_done_less_directory_and_transients() -> None:
    """DONE-less uncommitted dirs + partials go; DONE-bearing + non-matching stay."""
    import tempfile

    from voyage import paths
    from voyage.cli_generate import _discard_uncommitted_segments

    with tempfile.TemporaryDirectory() as raw_run:
        run_dir = Path(raw_run)
        segments = run_dir / paths.SEGMENTS_DIRNAME
        segments.mkdir(parents=True, exist_ok=True)
        _make_segment_directory(run_dir, "000000", with_done=True, done_text="")
        _make_segment_directory(run_dir, "000001", with_done=False)
        _make_segment_directory(run_dir, "000002", with_done=True, done_text="")
        (segments / "000003.partial").write_text("staging", encoding="utf-8")
        (segments / "000004.tmp.npy").write_bytes(b"staging")
        (segments / "notes.txt").write_text("operator notes", encoding="utf-8")
        kept_directory = segments / "operator-backup"
        kept_directory.mkdir(parents=True, exist_ok=True)
        removed = _discard_uncommitted_segments(run_dir, 1)
        assert not (segments / "000001").exists()
        assert not (segments / "000003.partial").exists()
        assert not (segments / "000004.tmp.npy").exists()
        assert (segments / "000000" / "DONE").exists()
        assert (segments / "000002" / "DONE").exists()
        assert (segments / "notes.txt").is_file()
        assert kept_directory.is_dir()
        assert removed == 3


def test_prune_stale_finalize_tmpdirs_keeps_non_matching() -> None:
    """Only `voyage-final-*` dirs go; files, symlinks, and other dirs stay (DESIGN §56)."""
    import tempfile

    from voyage.media import FINALIZE_TMPDIR_PREFIX, prune_stale_finalize_tmpdirs

    with tempfile.TemporaryDirectory() as raw_run:
        run_dir = Path(raw_run)
        stale_one = run_dir / f"{FINALIZE_TMPDIR_PREFIX}abc123"
        stale_one.mkdir(parents=True, exist_ok=True)
        (stale_one / "final.mp4").write_bytes(b"partial")
        stale_two = run_dir / f"{FINALIZE_TMPDIR_PREFIX}xyz789"
        stale_two.mkdir(parents=True, exist_ok=True)
        kept_segments = run_dir / "segments"
        kept_segments.mkdir(parents=True, exist_ok=True)
        kept_plain = run_dir / "voyage-final"
        kept_plain.mkdir(parents=True, exist_ok=True)
        kept_file = run_dir / f"{FINALIZE_TMPDIR_PREFIX}not-a-dir.txt"
        kept_file.write_text("not a directory", encoding="utf-8")
        target = run_dir / "real-target"
        target.mkdir(parents=True, exist_ok=True)
        (target / "video.mp4").write_bytes(b"real")
        link = run_dir / f"{FINALIZE_TMPDIR_PREFIX}link123"
        try:
            link.symlink_to(target, target_is_directory=True)
            link_created = True
        except OSError:
            link_created = False
        removed = prune_stale_finalize_tmpdirs(run_dir)
        assert not stale_one.exists()
        assert not stale_two.exists()
        assert kept_segments.is_dir()
        assert kept_plain.is_dir()
        assert kept_file.is_file()
        if link_created:
            assert link.is_symlink()
        assert removed == 2


def test_prune_missing_run_directory_is_best_effort() -> None:
    """An unreadable/missing run dir prunes zero and never raises (DESIGN §56)."""
    import tempfile

    from voyage.media import prune_stale_finalize_tmpdirs

    with tempfile.TemporaryDirectory() as raw_parent:
        missing = Path(raw_parent) / "no-such-run"
        assert prune_stale_finalize_tmpdirs(missing) == 0


def _empty_zero_result() -> PrewarmResult:
    return PrewarmResult(
        segments_seen=1,
        upscale_chunks_done=0,
        upscale_chunks_skipped=0,
        interp_chunks_done=0,
        interp_chunks_skipped=0,
        interp_chunks_waiting=0,
        upscale_frames_done=0,
        interp_frames_done=0,
        skip_reason="upscale skipped: 0.2 GiB free on cuda:1 < 1.0 GiB needed",
    )


def test_prewarm_nothing_emits_loud_note_without_verbose() -> None:
    """A zero-chunk zero-hit pass emits the loud line even when not verbose (DESIGN §59)."""
    from voyage.augment_background import PREWARM_NOTHING_LOUD_NOTE, report_prewarm_pass

    result = _empty_zero_result()
    sink = FakeProgressSink(verbose=False)
    report_prewarm_pass(result, sink)
    assert sink.notes == [PREWARM_NOTHING_LOUD_NOTE]


def test_prewarm_nothing_detail_stays_verbose_only() -> None:
    """The per-leg reason rides a second line only when verbose (DESIGN §59)."""
    from voyage.augment_background import report_prewarm_pass

    verbose_sink = FakeProgressSink(verbose=True)
    report_prewarm_pass(_empty_zero_result(), verbose_sink, verbose=True)
    assert len(verbose_sink.notes) == 2
    assert verbose_sink.notes[1].startswith("pre-warm detail: ")

    explicit_sink = FakeProgressSink(verbose=False)
    report_prewarm_pass(_empty_zero_result(), explicit_sink, verbose=False)
    assert len(explicit_sink.notes) == 1


def test_prewarm_useful_work_and_moot_stay_silent() -> None:
    """Ledger hits and moot passes never emit the loud nothing line (DESIGN §140)."""
    from voyage.augment_background import PrewarmResult, report_prewarm_pass

    skipped_hit = PrewarmResult(
        segments_seen=1,
        upscale_chunks_done=0,
        upscale_chunks_skipped=3,
        interp_chunks_done=0,
        interp_chunks_skipped=0,
        interp_chunks_waiting=0,
        upscale_frames_done=96,
        interp_frames_done=0,
    )
    silent_sink = FakeProgressSink()
    report_prewarm_pass(skipped_hit, silent_sink)
    assert silent_sink.notes == []

    moot_sink = FakeProgressSink()
    report_prewarm_pass(None, moot_sink)
    assert moot_sink.notes == []


def test_background_stop_report_nothing_all_run(tmp_path: Path) -> None:
    """Zero chunks + zero frames across passes emits the loud stop line (DESIGN §140)."""
    from voyage.augment_background import (
        PREWARM_NOTHING_LOUD_NOTE,
        BackgroundPrewarm,
        PrewarmResult,
    )
    from voyage.config import preset_config

    config = preset_config("prewarm-stop", "pastel neon line-art, peaceful", 11)
    driver = BackgroundPrewarm(tmp_path, config)
    driver._ledgered = (2, 0, 0, 0, 0, 0.0, 0.0)
    driver._last_result = PrewarmResult(
        segments_seen=1,
        upscale_chunks_done=0,
        upscale_chunks_skipped=0,
        interp_chunks_done=0,
        interp_chunks_skipped=0,
        interp_chunks_waiting=0,
        upscale_frames_done=0,
        interp_frames_done=0,
        skip_reason="director busy past 60s idle wait",
    )
    assert driver.did_nothing_all_run() is True
    sink = FakeProgressSink()
    driver.report_at_stop(sink)
    assert sink.notes == [PREWARM_NOTHING_LOUD_NOTE]


def test_background_stop_report_silent_when_helped_or_moot(tmp_path: Path) -> None:
    """Helped runs and moot-only runs never emit the stop nothing line (DESIGN §140)."""
    from voyage.augment_background import BackgroundPrewarm
    from voyage.config import preset_config

    config = preset_config("prewarm-stop-quiet", "pastel neon line-art, peaceful", 11)
    helped = BackgroundPrewarm(tmp_path, config)
    helped._ledgered = (2, 4, 0, 128, 0, 1.0, 0.0)
    helped._last_result = None
    assert helped.did_nothing_all_run() is False
    helped_sink = FakeProgressSink()
    helped.report_at_stop(helped_sink)
    assert helped_sink.notes == []

    moot_only = BackgroundPrewarm(tmp_path, config)
    moot_only._ledgered = (2, 0, 0, 0, 0, 0.0, 0.0)
    moot_only._last_result = None
    assert moot_only.did_nothing_all_run() is False
    moot_sink = FakeProgressSink()
    moot_only.report_at_stop(moot_sink)
    assert moot_sink.notes == []

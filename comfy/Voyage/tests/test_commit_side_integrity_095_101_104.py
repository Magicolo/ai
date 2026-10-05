"""Commit-side integrity regression tests (issues 095, 101, 104).

Why this module exists: the three Rank-2 commit-side integrity gaps share
one blast radius — a committed segment that validates clean while carrying
corrupt metadata, an unledgered take, or a poisoned audio walk. Each test
below fails on the pre-fix tree (watched fail in-container) and passes
after the fix, so the suite pins the repaired contract instead of the bug.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import initialize_run_directory
from voyage.audio.planner import AudioTake, append_take
from voyage.cli_validate import _check_segment_checksums, validate_run
from voyage.errors import MediaError, StateError
from voyage.persistence import read_effective_config

METADATA_ARTIFACTS = (
    "metrics.json",
    "transition.json",
    "prompt_plan.json",
    "audio_state.json",
    "world_state.json",
)
"""Segment JSON artifacts the checksum manifest must cover (issue 095)."""


def _write_segment_files(segment: Path) -> None:
    """Create a minimal segment with media bytes + all metadata files."""
    from voyage.hashing import sha256_file

    segment.mkdir(parents=True, exist_ok=True)
    (segment / "video.mp4").write_bytes(b"fake-video-bytes")
    (segment / "audio.wav").write_bytes(b"fake-audio-bytes")
    for name in METADATA_ARTIFACTS:
        (segment / name).write_text(json.dumps({"artifact": name}), encoding="utf-8")
    checksums = {
        name: sha256_file(segment / name)
        for name in ("video.mp4", "audio.wav", *METADATA_ARTIFACTS)
    }
    (segment / "sha256.json").write_text(json.dumps(checksums), encoding="utf-8")


def test_checksum_detects_corrupt_metrics() -> None:
    """Issue 095: a byte-preserving metrics mutation must trip the checksum."""
    segment = Path("/tmp") / "095-unused"
    del segment
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        segment_path = Path(scratch) / "000000"
        _write_segment_files(segment_path)
        payload = json.loads((segment_path / "metrics.json").read_text(encoding="utf-8"))
        payload["artifact"] = "tampered"
        (segment_path / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        errors = _check_segment_checksums(segment_path)
        assert any("metrics.json" in error for error in errors)


def test_legacy_checksum_without_metadata_stays_valid() -> None:
    """Issue 095: old two-entry manifests must not newly fail (additive)."""
    import tempfile

    from voyage.hashing import sha256_file

    with tempfile.TemporaryDirectory() as scratch:
        segment_path = Path(scratch) / "000000"
        segment_path.mkdir(parents=True, exist_ok=True)
        (segment_path / "video.mp4").write_bytes(b"fake-video-bytes")
        (segment_path / "audio.wav").write_bytes(b"fake-audio-bytes")
        for name in METADATA_ARTIFACTS:
            (segment_path / name).write_text(json.dumps({"artifact": name}), encoding="utf-8")
        (segment_path / "sha256.json").write_text(
            json.dumps(
                {
                    "video.mp4": sha256_file(segment_path / "video.mp4"),
                    "audio.wav": sha256_file(segment_path / "audio.wav"),
                }
            ),
            encoding="utf-8",
        )
        assert _check_segment_checksums(segment_path) == []


def test_commit_writes_full_checksum_manifest(tmp_path: Path) -> None:
    """Pruned layout: a real fake-backend commit writes manifest + DONE only.

    Video-only commit (all backends deferred): checksums cover video.mp4
    plus recovery.pt when the backend emits a tape; audio.wav is never
    required (old runs may still carry the entry, ignored by validators).
    """
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="checksum")
    from voyage.supervisor import Supervisor

    config = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    segment = run_dir / "segments" / "000000"
    manifest = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == 1
    checksums = manifest["checksums"]
    assert isinstance(checksums.get("video.mp4"), str) and checksums["video.mp4"]
    assert "audio.wav" not in checksums
    assert not (segment / "audio.wav").exists()
    for name, digest in checksums.items():
        assert isinstance(digest, str) and digest
        assert (segment / name).exists(), name
    for name in METADATA_ARTIFACTS:
        assert not (segment / name).exists(), name
    assert not (segment / "sha256.json").exists()
    assert validate_run(run_dir) == []


def test_commit_detects_post_commit_metrics_tamper(tmp_path: Path) -> None:
    """Flipping manifest metrics after commit must fail validate (timeline)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="tamper")
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest
    from voyage.supervisor import Supervisor

    config = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    segment = run_dir / "segments" / "000000"
    manifest = load_segment_manifest(segment)
    metrics = dict(manifest["metrics"])
    metrics["frames"] = int(metrics["frames"]) + 10
    write_segment_manifest(segment, {**manifest, "metrics": metrics})
    errors = validate_run(run_dir)
    assert errors, "mutated frame count must fail validate"
    assert any("frames" in error or "timeline" in error for error in errors)


def test_verify_segment_ignores_metadata_tamper(tmp_path: Path) -> None:
    """Manifest metadata is not checksummed: transition edits pass verify."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="verify")
    from voyage.media import _verify_segment
    from voyage.segment_manifest import load_segment_manifest, write_segment_manifest
    from voyage.supervisor import Supervisor

    config = read_effective_config(run_dir)
    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    segment = run_dir / "segments" / "000000"
    manifest = load_segment_manifest(segment)
    transition = dict(manifest["transition"])
    transition["tampered"] = True
    write_segment_manifest(segment, {**manifest, "transition": transition})
    _verify_segment(segment)


def test_append_take_syncs_directory_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue 101: the ledger append must persist the directory entry too."""
    import voyage.audio.planner as planner_module

    sync_calls: list[Path] = []
    original_fsync_dir = getattr(planner_module, "fsync_dir", None)
    assert original_fsync_dir is not None, "planner must import fsync_dir (issue 101)"

    def _recording_fsync(target: Path) -> None:
        sync_calls.append(target)
        import voyage.atomic as atomic_module

        atomic_module.fsync_dir(target)

    monkeypatch.setattr(planner_module, "fsync_dir", _recording_fsync)
    ledger = tmp_path / "audio" / "takes.jsonl"
    append_take(
        ledger,
        AudioTake(
            take_id="take_0000",
            path="audio/take_0000.wav",
            caption="ambient",
            seed=1,
            covers_from=0.0,
            duration=45.0,
            segment_index=0,
        ),
    )
    assert sync_calls == [ledger.parent]


def _take_record(**overrides: object) -> dict[str, object]:
    """Valid ledger record with caller overrides applied."""
    record: dict[str, object] = {
        "take_id": "take_0000",
        "path": "audio/take_0000.wav",
        "caption": "ambient",
        "seed": 1,
        "covers_from": 0.0,
        "duration": 45.0,
        "segment_index": 0,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "overrides",
    [
        {"duration": float("nan")},
        {"duration": float("inf")},
        {"duration": -5.0},
        {"duration": 0.0},
        {"covers_from": float("nan")},
        {"covers_from": float("inf")},
        {"covers_from": -1.0},
        {"bpm": float("nan")},
        {"bpm": float("inf")},
        {"bpm": -120.0},
        {"bpm": 0.0},
    ],
)
def test_take_from_dict_rejects_degenerate_geometry(overrides: dict[str, object]) -> None:
    """Issue 104: corrupt ledger geometry must fail loud at load, never in the walk."""
    with pytest.raises(StateError):
        AudioTake.from_dict(_take_record(**overrides))


def test_take_from_dict_accepts_tiny_positive_duration() -> None:
    """Issue 104: a 1 ms take loads (the walk bound, not the loader, owns tiny)."""
    take = AudioTake.from_dict(_take_record(duration=0.001))
    assert take.duration == pytest.approx(0.001)


def _sine_wav(path: Path, seconds: float) -> Path:
    """Render a mono sine WAV (real ffmpeg, CPU-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=8000:duration={seconds}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "8000",
            "-ac",
            "1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return path


def test_slice_take_rejects_tiny_piece(tmp_path: Path) -> None:
    """Issue 104: sub-50 ms slices are never legitimate music coverage."""
    from voyage.media import slice_take

    take_file = _sine_wav(tmp_path / "take.wav", 2.0)
    with pytest.raises(MediaError):
        slice_take(take_file, 0.0, 0.01, tmp_path / "slice.wav", 8000, 1)


def test_finalize_window_bounds_tiny_takes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue 104 finalize side: degenerate ledgers fail loud, not spawn thousands.

    Ledger-only finalize (no concat fallback): 10 ms slivers are never
    legitimate music coverage, so build_final_audio raises instead of
    slicing thousands of pieces.
    """
    import voyage.media as media_module
    from voyage.media import build_final_audio

    run_dir = tmp_path / "run"
    segments = [run_dir / "segments" / "000000", run_dir / "segments" / "000001"]
    for segment in segments:
        segment.mkdir(parents=True, exist_ok=True)
        (segment / "metrics.json").write_text(json.dumps({"frames": 48}), encoding="utf-8")
        _sine_wav(segment / "audio.wav", 2.0)
    take_file = _sine_wav(run_dir / "audio" / "take.wav", 4.0)
    assert take_file.exists()
    ledger = run_dir / "audio" / "takes.jsonl"
    for index in range(400):
        append_take(
            ledger,
            AudioTake(
                take_id=f"take_{index:04d}",
                path="audio/take.wav",
                caption="ambient",
                seed=index,
                covers_from=index * 0.01,
                duration=0.01,
                segment_index=0,
            ),
        )
    slice_calls: list[tuple[float, float]] = []
    original_cached = media_module._cached_slice_take

    def _counting_cached(
        slice_cache: dict[tuple[str, str, str], Path],
        take_path: Path,
        start_seconds: float,
        duration_seconds: float,
        dest: Path,
        sample_rate: int,
        channels: int,
    ) -> Path:
        slice_calls.append((start_seconds, duration_seconds))
        return original_cached(
            slice_cache, take_path, start_seconds, duration_seconds, dest, sample_rate, channels
        )

    monkeypatch.setattr(media_module, "_cached_slice_take", _counting_cached)
    with pytest.raises(MediaError, match="sliver|too many slices|no rendered takes"):
        build_final_audio(run_dir, segments, tmp_path, 24, 8000, 1)
    assert len(slice_calls) < 50

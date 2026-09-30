"""Run-file pruning: on-demand `video_tail.mp4` derivation (DESIGN §§5.3-5.4).

`generate_blocks` (ltxv + causvid) no longer persists `video_tail.mp4`;
resume derives the last 25 frames from the sibling segment `video.mp4`
via ffmpeg, writes the derived file to the recorded tail path, and
proceeds. The tape sha stays advisory (never a resume hard-fail).

CPU-only: segment videos are real ffmpeg `testsrc` renders on `tmp_path`
throwaway dirs (fake-backend-first, real media); sessions are namespace
doubles (resume touches resident fields only, never the GPU stack).
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

from voyage.fake_backends import FakeVideoBackend
from voyage.workers import video_causvid, video_common, video_ltxv

_TESTSRC = FakeVideoBackend()


def _render_segment_video(
    path: Path,
    *,
    frames: int = 40,
    fps: int = 16,
    width: int = 64,
    height: int = 64,
    seed: int = 7,
) -> Path:
    """Render a real `testsrc` segment video (repo-standard fake media)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _TESTSRC.generate_segment(
        path, prompt="tail probe", seed=seed, width=width, height=height, fps=fps, frames=frames
    )
    return path


def _count_frames(path: Path) -> int:
    """Frame count via ffprobe (decodes — exact, no container estimate)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ffprobe failed for {path}: {proc.stderr[-500:]}"
    return int(proc.stdout.strip())


def _ltxv_session_double() -> Any:
    """Resume-only ltxv session double (no GPU fields touched by resume)."""
    return types.SimpleNamespace(_conditioning_tail_path=None, _last_prompt=None)


def _causvid_session_double() -> Any:
    """Resume-only causvid session double (no GPU fields touched by resume)."""
    return types.SimpleNamespace(
        _pending_tail_path=None,
        _pending_overlap=3,
        _start_latents=None,
        _last_prompt=None,
        _latent_shape=[1, 21, 16, 60, 104],
    )


def _ltxv_tape(tail: Path, *, sha: str | None = "0" * 64) -> dict[str, Any]:
    """JSON-round-tripped ltxv tape pointing at `tail` (on-disk format)."""
    kwargs: dict[str, Any] = {
        "source_segment_id": "000007",
        "conditioning_tail_path": str(tail),
        "prompts": ["amber dunes"],
        "seeds": [7],
        "width": 768,
        "height": 512,
        "fps": 24,
    }
    if sha is not None:
        kwargs["conditioning_tail_sha256"] = sha
    tape: dict[str, Any] = json.loads(json.dumps(video_ltxv.build_recovery_tape(**kwargs)))
    return tape


def _causvid_tape(tail: Path, *, sha: str | None = "0" * 64) -> dict[str, Any]:
    """JSON-round-tripped causvid tape pointing at `tail` (on-disk format)."""
    kwargs: dict[str, Any] = {
        "source_segment_id": "000007",
        "conditioning_tail_path": str(tail),
        "overlap_frames": 3,
        "num_frame_per_block": 3,
        "decoded_frames_per_rollout": 81,
        "novel_frames_per_rollout_count": 72,
        "rollouts": 2,
        "latent_shape": [1, 21, 16, 60, 104],
        "prompts": ["amber dunes", "teal spires"],
        "seeds": [7, 8],
        "width": 832,
        "height": 480,
        "fps": 16,
        "config_sha256": "2" * 64,
    }
    if sha is not None:
        kwargs["conditioning_tail_sha256"] = sha
    tape: dict[str, Any] = json.loads(json.dumps(video_causvid.build_recovery_tape(**kwargs)))
    return tape


def test_derive_target_name_is_stable() -> None:
    """The derive target keeps the historical filename (later resumes hit it).

    Worker modules alias the shared constant (pinned by
    test_video_common.py); the value itself is pinned here.
    """
    assert video_common.TAIL_FILENAME == "video_tail.mp4"


def test_find_segment_video_prefers_video_mp4(tmp_path: Path) -> None:
    _render_segment_video(tmp_path / "video.mp4", frames=30)
    _render_segment_video(tmp_path / "other.mp4", frames=30, seed=8)
    assert video_common.find_segment_video(tmp_path) == tmp_path / "video.mp4"


def test_find_segment_video_falls_back_to_newest_mp4(tmp_path: Path) -> None:
    older = _render_segment_video(tmp_path / "alpha.mp4", frames=30)
    newer = _render_segment_video(tmp_path / "beta.mp4", frames=30, seed=9)
    assert video_common.find_segment_video(tmp_path) == newer
    assert older.exists()


def test_find_segment_video_ignores_the_tail_file(tmp_path: Path) -> None:
    (tmp_path / "video_tail.mp4").write_bytes(b"stale-tail")
    assert video_common.find_segment_video(tmp_path) is None


def test_find_segment_video_none_when_empty(tmp_path: Path) -> None:
    assert video_common.find_segment_video(tmp_path) is None


def test_derive_trims_exactly_25_frames(tmp_path: Path) -> None:
    source = _render_segment_video(tmp_path / "video.mp4", frames=40)
    dest = tmp_path / "video_tail.mp4"
    assert video_common.derive_tail_from_segment_video(source, dest) == dest
    assert dest.is_file()
    assert _count_frames(dest) == 25


def test_derive_full_video_when_exactly_25_frames(tmp_path: Path) -> None:
    source = _render_segment_video(tmp_path / "video.mp4", frames=25)
    dest = tmp_path / "video_tail.mp4"
    video_common.derive_tail_from_segment_video(source, dest)
    assert _count_frames(dest) == 25


def test_derive_rejects_short_source(tmp_path: Path) -> None:

    source = _render_segment_video(tmp_path / "video.mp4", frames=10)
    with pytest.raises(ValueError, match="need 25"):
        video_common.derive_tail_from_segment_video(source, tmp_path / "video_tail.mp4")


def test_derive_rejects_missing_source(tmp_path: Path) -> None:

    with pytest.raises(ValueError, match="segment video missing"):
        video_common.derive_tail_from_segment_video(
            tmp_path / "gone.mp4", tmp_path / "video_tail.mp4"
        )


def test_derive_rejects_nonpositive_tail_frames(tmp_path: Path) -> None:

    source = _render_segment_video(tmp_path / "video.mp4", frames=30)
    with pytest.raises(ValueError, match="tail_frames must be positive"):
        video_common.derive_tail_from_segment_video(source, tmp_path / "tail.mp4", tail_frames=0)


def test_derive_rejects_undecodable_source(tmp_path: Path) -> None:

    source = tmp_path / "video.mp4"
    source.write_bytes(b"not a video at all")
    with pytest.raises(RuntimeError, match="ffprobe frame count failed"):
        video_common.derive_tail_from_segment_video(source, tmp_path / "video_tail.mp4")


def test_ensure_uses_existing_tail_as_before(tmp_path: Path) -> None:
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"resident-tail")
    outcome = video_common.ensure_conditioning_tail(tail)
    assert outcome.path == tail
    assert outcome.derived is False
    assert tail.read_bytes() == b"resident-tail"


def test_ensure_derives_missing_tail_from_sibling_video(tmp_path: Path) -> None:
    _render_segment_video(tmp_path / "video.mp4", frames=40)
    tail = tmp_path / "video_tail.mp4"
    outcome = video_common.ensure_conditioning_tail(tail)
    assert outcome.path == tail
    assert outcome.derived is True
    assert _count_frames(tail) == 25


def test_ensure_raises_when_tail_and_video_are_both_missing(tmp_path: Path) -> None:

    with pytest.raises(ValueError, match="conditioning tail missing"):
        video_common.ensure_conditioning_tail(tmp_path / "video_tail.mp4")


def test_ltxv_resume_derives_missing_tail_from_segment_video(tmp_path: Path) -> None:
    """Spec (b): missing tail + sibling video.mp4 → derive 25 frames, succeed."""
    segment = tmp_path / "seg"
    _render_segment_video(segment / "video.mp4", frames=40)
    tail = segment / "video_tail.mp4"
    tape = _ltxv_tape(tail)
    session = _ltxv_session_double()
    result = video_ltxv.LTXVSession.resume_from_tape(session, tape)
    assert result == {"resumed": True, "conditioning_tail_path": str(tail)}
    assert session._conditioning_tail_path == str(tail)
    assert session._last_prompt == "amber dunes"
    assert tail.is_file()
    assert _count_frames(tail) == 25
    assert tape["conditioning_tail_sha256"] == video_ltxv.sha256_file(tail)
    assert tape["conditioning_tail_sha256"] != "0" * 64


def test_ltxv_resume_skips_rehash_when_tape_carries_no_tail_hash(tmp_path: Path) -> None:
    segment = tmp_path / "seg"
    _render_segment_video(segment / "video.mp4", frames=40)
    tail = segment / "video_tail.mp4"
    tape = _ltxv_tape(tail, sha=None)
    assert "conditioning_tail_sha256" not in tape
    session = _ltxv_session_double()
    result = video_ltxv.LTXVSession.resume_from_tape(session, tape)
    assert result["resumed"] is True
    assert tail.is_file()
    assert "conditioning_tail_sha256" not in tape


def test_ltxv_resume_uses_existing_tail_as_before(tmp_path: Path) -> None:
    """Spec (c): existing tail wins even when its bytes mismatch the tape sha."""
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"resident-tail")
    tape = _ltxv_tape(tail, sha="1" * 64)
    session = _ltxv_session_double()
    result = video_ltxv.LTXVSession.resume_from_tape(session, tape)
    assert result == {"resumed": True, "conditioning_tail_path": str(tail)}
    assert session._conditioning_tail_path == str(tail)
    assert tail.read_bytes() == b"resident-tail"
    assert tape["conditioning_tail_sha256"] == "1" * 64


def test_ltxv_resume_raises_when_tail_and_video_are_both_missing(tmp_path: Path) -> None:

    tape = _ltxv_tape(tmp_path / "video_tail.mp4")
    with pytest.raises(ValueError, match="conditioning tail missing"):
        video_ltxv.LTXVSession.resume_from_tape(_ltxv_session_double(), tape)


def test_causvid_resume_derives_missing_tail_from_segment_video(tmp_path: Path) -> None:
    """Spec (b): missing tail + sibling video.mp4 → derive 25 frames, succeed."""
    segment = tmp_path / "seg"
    _render_segment_video(segment / "video.mp4", frames=72)
    tail = segment / "video_tail.mp4"
    tape = _causvid_tape(tail)
    session = _causvid_session_double()
    result = video_causvid.CausvidSession.resume_from_tape(session, tape)
    assert result["resumed"] is True
    assert result["conditioning_tail_path"] == str(tail)
    assert result["start_latents_from"] == video_causvid.START_FROM_RESUME
    assert session._pending_tail_path == str(tail)
    assert tail.is_file()
    assert _count_frames(tail) == 25
    assert tape["conditioning_tail_sha256"] == video_causvid.sha256_file(tail)
    assert tape["conditioning_tail_sha256"] != "0" * 64


def test_causvid_resume_skips_rehash_when_tape_carries_no_tail_hash(tmp_path: Path) -> None:
    segment = tmp_path / "seg"
    _render_segment_video(segment / "video.mp4", frames=72)
    tail = segment / "video_tail.mp4"
    tape = _causvid_tape(tail, sha=None)
    assert "conditioning_tail_sha256" not in tape
    session = _causvid_session_double()
    result = video_causvid.CausvidSession.resume_from_tape(session, tape)
    assert result["resumed"] is True
    assert tail.is_file()
    assert "conditioning_tail_sha256" not in tape


def test_causvid_resume_uses_existing_tail_as_before(tmp_path: Path) -> None:
    """Spec (c): existing tail wins even when its bytes mismatch the tape sha."""
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"resident-tail")
    tape = _causvid_tape(tail, sha="1" * 64)
    session = _causvid_session_double()
    result = video_causvid.CausvidSession.resume_from_tape(session, tape)
    assert result["resumed"] is True
    assert result["conditioning_tail_path"] == str(tail)
    assert session._pending_tail_path == str(tail)
    assert tail.read_bytes() == b"resident-tail"
    assert tape["conditioning_tail_sha256"] == "1" * 64


def test_causvid_resume_raises_when_tail_and_video_are_both_missing(tmp_path: Path) -> None:

    tape = _causvid_tape(tmp_path / "video_tail.mp4")
    with pytest.raises(ValueError, match="conditioning tail missing"):
        video_causvid.CausvidSession.resume_from_tape(_causvid_session_double(), tape)


def test_build_tail_trim_argv_is_pure_frame_exact(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    dest = tmp_path / "video_tail.mp4"
    argv = video_common.build_tail_trim_argv(source, dest, start_frame=15, tail_frames=25)
    assert argv[-1] == str(dest)
    assert "-frames:v" in argv
    assert argv[argv.index("-frames:v") + 1] == "25"
    assert argv[argv.index("-vf") + 1] == "select='gte(n,15)'"
    assert "setpts" not in " ".join(argv)


def test_build_tail_trim_argv_rejects_bad_counts(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    dest = tmp_path / "video_tail.mp4"
    with pytest.raises(ValueError, match="tail_frames must be positive"):
        video_common.build_tail_trim_argv(source, dest, start_frame=0, tail_frames=0)
    with pytest.raises(ValueError, match="start_frame must be non-negative"):
        video_common.build_tail_trim_argv(source, dest, start_frame=-1, tail_frames=25)


def test_tail_start_frame_resolves_from_source(tmp_path: Path) -> None:
    source = _render_segment_video(tmp_path / "video.mp4", frames=40)
    assert video_common.tail_start_frame(source, 25) == 15


def test_tail_start_frame_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="segment video missing"):
        video_common.tail_start_frame(tmp_path / "gone.mp4", 25)

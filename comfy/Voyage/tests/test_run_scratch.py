"""Run-scoped generation scratch (boba /tmp-quota incident).

Every temp file a run produces — worker session work_roots, mux/assembly
staging — lives under `run_dir/tmp/`, never on the host /tmp tmpfs (31G
with usrquota; a 12-segment LTX25 run exhausted it mid-mux and tripped
the video circuit breaker with ENOSPC). The supervisor creates the dir,
passes it explicitly (`scratch_dir` init field) to the session-owning
workers, and points process TMPDIR at it as a backstop for bench
harnesses, bare `TemporaryDirectory` calls, and third-party libs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from voyage import paths
from voyage.config import AudioConfig, VideoConfig
from voyage.persistence import read_effective_config
from voyage.supervisor import Supervisor, point_temp_at_run_scratch
from voyage.workers import audio_acestep, sfx_mmaudio, video_common


def test_scratch_dir_lives_under_run_dir(tmp_path: Path) -> None:
    assert paths.scratch_dir(tmp_path / "run") == tmp_path / "run" / "tmp"
    assert paths.SCRATCH_DIRNAME == "tmp"


def test_ensure_scratch_dir_creates_and_prunes_stale_sessions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    scratch = paths.ensure_scratch_dir(run_dir)
    assert scratch.is_dir()
    # Stale session leftovers from a killed run (evict-time rmtree only
    # runs on clean shutdown) are pruned; anything else is kept.
    (scratch / "voyage-ltx25-deadbeef").mkdir()
    (scratch / "voyage-acestep-cwd-1234").mkdir()
    (scratch / "keep").mkdir()
    (scratch / "keep" / "note.txt").write_text("stay", encoding="utf-8")
    paths.ensure_scratch_dir(run_dir)
    assert not (scratch / "voyage-ltx25-deadbeef").exists()
    assert not (scratch / "voyage-acestep-cwd-1234").exists()
    assert (scratch / "keep" / "note.txt").is_file()


def test_session_scratch_parent_honors_payload(tmp_path: Path) -> None:
    scratch = tmp_path / "custom" / "tmp"
    parent = video_common.session_scratch_parent({"scratch_dir": str(scratch)})
    assert parent == scratch
    assert scratch.is_dir()


def test_session_scratch_parent_fallback_is_cwd_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Legacy/test callers without the field (workers always spawn with
    # CWD=run_dir) still land in the run, never /tmp.
    monkeypatch.chdir(tmp_path)
    parent = video_common.session_scratch_parent({})
    assert parent == tmp_path / "tmp"
    assert parent.is_dir()


def test_session_scratch_parent_rejects_non_str() -> None:
    with pytest.raises(TypeError, match="scratch_dir"):
        video_common.session_scratch_parent({"scratch_dir": 123})


def test_session_scratch_parent_prefers_stored_over_payload(tmp_path: Path) -> None:
    # Rebuild handlers pass their `_INIT_PARAMS` value (their payload
    # only carries the tape path); the stored dir wins when present.
    stored = tmp_path / "stored"
    from_payload = tmp_path / "payload"
    parent = video_common.session_scratch_parent(
        {"scratch_dir": str(from_payload)}, stored_scratch_dir=str(stored)
    )
    assert parent == stored
    assert stored.is_dir()
    assert not from_payload.exists()
    with pytest.raises(TypeError, match="scratch_dir"):
        video_common.session_scratch_parent({}, stored_scratch_dir=123)


def test_paths_staging_parent_makes_dir_or_defers_to_tmpdir(tmp_path: Path) -> None:
    assert paths.staging_parent(None) is None
    scratch = tmp_path / "run" / "tmp"
    assert paths.staging_parent(str(scratch)) == scratch
    assert scratch.is_dir()


def _scratch_config(tmp_path: Path):  # type: ignore[no-untyped-def]
    from tests.conftest import initialize_run_directory

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="run-scratch", style="probe", seed=11)
    config = read_effective_config(run_dir)
    config.video = VideoConfig(
        **{
            **config.video.model_dump(),
            "backend": "ltxv",
            "device": "cuda:0",
            "width": 768,
            "height": 512,
        }
    )
    config.audio = AudioConfig(
        **{**config.audio.model_dump(), "backend": "acestep", "device": "cuda:0"}
    )
    return run_dir, config


def test_supervisor_passes_scratch_to_video_and_audio(tmp_path: Path) -> None:
    run_dir, config = _scratch_config(tmp_path)
    supervisor = Supervisor(run_dir, config)
    assert supervisor._video._init_payload["scratch_dir"] == str(run_dir / "tmp")
    assert supervisor._audio._init_payload == {
        "models_dir": config.audio.models_dir,
        "device": "cuda:0",
        "scratch_dir": str(run_dir / "tmp"),
    }
    assert (run_dir / "tmp").is_dir()


def test_point_temp_at_run_scratch_redirects_bare_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "run" / "tmp"
    previous_tempdir = tempfile.tempdir
    monkeypatch.setenv("TMPDIR", "/nonexistent-before")
    try:
        point_temp_at_run_scratch(scratch)
        assert os.environ["TMPDIR"] == str(scratch)
        assert tempfile.tempdir == str(scratch)
        # A bare TemporaryDirectory (bench harnesses, third-party libs)
        # now lands in the run instead of host /tmp.
        with tempfile.TemporaryDirectory(prefix="voyage-probe-") as staged:
            assert Path(staged).parent == scratch
    finally:
        tempfile.tempdir = previous_tempdir


def test_acestep_init_records_scratch_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scratch = run_dir / "tmp"
    previous_cwd = Path.cwd()
    previous_models = audio_acestep._models_dir
    previous_device = audio_acestep._device
    previous_scratch = audio_acestep._scratch_dir
    previous_cache = audio_acestep._upstream_cache_dir
    os.chdir(run_dir)
    try:
        result = audio_acestep.handle_init(
            {"models_dir": "/models", "device": "cuda:0", "scratch_dir": str(scratch)}
        )
        assert result["status"] == "READY"
        assert audio_acestep._scratch_dir == str(scratch)
        assert paths.staging_parent(audio_acestep._scratch_dir) == scratch
        # The upstream-writes CWD redirect lands under the scratch too —
        # inside the run by design (never host /tmp), but away from the
        # run root so no `.cache/` litters the run directory.
        assert Path.cwd().parent == scratch
        assert Path.cwd() != run_dir
        assert not (run_dir / ".cache").exists()
        with pytest.raises(TypeError, match="scratch_dir"):
            audio_acestep.handle_init({"scratch_dir": 123})
    finally:
        os.chdir(previous_cwd)
        audio_acestep._models_dir = previous_models
        audio_acestep._device = previous_device
        audio_acestep._scratch_dir = previous_scratch
        audio_acestep._upstream_cache_dir = previous_cache


def test_acestep_staging_parent_falls_back_to_tmpdir_default() -> None:
    previous_scratch = audio_acestep._scratch_dir
    audio_acestep._scratch_dir = None
    try:
        assert paths.staging_parent(audio_acestep._scratch_dir) is None
    finally:
        audio_acestep._scratch_dir = previous_scratch


def test_sfx_init_records_scratch_dir(tmp_path: Path) -> None:
    scratch = tmp_path / "run" / "tmp"
    previous_models = sfx_mmaudio._models_dir
    previous_device = sfx_mmaudio._device
    previous_size = sfx_mmaudio._model_size
    previous_scratch = sfx_mmaudio._scratch_dir
    try:
        result = sfx_mmaudio.handle_init(
            {
                "models_dir": "/models",
                "device": "cuda:0",
                "model_size": "small_44k",
                "scratch_dir": str(scratch),
            }
        )
        assert result["status"] == "READY"
        assert sfx_mmaudio._scratch_dir == str(scratch)
        assert paths.staging_parent(sfx_mmaudio._scratch_dir) == scratch
        with pytest.raises(TypeError, match="scratch_dir"):
            sfx_mmaudio.handle_init({"scratch_dir": 123})
    finally:
        sfx_mmaudio._models_dir = previous_models
        sfx_mmaudio._device = previous_device
        sfx_mmaudio._model_size = previous_size
        sfx_mmaudio._scratch_dir = previous_scratch


def test_sfx_staging_parent_falls_back_to_tmpdir_default() -> None:
    previous_scratch = sfx_mmaudio._scratch_dir
    sfx_mmaudio._scratch_dir = None
    try:
        assert paths.staging_parent(sfx_mmaudio._scratch_dir) is None
    finally:
        sfx_mmaudio._scratch_dir = previous_scratch

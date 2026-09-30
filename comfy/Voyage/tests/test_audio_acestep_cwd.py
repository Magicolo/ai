"""Run-file pruning: the ACE-Step worker keeps upstream relative writes out (DESIGN §37).

The upstream ACE-Step library writes `.cache/acestep/progress_estimates.json`
relative to the worker process CWD, and workers spawn with CWD=run_dir — so
every music render littered the run directory. The worker redirects its own
CWD to a dedicated tmp dir at init (before any ACE-Step library call); all
voyage paths are absolute, so the chdir is side-effect free (the
`video_longlive`/`video_causvid` `_enter_*_tree` precedent).
"""

from __future__ import annotations

import os
from pathlib import Path

from voyage.workers import audio_acestep as audio_acestep_worker


def test_audio_acestep_init_redirects_cwd_away_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous_cwd = Path.cwd()
    previous_models_dir = audio_acestep_worker._models_dir
    previous_device = audio_acestep_worker._device
    previous_stack = audio_acestep_worker._stack
    previous_cache_dir = audio_acestep_worker._upstream_cache_dir
    os.chdir(run_dir)
    try:
        result = audio_acestep_worker.handle_init({"models_dir": "/models", "device": "cuda:0"})
        assert result["status"] == "READY"
        assert Path.cwd() != run_dir
        assert run_dir not in Path.cwd().parents
        assert not (run_dir / ".cache").exists()
    finally:
        os.chdir(previous_cwd)
        audio_acestep_worker._models_dir = previous_models_dir
        audio_acestep_worker._device = previous_device
        audio_acestep_worker._stack = previous_stack
        audio_acestep_worker._upstream_cache_dir = previous_cache_dir


def test_audio_acestep_init_redirect_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous_cwd = Path.cwd()
    previous_models_dir = audio_acestep_worker._models_dir
    previous_device = audio_acestep_worker._device
    previous_stack = audio_acestep_worker._stack
    previous_cache_dir = audio_acestep_worker._upstream_cache_dir
    os.chdir(run_dir)
    try:
        audio_acestep_worker.handle_init({})
        first = Path.cwd()
        audio_acestep_worker.handle_init({})
        assert Path.cwd() == first
        assert run_dir not in first.parents
    finally:
        os.chdir(previous_cwd)
        audio_acestep_worker._models_dir = previous_models_dir
        audio_acestep_worker._device = previous_device
        audio_acestep_worker._stack = previous_stack
        audio_acestep_worker._upstream_cache_dir = previous_cache_dir

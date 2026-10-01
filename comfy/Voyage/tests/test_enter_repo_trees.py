"""Repo-tree entry shim tolerates root-owned /opt checkouts (causvid perms).

`_enter_causvid_tree` links a `wan_models/`
symlink inside the upstream repo clone (`/opt/causvid`).
The clone is root-owned while the worker runs as the host user
(`--user` pin, issue 053 follow-up), so any write there raises
`PermissionError: [Errno 13]`. When the symlink is already correct the
shim must perform zero writes (skip the unlink/recreate) and just chdir.
(A removed backend's twin shim died with it, issue 079.)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voyage.workers import video_causvid

needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="read-only dirs stay writable for root"
)


def _restore_writable(target: Path) -> None:
    target.chmod(0o755)


@needs_non_root
def test_causvid_entry_skips_write_when_link_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "causvid"
    models_dir = tmp_path / "models"
    wanted = models_dir / "Wan2.1-T2V-1.3B"
    wanted.mkdir(parents=True)
    anchor_parent = repo_root / "wan_models"
    anchor_parent.mkdir(parents=True)
    anchor = anchor_parent / "Wan2.1-T2V-1.3B"
    anchor.symlink_to(wanted)
    monkeypatch.setenv("VOYAGE_CAUSVID_DIR", str(repo_root))
    anchor_parent.chmod(0o555)
    repo_root.chmod(0o555)
    previous = Path.cwd()
    try:
        video_causvid._enter_causvid_tree(models_dir)
        assert Path.cwd() == repo_root
        assert anchor.is_symlink()
        assert Path(os.readlink(anchor)) == wanted
    finally:
        os.chdir(previous)
        _restore_writable(anchor_parent)
        _restore_writable(repo_root)


def test_causvid_entry_repoints_stale_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "causvid"
    models_dir = tmp_path / "models"
    wanted = models_dir / "Wan2.1-T2V-1.3B"
    wanted.mkdir(parents=True)
    anchor_parent = repo_root / "wan_models"
    anchor_parent.mkdir(parents=True)
    anchor = anchor_parent / "Wan2.1-T2V-1.3B"
    anchor.symlink_to(tmp_path / "elsewhere")
    monkeypatch.setenv("VOYAGE_CAUSVID_DIR", str(repo_root))
    previous = Path.cwd()
    try:
        video_causvid._enter_causvid_tree(models_dir)
        assert Path.cwd() == repo_root
        assert Path(os.readlink(anchor)) == wanted
    finally:
        os.chdir(previous)

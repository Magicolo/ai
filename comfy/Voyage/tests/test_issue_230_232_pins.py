"""Dockerfile pin gates for issues 230/232 (text-only, no build).

CPU-only: plain text scans of the worker Dockerfiles — no docker, no GPU.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LTX_DOCKERFILE = REPO_ROOT / "worker" / "Dockerfile.ltx"
VIDEO_DOCKERFILE = REPO_ROOT / "worker" / "Dockerfile.video"


def test_ltx_sfx_venv_has_no_bare_packages() -> None:
    """Issue 232: the four bare SFX-venv rows stay `==`-pinned."""
    text = LTX_DOCKERFILE.read_text(encoding="utf-8")
    for pinned in (
        '"Pillow==',
        '"tqdm==',
        '"huggingface_hub==',
        '"numpy==',
    ):
        assert pinned in text, f"missing pin {pinned} in worker/Dockerfile.ltx"
    code_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    for bare in (" Pillow ", " tqdm ", " huggingface_hub ", " numpy "):
        assert bare not in code, f"bare package {bare.strip()} in worker/Dockerfile.ltx"


def test_ltx_apt_rows_carry_requery_procedure() -> None:
    """Issue 230: unpinned apt rows must document the exact re-query."""
    text = LTX_DOCKERFILE.read_text(encoding="utf-8")
    assert "dpkg-query -W" in text
    for package in ("python3.11", "ffmpeg", "curl", "ca-certificates"):
        assert package in text


def test_video_ffmpeg_stays_pinned_with_provenance_note() -> None:
    """Issue 230: jammy ffmpeg keeps its `=` pin plus the decision record."""
    text = VIDEO_DOCKERFILE.read_text(encoding="utf-8")
    assert "ffmpeg=7:4.4.2-0ubuntu0.22.04.1" in text
    assert "issue 230" in text

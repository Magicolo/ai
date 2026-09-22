"""Hardware / environment probing for `voyage doctor` (DESIGN §64).

Never trust a user-written profile blindly: query runtime facts.
GPU details are best-effort (may be absent in CPU-only containers);
ffmpeg presence is required.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any


def _capture(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def probe() -> dict[str, Any]:
    nvidia_smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpus: list[str] = nvidia_smi.splitlines() if nvidia_smi else []
    return {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "nvidia_smi": nvidia_smi is not None,
        "gpus": gpus,
        "torch_cuda": None,  # Resolved inside GPU worker envs (Phase 0: not required).
    }


def check_ffmpeg() -> tuple[bool, str]:
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found on PATH"
    if shutil.which("ffprobe") is None:
        return False, "ffprobe not found on PATH"
    return True, "ffmpeg + ffprobe present"

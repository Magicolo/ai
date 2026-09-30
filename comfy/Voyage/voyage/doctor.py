"""Hardware / environment probing for `voyage doctor` (DESIGN §64).

Never trust a user-written profile blindly: query runtime facts.
GPU details are best-effort (may be absent in CPU-only containers);
ffmpeg presence is required.

As-built note (issue 048): `probe()` covers python/ffmpeg/ffprobe/
nvidia-smi GPUs/torch-CUDA/disk/models-manifest presence; the remaining
§64 gaps (compute capability, CUDA runtime version, FlashAttention/Triton,
checkpoint compat, fs permissions, worker interpreters, ACE-Step) are
documented in `docs/TROUBLESHOOTING.md` and `docs/INSTALL.md` — full
model checks stay behind `voyage models verify`.

The supervisor package must never import torch/transformers/diffusers
at module scope (§83): torch is probed lazily behind a
`find_spec` guard so the slim image (no torch installed) pays nothing.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MODELS_DIR_ENVIRONMENT_VARIABLE = "VOYAGE_MODELS_DIR"
"""Env var naming the host models dir (mirrors `cli._models_dir`)."""

MODELS_DIR_DEFAULT = "/models"
"""Container mount point when the env var is unset."""

DIRECTOR_PYTHON_ENVIRONMENT_VARIABLE = "VOYAGE_DIRECTOR_PYTHON"
"""Env var naming the director venv interpreter (unified image, DESIGN §140).

The unified `voyage-video` image runs the director worker under
`/opt/venvs/director/bin/python` (CUDA torch + transformers 5.17 +
GPTQModel) while the supervisor stays on the video venv. Absent on the
slim image and on hosts — that simply means the legacy in-process
interpreter serves the director (CPU path).
"""

_SUBPROCESS_TIMEOUT_SECONDS = 15
"""Wall-clock cap per probe subprocess (nvidia-smi/ffmpeg must fail fast)."""

_BYTES_PER_GIB = 1024**3
"""Byte-to-GiB divisor for the disk-free fact (matches `cli.cmd_status`)."""


def _capture(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _torch_cuda() -> bool | None:
    """True/False when torch is importable, None when it is absent/broken."""
    try:
        if importlib.util.find_spec("torch") is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        import torch  # noqa: PLC0415 — lazy: supervisor image has no torch.

        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _disk_free_gib(path: Path) -> float | None:
    """Free space at `path` in GiB, None when the stat fails."""
    try:
        return shutil.disk_usage(path).free / _BYTES_PER_GIB
    except OSError:
        return None


def models_dir() -> Path:
    """Resolve the models dir the same way `cli._models_dir` does."""
    raw = os.environ.get(MODELS_DIR_ENVIRONMENT_VARIABLE, MODELS_DIR_DEFAULT)
    return Path(raw)


def director_python() -> Path | None:
    """Resolve the director venv interpreter, None when unset/empty."""
    raw = os.environ.get(DIRECTOR_PYTHON_ENVIRONMENT_VARIABLE, "")
    if not raw.strip():
        return None
    return Path(raw.strip())


def check_director_python(python: Path | None = None) -> dict[str, Any]:
    """Presence summary for the director venv interpreter (never raises)."""
    target = python if python is not None else director_python()
    if target is None:
        return {"path": None, "exists": False}
    try:
        exists = target.is_file()
    except OSError:
        exists = False
    return {"path": str(target), "exists": exists}


def check_models(present_dir: Path | None = None) -> dict[str, Any]:
    """Presence summary for the models dir (issue 048).

    Reports dir existence plus the per-backend `verify_*` results from
    `model_registry` (presence + size sanity only — never downloads).
    Never raises: registry import/stat failures degrade to `ok: False`.
    """
    target = present_dir if present_dir is not None else models_dir()
    try:
        dir_exists = target.is_dir()
    except OSError:
        dir_exists = False
    try:
        manifest_present = (target / "manifest.json").is_file()
    except OSError:
        manifest_present = False
    checks: dict[str, dict[str, Any]] = {}
    try:
        from voyage import model_registry  # noqa: PLC0415 — lazy: keep import light.
    except ImportError:
        return {
            "dir": str(target),
            "exists": dir_exists,
            "manifest": manifest_present,
            "checks": checks,
        }
    verifiers = (
        ("longlive2-bf16", model_registry.verify_longlive2_bf16),
        ("ltxv-2b", model_registry.verify_ltxv_models),
        ("causvid", model_registry.verify_causvid_models),
        ("director-qwen8b", model_registry.verify_director_models),
        ("audio-acestep", model_registry.verify_audio_models),
        ("inspector-qwen35", model_registry.verify_inspector_models),
    )
    for name, verify in verifiers:
        try:
            ok, message = verify(target)
        except Exception as error:
            ok, message = False, f"{name} check failed: {error}"
        checks[name] = {"ok": bool(ok), "message": str(message)}
    return {
        "dir": str(target),
        "exists": dir_exists,
        "manifest": manifest_present,
        "checks": checks,
    }


def probe() -> dict[str, Any]:
    nvidia_smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpus: list[str] = nvidia_smi.splitlines() if nvidia_smi else []
    ffmpeg_version: str | None = None
    version_output = _capture(["ffmpeg", "-version"])
    if version_output:
        ffmpeg_version = version_output.splitlines()[0]
    model_facts = check_models()
    models_ok = (
        bool(model_facts["exists"]) and all(check["ok"] for check in model_facts["checks"].values())
        if model_facts["checks"]
        else False
    )
    director = check_director_python()
    return {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "ffmpeg_version": ffmpeg_version,
        "nvidia_smi": nvidia_smi is not None,
        "gpus": gpus,
        "torch_cuda": _torch_cuda(),  # None when torch is absent/broken.
        "disk_free_gib": _disk_free_gib(Path("/")),
        "models": model_facts,
        "models_ok": models_ok,
        "director_python": director["path"],
        "director_python_exists": director["exists"],
    }


def check_ffmpeg() -> tuple[bool, str]:
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found on PATH"
    if shutil.which("ffprobe") is None:
        return False, "ffprobe not found on PATH"
    return True, "ffmpeg + ffprobe present"

"""Hardware / environment probing for `voyage doctor` (DESIGN §64).

Never trust a user-written profile blindly: query runtime facts.
GPU details are best-effort (may be absent in CPU-only containers);
ffmpeg presence is required.

As-built note (issue 048): `probe()` covers python/ffmpeg/ffprobe/
nvidia-smi GPUs/torch-CUDA/disk/models-manifest presence; the remaining
§64 gaps (FlashAttention/Triton, checkpoint compat, fs permissions,
worker interpreters, ACE-Step) are documented in
`docs/TROUBLESHOOTING.md` and `docs/INSTALL.md` — full model checks stay
behind `voyage models verify`.

As-built note (issue 066): `probe()` additionally reports per-mount
disks (`disk_by_mount`: root/models/tmp), per-GPU VRAM + compute
capability + temperature + driver, the torch CUDA runtime, and splits
`models_ok` into `models_ok_required` (optional stacks excluded) /
`models_ok_all`. `health_alerts` + `meets_reserve` turn those facts into
threshold WARN/CRIT lines (85%/90% disk, 15% VRAM, 84 C, reserve floor).

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

_MIB_PER_GIB = 1024.0
"""MiB-to-GiB divisor for the `nvidia-smi` memory columns."""

_OPTIONAL_MODEL_STACKS = frozenset({"inspector-qwen35"})
"""Model stacks that never flip the required flag (issue 066).

The VLM inspector stack is standalone tooling (the supervisor
piggyback was removed); a run must not read red just because its
snapshot is absent. Every other `check_models` entry is load-bearing
for some backend and stays required.
"""

_DISK_WARN_USED_FRACTION = 0.85
"""Disk usage fraction that earns a WARN (monitoring guidance, issue 066)."""

_DISK_CRIT_USED_FRACTION = 0.90
"""Disk usage fraction that earns a CRIT (past this, pause is near)."""

_VRAM_PRESSURE_FREE_FRACTION = 0.15
"""Free-VRAM fraction below which the GPU is flagged pressured (issue 066)."""

_GPU_TEMP_WARN_CELSIUS = 84.0
"""Sustained operating temp that earns a WARN (guidance threshold)."""

_SMI_COLUMN_TOTAL = 1
"""`nvidia-smi --query-gpu` CSV column holding memory.total (column 0 is the name)."""

_SMI_COLUMN_FREE = 2
"""`nvidia-smi --query-gpu` CSV column holding memory.free."""

_SMI_COLUMN_DRIVER = 3
"""`nvidia-smi --query-gpu` CSV column holding driver_version."""

_SMI_COLUMN_COMPUTE_CAP = 4
"""`nvidia-smi --query-gpu` CSV column holding compute_cap."""

_SMI_COLUMN_TEMP = 5
"""`nvidia-smi --query-gpu` CSV column holding temperature.gpu."""


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


def disk_mount_facts(path: Path) -> dict[str, float | None]:
    """Free/total/used-fraction at one mount point, never raises (issue 066).

    `used_fraction` is None when the stat fails (callers treat unknown as
    unknown, never as healthy).
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {"free_gib": None, "total_gib": None, "used_fraction": None}
    total = usage.total / _BYTES_PER_GIB
    free = usage.free / _BYTES_PER_GIB
    used = 1.0 - free / total if total > 0 else None
    return {"free_gib": free, "total_gib": total, "used_fraction": used}


def meets_reserve(free_gib: float | None, min_free_gib: float) -> bool | None:
    """Whether free space clears the run reserve (issue 066).

    None when free space is unknown — the caller prints "unknown", never
    a confident OK.
    """
    if free_gib is None:
        return None
    return free_gib >= min_free_gib


def _parse_mib_gib(raw: str) -> float | None:
    """Parse an `nvidia-smi` `"<n> MiB"` cell to GiB, None when unparsable."""
    text = raw.strip().lower().removesuffix("mib").strip()
    try:
        return float(text) / _MIB_PER_GIB
    except ValueError:
        return None


def _parse_float(raw: str) -> float | None:
    """Parse a bare numeric `nvidia-smi` cell, None when unparsable."""
    try:
        return float(raw.strip())
    except ValueError:
        return None


def _parse_gpu_details(csv_text: str | None) -> list[dict[str, Any]]:
    """One dict per GPU from the extended `nvidia-smi` CSV (issue 066).

    Columns: name,memory.total,memory.free,driver_version,compute_cap,
    temperature.gpu. Short rows (older drivers hiding new columns) degrade
    per-field to None instead of dropping the GPU.
    """
    details: list[dict[str, Any]] = []
    if not csv_text:
        return details
    for line in csv_text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if not cells or not cells[0]:
            continue
        name = cells[0]
        total = _parse_mib_gib(cells[_SMI_COLUMN_TOTAL]) if len(cells) > _SMI_COLUMN_TOTAL else None
        free = _parse_mib_gib(cells[_SMI_COLUMN_FREE]) if len(cells) > _SMI_COLUMN_FREE else None
        driver = cells[_SMI_COLUMN_DRIVER] or None if len(cells) > _SMI_COLUMN_DRIVER else None
        compute_cap = (
            cells[_SMI_COLUMN_COMPUTE_CAP] or None if len(cells) > _SMI_COLUMN_COMPUTE_CAP else None
        )
        temp_c = _parse_float(cells[_SMI_COLUMN_TEMP]) if len(cells) > _SMI_COLUMN_TEMP else None
        details.append(
            {
                "name": name,
                "vram_total_gib": total,
                "vram_free_gib": free,
                "driver": driver,
                "compute_cap": compute_cap,
                "temp_c": temp_c,
            }
        )
    return details


def _cuda_runtime() -> str | None:
    """CUDA runtime torch was built against, None when torch is absent."""
    try:
        if importlib.util.find_spec("torch") is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        import torch  # noqa: PLC0415 — lazy: supervisor image has no torch.

        runtime = torch.version.cuda
        return str(runtime) if runtime else None
    except Exception:
        return None


def _ffmpeg_version() -> str | None:
    """First line of `ffmpeg -version`, None when ffmpeg is absent/broken."""
    version_output = _capture(["ffmpeg", "-version"])
    if not version_output:
        return None
    return version_output.splitlines()[0]


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
        ("ltxv-2b", model_registry.verify_ltxv_models),
        ("causvid", model_registry.verify_causvid_models),
        ("director-qwen8b", model_registry.verify_director_models),
        ("director-qwen4b-awq", model_registry.verify_director_awq_models),
        ("audio-acestep", model_registry.verify_audio_models),
        ("sfx-mmaudio", model_registry.verify_sfx_models),
        ("film", model_registry.verify_film_models),
        ("rife", model_registry.verify_rife_models),
        ("realesrgan-anime", model_registry.verify_realesrgan_models),
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


def health_alerts(facts: dict[str, Any], *, min_free_gib: float | None = None) -> list[str]:
    """Threshold alerts over probe facts (issue 066, pure function).

    Disk is judged per mount (>85% WARN, >90% CRIT, below-reserve WARN when
    `min_free_gib` is given and free space is known); VRAM pressure is
    free < 15% of total; GPU temp >= 84 C warns; a failing *required*
    model stack warns (optional stacks never alert here). Unknown facts
    stay silent — an absent probe is not evidence of health or sickness.
    """
    alerts: list[str] = []
    by_mount = facts.get("disk_by_mount")
    if isinstance(by_mount, dict):
        for mount, entry in by_mount.items():
            if not isinstance(entry, dict):
                continue
            used = entry.get("used_fraction")
            if isinstance(used, bool):
                continue
            if isinstance(used, (int, float)):
                if used >= _DISK_CRIT_USED_FRACTION:
                    alerts.append(
                        f"CRIT: disk {mount} {used * 100.0:.0f}% used "
                        f"(free {entry.get('free_gib')})"
                    )
                elif used >= _DISK_WARN_USED_FRACTION:
                    alerts.append(
                        f"WARN: disk {mount} {used * 100.0:.0f}% used "
                        f"(free {entry.get('free_gib')})"
                    )
            if min_free_gib is not None:
                free = entry.get("free_gib")
                if isinstance(free, bool):
                    continue
                if isinstance(free, (int, float)) and free < min_free_gib:
                    alerts.append(
                        f"WARN: disk {mount} free {free:.1f} GiB "
                        f"below reserve {min_free_gib:.1f} GiB"
                    )
    details = facts.get("gpu_details")
    if isinstance(details, list):
        for gpu in details:
            if not isinstance(gpu, dict):
                continue
            free = gpu.get("vram_free_gib")
            total = gpu.get("vram_total_gib")
            if (
                isinstance(free, (int, float))
                and isinstance(total, (int, float))
                and not isinstance(free, bool)
                and not isinstance(total, bool)
                and total > 0
                and free / total < _VRAM_PRESSURE_FREE_FRACTION
            ):
                alerts.append(
                    f"WARN: VRAM pressure on {gpu.get('name')}: {free:.1f}/{total:.1f} GiB free"
                )
            temp = gpu.get("temp_c")
            if (
                isinstance(temp, (int, float))
                and not isinstance(temp, bool)
                and temp >= _GPU_TEMP_WARN_CELSIUS
            ):
                alerts.append(f"WARN: GPU {gpu.get('name')} temp {temp:.0f} C")
    models = facts.get("models")
    if isinstance(models, dict):
        checks = models.get("checks")
        if isinstance(checks, dict):
            missing = sorted(
                name
                for name, check in checks.items()
                if name not in _OPTIONAL_MODEL_STACKS
                and isinstance(check, dict)
                and not check.get("ok")
            )
            if missing:
                alerts.append(f"WARN: required model stacks missing: {', '.join(missing)}")
    return alerts


def probe() -> dict[str, Any]:
    nvidia_smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap,temperature.gpu",
            "--format=csv,noheader",
        ]
    )
    gpus: list[str] = nvidia_smi.splitlines() if nvidia_smi else []
    gpu_details = _parse_gpu_details(nvidia_smi)
    driver = next((gpu["driver"] for gpu in gpu_details if gpu.get("driver")), None)
    compute_cap = [gpu.get("compute_cap") for gpu in gpu_details] or None
    models_dir_path = models_dir()
    model_facts = check_models()
    required_ok = all(
        check["ok"]
        for name, check in model_facts["checks"].items()
        if name not in _OPTIONAL_MODEL_STACKS
    )
    models_ok = (
        bool(model_facts["exists"]) and all(check["ok"] for check in model_facts["checks"].values())
        if model_facts["checks"]
        else False
    )
    models_ok_required = bool(model_facts["exists"]) and required_ok
    director = check_director_python()
    return {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "ffmpeg_version": ffmpeg_version if (ffmpeg_version := _ffmpeg_version()) else None,
        "nvidia_smi": nvidia_smi is not None,
        "gpus": gpus,
        "gpu_details": gpu_details,
        "driver": driver,
        "compute_cap": compute_cap,
        "cuda_runtime": _cuda_runtime(),
        "torch_cuda": _torch_cuda(),  # None when torch is absent/broken.
        "disk_free_gib": _disk_free_gib(Path("/")),
        "disk_by_mount": {
            "root": disk_mount_facts(Path("/")),
            "models": disk_mount_facts(models_dir_path),
            "tmp": disk_mount_facts(Path("/tmp")),
        },
        "models": model_facts,
        "models_ok": models_ok,
        "models_ok_required": models_ok_required,
        "models_ok_all": models_ok,
        "director_python": director["path"],
        "director_python_exists": director["exists"],
    }


def check_ffmpeg() -> tuple[bool, str]:
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found on PATH"
    if shutil.which("ffprobe") is None:
        return False, "ffprobe not found on PATH"
    return True, "ffmpeg + ffprobe present"

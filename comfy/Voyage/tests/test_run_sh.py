"""run.sh image/GPU auto-selection via the VOYAGE_DRY_RUN seam (no docker).

Bare `run.sh` launches the launcher TUI, where the backend is picked
interactively — so no CLI signal exists and a GPU box must default to the
CUDA stack (voyage-video + --gpus all) for ltxv to work with no explicit
variables. These tests pin that matrix with an isolated PATH/HOME: only
symlinked coreutils plus an optional fake nvidia-smi are visible, so host
state (real GPUs, real HOME) cannot leak in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest

RUN_SH = Path(__file__).resolve().parent.parent / "scripts" / "run.sh"
# run.sh resolves --user from `id -u`/`id -g` on every path (including the
# dry-run seam), so the isolated PATH must provide it alongside coreutils.
# python3 is needed for the stored-config manifest sniff (run.sh parses
# DIR/manifest.json with the stdlib parser when --backend is absent).
_CORE_TOOLS = ("mkdir", "grep", "sed", "head", "dirname", "id", "python3")
_SMI_MODE = Literal["present-ok", "present-fail", "absent"]

needs_bash = pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="run.sh tests need /bin/bash"
)


def _write_fake_smi(bin_dir: Path, succeed: bool) -> None:
    """Fake nvidia-smi: lists a GPU (exit 0) or reports none (exit 1)."""
    bin_dir.joinpath("nvidia-smi").write_text(
        "#!/bin/sh\n" + ("echo 'GPU 0: Fake Test GPU'; exit 0\n" if succeed else "exit 1\n"),
        encoding="utf-8",
    )
    bin_dir.joinpath("nvidia-smi").chmod(0o755)


def _dry_run(
    tmp_path: Path,
    args: list[str],
    smi: _SMI_MODE,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Run run.sh with VOYAGE_DRY_RUN=1; return the image=/gpus= lines."""
    import shutil
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in _CORE_TOOLS:
        resolved = shutil.which(tool)
        assert resolved is not None, f"core tool missing: {tool}"
        (bin_dir / tool).symlink_to(resolved)
    if smi != "absent":
        _write_fake_smi(bin_dir, smi == "present-ok")
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": str(bin_dir), "HOME": str(home), "VOYAGE_DRY_RUN": "1"}
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        ["/bin/bash", str(RUN_SH), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition("=")
        parsed[key] = value
    return parsed


@needs_bash
def test_bare_with_gpu_defaults_to_video_image(tmp_path: Path) -> None:
    """Bare TUI launch on a GPU box: CUDA stack, no explicit variables."""
    selection = _dry_run(tmp_path, [], "present-ok")
    assert selection["image"] == "voyage-video:latest"
    assert selection["gpus"] == "--gpus all"


@needs_bash
def test_bare_without_gpu_stays_slim(tmp_path: Path) -> None:
    """Bare TUI launch with no GPU: slim image (fake-backend smoke runs)."""
    selection = _dry_run(tmp_path, [], "absent")
    assert selection["image"] == "voyage:latest"
    assert selection["gpus"] == "none"


@needs_bash
def test_bare_with_failing_smi_stays_slim(tmp_path: Path) -> None:
    """nvidia-smi present but failing (no GPU visible): slim, not CUDA."""
    selection = _dry_run(tmp_path, [], "present-fail")
    assert selection["image"] == "voyage:latest"
    assert selection["gpus"] == "none"


@needs_bash
def test_explicit_image_wins_over_gpu_default(tmp_path: Path) -> None:
    """VOYAGE_IMAGE is explicit: kept, while --gpus still auto-enables."""
    selection = _dry_run(tmp_path, [], "present-ok", {"VOYAGE_IMAGE": "custom:1"})
    assert selection["image"] == "custom:1"
    assert selection["gpus"] == "--gpus all"


@needs_bash
def test_explicit_gpus_zero_disables_gpu_flag(tmp_path: Path) -> None:
    """VOYAGE_GPUS=0 is explicit: image default stays, flag is dropped."""
    selection = _dry_run(tmp_path, [], "present-ok", {"VOYAGE_GPUS": "0"})
    assert selection["image"] == "voyage-video:latest"
    assert selection["gpus"] == "none"


@needs_bash
def test_generate_selects_cuda_without_host_gpu(tmp_path: Path) -> None:
    """CLI generate defaults to ltx25: LTX image regardless of host probe."""
    selection = _dry_run(tmp_path, ["generate"], "absent")
    assert selection["image"] == "voyage-ltx:latest"
    assert selection["gpus"] == "--gpus all"


@needs_bash
def test_explicit_fake_backend_stays_slim_despite_gpu(tmp_path: Path) -> None:
    """An explicit verb backend is a real signal: fake stays slim on GPU box."""
    selection = _dry_run(tmp_path, ["--backend", "fake"], "present-ok")
    assert selection["image"] == "voyage:latest"
    assert selection["gpus"] == "none"


@needs_bash
def test_dry_run_reports_host_user_mapping(tmp_path: Path) -> None:
    """The container runs as the host ids so bind-mount writes stay owned.

    run.sh pins --user to the invoking host's uid:gid (issue 053
    follow-up); the dry-run seam reports it as user=--user=UID:GID.
    """
    selection = _dry_run(tmp_path, ["generate"], "absent")
    assert selection["user"] == f"--user={os.getuid()}:{os.getgid()}"


@needs_bash
def test_dry_run_pins_direct_entrypoint(tmp_path: Path) -> None:
    """The nvidia/cuda entrypoint banner is bypassed on every path.

    voyage-video inherits /opt/nvidia/nvidia_entrypoint.sh, which prints
    a large CUDA banner + license block on every run. run.sh sets
    --entrypoint voyage and execs the CLI directly (equivalent on the
    entrypoint-less slim image); the dry-run seam reports it.
    """
    assert _dry_run(tmp_path, ["generate"], "absent")["entrypoint"] == "voyage"
    second = tmp_path / "second"
    second.mkdir()
    assert _dry_run(second, [], "absent")["entrypoint"] == "voyage"


@needs_bash
def test_generate_ltx25_selects_ltx_image_without_host_gpu(tmp_path: Path) -> None:
    """Explicit ltx25 backend: ComfyUI worker image, GPU flag on."""
    selection = _dry_run(tmp_path, ["generate", "--backend", "ltx25"], "absent")
    assert selection["image"] == "voyage-ltx:latest"
    assert selection["gpus"] == "--gpus all"


@needs_bash
def test_generate_ltx23_selects_ltx_image_with_equals_form(tmp_path: Path) -> None:
    """Equals-form --backend= works for the ltx image selection too."""
    selection = _dry_run(tmp_path, ["generate", "--backend=ltx23"], "present-ok")
    assert selection["image"] == "voyage-ltx:latest"
    assert selection["gpus"] == "--gpus all"


@needs_bash
def test_explicit_ltxv_backend_stays_video_image_despite_gpu(tmp_path: Path) -> None:
    """ltxv keeps the video image (regression guard for the ltx split)."""
    selection = _dry_run(tmp_path, ["generate", "--backend", "ltxv"], "present-ok")
    assert selection["image"] == "voyage-video:latest"
    assert selection["gpus"] == "--gpus all"


@needs_bash
def test_run_dir_with_ltx_manifest_selects_ltx_image(tmp_path: Path) -> None:
    """Stored-config runs sniff manifest video.backend for ltx."""
    import json

    run_dir = tmp_path / "rundir"
    run_dir.mkdir()
    run_dir.joinpath("manifest.json").write_text(
        json.dumps({"video": {"backend": "ltx25"}}),
        encoding="utf-8",
    )
    selection = _dry_run(tmp_path, ["run", "--run", str(run_dir)], "absent")
    assert selection["image"] == "voyage-ltx:latest"
    assert selection["gpus"] == "--gpus all"

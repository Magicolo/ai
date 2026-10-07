"""Track D run.sh sniff (augment/mastering, fail-closed, image check, env prefix).

Uses the VOYAGE_DRY_RUN seam (no docker), isolated PATH/HOME.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import pytest

RUN_SH = Path(__file__).resolve().parent.parent / "scripts" / "run.sh"
_CORE_TOOLS = ("mkdir", "grep", "sed", "head", "dirname", "id", "python3")
_SMI_MODE = Literal["present-ok", "present-fail", "absent"]

needs_bash = pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="run.sh tests need /bin/bash"
)


def _write_fake_smi(bin_dir: Path, succeed: bool) -> None:
    bin_dir.joinpath("nvidia-smi").write_text(
        "#!/bin/sh\n" + ("echo 'GPU 0: Fake Test GPU'; exit 0\n" if succeed else "exit 1\n"),
        encoding="utf-8",
    )
    bin_dir.joinpath("nvidia-smi").chmod(0o755)


def _dry_run_full(
    tmp_path: Path,
    args: list[str],
    smi: _SMI_MODE,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    """Run run.sh dry-run; return (returncode, parsed stdout, stderr)."""
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
    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition("=")
        parsed[key] = value
    return completed.returncode, parsed, completed.stderr


@needs_bash
def test_missing_manifest_still_falls_back_to_ltx(tmp_path: Path) -> None:
    code, selection, _ = _dry_run_full(tmp_path, ["generate", "__track_d_missing__"], "absent")
    assert code == 0
    assert selection["image"] == "voyage-ltx:latest"


@needs_bash
def test_unreadable_manifest_fails_closed_for_generate(tmp_path: Path) -> None:
    import shutil as _shutil

    staged = RUN_SH.parent.parent / "output" / "__track_d_torn__"
    staged.mkdir(parents=True, exist_ok=True)
    staged.joinpath("manifest.json").write_text("{not-json", encoding="utf-8")
    try:
        bin_dir = tmp_path / "bin2"
        bin_dir.mkdir()
        for tool in _CORE_TOOLS:
            resolved = shutil.which(tool)
            assert resolved is not None
            (bin_dir / tool).symlink_to(resolved)
        home = tmp_path / "home2"
        home.mkdir()
        import subprocess

        completed = subprocess.run(
            ["/bin/bash", str(RUN_SH), "generate", staged.name],
            capture_output=True,
            text=True,
            env={"PATH": str(bin_dir), "HOME": str(home), "VOYAGE_DRY_RUN": "1"},
            timeout=60,
            check=False,
        )
        assert completed.returncode == 2
        assert "refusing to guess" in completed.stderr
    finally:
        _shutil.rmtree(staged, ignore_errors=True)


@needs_bash
def test_explicit_image_mismatch_warns(tmp_path: Path) -> None:
    code, selection, stderr = _dry_run_full(
        tmp_path, ["generate", "--backend", "ltx25"], "absent", {"VOYAGE_IMAGE": "voyage:latest"}
    )
    assert code == 0
    assert selection["image"] == "voyage:latest"
    assert selection.get("image_warning", "none") != "none"
    assert "disagrees" in stderr


@needs_bash
def test_ltx_prefix_forwarding_by_prefix(tmp_path: Path) -> None:
    _, selection, _ = _dry_run_full(
        tmp_path,
        ["generate", "--backend", "ltx25"],
        "present-ok",
        {"VOYAGE_LTX_CUSTOM_KNOB": "7", "VOYAGE_LTX_EMPTY": ""},
    )
    assert "VOYAGE_LTX_CUSTOM_KNOB=7" in selection["env"]
    assert "VOYAGE_LTX_EMPTY" not in selection["env"]


@needs_bash
def test_python_overrides_forwarded(tmp_path: Path) -> None:
    _, selection, _ = _dry_run_full(
        tmp_path,
        ["generate", "--backend", "ltx25"],
        "present-ok",
        {"VOYAGE_DIRECTOR_PYTHON": "/opt/venvs/director/bin/python"},
    )
    assert "VOYAGE_DIRECTOR_PYTHON=/opt/venvs/director/bin/python" in selection["env"]

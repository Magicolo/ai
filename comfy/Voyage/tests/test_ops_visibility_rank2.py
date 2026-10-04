"""Ops-visibility Rank-2 tests (issues 060/064/066).

CPU-only, fake backends: report document shape (060), qualify.sh
gate hygiene (064), and doctor per-mount disks + model split + alert
thresholds (066).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from voyage.bench import report_document
from voyage.doctor import health_alerts, meets_reserve, probe


def test_report_document_builds_json_artifact() -> None:
    """060: report_document is a JSON-serializable setup+measured document."""
    document = report_document("video", {"backend": "fake"}, {"count": 2})
    assert document["title"] == "video"
    assert document["setup"] == {"backend": "fake"}
    assert document["measured"] == {"count": 2}
    json.dumps(document)


def test_doctor_reports_disks_per_mount() -> None:
    """066: probe reports free space per mount (root/models/tmp), not just `/`."""
    facts = probe()
    by_mount = facts["disk_by_mount"]
    assert {"root", "models", "tmp"} <= set(by_mount)
    for mount in ("root", "models", "tmp"):
        entry = by_mount[mount]
        assert "free_gib" in entry and "used_fraction" in entry, mount
    # Legacy single number stays for backward compatibility.
    assert facts["disk_free_gib"] is None or isinstance(facts["disk_free_gib"], float)


def test_meets_reserve_compares_against_config() -> None:
    """066: reserve comparison is explicit; unknown free space stays unknown."""
    assert meets_reserve(10.0, 5.0) is True
    assert meets_reserve(1.0, 5.0) is False
    assert meets_reserve(None, 5.0) is None


def test_doctor_splits_required_vs_all_models() -> None:
    """066: the optional inspector stack no longer flips the required flag."""
    facts = probe()
    assert isinstance(facts["models_ok_required"], bool)
    assert isinstance(facts["models_ok_all"], bool)
    assert facts["models_ok"] == facts["models_ok_all"]


def test_doctor_reports_vram_compute_facts() -> None:
    """066: VRAM/compute-capability/CUDA-runtime facts exist (None off-GPU)."""
    facts = probe()
    assert "gpu_details" in facts
    assert "cuda_runtime" in facts
    assert isinstance(facts["gpu_details"], list)


def test_health_alerts_fire_on_thresholds() -> None:
    """066: >85%/90% disk, low reserve, VRAM pressure, and heat all alert."""
    facts = {
        "disk_by_mount": {
            "root": {"free_gib": 100.0, "total_gib": 100.0, "used_fraction": 0.0},
            "models": {"free_gib": 4.0, "total_gib": 100.0, "used_fraction": 0.96},
            "tmp": {"free_gib": 50.0, "total_gib": 100.0, "used_fraction": 0.50},
        },
        "gpu_details": [
            {"name": "gpu0", "vram_total_gib": 16.0, "vram_free_gib": 1.0, "temp_c": 86.0}
        ],
        "models": {"checks": {"ltxv-2b": {"ok": False, "message": "missing"}}},
        "models_ok_required": False,
    }
    alerts = health_alerts(facts, min_free_gib=5.0)
    joined = "\n".join(alerts)
    assert "models" in joined and "CRIT" in joined
    assert "VRAM" in joined or "vram" in joined.lower()
    assert "86" in joined
    assert "reserve" in joined.lower()
    assert health_alerts({"disk_by_mount": {}, "gpu_details": []}) == []


def _qualify_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "qualify.sh"


def test_qualify_sh_passes_syntax_check() -> None:
    """064: the harness is at least syntactically valid bash."""
    completed = subprocess.run(
        ["bash", "-n", str(_qualify_path())], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr


def test_qualify_sh_fails_closed_without_nvidia_smi(tmp_path: Path) -> None:
    """064: no nvidia-smi → explicit exit 4 before any docker/benchmark work."""
    import shutil

    bash = shutil.which("bash")
    assert bash is not None
    completed = subprocess.run(
        [bash, str(_qualify_path()), str(tmp_path / "run")],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/tmp/emptybin-qualify-060-test", "HOME": "/root"},
    )
    assert completed.returncode == 4
    assert "nvidia-smi" in completed.stderr


def test_qualify_sh_rejects_relative_run_dir(tmp_path: Path) -> None:
    """064: relative run dirs exit 2 with an absolute-path message (no docker)."""
    import shutil
    import stat

    text = _qualify_path().read_text(encoding="utf-8")
    assert "absolute" in text.lower()
    assert "tee" in text
    assert "df" in text or "disk" in text.lower()
    bash = shutil.which("bash")
    assert bash is not None
    stub = tmp_path / "stubbin-rel"
    stub.mkdir()
    nvidia_smi = stub / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 0\n", encoding="utf-8")
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    completed = subprocess.run(
        [bash, str(_qualify_path()), "output/rel-path"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": f"{stub}:/usr/bin:/bin", "HOME": "/root"},
    )
    assert completed.returncode == 2
    assert "absolute" in completed.stderr.lower()


def test_qualify_sh_missing_run_dir_reaches_generate_hint(tmp_path: Path) -> None:
    """064: a missing absolute run dir exits 2 with the generate hint (no silent 1).

    Regression: the `df` preflight on a nonexistent dir once tripped
    `set -e` (pipefail) inside the command substitution, so the script
    died rc=1 before the manifest.json check. `|| true` keeps the gate
    total — unknown space skips the preflight, it never aborts it.
    """
    import shutil
    import stat

    bash = shutil.which("bash")
    assert bash is not None
    stub = tmp_path / "stubbin"
    stub.mkdir()
    nvidia_smi = stub / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\necho 0\n", encoding="utf-8")
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = {
        "PATH": f"{stub}:/usr/bin:/bin",
        "HOME": "/root",
        "QUALIFY_MIN_FREE_GIB": "5",
    }
    completed = subprocess.run(
        [bash, str(_qualify_path()), str(tmp_path / "no-such-run")],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert completed.returncode == 2
    assert "manifest.json" in completed.stderr

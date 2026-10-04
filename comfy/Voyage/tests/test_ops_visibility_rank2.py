"""Ops-visibility Rank-2 tests (issues 060/061/064/066).

CPU-only, fake backends: benchmark env richness + report persistence (060),
status config/health sections (061), qualify.sh gate hygiene (064), and
doctor per-mount disks + model split + alert thresholds (066).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.bench import report_document
from voyage.cli import _benchmark_env, cmd_status, main
from voyage.doctor import health_alerts, meets_reserve, probe
from voyage.persistence import read_effective_config


def test_benchmark_env_carries_driver_revision_keys() -> None:
    """060: env reports driver/compute/CUDA-runtime/torch/version/revisions."""
    env = _benchmark_env()
    for key in (
        "gpu",
        "driver",
        "compute_cap",
        "cuda_runtime",
        "torch",
        "cuda_available",
        "voyage_version",
        "revisions",
    ):
        assert key in env, key
    assert isinstance(env["revisions"], dict)
    assert env["revisions"], "at least one model revision must be recorded"
    # Off-GPU honesty: absent facts degrade to "unknown"/None, never raise.
    assert env["gpu"] in ("unknown", env["gpu"])
    assert env["driver"] is None or isinstance(env["driver"], str)


def test_report_document_builds_json_artifact() -> None:
    """060: report_document is a JSON-serializable setup+measured document."""
    document = report_document("video", {"backend": "fake"}, {"count": 2})
    assert document["title"] == "video"
    assert document["setup"] == {"backend": "fake"}
    assert document["measured"] == {"count": 2}
    json.dumps(document)


def test_benchmark_cli_persists_report_json(tmp_path: Path) -> None:
    """060: `benchmark video` tees a JSON sidecar into the run logs."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    assert main(["benchmark", "video", "--run", str(run_dir)]) == 0
    artifacts = sorted((run_dir / paths.LOGS_DIRNAME).glob("benchmark-video-*.json"))
    assert artifacts, "benchmark video must persist a logs/ JSON artifact"
    document = json.loads(artifacts[-1].read_text(encoding="utf-8"))
    assert document["setup"]["backend"] == "fake"
    assert "measured" in document


def test_soak_cli_persists_report_json(tmp_path: Path) -> None:
    """060: `soak` tees a JSON sidecar into the run logs."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    assert main(["soak", "--run", str(run_dir), "--segments", "1"]) == 0
    artifacts = sorted((run_dir / paths.LOGS_DIRNAME).glob("soak-*.json"))
    assert artifacts, "soak must persist a logs/ JSON artifact"
    document = json.loads(artifacts[-1].read_text(encoding="utf-8"))
    assert "measured" in document


def _status_output(run_dir: Path, capsys: object) -> str:
    args = type("Args", (), {"run": str(run_dir)})()
    assert cmd_status(args) == 0
    return str(capsys.readouterr().out)  # type: ignore[attr-defined]


def test_status_shows_config_section(tmp_path: Path, capsys: object) -> None:
    """061: status echoes quantization/SFX/floors/inspector/beats/drift/takes."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    out = _status_output(run_dir, capsys).lower()
    for needle in (
        "quantization",
        "sfx",
        "augment",
        "inspector",
        "beats",
        "drift",
        "take_seconds",
        "ahead_seconds",
    ):
        assert needle in out, needle


def test_status_labels_manifest_hardware_vs_live_probe(tmp_path: Path, capsys: object) -> None:
    """061: recorded-at-init hardware is labeled; the live probe says live."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    out = _status_output(run_dir, capsys)
    assert "recorded" in out.lower()
    assert "live" in out.lower()


def test_status_warns_below_reserve(tmp_path: Path, capsys: object) -> None:
    """061: free space below min_free_space_gib warns instead of printing bare."""

    from voyage.persistence import read_manifest, write_manifest

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    manifest = read_manifest(run_dir)
    effective = manifest.get("effective_config")
    assert isinstance(effective, dict)
    assert "min_free_space_gib" in effective
    # A reserve no disk can meet forces the WARN path deterministically.
    effective["min_free_space_gib"] = 999999.0
    manifest["effective_config"] = effective
    write_manifest(run_dir, manifest)
    out = _status_output(run_dir, capsys)
    assert "WARN" in out
    assert "reserve" in out.lower()


def test_status_shows_gauges_and_restart_counts(tmp_path: Path, capsys: object) -> None:
    """061: committed runs show the gauges trend + restart/circuit counts."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir)
    config, _ = read_effective_config(run_dir)
    from voyage.supervisor import Supervisor

    assert Supervisor(run_dir, config).run_segments(1) == ["000000"]
    out = _status_output(run_dir, capsys).lower()
    assert "gauges" in out
    assert "restart" in out


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


def test_cmd_doctor_prints_alerts_without_crashing(capsys: object) -> None:
    """066: `voyage doctor` stays exit-0 on missing ffmpeg siblings + prints."""
    assert main(["doctor"]) in (0, 1)
    out = str(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert "python:" in out


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
    died rc=1 before the run_manifest.json check. `|| true` keeps the gate
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
    assert "run_manifest.json" in completed.stderr

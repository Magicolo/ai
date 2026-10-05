"""Track A Phase-0 boundary metrics: CPU-only seam continuity + prompt audit.

Covers `voyage.boundary_metrics` on synthetic media (fake-backend style:
real ffmpeg `testsrc` frames under `tmp_path` — no GPU, no network, no
model weights). Also pins the CPU-only contract (no torch/transformers
imports) and the read-only promise (the tool never writes into the run).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voyage import boundary_metrics
from voyage.boundary_metrics import (
    audit_prompts,
    boundary_verdict,
    committed_segment_videos,
    format_boundary_report,
    format_prompt_audit,
    main,
    mean_abs_diff,
    summarize_boundaries,
)


def _render_testsrc_clip(
    destination: Path, *, seed_hue: int = 0, frames: int = 12, fps: int = 12
) -> Path:
    """Real ffmpeg `testsrc` clip; hue varies the palette per seed."""
    duration = frames / fps
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x64:rate={fps}:duration={duration}",
            "-vf",
            f"hue=h={seed_hue}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"testsrc render failed: {proc.stderr[-500:]}"
    return destination


def _commit_segment(
    run_dir: Path,
    index: int,
    *,
    seed_hue: int = 0,
    prompt: str = "teal dunes under a slow wind",
    mechanism: str = "environmental_transformation",
) -> Path:
    """Committed-shape segment: DONE + testsrc video + production-shape manifest."""
    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    _render_testsrc_clip(segment / "video.mp4", seed_hue=seed_hue)
    manifest = {
        "format": 1,
        "transition": {
            "destination": {
                "canonical_name": "teal dunes",
                "summary": "deterministic continuation",
            },
            "transition": {
                "mechanism": mechanism,
                "intermediate_stages": ["the dunes hold and deepen"],
            },
        },
        "prompt_plan": {
            "segment_id": f"{index:06d}",
            "stages": [
                {
                    "stage": 0,
                    "block_start": 0,
                    "block_end": 0,
                    "prompt": prompt,
                }
            ],
        },
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": 12},
        "checksums": {"video.mp4": "synthetic"},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment / "DONE").write_bytes(b"")
    return segment


def _make_run(run_dir: Path, *, hues: tuple[int, ...] = (0, 90)) -> Path:
    """Run directory with one committed synthetic segment per hue."""
    (run_dir / "segments").mkdir(parents=True)
    for index, hue in enumerate(hues):
        _commit_segment(run_dir, index, seed_hue=hue)
    return run_dir


def test_mean_abs_diff_identical_is_zero() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert mean_abs_diff(frame, frame) == 0.0


def test_mean_abs_diff_black_white_is_one() -> None:
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert mean_abs_diff(black, white) == pytest.approx(1.0)


def test_boundary_verdict_pass_and_fail() -> None:
    passing = boundary_verdict(0.01, 0.02)
    assert passing["verdict"] == "PASS"
    assert passing["boundary_ratio"] == pytest.approx(2.0)
    failing = boundary_verdict(0.01, 0.10)
    assert failing["verdict"] == "FAIL"
    assert failing["boundary_ratio"] == pytest.approx(10.0)


def test_boundary_verdict_zero_within_is_inf_fail() -> None:
    """A frozen segment meeting any visible jump is a cut (harness parity)."""
    verdict = boundary_verdict(0.0, 0.05)
    assert verdict["boundary_ratio"] == float("inf")
    assert verdict["verdict"] == "FAIL"


def test_committed_segment_videos_in_order(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run", hues=(0, 90, 180))
    pairs = committed_segment_videos(run_dir)
    assert [segment_id for segment_id, _ in pairs] == ["000000", "000001", "000002"]
    assert all(video.name == "video.mp4" for _, video in pairs)


def test_committed_segment_videos_empty_run_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "segments").mkdir(parents=True)
    with pytest.raises(ValueError, match="no committed segments"):
        committed_segment_videos(run_dir)


def test_committed_segment_videos_missing_video_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    segment = run_dir / "segments" / "000000"
    segment.mkdir(parents=True)
    (segment / "DONE").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="missing segment video"):
        committed_segment_videos(run_dir)


def test_summarize_boundaries_two_segments(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    summary = summarize_boundaries(run_dir, samples_per_segment=3)
    assert summary["segments"] == ["000000", "000001"]
    assert summary["samples_per_segment"] == 3
    within_mean = summary["within_mean"]
    boundary_mean = summary["boundary_mean"]
    assert isinstance(within_mean, float) and within_mean >= 0
    assert isinstance(boundary_mean, float) and boundary_mean >= 0
    assert summary["verdict"] in ("PASS", "FAIL")
    boundaries = summary["boundaries"]
    assert isinstance(boundaries, list) and len(boundaries) == 1
    assert boundaries[0]["previous"] == "000000"
    assert boundaries[0]["current"] == "000001"


def test_summarize_boundaries_single_segment_is_na(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run", hues=(0,))
    summary = summarize_boundaries(run_dir, samples_per_segment=3)
    assert summary["boundary_mean"] is None
    assert summary["boundary_ratio"] is None
    assert summary["verdict"] == "N/A (single segment)"


def test_summarize_boundaries_rejects_bad_sampling() -> None:
    with pytest.raises(ValueError, match="samples_per_segment"):
        summarize_boundaries(Path("/nonexistent"), samples_per_segment=0)
    with pytest.raises(ValueError, match="decode_width"):
        summarize_boundaries(Path("/nonexistent"), decode_width=0)


def test_audit_prompts_production_shape(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    entries = audit_prompts(run_dir)
    assert [entry["segment_id"] for entry in entries] == ["000000", "000001"]
    first = entries[0]
    assert first["error"] is None
    assert first["manifest_present"] is True
    assert first["destination"] == "teal dunes"
    assert first["destination_summary"] == "deterministic continuation"
    assert first["mechanism"] == "environmental_transformation"
    assert first["intermediate_stages"] == ["the dunes hold and deepen"]
    stages = first["stages"]
    assert isinstance(stages, list) and len(stages) == 1
    assert stages[0]["prompt"] == "teal dunes under a slow wind"
    assert (stages[0]["block_start"], stages[0]["block_end"]) == (0, 0)


def test_audit_prompts_torn_manifest_records_error(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run", hues=(0,))
    (run_dir / "segments" / "000000" / "manifest.json").write_text("{torn", encoding="utf-8")
    entries = audit_prompts(run_dir)
    assert len(entries) == 1
    assert isinstance(entries[0]["error"], str)
    assert entries[0]["stages"] == []


def test_format_reports_render_segments_and_prompts(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    report = format_boundary_report(summarize_boundaries(run_dir, samples_per_segment=3))
    assert "segments: ['000000', '000001']" in report
    assert "boundary 000000 -> 000001" in report
    assert "verdict:" in report
    audit = format_prompt_audit(audit_prompts(run_dir))
    assert "== segment 000000 ==" in audit
    assert "teal dunes under a slow wind" in audit
    assert "environmental_transformation" in audit


def test_main_text_report_exit_matches_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _make_run(tmp_path / "run")
    expected = 1 if summarize_boundaries(run_dir)["verdict"] == "FAIL" else 0
    assert main(["--run", str(run_dir)]) == expected
    out = capsys.readouterr().out
    assert "verdict:" in out


def test_main_prompts_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _make_run(tmp_path / "run")
    assert main(["--run", str(run_dir), "--prompts"]) == main(["--run", str(run_dir)])
    out = capsys.readouterr().out
    assert "teal dunes under a slow wind" in out
    assert main(["--run", str(run_dir), "--json"]) in (0, 1)
    payload = json.loads(capsys.readouterr().out)
    assert payload["boundaries"]["segments"] == ["000000", "000001"]
    assert payload["prompts"][0]["stages"][0]["prompt"] == "teal dunes under a slow wind"


def test_main_missing_run_is_tool_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--run", "/nonexistent-run-dir"]) == 1
    assert "error" in capsys.readouterr().err


def test_tool_is_read_only(tmp_path: Path) -> None:
    """The tool writes nothing into the run (before/after tree identical)."""
    run_dir = _make_run(tmp_path / "run")
    before = sorted(str(path) for path in run_dir.rglob("*"))
    summarize_boundaries(run_dir)
    audit_prompts(run_dir)
    assert main(["--run", str(run_dir), "--prompts"]) in (0, 1)
    assert sorted(str(path) for path in run_dir.rglob("*")) == before


def test_module_has_no_gpu_imports() -> None:
    """CPU-only contract: no torch/transformers imports in the new module."""
    source = Path(boundary_metrics.__file__).resolve().read_text(encoding="utf-8")
    for banned in ("import torch", "from torch", "import transformers", "from transformers"):
        assert banned not in source
    assert "torch" not in sys.modules

"""`benchmark sfx` / `benchmark augment` verbs + soak SFX/presentation sections (154/163/194).

CPU-only: the fake SFX probe renders seeded noise windows (ffmpeg
anoisesrc) and the augment probe times the real ffmpeg chunk-encode
orchestration — both genuine in `voyage:latest`. CUDA/model paths
degrade to skip notes, never tracebacks.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import initialize_run_directory
from voyage.cli import build_parser, main


def test_benchmark_parser_pins_all_five_targets() -> None:
    """154 fix candidate 3: the choice list pins every probe so no worker
    ships a headless benchmark again."""
    parser = build_parser()
    for target in ("video", "audio", "sfx", "augment", "end-to-end"):
        namespace = parser.parse_args(["benchmark", target, "--run", "x"])
        assert namespace.benchmark_target == target


def test_benchmark_sfx_fake_with_run_reports_windows(tmp_path: Path, capsys: object) -> None:
    """154/163: `benchmark sfx` on a fake-config run spawns the SFX worker
    directly and reports the window ladder fields."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    assert (
        main(["benchmark", "sfx", "--run", str(run_dir), "--warmup", "0", "--measured", "1"]) == 0
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "windows_per_second" in out
    assert "model_size" in out
    assert "sfx_workers" in out
    artifacts = list((run_dir / "logs").glob("benchmark-sfx-*.json"))
    assert len(artifacts) == 1


def test_benchmark_sfx_fake_needs_no_run_dir(capsys: object) -> None:
    """154/163: bare `benchmark sfx` probes fake with no run dir (stdout
    is the record, mirroring the end-to-end throwaway note)."""
    assert main(["benchmark", "sfx", "--warmup", "0", "--measured", "1"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "windows_per_second" in out
    assert "no --run" in out


def test_benchmark_sfx_rejects_bad_counts(tmp_path: Path) -> None:
    """154/163: the shared count guard applies to the sfx target too."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    assert (
        main(["benchmark", "sfx", "--run", str(run_dir), "--warmup", "-1", "--measured", "1"]) == 2
    )
    assert (
        main(["benchmark", "sfx", "--run", str(run_dir), "--warmup", "0", "--measured", "0"]) == 2
    )


def test_benchmark_augment_cpu_reports_probe_and_model_skip(capsys: object) -> None:
    """154: `benchmark augment` on a torch-free box times the ffmpeg chunk
    path and records the model path as skipped, never tracebacks."""
    assert main(["benchmark", "augment", "--warmup", "0", "--measured", "1"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "chunk_encode_mean_seconds" in out
    assert "veryfast" in out
    assert "skipped" in out


def test_benchmark_augment_with_run_records_run_axes(tmp_path: Path, capsys: object) -> None:
    """154: with --run the augment setup records the run's floors."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    assert (
        main(["benchmark", "augment", "--run", str(run_dir), "--warmup", "0", "--measured", "1"])
        == 0
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "min_fps" in out
    assert "chunk_frames" in out
    artifacts = list((run_dir / "logs").glob("benchmark-augment-*.json"))
    assert len(artifacts) == 1


def test_benchmark_video_setup_records_presentation_floors(tmp_path: Path, capsys: object) -> None:
    """194: the video setup block carries the floor triple + resolved plan
    so re-encode and stream-copy reports are never silently compared."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    assert main(["benchmark", "video", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "min_fps" in out
    assert "min_width" in out
    assert "min_height" in out
    assert "needs_reencode" in out
    assert "needs_minterpolate" in out


def test_soak_setup_records_sfx_and_presentation_axes(tmp_path: Path, capsys: object) -> None:
    """163 + 194: soak setup carries the sfx/augment axes and metrics carry
    the post-run SFX section (zeros + clean verdict on an SFX-less run)."""
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    assert main(["soak", "--run", str(run_dir), "--segments", "1"]) == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "sfx_backend" in out
    assert "sfx_model_size" in out
    assert "augment_chunk_frames" in out
    assert "min_fps" in out
    assert "ledger_errors" in out


def test_soak_sfx_section_reports_missing_stems(tmp_path: Path) -> None:
    """163: a ledger whose stems never rendered fails loud on the missing
    file (read-only post-run pass — the verdict needs no extra renders)."""
    import json

    from voyage.cli_observe import _soak_sfx_section

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    assert main(["soak", "--run", str(run_dir), "--segments", "1"]) == 0
    sfx_dir = run_dir / "audio" / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    (sfx_dir / "sfx.jsonl").write_text(
        json.dumps(
            {
                "window_id": "w0000",
                "start": 0.0,
                "duration": 8.0,
                "caption": "probing",
                "seed": 7,
                "path": "audio/sfx/w0000.wav",
                "model_size": "large_44k_v2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    section = _soak_sfx_section(run_dir)
    assert section["windows"] == 1
    assert section["total_audio_seconds"] == 8.0
    assert section["stems"] == 0
    ledger_errors = section["ledger_errors"]
    assert isinstance(ledger_errors, list)
    assert any(isinstance(error, str) and "missing" in error for error in ledger_errors)


def test_soak_sfx_section_reports_torn_ledger(tmp_path: Path) -> None:
    """163: a torn ledger is one error entry, never a soak crash."""
    from voyage.cli_observe import _soak_sfx_section

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="bench-sfx")
    sfx_dir = run_dir / "audio" / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    (sfx_dir / "sfx.jsonl").write_text('{"window_id": "w0000",\n', encoding="utf-8")
    section = _soak_sfx_section(run_dir)
    assert section["windows"] == 0
    assert section["stems"] == 0
    ledger_errors = section["ledger_errors"]
    assert isinstance(ledger_errors, list)
    assert any(isinstance(error, str) and "unreadable" in error for error in ledger_errors)

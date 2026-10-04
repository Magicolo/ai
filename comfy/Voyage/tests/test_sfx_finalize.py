"""Finalize-time SFX: windowing, ledger, validate (slice 3, TDD).

The SFX pass runs AFTER `finalize_run` publishes the music-only final:
windows condition on the shipped pixels, the bed joins with manual
fades (never acrossfade), and amix lays it under the music. The fake
backend drives the whole path in slim gates (no GPU); the ladder
pinned the real models in slice 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voyage.sfx_finalize import (
    SFX_WINDOW_OVERLAP,
    SFX_WINDOW_SECONDS,
    SfxWindow,
    load_sfx_ledger,
    plan_sfx_windows,
    validate_sfx_ledger,
)


def test_plan_single_segment_timeline() -> None:
    windows = plan_sfx_windows(20.0, [(0.0, 20.0, "rain on glass")], seed_base=7)
    assert [(w.start, w.duration) for w in windows] == [
        (0.0, 8.0),
        (7.0, 8.0),
        (14.0, 6.0),
    ]
    assert all(w.caption == "rain on glass" for w in windows)
    assert [w.window_id for w in windows] == ["w0000", "w0001", "w0002"]
    assert windows[0].seed == 7


def test_plan_junction_window_gets_both_captions() -> None:
    windows = plan_sfx_windows(
        16.0,
        [(0.0, 8.0, "glass chimes"), (8.0, 16.0, "deep rumble")],
        seed_base=0,
    )
    assert [(w.start, w.duration) for w in windows] == [(0.0, 8.0), (7.0, 8.0), (14.0, 2.0)]
    assert windows[0].caption == "glass chimes"
    assert "glass chimes" in windows[1].caption and "deep rumble" in windows[1].caption
    assert windows[2].caption == "deep rumble"


def test_plan_short_timeline_is_one_clamped_window() -> None:
    windows = plan_sfx_windows(5.0, [(0.0, 5.0, "wind")], seed_base=3)
    assert len(windows) == 1
    assert windows[0].start == 0.0 and windows[0].duration == 5.0
    assert windows[0].seed == 3


def test_plan_empty_caption_stays_empty_for_worker_default() -> None:
    windows = plan_sfx_windows(4.0, [(0.0, 4.0, "")], seed_base=0)
    assert windows[0].caption == ""


def test_plan_rejects_non_positive_timeline() -> None:
    with pytest.raises(ValueError, match="timeline"):
        plan_sfx_windows(0.0, [(0.0, 0.0, "x")], seed_base=0)


def test_ledger_round_trip_and_validate(tmp_path: Path) -> None:
    from voyage.sfx_finalize import append_sfx_window

    ledger = tmp_path / "audio" / "sfx" / "sfx.jsonl"
    append_sfx_window(
        ledger,
        SfxWindow("w0000", 0.0, 8.0, "rain", 7),
        path="audio/sfx/w0000.wav",
        model_size="small_44k",
    )
    (tmp_path / "audio" / "sfx" / "w0000.wav").write_bytes(b"RIFF" + b"\0" * 100)
    records = load_sfx_ledger(ledger)
    assert len(records) == 1 and records[0]["window_id"] == "w0000"
    assert validate_sfx_ledger(tmp_path, 8.0) == []


def test_validate_missing_ledger_is_clean_for_old_runs(tmp_path: Path) -> None:
    assert validate_sfx_ledger(tmp_path, 10.0) == []


def test_validate_flags_missing_file_and_gap(tmp_path: Path) -> None:
    from voyage.sfx_finalize import append_sfx_window

    ledger = tmp_path / "audio" / "sfx" / "sfx.jsonl"
    append_sfx_window(
        ledger,
        SfxWindow("w0000", 0.0, 8.0, "rain", 7),
        path="audio/sfx/w0000.wav",
        model_size="small_44k",
    )
    errors = validate_sfx_ledger(tmp_path, 16.0)
    assert any("missing" in error for error in errors)
    assert any("coverage" in error or "gap" in error for error in errors)


def test_plan_stub_tail_merges_into_predecessor() -> None:
    windows = plan_sfx_windows(14.5, [(0.0, 14.5, "rain")], seed_base=0)
    assert [(w.start, w.duration) for w in windows] == [(0.0, 8.0), (7.0, 7.5)]
    assert [w.window_id for w in windows] == ["w0000", "w0001"]


def test_plan_degenerate_timeline_raises() -> None:
    with pytest.raises(ValueError, match="minimum"):
        plan_sfx_windows(0.5, [(0.0, 0.5, "rain")], seed_base=0)


def test_sfx_window_constants_match_ladder() -> None:
    assert SFX_WINDOW_SECONDS == 8.0
    assert SFX_WINDOW_OVERLAP == 1.0


def test_sfx_verb_parses_with_caption_override() -> None:
    from voyage.cli import build_parser

    args = build_parser().parse_args(
        [
            "sfx",
            "--run",
            "/tmp/poulah",
            "--sfx-backend",
            "mmaudio",
            "--sfx-caption",
            "frenetic glitch foley",
            "--sfx-workers",
            "2",
        ]
    )
    assert args.func.__name__ == "cmd_sfx"
    assert args.sfx_backend == "mmaudio"
    assert args.sfx_caption == "frenetic glitch foley"
    assert args.sfx_workers == 2
    assert not hasattr(args, "no_sfx")
    assert args.video is None and args.output is None


def test_finalize_verb_accepts_new_sfx_flags() -> None:
    from voyage.cli import build_parser

    args = build_parser().parse_args(
        [
            "finalize",
            "--run",
            "/tmp/x",
            "--output",
            "/tmp/x.mp4",
            "--sfx-backend",
            "mmaudio",
            "--sfx-caption",
            "wind",
        ]
    )
    assert args.sfx_backend == "mmaudio"
    assert args.sfx_caption == "wind"


def test_caption_override_replaces_segment_captions(tmp_path: Path) -> None:
    from voyage.sfx_finalize import segment_sfx_bounds

    run_dir = tmp_path / "run"
    _make_finalize_segment(run_dir, "000000", 4.0, "glass chimes")
    usable = [run_dir / "segments" / "000000"]
    bounds = segment_sfx_bounds(run_dir, usable, 24, caption_override="forced dub")
    assert bounds[0][2] == "forced dub"
    plain = segment_sfx_bounds(run_dir, usable, 24)
    assert plain[0][2] == "glass chimes"


def test_cmd_sfx_rejects_missing_video(tmp_path: Path) -> None:
    import argparse

    from tests.conftest import initialize_run_directory
    from voyage.cli import cmd_sfx

    initialize_run_directory(tmp_path, run_id="x", style="y", seed=0)
    code = cmd_sfx(
        argparse.Namespace(
            run=str(tmp_path),
            video=str(tmp_path / "nope.mp4"),
            output=None,
            sfx_backend="fake",
            sfx_caption=None,
            sfx_device=None,
            sfx_model_size=None,
            sfx_workers=1,
        )
    )
    assert code == 1


def _make_finalize_segment(run_dir: Path, seg_id: str, duration: float, sfx_caption: str) -> None:
    import subprocess

    from voyage.director import deterministic_decision

    segment = run_dir / "segments" / seg_id
    segment.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=24:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(segment / "video.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(segment / "audio.wav"),
        ],
        check=True,
    )
    decision = deterministic_decision(0, "reef", "reef", "ESTABLISH", "pastel neon")
    dumped = decision.model_dump()
    dumped["audio"]["sfx_caption"] = sfx_caption
    (segment / "transition.json").write_text(__import__("json").dumps(dumped))
    (segment / "DONE").write_text("", encoding="utf-8")


def test_run_and_generate_accept_caption_pins() -> None:
    from voyage.cli import build_parser

    run_args = build_parser().parse_args(
        ["run", "--run", "r", "--music-caption", "brass", "--video-caption", "dune"]
    )
    assert run_args.music_caption == "brass"
    assert run_args.video_caption == "dune"
    gen_args = build_parser().parse_args(
        ["configure", "x", "--segments", "1", "--music-caption", "brass"]
    )
    assert gen_args.music_caption == "brass"
    assert gen_args.video_caption is None


def test_fake_bed_end_to_end_over_junctions(tmp_path: Path) -> None:
    import subprocess

    from voyage.sfx_finalize import (
        mix_music_and_sfx,
        render_sfx_bed,
        segment_sfx_bounds,
    )

    run_dir = tmp_path / "run"
    _make_finalize_segment(run_dir, "000000", 4.0, "glass chimes")
    _make_finalize_segment(run_dir, "000001", 4.0, "deep rumble")
    _make_finalize_segment(run_dir, "000002", 4.0, "soft wind")
    usable = [run_dir / "segments" / name for name in ("000000", "000001", "000002")]
    final_video = tmp_path / "final_video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(usable[0] / "video.mp4"),
            "-i",
            str(usable[1] / "video.mp4"),
            "-i",
            str(usable[2] / "video.mp4"),
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(final_video),
        ],
        check=True,
    )
    music = tmp_path / "music.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=12",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(music),
        ],
        check=True,
    )
    bounds = segment_sfx_bounds(run_dir, usable, 24)
    assert [caption for _, _, caption in bounds] == ["glass chimes", "deep rumble", "soft wind"]
    bed = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        bounds,
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "large_44k_v2",
        11,
        48000,
        2,
        1,
    )
    assert bed.exists()
    mixed = mix_music_and_sfx(music, bed, tmp_path / "mixed.wav", 48000, 2)
    assert mixed.exists()
    assert validate_sfx_ledger(tmp_path / "run", 12.0) == []
    records = load_sfx_ledger(tmp_path / "run" / "audio" / "sfx" / "sfx.jsonl")
    assert len(records) == 2
    assert "glass chimes" in records[0]["caption"]
    assert "deep rumble" in records[0]["caption"]


def test_finalize_sfx_pass_emits_timing_metric(tmp_path: Path) -> None:
    import json
    import subprocess

    from voyage.sfx_finalize import finalize_sfx_pass

    run_dir = tmp_path / "run"
    _make_finalize_segment(run_dir, "000000", 4.0, "glass chimes")
    final_video = tmp_path / "final.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(run_dir / "segments" / "000000" / "video.mp4"),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=4",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(final_video),
        ],
        check=True,
    )
    out = finalize_sfx_pass(
        run_dir, final_video, "fake", "/models", "cpu", "large_44k_v2", 11, 48000, 2, 1, 24
    )
    assert out == final_video
    events = [
        json.loads(line)
        for line in (run_dir / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    timed = [entry for entry in events if entry.get("event") == "sfx_pass_completed"]
    assert len(timed) == 1
    assert timed[0]["sfx_pass_s"] >= 0.0
    assert timed[0]["windows"] == 1
    assert timed[0]["backend"] == "fake"

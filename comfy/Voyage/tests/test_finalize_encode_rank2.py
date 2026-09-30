"""Rank-2 finalize encode (050) + concat quoting / ffmpeg hygiene (053).

050: non-native finalize double-encodes (N libx264 parts + final vf
re-encode), hardcodes `veryfast` with no CRF knob and no per-stage
timings. The fix is a single vf encode over the concat demuxer plus
`crf`/`preset` on `FinalizeOptions` and a `finalize_completed` metrics
event carrying the stage timings.

053: concat demuxer lists interpolate raw paths (a single quote breaks the
list) and `audio_acestep._convert` omits `-hide_banner`/`-nostdin`.
NOTE: 102's quoting legs were deferred to 053 — this module owns the
quoting fix.

Real ffmpeg throughout (slim image ships it); segments come from the
fake-backend commit path like the existing finalize tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config
from voyage.supervisor import Supervisor


def _init_run(run_dir: Path) -> None:
    initialize_run_directory(run_dir, run_id="rank2encode", style="pastel neon line-art, peaceful")


def _commit(run_dir: Path, count: int) -> None:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        for _ in range(count):
            supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _metrics_events(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                events.append(loaded)
    return events


# ---------------------------------------------------------------------------
# 050: FinalizeOptions exposes crf/preset
# ---------------------------------------------------------------------------


def test_finalize_options_exposes_crf_preset_defaults() -> None:
    from voyage.media import FinalizeOptions

    options = FinalizeOptions()
    assert options.crf == 15
    assert options.preset == "veryfast"


def test_finalize_options_rejects_bad_crf_preset() -> None:
    from voyage.media import FinalizeOptions

    with pytest.raises(ValueError, match="crf"):
        FinalizeOptions(crf=-1)
    with pytest.raises(ValueError, match="crf"):
        FinalizeOptions(crf=52)
    with pytest.raises(ValueError, match="preset"):
        FinalizeOptions(preset="turbo")


# ---------------------------------------------------------------------------
# 053: concat escaping helper
# ---------------------------------------------------------------------------


def test_escape_concat_path_handles_single_quote() -> None:
    from voyage.media import escape_concat_path

    assert escape_concat_path(Path("/tmp/a'b/c.mp4")) == "/tmp/a'\\''b/c.mp4"
    assert escape_concat_path(Path("/tmp/plain/c.mp4")) == "/tmp/plain/c.mp4"
    assert escape_concat_path("o'brien") == "o'\\''brien"


def test_write_concat_list_escapes_quote(tmp_path: Path) -> None:
    from voyage.media import write_concat_list

    dest = tmp_path / "concat.txt"
    write_concat_list([Path("/tmp/voyage o'brien/segments/000000/video.mp4")], dest)
    text = dest.read_text(encoding="utf-8")
    assert text == "file '/tmp/voyage o'brien/segments/000000/video.mp4'\n".replace(
        "o'brien", "o'\\''brien"
    )


def test_audio_acestep_convert_uses_hygiene_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage.workers import audio_acestep

    seen: dict[str, Any] = {}
    real_run = subprocess.run

    def _recording(*args: Any, **kwargs: Any) -> Any:
        seen["argv"] = args[0] if args else kwargs.get("args")
        return real_run(["true"], capture_output=True, text=True, check=False)

    monkeypatch.setattr(audio_acestep.subprocess, "run", _recording)
    rendered = tmp_path / "take.flac"
    rendered.write_bytes(b"fake")
    output = tmp_path / "out.wav"
    audio_acestep._convert(rendered, output, sample_rate=48000, channels=2)
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert "-hide_banner" in argv
    assert "-nostdin" in argv


# ---------------------------------------------------------------------------
# 050: single-pass re-encode (no intermediate libx264 parts)
# ---------------------------------------------------------------------------


def test_reencode_is_single_pass_with_crf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-native finalize runs one libx264 encode carrying -crf/-preset."""
    import voyage.media as media_module

    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)

    calls: list[list[str]] = []
    real_run_capture = media_module.run_capture

    def _recording(argv: list[str], timeout: float = 600.0) -> Any:
        calls.append(list(argv))
        return real_run_capture(argv, timeout=timeout)

    monkeypatch.setattr(media_module, "run_capture", _recording)
    out = tmp_path / "final-single.mp4"
    options = media_module.FinalizeOptions(crf=18, preset="fast")
    assert media_module.finalize_run(
        run_dir,
        out,
        width=640,
        height=352,
        fps=24,
        min_fps=0,
        min_width=0,
        min_height=0,
        options=options,
    ).exists()
    video_encodes = [argv for argv in calls if "-c:v" in argv and "libx264" in argv]
    assert len(video_encodes) == 1, f"expected single encode, got {len(video_encodes)}: {calls!r}"
    only = video_encodes[0]
    assert "-crf" in only and "18" in only
    assert "-preset" in only and "fast" in only
    assert "-vf" in only


def test_finalize_emits_stage_timings(tmp_path: Path) -> None:
    """`finalize_completed` carries parts/audio/final timings + effective knobs."""
    import voyage.media as media_module

    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    out = tmp_path / "final-timed.mp4"
    assert media_module.finalize_run(
        run_dir,
        out,
        width=640,
        height=352,
        fps=24,
        min_fps=0,
        min_width=0,
        min_height=0,
    ).exists()
    events = [e for e in _metrics_events(run_dir) if e.get("event") == "finalize_completed"]
    assert events, "finalize must emit a finalize_completed metrics event"
    event = events[-1]
    for key in ("parts_encode_ms", "audio_blend_ms", "final_encode_ms"):
        value = event.get(key)
        assert isinstance(value, (int, float)), key
        assert float(value) >= 0.0, key
    assert event.get("crf") == 15
    assert event.get("preset") == "veryfast"


def test_finalize_adversarial_path_with_quote(tmp_path: Path) -> None:
    """A run dir containing `'` and spaces still finalizes (053 quoting)."""
    import voyage.media as media_module

    run_dir = tmp_path / "o'brien run" / "run"
    _init_run(run_dir)
    _commit(run_dir, 1)
    out = tmp_path / "o'brien run" / "final-quote.mp4"
    assert media_module.finalize_run(run_dir, out, min_fps=0, min_width=0, min_height=0).exists()
    assert out.exists() and out.stat().st_size > 0

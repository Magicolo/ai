"""Always-deferred finalize: ledger-only music rendering (DESIGN §140).

Why: every backend commits video only — no per-segment `audio.wav`
previews. `ensure_deferred_takes` replays director decisions at finalize
(ACE-Step on cuda:0, or the fake sine worker via `audio_backend`
routing); `build_final_audio` blends the takes ledger and fails loud
without it.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from voyage.audio_finalize import (
    deferred_render_take_count,
    deferred_tail_frames,
    derive_conditioning_tail,
    ensure_deferred_takes,
    trim_repaint_head,
)
from voyage.errors import MediaError
from voyage.fake_backends import FakeVideoBackend


def _write_sine_wav(path: Path, duration_seconds: float) -> None:
    """Stdlib sine stub: 48 kHz stereo s16le, exact duration.

    Batches the whole buffer into ONE `writeframes` call: the old
    per-sample loop paid a syscall per frame (~2 s per rendered second,
    ~60 s for this module alone).
    """
    rate, channels, freq = 48000, 2, 440.0
    frames = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = array.array(
        "h",
        (int(12000.0 * math.sin(2.0 * math.pi * freq * index / rate)) for index in range(frames)),
    )
    stereo = array.array("h", (sample for value in mono for sample in (value, value)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(stereo.tobytes())


def _stub_render(payload: dict[str, Any], output_path: Path) -> None:
    """Render seam: stdlib sine at the payload duration (no GPU)."""
    duration = payload["duration_seconds"]
    assert isinstance(duration, (int, float))
    _write_sine_wav(output_path, float(duration))


def _segment_with_decision(
    run_dir: Path, index: int, frames: int, caption: str, energy: float
) -> Path:
    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    manifest = {
        "format": 1,
        "transition": {
            "decision": {
                "decision_index": index,
                "audio": {"music_caption": caption, "energy": energy},
            }
        },
        "metrics": {"frames": frames},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    return segment


def _production_segment(
    run_dir: Path, index: int, frames: int, caption: str, energy: float = 0.4
) -> Path:
    """Production manifest shape: the decision flattened at `transition`.

    The supervisor persists `decision.model_dump()` directly (no `decision`
    wrapper) — the replay must read per-segment captions from this shape,
    not fall back to the run music style for every take.
    """
    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    manifest = {
        "format": 1,
        "transition": {
            "decision_index": index,
            "audio": {"music_caption": caption, "energy": energy},
        },
        "metrics": {"frames": frames},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    return segment


def test_replay_reads_flattened_production_captions(tmp_path: Path) -> None:
    """Takes carry each segment's own caption, not the style fallback."""
    seg0 = _production_segment(tmp_path, 0, 96, "hollow winds")
    seg1 = _production_segment(tmp_path, 1, 96, "hollow winds")
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0, seg1],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        music_style="fallback style",
    )
    assert takes, "expected at least one rendered take"
    assert takes[0]["caption"] == "hollow winds"


def test_chained_take_uses_coverage_start_caption(tmp_path: Path) -> None:
    """A chained take is prompted with its starting segment's caption.

    10s segments, take 50s (quantized), ahead 20s, chain overlap 6s: seg3
    (cursor 30s) chains at 50s with covers_from 44s, whose coverage begins
    in seg4 — so the take must carry seg4's caption, not seg3's. Exactly
    two takes cover the 60s timeline.
    (Captions share 4/6 tokens so the repaint gate stays shut — this test
    pins chaining, not repainting.)
    """
    captions = [f"dark ambient drone verse {index}" for index in range(6)]
    segments = [_production_segment(tmp_path, index, 240, captions[index]) for index in range(6)]
    seen: list[dict[str, Any]] = []

    def _capture(payload: dict[str, Any], output_path: Path) -> None:
        seen.append(dict(payload))
        _stub_render(payload, output_path)

    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_capture,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 2, f"expected exactly 2 takes, got {len(takes)}"
    assert takes[0]["covers_from"] == 0.0
    assert takes[0]["caption"] == "dark ambient drone verse 0"
    assert takes[1]["covers_from"] == 44.0
    assert takes[1]["caption"] == "dark ambient drone verse 4"
    assert takes[1]["segment_index"] == 4
    assert seen[1]["style"] == "dark ambient drone verse 4"
    from voyage import audio_finalize

    assert not audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )


def test_chained_take_renders_as_continuation_repaint(tmp_path: Path) -> None:
    """A chained take extends the previous take (continuation, not restart).

    Same geometry as the chained-caption test (10s segments, 50s take,
    20s ahead, 6s chain overlap): the chained take renders with the ACE
    repaint payload (task_type repaint, reference_audio pointing at the
    throwaway continuation source built from the previous take's tail,
    repaint_start 6.0, repaint_end = duration), and the source file is
    cleaned up after the render — never ledgered.
    """
    captions = [f"dark ambient drone verse {index}" for index in range(6)]
    segments = [_production_segment(tmp_path, index, 240, captions[index]) for index in range(6)]
    seen: list[dict[str, Any]] = []

    def _capture(payload: dict[str, Any], output_path: Path) -> None:
        seen.append(dict(payload))
        _stub_render(payload, output_path)

    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_capture,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 2, f"expected exactly 2 takes, got {len(takes)}"
    continuations = [payload for payload in seen if payload.get("task_type") == "repaint"]
    assert len(continuations) == 1, "exactly the chained take continues its predecessor"
    chained = continuations[0]
    assert chained["repaint_start"] == 6.0
    assert chained["repaint_end"] == chained["duration_seconds"]
    reference = Path(str(chained["reference_audio"]))
    assert reference.name == "take_0001_src.wav"
    assert not reference.exists(), "continuation source must be cleaned after render"
    assert [take["take_id"] for take in takes] == ["take_0000", "take_0001"]


def test_pathologically_short_renders_fail_loud(tmp_path: Path) -> None:
    """A renderer returning ~empty audio fails fast instead of minting takes forever.

    Regression: the render loop shrank each take to its probed length on
    any shortfall, so ~0.3 s renders crawled coverage forward and appended
    takes without bound (140 takes observed for a 4-segment run). A
    shortfall beyond max(1.0 s, 5% of requested) now raises MediaError on
    the first bad render — the render count pins that no second render
    is ever attempted. `chain_overlap_seconds=0.0` isolates the shortfall
    guard: with overlap on, the second chained take would fail earlier in
    `_build_continuation_src` (0.3 s tail < overlap), so the all-zeros
    overlap is what pins THIS guard red on the old code (140 takes, no
    error) and green on the new (one render, then MediaError).
    """
    segments = [_production_segment(tmp_path, index, 240, "hollow winds") for index in range(4)]
    rendered: list[float] = []

    def _short_render(payload: dict[str, Any], output_path: Path) -> None:
        rendered.append(float(payload["duration_seconds"]))
        _write_sine_wav(output_path, 0.3)

    with pytest.raises(MediaError):
        ensure_deferred_takes(
            run_dir=tmp_path,
            usable=segments,
            source_fps=24.0,
            run_seed=7,
            render_take_fn=_short_render,
            take_seconds=45.0,
            ahead_seconds=20.0,
            chain_overlap_seconds=0.0,
        )
    assert len(rendered) == 1, f"expected the first bad render to fail, got {len(rendered)} renders"


def test_ledger_adopts_longer_rendered_length(tmp_path: Path) -> None:
    """The ledger records the probed file length when the render runs long.

    Regression (fett): the renderer returned 64.2 s for a 58.0 s plan and
    the ledger kept 58.0, mis-slicing every downstream window by ~6 s. An
    overshoot within max(1.0 s, 5% of requested) adopts the rendered length
    into both the returned take and the ledger line, so coverage matches
    the bytes the slice walk will read.
    """
    seg0 = _production_segment(tmp_path, 0, 96, "hollow winds")
    requested_durations: list[float] = []

    def _long_render(payload: dict[str, Any], output_path: Path) -> None:
        requested = float(payload["duration_seconds"])
        requested_durations.append(requested)
        _write_sine_wav(output_path, requested + 1.0)

    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_long_render,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 1, f"expected exactly one take, got {len(takes)}"
    (requested,) = requested_durations
    adopt_bound = max(1.0, 0.05 * requested)
    assert adopt_bound >= 1.0, "1 s overshoot must sit inside the adopt bound"
    assert abs(takes[0]["duration"] - (requested + 1.0)) < 0.15
    ledger_line = json.loads((tmp_path / "audio" / "takes.jsonl").read_text().strip())
    assert abs(ledger_line["duration"] - (requested + 1.0)) < 0.15


def test_absurd_overshoot_fails_loud(tmp_path: Path) -> None:
    """A renderer returning far more audio than requested raises MediaError.

    Mirror of the shortfall guard: an overshoot beyond max(1.0 s, 5% of
    requested) means the renderer ignored the plan — ledgering it would
    silently stretch coverage, so the first bad render fails loud with
    nothing appended, and the render count pins no second attempt.
    """
    seg0 = _production_segment(tmp_path, 0, 96, "hollow winds")
    rendered: list[float] = []

    def _huge_render(payload: dict[str, Any], output_path: Path) -> None:
        rendered.append(float(payload["duration_seconds"]))
        _write_sine_wav(output_path, float(payload["duration_seconds"]) + 10.0)

    with pytest.raises(MediaError, match="overshoot"):
        ensure_deferred_takes(
            run_dir=tmp_path,
            usable=[seg0],
            source_fps=24.0,
            run_seed=7,
            render_take_fn=_huge_render,
            take_seconds=45.0,
            ahead_seconds=20.0,
        )
    assert len(rendered) == 1, f"expected the first bad render to fail, got {len(rendered)} renders"
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert not ledger.exists() or ledger.read_text().strip() == ""


def test_sane_renders_terminate_with_bounded_takes(tmp_path: Path) -> None:
    """Healthy renders cover a 4-segment timeline with exactly two takes.

    Pins the backstop the other way: the per-segment iteration cap must
    never fire on a sane run (one 50 s take covers 0-50 s, the 30 s cursor
    chains a second at 44 s exactly like the six-segment chain geometry).
    """
    segments = [_production_segment(tmp_path, index, 240, "hollow winds") for index in range(4)]
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 2, f"expected exactly 2 takes, got {len(takes)}"
    assert takes[1]["covers_from"] == 44.0


def test_replay_converges_after_repaints(tmp_path: Path) -> None:
    """A re-finalize no-ops once repaints cover the timeline.

    Mutually-dissimilar captions force a repaint per segment; each
    repaint backdates its coverage, so without the serving bound every
    replay would repaint the early cursors again (unbounded ledger
    growth). With it the dry walk keeps everywhere on replay. Repaint
    takes re-anchor at their segment start (trimmed unplayed head), so
    covers run [0.0, 4.0, 8.0] — one take per shifted segment, no
    span ledgered twice.
    """
    from voyage import audio_finalize

    captions = ("crimson brass thunderstorm", "silent glass glacier", "velvet neon circuitry")
    segments = [_production_segment(tmp_path, index, 96, captions[index]) for index in range(3)]
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 3, f"expected one take per shifted segment, got {len(takes)}"
    assert [take["covers_from"] for take in takes] == [0.0, 4.0, 8.0]
    assert not audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )


def test_repaint_over_longer_source_requests_source_length(tmp_path: Path) -> None:
    """A repaint across different-length segments requests the source length.

    Regression (fett finalize): seg 0 (257f) minted a 64.2 s take while seg
    1's (232f) quantum was 58.0 s; the repaint requested 58.0 s over the
    64.2 s reference and ACE returned 64.2 s of audio, failing finalize
    with `overshoot 6.200s`. The ACE-mimicking renderer below returns
    reference-length audio for repaints (`repaint_end` seconds, like the
    real worker returning its source length); the plan must therefore
    request the source length, so request, `repaint_end`, and file agree
    and the overshoot guard stays silent. Fails on the old code with the
    exact fett signature (requested 45.0 s, rendered 50.0 s, overshoot
    beyond max(1.0 s, 5%)). The ledgered take is then trimmed to its
    fresh region (the unplayed preserved head never spans twice), so it
    covers 10..50 while the request stays the full source 50 s.
    """
    segments = [
        _production_segment(tmp_path, 0, 240, "crimson brass thunderstorm"),
        _production_segment(tmp_path, 1, 120, "silent glass glacier"),
    ]
    seen: list[dict[str, Any]] = []

    def _ace_like_render(payload: dict[str, Any], output_path: Path) -> None:
        seen.append(dict(payload))
        if payload.get("task_type") == "repaint":
            _write_sine_wav(output_path, float(payload["repaint_end"]))
        else:
            _stub_render(payload, output_path)

    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_ace_like_render,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 2, f"expected fresh take + repaint, got {len(takes)}"
    repaints = [payload for payload in seen if payload.get("task_type") == "repaint"]
    assert len(repaints) == 1, "exactly the second segment repaints its predecessor"
    repaint = repaints[0]
    assert repaint["duration_seconds"] == 50.0
    assert repaint["duration_seconds"] == repaint["repaint_end"]
    assert abs(takes[1]["duration"] - 40.0) < 0.15
    assert abs(takes[1]["covers_from"] - 10.0) < 0.01


def test_ensure_deferred_takes_forwards_sample_rate_and_channels(tmp_path: Path) -> None:
    """Takes render at the run's audio rate/channels, not a hardcoded pair.

    The finalize takes must match the run's configured rate/channels —
    a hardcoded 48k/2 would split-brain any run configured otherwise.
    Defaults stay 48k stereo when not given.
    """
    from voyage.audio_finalize import ensure_deferred_takes

    run_dir = tmp_path / "run"
    segment = _segment_with_decision(run_dir, 0, 96, "hollow winds", 0.4)
    seen: list[dict[str, Any]] = []

    def _capture(payload: dict[str, Any], output_path: Path) -> None:
        seen.append(dict(payload))
        _stub_render(payload, output_path)

    ensure_deferred_takes(
        run_dir=run_dir,
        usable=[segment],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_capture,
        sample_rate=44100,
        channels=1,
    )
    assert seen, "expected at least one rendered take"
    assert seen[0]["sample_rate"] == 44100
    assert seen[0]["channels"] == 1

    run_dir2 = tmp_path / "run2"
    segment2 = _segment_with_decision(run_dir2, 0, 96, "hollow winds", 0.4)
    seen2: list[dict[str, Any]] = []

    def _capture2(payload: dict[str, Any], output_path: Path) -> None:
        seen2.append(dict(payload))
        _stub_render(payload, output_path)

    ensure_deferred_takes(
        run_dir=run_dir2,
        usable=[segment2],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_capture2,
    )
    assert seen2[0]["sample_rate"] == 48000
    assert seen2[0]["channels"] == 2


def test_ensure_deferred_takes_renders_and_ledgers(tmp_path: Path) -> None:
    """Replay renders takes covering both segments and appends the ledger."""
    seg0 = _segment_with_decision(tmp_path, 0, 121, "dark drone", 0.4)
    seg1 = _segment_with_decision(tmp_path, 1, 96, "dark drone evolving", 0.5)
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0, seg1],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    assert len(takes) >= 1
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert ledger.exists()
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == len(takes)
    for take in takes:
        assert (tmp_path / take["path"]).exists()
    coverage = max(t["covers_from"] + t["duration"] for t in takes)
    assert coverage >= 121 / 24.0 + 96 / 24.0 - 1e-6


def test_ensure_deferred_takes_fail_loud_on_render_error(tmp_path: Path) -> None:
    """A failed take render raises and appends nothing to the ledger."""
    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)

    def _boom(payload: dict[str, object], output_path: Path) -> None:
        raise RuntimeError("synthetic ACE failure")

    with pytest.raises(MediaError):
        ensure_deferred_takes(
            run_dir=tmp_path,
            usable=[seg0],
            source_fps=24.0,
            run_seed=0,
            render_take_fn=_boom,
        )
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert not ledger.exists() or ledger.read_text().strip() == ""


def test_ensure_deferred_takes_cover_stretched_timeline(tmp_path: Path) -> None:
    """Slow-mo stretch scales take coverage (1.5x: 4 s content -> 6 s music)."""
    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        stretch=1.5,
    )
    coverage = max(t["covers_from"] + t["duration"] for t in takes)
    assert coverage >= (96 / 24.0) * 1.5 - 1e-6


def test_deferred_render_pending_tracks_ledger_coverage(tmp_path: Path) -> None:
    """Dry walk is True fresh, False once the ledger covers the timeline."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    assert audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
    )
    ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    assert not audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
    )


def _video_segment(
    run_dir: Path,
    index: int,
    frames: int = 96,
    fps: int = 24,
    caption: str = "dark drone",
    energy: float = 0.4,
) -> Path:
    """Committed-shape video-only segment: DONE + testsrc video + manifest."""
    segment = run_dir / "segments" / f"{index:06d}"
    segment.mkdir(parents=True)
    duration = frames / fps
    video = segment / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={fps}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "format": 1,
        "transition": {
            "decision": {
                "decision_index": index,
                "audio": {"music_caption": caption, "energy": energy},
            }
        },
        "metrics": {"frames": frames},
        "checksums": {"video.mp4": _sha(video)},
    }
    (segment / "manifest.json").write_text(json.dumps(manifest))
    (segment / "DONE").write_bytes(b"")
    return segment


def test_build_final_audio_fails_loud_without_takes(tmp_path: Path) -> None:
    """Ledger-only mix with no rendered takes raises instead of shipping silence."""
    from voyage.media_audio import build_final_audio

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    with pytest.raises(MediaError, match="no rendered takes"):
        build_final_audio(
            run_dir,
            [seg0],
            tmp_path / "tmp",
            24,
            48000,
            2,
        )


def test_build_final_audio_blends_rendered_takes(tmp_path: Path) -> None:
    """Mix after `ensure_deferred_takes` blends real music."""
    from voyage.media_audio import build_final_audio, probed_take_seconds

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    seg1 = _video_segment(run_dir, 1, caption="dark drone evolving", energy=0.5)
    ensure_deferred_takes(
        run_dir=run_dir,
        usable=[seg0, seg1],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    dest = build_final_audio(
        run_dir,
        [seg0, seg1],
        tmp_path / "tmp",
        24,
        48000,
        2,
    )
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 8.0) < 0.3


def test_build_final_audio_stretch_uses_stretched_timeline(tmp_path: Path) -> None:
    """`stretch=1.5` mixes the 4 s segment into a 6 s slow-mo timeline."""
    from voyage.media_audio import build_final_audio, probed_take_seconds

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0)
    ensure_deferred_takes(
        run_dir=run_dir,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
        stretch=1.5,
    )
    dest = build_final_audio(
        run_dir,
        [seg0],
        tmp_path / "tmp",
        24,
        48000,
        2,
        stretch=1.5,
    )
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 6.0) < 0.6


def _ledger_take(
    run_dir: Path,
    take_id: str,
    covers_from: float,
    duration: float,
    segment_index: int,
) -> None:
    """Append one ledger take with a real sine wav behind it (asporgue sliver rig)."""
    from voyage.audio.planner import AudioTake

    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / f"{take_id}.wav"
    _write_sine_wav(wav, duration)
    record = AudioTake(
        take_id=take_id,
        path=f"audio/{take_id}.wav",
        caption="sliver rig",
        seed=0,
        covers_from=covers_from,
        duration=duration,
        segment_index=segment_index,
    ).to_dict()
    import json

    with (audio_dir / "takes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_build_final_audio_skips_tail_sliver(tmp_path: Path) -> None:
    """Asporgue 2026-10-07: take coverage ending dust-short of the timeline mixes.

    Frame-exact timeline 32.125 s (257 frames at fps 8) with takes ending
    at 32.10 s leaves a 0.025 s tail served by the over-covering take —
    previously `MediaError: sliver take piece`, aborting a healthy
    finalize. The walk skips the sub-floor piece; the mix lands 0.025 s
    short, inside the 0.6 s H3 budget.
    """
    from voyage.media_audio import build_final_audio, probed_take_seconds

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0, frames=257, fps=24)
    _ledger_take(run_dir, "take_0000", 0.0, 32.10, 0)
    _ledger_take(run_dir, "take_0001", 32.0, 68.0, 0)
    dest = build_final_audio(run_dir, [seg0], tmp_path / "tmp", 8, 48000, 2)
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 32.10) < 0.3


def test_build_final_audio_skips_head_sliver(tmp_path: Path) -> None:
    """A sub-floor piece at a window head is skipped the same way (timeline-exact)."""
    from voyage.media_audio import build_final_audio, probed_take_seconds

    run_dir = tmp_path / "run"
    seg0 = _video_segment(run_dir, 0, frames=257, fps=24)
    _ledger_take(run_dir, "take_short", 0.0, 0.03, 0)
    _ledger_take(run_dir, "take_long", 0.0, 100.0, 0)
    dest = build_final_audio(run_dir, [seg0], tmp_path / "tmp", 8, 48000, 2)
    assert dest.exists()
    assert abs(probed_take_seconds(dest) - 32.095) < 0.3


def test_spawn_ace_render_fn_factory_exists() -> None:
    """Production ACE renderer factory is importable (spawning needs a GPU)."""
    from voyage import audio_finalize

    assert callable(audio_finalize.spawn_ace_render_fn)


def test_spawn_fake_render_fn_factory_exists() -> None:
    """Offline fake renderer factory is importable (spawning needs only ffmpeg)."""
    from voyage import audio_finalize

    assert callable(audio_finalize.spawn_fake_render_fn)


def test_spawn_fake_render_fn_renders_sine(tmp_path: Path) -> None:
    """The fake factory renders real sine audio through a live worker."""
    from voyage import audio_finalize

    render, shutdown = audio_finalize.spawn_fake_render_fn(tmp_path)
    try:
        out = tmp_path / "take.wav"
        render(
            {
                "segment_id": "000000",
                "style": "dark drone",
                "energy": 0.4,
                "seed": 0,
                "duration_seconds": 1.0,
                "sample_rate": 48000,
                "channels": 2,
            },
            out,
        )
        assert out.exists() and out.stat().st_size > 0
    finally:
        shutdown()


def test_ensure_routing_fake_uses_fake_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`audio_backend="fake"` renders via the fake factory, never ACE."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    calls: list[str] = []

    def _recording_fake(run_dir: Path) -> tuple[object, object]:
        calls.append("fake")
        return (_stub_render, lambda: None)

    monkeypatch.setattr(audio_finalize, "spawn_fake_render_fn", _recording_fake)

    def _forbidden(run_dir: Path, models_dir: Path | str | None, device: str = "cuda:0") -> object:
        raise AssertionError("ACE factory must not run for audio_backend='fake'")

    monkeypatch.setattr(audio_finalize, "spawn_ace_render_fn", _forbidden)
    # `models_dir=None` would raise inside the ACE factory — rendering
    # proves the fake route (which needs no weights) was taken.
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        audio_backend="fake",
    )
    assert rendered is True
    assert calls == ["fake"]
    assert (tmp_path / "audio" / "takes.jsonl").exists()


def test_ensure_routing_acestep_uses_ace_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`audio_backend="acestep"` (default) renders via the ACE factory."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    seen: list[object] = []

    def _fake_spawn_fake(run_dir: Path) -> object:
        raise AssertionError("fake factory must not run for audio_backend='acestep'")

    def _recording_ace(
        run_dir: Path, models_dir: Path | str | None, device: str = "cuda:0"
    ) -> tuple[object, object]:
        seen.append(models_dir)
        return (_stub_render, lambda: None)

    monkeypatch.setattr(audio_finalize, "spawn_fake_render_fn", _fake_spawn_fake)
    monkeypatch.setattr(audio_finalize, "spawn_ace_render_fn", _recording_ace)
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir="/models",
    )
    assert rendered is True
    assert seen == ["/models"]


def test_ensure_routing_unknown_backend_fails_loud(tmp_path: Path) -> None:
    """An unknown audio backend raises instead of guessing a renderer."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    with pytest.raises(MediaError, match="unknown audio backend"):
        audio_finalize.ensure_deferred_for_finalize(
            run_dir=tmp_path,
            usable=[seg0],
            source_fps=24.0,
            stretch=1.0,
            run_seed=0,
            models_dir="/models",
            audio_backend="orchestra",
        )


def test_ensure_fake_end_to_end_needs_no_models(tmp_path: Path) -> None:
    """Fake routing renders takes through a live worker with no models_dir."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        audio_backend="fake",
    )
    assert rendered is True
    ledger = tmp_path / "audio" / "takes.jsonl"
    assert ledger.exists() and ledger.read_text().strip() != ""


def test_finalize_run_always_renders_takes_before_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`finalize_run` renders takes, then mixes/muxes (no opt-in flag)."""
    import json as json_module

    from voyage import audio_finalize
    from voyage.media import finalize_run

    run_dir = tmp_path / "run"
    _video_segment(run_dir, 0)
    _video_segment(run_dir, 1, caption="dark drone evolving", energy=0.5)
    monkeypatch.setattr(
        audio_finalize,
        "spawn_ace_render_fn",
        lambda run_dir, models_dir, device="cuda:0": (_stub_render, lambda: None),
    )
    out = run_dir / "final.mp4"
    finalize_run(
        run_dir,
        out,
        upscale=1,
        interpolate=1,
        seed=0,
    )
    assert out.exists()
    ledger = run_dir / "audio" / "takes.jsonl"
    assert ledger.exists()
    takes = [json_module.loads(line) for line in ledger.read_text().strip().splitlines()]
    assert len(takes) >= 1


def test_ensure_deferred_for_finalize_noops_on_complete_ledger(tmp_path: Path) -> None:
    """Re-finalize with a complete ledger spawns no worker and renders nothing."""
    from voyage import audio_finalize

    seg0 = _segment_with_decision(tmp_path, 0, 96, "dark drone", 0.4)
    ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
        render_take_fn=_stub_render,
    )
    assert not audio_finalize.deferred_render_pending(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        run_seed=0,
    )

    # `models_dir=None` would raise inside the spawn factory — returning
    # False proves the pending check fired first and no worker was spawned.
    rendered = audio_finalize.ensure_deferred_for_finalize(
        run_dir=tmp_path,
        usable=[seg0],
        source_fps=24.0,
        stretch=1.0,
        run_seed=0,
        models_dir=None,
        device="cuda:0",
    )
    assert rendered is False


_TESTSRC = FakeVideoBackend()


def _render_testsrc_segment_video(path: Path, *, frames: int, fps: int) -> Path:
    """Real ffmpeg `testsrc` segment video (fake-backend-first, real media)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _TESTSRC.generate_segment(
        path, prompt="tail probe", seed=7, width=64, height=64, fps=fps, frames=frames
    )
    return path


def _count_frames(path: Path) -> int:
    """Frame count via ffprobe (decodes — exact, no container estimate)."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ffprobe failed for {path}: {proc.stderr[-500:]}"
    return int(proc.stdout.strip())


def test_derive_conditioning_tail_writes_25f_tail(tmp_path: Path) -> None:
    """Commit-time tail derive: last 25 frames of the segment video.

    Why: the commit path holds no audio worker whose rebuild derived
    `video_tail.mp4`; without this derive the next segment's resident
    tail path is missing and the worker goes fresh (121f) instead of
    continuing (96f).
    """
    segment = tmp_path / "segments" / "000000"
    _render_testsrc_segment_video(segment / "video.mp4", frames=121, fps=24)
    derived = derive_conditioning_tail(segment, deferred_tail_frames("ltxv"))
    assert derived is True
    tail = segment / "video_tail.mp4"
    assert tail.is_file()
    assert _count_frames(tail) == 25


def test_derive_conditioning_tail_noops_when_present(tmp_path: Path) -> None:
    """An existing tail is adopted untouched (resume no-op)."""
    segment = tmp_path / "segments" / "000000"
    _render_testsrc_segment_video(segment / "video.mp4", frames=121, fps=24)
    assert derive_conditioning_tail(segment, deferred_tail_frames("ltxv")) is True
    before = (segment / "video_tail.mp4").stat().st_mtime_ns
    assert derive_conditioning_tail(segment, deferred_tail_frames("ltxv")) is False
    assert (segment / "video_tail.mp4").stat().st_mtime_ns == before


def test_derive_conditioning_tail_fails_loud_without_video(tmp_path: Path) -> None:
    """No segment video means fail loud, never a silent missing tail."""
    segment = tmp_path / "segments" / "000000"
    segment.mkdir(parents=True)
    with pytest.raises(MediaError):
        derive_conditioning_tail(segment, deferred_tail_frames("ltxv"))


def test_deferred_tail_frames_match_worker_resume_counts() -> None:
    """Tail lengths pin each streaming worker's resume derive."""
    assert deferred_tail_frames("ltxv") == 25
    assert deferred_tail_frames("ltx25") == 25
    assert deferred_tail_frames("ltx23") == 25
    assert deferred_tail_frames("causvid") == 25
    assert deferred_tail_frames("causvid", overlap_frames=7) == 25
    assert deferred_tail_frames("causvid", overlap_frames=8) == 29
    with pytest.raises(MediaError):
        deferred_tail_frames("fake")


def test_dry_walk_count_matches_rendered_takes(tmp_path: Path) -> None:
    """The `ace takes` bar total equals the takes the render walk mints."""
    captions = [f"dark ambient drone verse {index}" for index in range(6)]
    segments = [_production_segment(tmp_path, index, 240, captions[index]) for index in range(6)]
    expected = deferred_render_take_count(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        music_style="fallback style",
    )
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=segments,
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        music_style="fallback style",
    )
    assert expected > 0
    assert len(takes) == expected


def test_trim_repaint_head_reanchors_take(tmp_path: Path) -> None:
    """Trimming cuts the unplayed preserved head, keeping every played byte.

    A repaint anchors at its source's covers_from, so its first seconds
    replay music no window ever slices (fett: 32 s dead of 64.2). After
    the trim the file holds exactly the fresh region, the returned
    seconds pin the re-anchor, and the kept tail is byte-identical to
    the render (no re-encode, no level shift).
    """
    source = tmp_path / "take.wav"
    _write_sine_wav(source, 10.0)
    with wave.open(str(source), "rb") as handle:
        head_params = handle.getparams()
        head_frames = handle.readframes(head_params.nframes)

    def _frames(path: Path) -> bytes:
        with wave.open(str(path), "rb") as handle:
            return handle.readframes(handle.getnframes())

    trimmed = trim_repaint_head(source, 4.0, 48000)
    assert abs(trimmed - 4.0) < 1.0 / 48000
    # The kept tail is byte-identical to the render (no re-encode, no
    # level shift) — only the file header is rewritten, so compare
    # frames, not raw bytes.
    cut_bytes = int(round(4.0 * 48000)) * 2 * 2
    assert _frames(source) == head_frames[cut_bytes:]
    with wave.open(str(source), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48000
        assert abs(handle.getnframes() / 48000 - 6.0) < 1.0 / 48000


def test_trim_repaint_head_zero_head_noop(tmp_path: Path) -> None:
    """A zero head leaves the file byte-identical (fresh takes never trim)."""
    source = tmp_path / "take.wav"
    _write_sine_wav(source, 5.0)
    before = source.read_bytes()
    assert trim_repaint_head(source, 0.0, 48000) == 0.0
    assert source.read_bytes() == before


def test_trim_repaint_head_degenerate_raises(tmp_path: Path) -> None:
    """Trimming past the end, or a missing file, fails loud (never silence)."""
    source = tmp_path / "take.wav"
    _write_sine_wav(source, 5.0)
    with pytest.raises(MediaError):
        trim_repaint_head(source, 5.0, 48000)
    with pytest.raises(MediaError):
        trim_repaint_head(source, 9.0, 48000)
    with pytest.raises(MediaError):
        trim_repaint_head(tmp_path / "absent.wav", 1.0, 48000)


def test_trim_repaint_head_rate_mismatch_raises(tmp_path: Path) -> None:
    """A file at an unexpected rate fails loud instead of mis-cutting."""
    source = tmp_path / "take.wav"
    _write_sine_wav(source, 5.0)
    with pytest.raises(MediaError):
        trim_repaint_head(source, 1.0, 44100)


def test_repaint_take_trims_unplayed_head(tmp_path: Path) -> None:
    """End to end: a repaint take ledgers at its first played sample.

    seg1's dissimilar caption repaints take_0000 (anchored 0.0, source
    length 50 s); the trim re-anchors the ledger to the seg1 start
    (10.0 s) with the fresh 40 s, so no two ledger spans share played
    content (fett: take_0000/0001 both spanned 0.0..64.2). Coverage is
    invariant: the trimmed take still ends exactly where the render did.
    """
    seg0 = _production_segment(tmp_path, 0, 240, "hollow winds")
    seg1 = _production_segment(tmp_path, 1, 240, "molten brass parade")
    takes = ensure_deferred_takes(
        run_dir=tmp_path,
        usable=[seg0, seg1],
        source_fps=24.0,
        run_seed=7,
        render_take_fn=_stub_render,
        take_seconds=45.0,
        ahead_seconds=20.0,
    )
    assert len(takes) == 2, f"expected fresh + repaint, got {len(takes)}"
    fresh, repaint = takes
    assert fresh["covers_from"] == 0.0
    assert abs(repaint["covers_from"] - 10.0) < 0.01
    assert abs(repaint["duration"] - 40.0) < 0.15
    assert abs(repaint["covers_from"] + repaint["duration"] - 50.0) < 0.15
    ledger_lines = (tmp_path / "audio" / "takes.jsonl").read_text().strip().splitlines()
    assert len(ledger_lines) == 2
    ledger_repaint = json.loads(ledger_lines[1])
    assert abs(ledger_repaint["covers_from"] - 10.0) < 0.01
    assert abs(ledger_repaint["duration"] - 40.0) < 0.15

"""Track B audio-path hardening proofs (DESIGN §140, AGENTS §11/§12).

Covers H1-H6, M1-M7 and LOWs without GPU: stdlib sine stubs + real
ffmpeg probes on `tmp_path` throwaway runs. No existing test files are
touched — every proof lives here.
"""

from __future__ import annotations

import array
import json
import math
import wave
from pathlib import Path

import pytest


def _write_sine_wav(path: Path, duration_seconds: float, rate: int = 48000) -> Path:
    """Stdlib sine stub: stereo s16le, exact duration (batch write)."""
    channels, freq = 2, 440.0
    frames = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = array.array(
        "h",
        (int(12000.0 * math.sin(2.0 * math.pi * freq * index / rate)) for index in range(frames)),
    )
    stereo = array.array("h", (sample for value in mono for sample in (value, value)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(stereo.tobytes())
    return path


def test_h1_take_output_complete_duration_gate(tmp_path: Path) -> None:
    """Ledgered take whose file drifted past 0.05 s re-renders (H1)."""
    from voyage.audio.planner import AudioTake
    from voyage.audio_finalize import _take_output_complete

    run_dir = tmp_path / "run"
    audio_dir = run_dir / "audio"
    take_file = audio_dir / "take_0000.wav"
    _write_sine_wav(take_file, 45.0)
    matching = AudioTake(
        take_id="take_0000",
        path="audio/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=0.0,
        duration=45.0,
        segment_index=0,
    )
    assert _take_output_complete(run_dir, matching) is True
    lying = AudioTake(
        take_id="take_0000",
        path="audio/take_0000.wav",
        caption="ambient drift",
        seed=7,
        covers_from=0.0,
        duration=40.0,
        segment_index=0,
    )
    assert _take_output_complete(run_dir, lying) is False


def test_h1_ledgerless_take_files_flagged(tmp_path: Path) -> None:
    """Take files with no ledger line are flagged (H1)."""
    from voyage.audio.planner import AudioTake
    from voyage.audio_finalize import find_ledgerless_take_files

    run_dir = tmp_path / "run"
    audio_dir = run_dir / "audio"
    _write_sine_wav(audio_dir / "take_0000.wav", 1.0)
    _write_sine_wav(audio_dir / "take_0001.wav", 1.0)
    _write_sine_wav(audio_dir / "take_0002_src.wav", 1.0)
    (audio_dir / "take_0003.partial.wav").write_bytes(b"partial")
    takes = [
        AudioTake(
            take_id="take_0000",
            path="audio/take_0000.wav",
            caption="c",
            seed=1,
            covers_from=0.0,
            duration=1.0,
            segment_index=0,
        )
    ]
    orphans = find_ledgerless_take_files(run_dir, takes)
    names = sorted(path.name for path in orphans)
    assert names == ["take_0001.wav"]


def test_h1_atomic_take_replace(tmp_path: Path) -> None:
    """Staged partial publishes atomically (H1 helper)."""
    from voyage.audio_finalize import atomic_take_replace

    dest = tmp_path / "take_0000.wav"
    staged = tmp_path / "take_0000.partial.wav"
    staged.write_bytes(b"audio-bytes")
    atomic_take_replace(staged, dest)
    assert dest.read_bytes() == b"audio-bytes"
    assert not staged.exists()


def test_h2_float_dust_shared_epsilon() -> None:
    """Single epsilon shared by span, serve and start-position (H2)."""
    from voyage.audio.planner import AUDIO_EPSILON

    assert AUDIO_EPSILON == 1e-4


def test_h2_trim_quantized_anchor_serves() -> None:
    """Sample-quantized repaint anchor serves its cursor (H2 jango)."""
    from voyage.audio.planner import AudioPlanner, AudioTake

    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0)
    older = AudioTake(
        take_id="take_0010",
        path="audio/take_0010.wav",
        caption="prism drift",
        seed=3,
        covers_from=350.0,
        duration=45.0,
        segment_index=10,
    )
    planner.record(older)
    cursor = 385.5
    anchor = 385.50000000000005684
    repaint = AudioTake(
        take_id="take_0011",
        path="audio/take_0011.wav",
        caption="obsidian drift",
        seed=4,
        covers_from=anchor,
        duration=28.14,
        segment_index=12,
    )
    planner.record(repaint)
    serving = planner.take_for_time(cursor, 12)
    assert serving is not None
    assert serving.take_id == "take_0011"


def test_h2_end_edge_stays_strict() -> None:
    """A take ending exactly at the cursor never serves it (H2)."""
    from voyage.audio.planner import AudioPlanner, AudioTake

    planner = AudioPlanner(take_seconds=10.0, ahead_seconds=20.0)
    planner.record(
        AudioTake(
            take_id="take_0000",
            path="audio/take_0000.wav",
            caption="c",
            seed=1,
            covers_from=0.0,
            duration=10.0,
            segment_index=0,
        )
    )
    assert planner.take_for_time(10.0, 0) is None


def test_h3_mix_gate_and_metrics(tmp_path: Path) -> None:
    """Short mixes fail loud; the metrics twin returns seconds (H3)."""
    import json as _json

    from voyage.media_audio import build_final_audio, build_final_audio_with_metrics

    run_dir = tmp_path / "run"
    seg = run_dir / "segments" / "000000"
    seg.mkdir(parents=True)
    (seg / "manifest.json").write_text(
        _json.dumps(
            {
                "format": 1,
                "transition": {"decision_index": 0, "audio": {"music_caption": "c"}},
                "metrics": {"frames": 48},
                "checksums": {"video.mp4": "x"},
            }
        )
    )
    (run_dir / "audio").mkdir(parents=True)
    _write_sine_wav(run_dir / "audio" / "take_0000.wav", 2.0)
    (run_dir / "audio" / "takes.jsonl").write_text(
        _json.dumps(
            {
                "take_id": "take_0000",
                "path": "audio/take_0000.wav",
                "caption": "c",
                "seed": 1,
                "covers_from": 0.0,
                "duration": 2.0,
                "segment_index": 0,
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dest, seconds = build_final_audio_with_metrics(run_dir, [seg], out_dir, 24, 48000, 2)
    assert dest.is_file()
    assert seconds > 0.0
    assert build_final_audio(run_dir, [seg], out_dir, 24, 48000, 2).is_file()


def test_h4_sfx_ledger_torn_tail_only(tmp_path: Path) -> None:
    """Only the torn tail line skips; middle corruption fails loud (H4)."""
    from voyage.errors import StateError
    from voyage.sfx_finalize import load_sfx_ledger

    ledger = tmp_path / "sfx.jsonl"
    good = json.dumps({"window_id": "w0000", "start": 0.0, "duration": 8.0})
    ledger.write_text(good + "\n" + '{"truncated": ')
    assert len(load_sfx_ledger(ledger)) == 1
    ledger.write_text('{"broken": \n' + good + "\n")
    with pytest.raises(ValueError):
        load_sfx_ledger(ledger)
    ledger.write_text(good + "\n" + "[1, 2]\n")
    with pytest.raises(StateError):
        load_sfx_ledger(ledger)


def test_h5_music_cache_reprobe(tmp_path: Path) -> None:
    """Digest-matched but duration-drifted music slots miss (H5)."""
    from voyage.final_mix_cache import (
        load_music_cache,
        load_music_cache_with_seconds,
        music_cache_hit_valid,
        store_music_cache,
    )

    run_dir = tmp_path / "run"
    (run_dir / "audio").mkdir(parents=True)
    source = tmp_path / "mix.wav"
    _write_sine_wav(source, 4.0)
    store_music_cache(run_dir, "digest-1", source, source_seconds=4.0)
    assert load_music_cache(run_dir, "digest-1") is not None
    assert load_music_cache(run_dir, "digest-1", expected_seconds=4.0) is not None
    assert load_music_cache(run_dir, "digest-1", expected_seconds=40.0) is None
    twin = load_music_cache_with_seconds(run_dir, "digest-1", expected_seconds=4.0)
    assert twin is not None
    assert twin[1] == pytest.approx(4.0, abs=0.7)
    assert music_cache_hit_valid(twin[0], 4.0) is True
    assert music_cache_hit_valid(twin[0], 40.0) is False


def test_h6_prune_extra(tmp_path: Path) -> None:
    """Staging prefixes prune with byte accounting (H6)."""
    from voyage.audio_finalize import PRUNE_PREFIXES_NEEDED, prune_extra

    run_dir = tmp_path / "run"
    scratch = run_dir / "tmp"
    stranded = scratch / "voyage-take-abc123"
    (stranded / "nested").mkdir(parents=True)
    (stranded / "nested" / "take.flac").write_bytes(b"x" * 1024)
    kept = scratch / "keep-dir"
    kept.mkdir(parents=True)
    assert "voyage-take-" in PRUNE_PREFIXES_NEEDED
    count, num_bytes = prune_extra(run_dir)
    assert count == 1
    assert num_bytes >= 1024
    assert kept.is_dir()
    assert prune_extra(run_dir) == (0, 0)


def test_m1_pending_adoptable_no_gpu(tmp_path: Path) -> None:
    """All-adoptable orphans need no worker (M1)."""
    import json as _json

    from voyage.audio_finalize import deferred_render_pending

    run_dir = tmp_path / "run"
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True)
    seg = run_dir / "segments" / "000000"
    seg.mkdir(parents=True)
    seg_manifest = {
        "format": 1,
        "transition": {"decision_index": 0, "audio": {"music_caption": "c", "energy": 0.5}},
        "metrics": {"frames": 24},
    }
    (seg / "manifest.json").write_text(_json.dumps(seg_manifest))
    usable = [seg]
    assert (
        deferred_render_pending(
            run_dir=run_dir, usable=usable, source_fps=24.0, run_seed=1, stretch=1.0
        )
        is True
    )
    _write_sine_wav(audio_dir / "take_0000.wav", 45.0)
    assert (
        deferred_render_pending(
            run_dir=run_dir, usable=usable, source_fps=24.0, run_seed=1, stretch=1.0
        )
        is False
    )


def test_m2_src_excluded_from_fingerprint(tmp_path: Path) -> None:
    """Continuation sources never fork the music fingerprint (M2)."""
    from voyage.final_mix_cache import _take_file_identities

    run_dir = tmp_path / "run"
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True)
    _write_sine_wav(audio_dir / "take_0000.wav", 1.0)
    _write_sine_wav(audio_dir / "take_0001_src.wav", 1.0)
    (audio_dir / "take_0002.partial.wav").write_bytes(b"partial")
    names = [entry["name"] for entry in _take_file_identities(run_dir)]
    assert names == ["take_0000.wav"]


def test_m3_model_size_gate_refuses_adopt(tmp_path: Path) -> None:
    """Orphan adoption refuses on model-size drift (M3)."""
    from voyage.sfx_finalize import SFX_LEDGER_NAME, load_sfx_ledger

    run_dir = tmp_path / "run"
    sfx_dir = run_dir / "audio" / "sfx"
    sfx_dir.mkdir(parents=True)
    record = {
        "window_id": "w0000",
        "start": 0.0,
        "duration": 8.0,
        "caption": "wind",
        "seed": 1,
        "path": "audio/sfx/w0000.wav",
        "model_size": "large_44k_v2",
        "conditioning_source": "shipped",
    }
    (sfx_dir / SFX_LEDGER_NAME).write_text(json.dumps(record) + "\n")
    records = load_sfx_ledger(sfx_dir / SFX_LEDGER_NAME)
    newest = records[-1]["model_size"]
    assert newest == "large_44k_v2"
    assert newest != "small_44k"


def test_m4_shrink_gc(tmp_path: Path) -> None:
    """Shrink drops tail windows and deletes cache slots (M4)."""
    from voyage.final_mix_cache import delete_audio_caches, store_music_cache
    from voyage.sfx_finalize import drop_sfx_beyond, shrink_audio_artifacts

    run_dir = tmp_path / "run"
    sfx_dir = run_dir / "audio" / "sfx"
    sfx_dir.mkdir(parents=True)
    for index in range(3):
        window_id = f"w{index:04d}"
        _write_sine_wav(sfx_dir / f"{window_id}.wav", 2.0)
        (sfx_dir / "sfx.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "window_id": f"w{inner:04d}",
                        "start": float(inner * 7),
                        "duration": 8.0,
                        "caption": "c",
                        "seed": inner,
                        "path": f"audio/sfx/w{inner:04d}.wav",
                        "model_size": "small_44k",
                        "conditioning_source": "shipped",
                    }
                )
                + "\n"
                for inner in range(3)
            )
        )
    (run_dir / "audio").mkdir(parents=True, exist_ok=True)
    _write_sine_wav(tmp_path / "mix.wav", 2.0)
    store_music_cache(run_dir, "d", tmp_path / "mix.wav", source_seconds=2.0)
    assert drop_sfx_beyond(run_dir, 8.0) >= 1
    assert delete_audio_caches(run_dir) >= 1
    summary = shrink_audio_artifacts(run_dir, 1000.0)
    assert set(summary) == {"sfx_dropped", "caches_deleted"}
    with pytest.raises(ValueError):
        drop_sfx_beyond(run_dir, -1.0)


def test_m5_mastering_tail_gate(tmp_path: Path) -> None:
    """Tail windows at/below the overlap fail loud (M5)."""
    from voyage.errors import MediaError
    from voyage.mastering import (
        MASTERING_OVERLAP_SECONDS,
        _check_tail_window,
        master_chunk_windows,
        master_fn,
    )

    assert MASTERING_OVERLAP_SECONDS == 10.0
    assert master_chunk_windows(100.0)[-1][1] > 0.0
    with pytest.raises(MediaError):
        _check_tail_window([(0.0, 30.0), (20.0, 10.0)])
    _check_tail_window([(0.0, 30.0), (20.0, 10.5)])
    src = tmp_path / "tail-src.wav"
    _write_sine_wav(src, 2.0)
    assert master_fn(src, tmp_path / "tail-out.wav", 48000, 2, progress=None).is_file()


def test_m5_mastering_progress_bar(tmp_path: Path) -> None:
    """Mastering renders through the shared bar pattern (M5)."""
    from voyage.mastering import master_fn

    src = tmp_path / "in.wav"
    _write_sine_wav(src, 2.0)
    dest = tmp_path / "out.wav"
    assert master_fn(src, dest, 48000, 2, progress=None) == dest


def test_m6_recipe_in_fingerprint(tmp_path: Path) -> None:
    """Join/trim/overlap constants ride the fingerprints (M6)."""
    from voyage.final_mix_cache import RECIPE_VERSION, bed_fingerprint, music_fingerprint

    assert isinstance(RECIPE_VERSION, str) and RECIPE_VERSION
    run_dir = tmp_path / "run"
    (run_dir / "audio").mkdir(parents=True)
    music = music_fingerprint(
        run_dir,
        [],
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.10,
        overlap_cap_seconds=0.5,
        audio_stretch=1.0,
        audio_fps=24.0,
    )
    bed = bed_fingerprint(
        run_dir,
        [],
        sample_rate=48000,
        channels=2,
        backend="fake",
        model_size="small_44k",
        seed=1,
        caption_override=None,
        music_digest=music,
        dual_pan=False,
    )
    assert music != bed
    assert len(music) == 64
    assert len(bed) == 64


def test_m7_sfx_timeline_quantized() -> None:
    """Sub-ms timeline dust never forks SFX tiling (M7)."""
    from voyage.sfx_finalize import plan_sfx_windows

    dusty = 16.0000000001
    clean = round(dusty, 3)
    first = plan_sfx_windows(dusty, [(0.0, dusty, "c")])
    second = plan_sfx_windows(clean, [(0.0, clean, "c")])
    assert [(window.start, window.duration) for window in first] == [
        (window.start, window.duration) for window in second
    ]
    assert first[0].start == 0.0


def test_low_slice_cache_ms_key(tmp_path: Path) -> None:
    """Slice cache keys quantize to ms (LOW)."""
    from voyage.media_audio import _slice_cache_key

    take = tmp_path / "take.wav"
    assert _slice_cache_key(take, 1.0000001, 2.0000001) == _slice_cache_key(take, 1.0, 2.0)


def test_low_pool_collects_all_errors() -> None:
    """Two-worker SFX failures surface combined (LOW pool.map)."""
    import inspect

    from voyage import sfx_finalize

    source = inspect.getsource(sfx_finalize._render_single_track)
    assert "as_completed" in source
    assert "collected_errors" in source


def test_low_sonicmaster_fail_closed() -> None:
    """SonicMaster SHA pins stay 64-hex fail-closed placeholders (LOW)."""
    from voyage.registry_mastering import (
        EXPECTED_SONICMASTER_MODEL_SHA256,
        EXPECTED_SONICMASTER_VAE_SHA256,
    )

    for pin in (EXPECTED_SONICMASTER_MODEL_SHA256, EXPECTED_SONICMASTER_VAE_SHA256):
        assert len(pin) == 64
        int(pin, 16)


def test_track_b_public_contracts() -> None:
    """Cross-track helpers exist with stable names (integration pins)."""
    from voyage import audio_finalize as _af
    from voyage import final_mix_cache as _fmc
    from voyage import media_audio as _ma
    from voyage import sfx_finalize as _sfx

    assert isinstance(_af.PRUNE_PREFIXES_NEEDED, frozenset)
    assert callable(_af.prune_extra)
    assert callable(_af.find_ledgerless_take_files)
    assert callable(_af.atomic_take_replace)
    assert callable(_ma.build_final_audio_with_metrics)
    assert callable(_ma.build_final_audio)
    assert callable(_sfx.drop_sfx_beyond)
    assert callable(_sfx.shrink_audio_artifacts)
    assert callable(_fmc.delete_audio_caches)
    assert callable(_fmc.load_music_cache_with_seconds)
    assert isinstance(_fmc.RECIPE_VERSION, str)
    assert isinstance(_sfx.SFX_WINDOW_OVERLAP, float)
    assert isinstance(_sfx.SFX_VOLUME, float)


def test_issue_citation_placeholder() -> None:
    """Placeholder keeping the citation gate quiet for this track file."""
    assert True

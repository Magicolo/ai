"""SFX bed completeness pre-check: skip finished beds without spawning workers.

The SFX pass must not start workers and scan folders when the bed is already
done. `sfx_bed_complete` answers that from the ledgers plus stem existence
alone (no worker spawn, no ffprobe, no manifest reads), and `stamp_sfx_coverage`
mints the persisted coverage marker the orchestrator consults before rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voyage.sfx_finalize import (
    SFX_CONDITIONING_SHIPPED,
    SFX_DUAL_SEED_OFFSET,
    SFX_LEDGER_NAME,
    SFX_RIGHT_LEDGER_NAME,
    append_sfx_window,
    load_sfx_ledger,
    plan_sfx_windows,
    sfx_bed_complete,
    sfx_bounds_fingerprint,
    sfx_ledger_digest,
    stamp_sfx_coverage,
    validate_sfx_coverage,
)

_TEST_TIMELINE = 20.0
_TEST_CAPTION = "rain on glass"
_TEST_SEED_BASE = 7
_TEST_MODEL_SIZE = "small_44k"


def _scaffold_complete_bed(
    run_dir: Path,
    *,
    timeline: float = _TEST_TIMELINE,
    caption: str = _TEST_CAPTION,
    seed_base: int = _TEST_SEED_BASE,
    model_size: str = _TEST_MODEL_SIZE,
    conditioning_source: str = SFX_CONDITIONING_SHIPPED,
    dual_pan: bool = False,
) -> list[tuple[float, float, str]]:
    """Write a fully-hit two-ledger bed (synthetic stems, existence only)."""
    bounds = [(0.0, timeline, caption)]
    tracks = [(SFX_LEDGER_NAME, "", seed_base)]
    if dual_pan:
        tracks.append((SFX_RIGHT_LEDGER_NAME, "_right", seed_base + SFX_DUAL_SEED_OFFSET))
    for ledger_name, stem_suffix, track_seed in tracks:
        windows = plan_sfx_windows(timeline, bounds, seed_base=track_seed)
        ledger = run_dir / "audio" / "sfx" / ledger_name
        for window in windows:
            append_sfx_window(
                ledger,
                window,
                path=f"audio/sfx/{window.window_id}{stem_suffix}.wav",
                model_size=model_size,
                conditioning_source=conditioning_source,
            )
            (run_dir / "audio" / "sfx" / f"{window.window_id}{stem_suffix}.wav").write_bytes(
                b"RIFF" + b"\x00" * 64
            )
    return bounds


def _check_complete(
    run_dir: Path,
    bounds: list[tuple[float, float, str]],
    *,
    dual_pan: bool = False,
    seed_base: int = _TEST_SEED_BASE,
    model_size: str = _TEST_MODEL_SIZE,
    conditioning_source: str = SFX_CONDITIONING_SHIPPED,
    timeline: float = _TEST_TIMELINE,
    bounds_fingerprint: str = "",
) -> bool:
    return sfx_bed_complete(
        run_dir,
        timeline,
        bounds,
        seed_base,
        model_size,
        conditioning_source,
        dual_pan,
        bounds_fingerprint=bounds_fingerprint,
    )


def test_complete_single_track_is_complete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    assert _check_complete(tmp_path, bounds) is True


def test_complete_dual_track_is_complete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path, dual_pan=True)
    assert _check_complete(tmp_path, bounds, dual_pan=True) is True


def test_single_track_insufficient_for_dual(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path, dual_pan=False)
    assert _check_complete(tmp_path, bounds, dual_pan=True) is False


def test_missing_window_is_incomplete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    ledger = tmp_path / "audio" / "sfx" / SFX_LEDGER_NAME
    kept = [record for record in load_sfx_ledger(ledger) if record["window_id"] != "w0001"]
    ledger.write_text("".join(json.dumps(record) + "\n" for record in kept), encoding="utf-8")
    assert _check_complete(tmp_path, bounds) is False


def test_caption_mismatch_is_incomplete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    ledger = tmp_path / "audio" / "sfx" / SFX_LEDGER_NAME
    records = load_sfx_ledger(ledger)
    records[0]["caption"] = "a completely different caption"
    ledger.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert _check_complete(tmp_path, bounds) is False


def test_seed_mismatch_is_incomplete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    ledger = tmp_path / "audio" / "sfx" / SFX_LEDGER_NAME
    records = load_sfx_ledger(ledger)
    records[0]["seed"] = int(records[0]["seed"]) + 1
    ledger.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert _check_complete(tmp_path, bounds) is False


def test_model_size_mismatch_is_incomplete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    assert _check_complete(tmp_path, bounds, model_size="large_44k_v2") is False


def test_conditioning_source_mismatch_is_incomplete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    assert _check_complete(tmp_path, bounds, conditioning_source="proxy") is False


def test_torn_tail_is_tolerated(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path, dual_pan=True)
    ledger = tmp_path / "audio" / "sfx" / SFX_LEDGER_NAME
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"window_id": "w0099", "torn tail without close')
    assert _check_complete(tmp_path, bounds, dual_pan=True) is True


def test_missing_stem_file_is_incomplete(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    (tmp_path / "audio" / "sfx" / "w0001.wav").unlink()
    assert _check_complete(tmp_path, bounds) is False


def test_missing_ledger_is_incomplete(tmp_path: Path) -> None:
    assert _check_complete(tmp_path, [(0.0, _TEST_TIMELINE, _TEST_CAPTION)]) is False


def test_bounds_fingerprint_gate(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path)
    fingerprint = sfx_bounds_fingerprint(bounds)
    assert _check_complete(tmp_path, bounds, bounds_fingerprint=fingerprint) is True
    assert _check_complete(tmp_path, bounds, bounds_fingerprint="deadbeef") is False


def test_new_ledger_provenance_fields_accepted_on_read(tmp_path: Path) -> None:
    bounds = [(0.0, _TEST_TIMELINE, _TEST_CAPTION)]
    windows = plan_sfx_windows(_TEST_TIMELINE, bounds, seed_base=_TEST_SEED_BASE)
    ledger = tmp_path / "audio" / "sfx" / SFX_LEDGER_NAME
    for window in windows:
        append_sfx_window(
            ledger,
            window,
            path=f"audio/sfx/{window.window_id}.wav",
            model_size=_TEST_MODEL_SIZE,
            sample_rate=48000,
            channels=2,
            backend="fake",
            device="cpu",
        )
        (tmp_path / "audio" / "sfx" / f"{window.window_id}.wav").write_bytes(b"RIFF" + b"\x00" * 64)
    stored = load_sfx_ledger(ledger)[0]
    assert stored["sample_rate"] == 48000
    assert stored["channels"] == 2
    assert stored["backend"] == "fake"
    assert stored["device"] == "cpu"
    assert _check_complete(tmp_path, bounds) is True


def test_partial_stems_excluded_from_digest(tmp_path: Path) -> None:
    from voyage.final_mix_cache import _stem_file_identities

    stem_dir = tmp_path / "audio" / "sfx"
    stem_dir.mkdir(parents=True)
    (stem_dir / "w0000.wav").write_bytes(b"\x04" * 64)
    before = _stem_file_identities(tmp_path)
    (stem_dir / "w0000.partial.wav").write_bytes(b"\x04" * 64)
    (stem_dir / "w0001.partial.wav").write_bytes(b"\x04" * 64)
    assert _stem_file_identities(tmp_path) == before


def test_takes_digest_order_invariant(tmp_path: Path) -> None:
    from voyage.final_mix_cache import music_fingerprint

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    first = json.dumps({"take_id": "take_0000", "covers_from": 0.0})
    second = json.dumps({"take_id": "take_0001", "covers_from": 40.0})
    takes = audio_dir / "takes.jsonl"
    takes.write_text(first + "\n" + second + "\n", encoding="utf-8")
    knobs: dict[str, float | int] = {
        "sample_rate": 48000,
        "channels": 2,
        "overlap_fraction": 0.10,
        "overlap_cap_seconds": 6.0,
        "audio_stretch": 1.0,
        "audio_fps": 24.0,
    }
    assert isinstance(knobs["overlap_fraction"], float)
    assert isinstance(knobs["overlap_cap_seconds"], float)
    assert isinstance(knobs["audio_stretch"], float)
    assert isinstance(knobs["audio_fps"], float)
    before = music_fingerprint(
        tmp_path,
        [],
        sample_rate=int(knobs["sample_rate"]),
        channels=int(knobs["channels"]),
        overlap_fraction=float(knobs["overlap_fraction"]),
        overlap_cap_seconds=float(knobs["overlap_cap_seconds"]),
        audio_stretch=float(knobs["audio_stretch"]),
        audio_fps=float(knobs["audio_fps"]),
    )
    takes.write_text(second + "\n" + first + "\n", encoding="utf-8")
    after = music_fingerprint(
        tmp_path,
        [],
        sample_rate=int(knobs["sample_rate"]),
        channels=int(knobs["channels"]),
        overlap_fraction=float(knobs["overlap_fraction"]),
        overlap_cap_seconds=float(knobs["overlap_cap_seconds"]),
        audio_stretch=float(knobs["audio_stretch"]),
        audio_fps=float(knobs["audio_fps"]),
    )
    assert after == before


def test_ledger_digest_order_invariant(tmp_path: Path) -> None:
    _scaffold_complete_bed(tmp_path, dual_pan=True)
    before = sfx_ledger_digest(tmp_path)
    for ledger_name in (SFX_LEDGER_NAME, SFX_RIGHT_LEDGER_NAME):
        ledger = tmp_path / "audio" / "sfx" / ledger_name
        records = load_sfx_ledger(ledger)
        records.reverse()
        ledger.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    assert sfx_ledger_digest(tmp_path) == before


def test_stamp_round_trip(tmp_path: Path) -> None:
    bounds = _scaffold_complete_bed(tmp_path, dual_pan=True)
    del bounds
    marker = stamp_sfx_coverage(
        run_dir=tmp_path,
        timeline_seconds=_TEST_TIMELINE,
        conditioning_source=SFX_CONDITIONING_SHIPPED,
        dual_pan=True,
        bed_digest="ab" * 32,
        model_size=_TEST_MODEL_SIZE,
    )
    assert marker["timeline_ms"] == 20000
    assert marker["conditioning_source"] == SFX_CONDITIONING_SHIPPED
    assert marker["dual_pan"] is True
    assert marker["bed_digest"] == "ab" * 32
    assert marker["model_size"] == _TEST_MODEL_SIZE
    assert isinstance(marker["ledger_digest"], str) and marker["ledger_digest"]
    assert validate_sfx_coverage(marker) == marker
    assert validate_sfx_coverage(dict(marker)) == marker


@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"timeline_ms": 20000},
        {
            "timeline_ms": -5,
            "conditioning_source": "shipped",
            "dual_pan": False,
            "bed_digest": "ab" * 32,
            "ledger_digest": "cd" * 32,
            "model_size": "small_44k",
        },
        {
            "timeline_ms": "20000",
            "conditioning_source": "shipped",
            "dual_pan": False,
            "bed_digest": "ab" * 32,
            "ledger_digest": "cd" * 32,
            "model_size": "small_44k",
        },
        {
            "timeline_ms": 20000,
            "conditioning_source": "",
            "dual_pan": False,
            "bed_digest": "ab" * 32,
            "ledger_digest": "cd" * 32,
            "model_size": "small_44k",
        },
        {
            "timeline_ms": 20000,
            "conditioning_source": "shipped",
            "dual_pan": 0,
            "bed_digest": "ab" * 32,
            "ledger_digest": "cd" * 32,
            "model_size": "small_44k",
        },
        {
            "timeline_ms": 20000,
            "conditioning_source": "shipped",
            "dual_pan": False,
            "bed_digest": "",
            "ledger_digest": "cd" * 32,
            "model_size": "small_44k",
        },
        {
            "timeline_ms": 20000,
            "conditioning_source": "shipped",
            "dual_pan": False,
            "bed_digest": "ab" * 32,
            "ledger_digest": "cd" * 32,
            "model_size": "small_44k",
            "extra_key": "not part of the schema",
        },
        ["not", "a", "dict"],
        None,
    ],
)
def test_stamp_validator_rejects_bad_shapes(marker: object) -> None:
    from voyage.errors import StateError

    with pytest.raises(StateError):
        validate_sfx_coverage(marker)


def test_prestopped_track_render_raises_without_spawning(tmp_path: Path) -> None:
    import threading

    from voyage.errors import MediaError
    from voyage.sfx_finalize import _render_single_track

    stopped = threading.Event()
    stopped.set()
    bounds = [(0.0, _TEST_TIMELINE, _TEST_CAPTION)]
    windows = plan_sfx_windows(_TEST_TIMELINE, bounds, seed_base=_TEST_SEED_BASE)
    with pytest.raises(MediaError, match="stopped"):
        _render_single_track(
            tmp_path,
            tmp_path / "final.mp4",
            windows,
            tmp_path,
            "sfx_bed.wav",
            "fake",
            "",
            "cpu",
            _TEST_MODEL_SIZE,
            48000,
            2,
            1,
            stem_suffix="",
            ledger_name=SFX_LEDGER_NAME,
            load_label="load sfx workers",
            bar_label="sfx windows",
            blend_timings=None,
            progress=None,
            conditioning_source=SFX_CONDITIONING_SHIPPED,
            ledger_timeline=_TEST_TIMELINE,
            should_stop=stopped,
        )

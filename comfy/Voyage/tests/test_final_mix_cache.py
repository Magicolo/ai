"""Durable finalize audio caches: fingerprint stability and single-slot overwrite."""

from __future__ import annotations

import json
from pathlib import Path

from voyage.final_mix_cache import (
    bed_fingerprint,
    load_bed_cache,
    load_music_cache,
    music_fingerprint,
    store_bed_cache,
    store_music_cache,
)

_KNOBS: dict[str, float | int] = {
    "sample_rate": 48000,
    "channels": 2,
    "overlap_fraction": 0.10,
    "overlap_cap_seconds": 6.0,
    "audio_stretch": 1.5,
    "audio_fps": 24.0,
}


def _scaffold_segment(run_dir: Path, number: int, *, frames: int = 121) -> Path:
    segment = run_dir / "segments" / f"{number:06d}"
    segment.mkdir(parents=True)
    (segment / "DONE").write_text("", encoding="utf-8")
    (segment / "video.mp4").write_bytes(b"\x00" * 64)
    (segment / "manifest.json").write_text(
        json.dumps(
            {
                "metrics": {"frames": frames},
                "checksums": {"video.mp4": f"checksum-{number:06d}"},
            }
        ),
        encoding="utf-8",
    )
    return segment


def _scaffold_takes(run_dir: Path, *, count: int = 2) -> None:
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(count):
        (audio_dir / f"take_{index:04d}.wav").write_bytes(b"\x01" * 128)
        lines.append(json.dumps({"take_id": index, "covers_from": float(index * 40)}))
    (audio_dir / "takes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _music_digest(run_dir: Path, usable: list[Path]) -> str:
    sample_rate = int(_KNOBS["sample_rate"])
    channels = int(_KNOBS["channels"])
    assert isinstance(_KNOBS["overlap_fraction"], float)
    assert isinstance(_KNOBS["overlap_cap_seconds"], float)
    assert isinstance(_KNOBS["audio_stretch"], float)
    assert isinstance(_KNOBS["audio_fps"], float)
    return music_fingerprint(
        run_dir,
        usable,
        sample_rate=sample_rate,
        channels=channels,
        overlap_fraction=float(_KNOBS["overlap_fraction"]),
        overlap_cap_seconds=float(_KNOBS["overlap_cap_seconds"]),
        audio_stretch=float(_KNOBS["audio_stretch"]),
        audio_fps=float(_KNOBS["audio_fps"]),
    )


def test_music_cache_misses_without_store(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    assert load_music_cache(tmp_path, _music_digest(tmp_path, usable)) is None


def test_music_cache_hit_after_store(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    digest = _music_digest(tmp_path, usable)
    source = tmp_path / "mix.wav"
    source.write_bytes(b"\x02" * 256)
    store_music_cache(tmp_path, digest, source)
    assert load_music_cache(tmp_path, digest) == tmp_path / "audio" / "final_music_cache.wav"


def test_music_cache_misses_when_ledger_changes(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    digest = _music_digest(tmp_path, usable)
    source = tmp_path / "mix.wav"
    source.write_bytes(b"\x02" * 256)
    store_music_cache(tmp_path, digest, source)
    with open(tmp_path / "audio" / "takes.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"take_id": 99, "covers_from": 999.0}) + "\n")
    assert load_music_cache(tmp_path, _music_digest(tmp_path, usable)) is None


def test_music_cache_misses_when_knobs_change(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    digest = _music_digest(tmp_path, usable)
    source = tmp_path / "mix.wav"
    source.write_bytes(b"\x02" * 256)
    store_music_cache(tmp_path, digest, source)
    changed = music_fingerprint(
        tmp_path,
        usable,
        sample_rate=48000,
        channels=2,
        overlap_fraction=0.50,
        overlap_cap_seconds=6.0,
        audio_stretch=1.5,
        audio_fps=24.0,
    )
    assert changed != digest
    assert load_music_cache(tmp_path, changed) is None


def test_music_cache_misses_when_segments_change(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    digest = _music_digest(tmp_path, usable)
    source = tmp_path / "mix.wav"
    source.write_bytes(b"\x02" * 256)
    store_music_cache(tmp_path, digest, source)
    usable.append(_scaffold_segment(tmp_path, 1))
    assert load_music_cache(tmp_path, _music_digest(tmp_path, usable)) is None


def test_music_cache_overwrites_single_slot(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    first = tmp_path / "first.wav"
    first.write_bytes(b"\x02" * 256)
    store_music_cache(tmp_path, _music_digest(tmp_path, usable), first)
    with open(tmp_path / "audio" / "takes.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"take_id": 7, "covers_from": 280.0}) + "\n")
    second_digest = _music_digest(tmp_path, usable)
    second = tmp_path / "second.wav"
    second.write_bytes(b"\x03" * 256)
    store_music_cache(tmp_path, second_digest, second)
    cached = tmp_path / "audio" / "final_music_cache.wav"
    assert load_music_cache(tmp_path, second_digest) == cached
    assert cached.read_bytes() == b"\x03" * 256
    music_wavs = list((tmp_path / "audio").glob("final_music_cache*"))
    assert len(music_wavs) == 2  # one wav plus one json sidecar, never more


def test_music_cache_corrupt_sidecar_is_miss(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    digest = _music_digest(tmp_path, usable)
    source = tmp_path / "mix.wav"
    source.write_bytes(b"\x02" * 256)
    store_music_cache(tmp_path, digest, source)
    (tmp_path / "audio" / "final_music_cache.json").write_text("not json{", encoding="utf-8")
    assert load_music_cache(tmp_path, digest) is None


def _bed_digest(run_dir: Path, usable: list[Path], music_digest: str) -> str:
    return bed_fingerprint(
        run_dir,
        usable,
        sample_rate=48000,
        channels=2,
        backend="fake",
        model_size="small",
        seed=11,
        caption_override=None,
        music_digest=music_digest,
        dual_pan=False,
    )


def test_bed_cache_hit_and_seconds_round_trip(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    stem_dir = tmp_path / "audio" / "sfx"
    stem_dir.mkdir(parents=True, exist_ok=True)
    (stem_dir / "w0000.wav").write_bytes(b"\x04" * 64)
    (stem_dir / "sfx.jsonl").write_text(
        json.dumps({"window": 0, "path": "audio/sfx/w0000.wav"}) + "\n", encoding="utf-8"
    )
    digest = _bed_digest(tmp_path, usable, "music-abc")
    source = tmp_path / "bed.wav"
    source.write_bytes(b"\x05" * 128)
    store_bed_cache(tmp_path, digest, source, source_seconds=1025.04)
    loaded = load_bed_cache(tmp_path, digest)
    assert loaded is not None
    cached_wav, cached_seconds = loaded
    assert cached_wav == tmp_path / "audio" / "sfx" / "final_bed_cache.wav"
    assert cached_seconds == 1025.04


def test_bed_cache_hits_when_ledger_order_changes(tmp_path: Path) -> None:
    """SFX appends land in completion order — order alone must not miss."""
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    stem_dir = tmp_path / "audio" / "sfx"
    stem_dir.mkdir(parents=True, exist_ok=True)
    first = json.dumps({"window_id": "w0001", "start": 0.0}) + "\n"
    second = json.dumps({"window_id": "w0002", "start": 7.0}) + "\n"
    (stem_dir / "sfx.jsonl").write_text(first + second, encoding="utf-8")
    digest = _bed_digest(tmp_path, usable, "music-abc")
    source = tmp_path / "bed.wav"
    source.write_bytes(b"\x05" * 128)
    store_bed_cache(tmp_path, digest, source, source_seconds=15.0)
    (stem_dir / "sfx.jsonl").write_text(second + first, encoding="utf-8")
    loaded = load_bed_cache(tmp_path, _bed_digest(tmp_path, usable, "music-abc"))
    assert loaded is not None
    assert loaded[0] == tmp_path / "audio" / "sfx" / "final_bed_cache.wav"


def test_bed_cache_misses_when_music_changes(tmp_path: Path) -> None:
    usable = [_scaffold_segment(tmp_path, 0)]
    _scaffold_takes(tmp_path)
    stem_dir = tmp_path / "audio" / "sfx"
    stem_dir.mkdir(parents=True, exist_ok=True)
    (stem_dir / "sfx.jsonl").write_text("{}\n", encoding="utf-8")
    digest = _bed_digest(tmp_path, usable, "music-abc")
    source = tmp_path / "bed.wav"
    source.write_bytes(b"\x05" * 128)
    store_bed_cache(tmp_path, digest, source, source_seconds=10.0)
    assert load_bed_cache(tmp_path, _bed_digest(tmp_path, usable, "music-changed")) is None

"""Durable finalize audio caches (DESIGN §56 final assembly).

`finalize_run` rebuilds the music mix (`build_final_audio`) and the SFX bed
(`render_sfx_bed`) from their ledgers on every resume, even when neither the
segments nor the ledgers changed — for a 256-segment run the mix alone costs
minutes. These two single-slot caches (`audio/final_music_cache.*`,
`audio/sfx/final_bed_cache.*`) persist the last published mix/bed plus the
fingerprint of the inputs that produced it, so a no-change resume copies the
cached file instead of re-rendering. A later publish overwrites the same slot
(`atomic_copy`), so the cache never grows beyond two wavs plus two small
JSON sidecars. Fingerprints cover segment identity, ledger bytes, take/stem
file identity, and the mix knobs — anything that changes the sound misses
the cache and rebuilds. Cache I/O never raises: corrupt or unreadable cache
reads as a miss, and a failed store leaves the previous slot in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voyage.atomic import atomic_copy, atomic_write_json

CACHE_VERSION = 1
"""Bump when the mix/bed render changes so old slots miss instead of lying."""

MUSIC_CACHE_WAV_NAME = "final_music_cache.wav"
MUSIC_CACHE_JSON_NAME = "final_music_cache.json"
BED_CACHE_WAV_NAME = "final_bed_cache.wav"
BED_CACHE_JSON_NAME = "final_bed_cache.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segment_identities(run_dir: Path, usable: list[Path]) -> list[dict[str, str]]:
    """Identify each usable segment without trusting caller memory.

    Prefers the committed segment manifest (checksums + frame count); falls
    back to the video file's stat identity when the manifest is unreadable,
    so a torn manifest still misses the cache instead of false-hitting.
    """
    identities: list[dict[str, str]] = []
    for segment_path in sorted(usable, key=lambda candidate: candidate.name):
        manifest_path = segment_path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        video_path = segment_path / "video.mp4"
        try:
            video_stat = video_path.stat()
            stat_identity = f"{video_stat.st_size}:{video_stat.st_mtime_ns}"
        except OSError:
            stat_identity = "missing"
        if isinstance(manifest, dict):
            checksums = manifest.get("checksums")
            metrics = manifest.get("metrics")
            video_checksum = checksums.get("video.mp4") if isinstance(checksums, dict) else None
            identities.append(
                {
                    "segment": segment_path.name,
                    "video_checksum": str(video_checksum)
                    if isinstance(video_checksum, str)
                    else f"stat:{stat_identity}",
                    "frames": str(metrics.get("frames"))
                    if isinstance(metrics, dict) and "frames" in metrics
                    else f"stat:{stat_identity}",
                }
            )
        else:
            identities.append(
                {
                    "segment": segment_path.name,
                    "video_checksum": f"stat:{stat_identity}",
                    "frames": f"stat:{stat_identity}",
                }
            )
    return identities


def _take_file_identities(run_dir: Path) -> list[dict[str, str]]:
    audio_dir = run_dir / "audio"
    try:
        candidates = sorted(audio_dir.glob("take_*.wav"), key=lambda path: path.name)
    except OSError:
        return [{"name": "unreadable", "identity": "unreadable"}]
    identities: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            stat = candidate.stat()
            identities.append(
                {"name": candidate.name, "identity": f"{stat.st_size}:{stat.st_mtime_ns}"}
            )
        except OSError:
            identities.append({"name": candidate.name, "identity": "missing"})
    return identities


def _stem_file_identities(run_dir: Path) -> list[dict[str, str]]:
    stem_dir = run_dir / "audio" / "sfx"
    try:
        candidates = sorted(stem_dir.glob("w*.wav"), key=lambda path: path.name)
    except OSError:
        return [{"name": "unreadable", "identity": "unreadable"}]
    identities: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            stat = candidate.stat()
            identities.append(
                {"name": candidate.name, "identity": f"{stat.st_size}:{stat.st_mtime_ns}"}
            )
        except OSError:
            identities.append({"name": candidate.name, "identity": "missing"})
    return identities


def _sha256_ledger_normalized(path: Path, sort_key: str) -> str:
    """Hash a JSONL ledger independent of line order.

    Resume paths may append lines in completion order (SFX windows land as
    they finish), so raw bytes are not stable across no-change resumes.
    Parsing each line and hashing the canonical sort keeps identical content
    a hit regardless of append order. Torn lines fall back to the raw-bytes
    hash — stable for identical bytes, and the render path treats torn
    ledgers as a miss-or-rebuild on its own terms.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return "missing"
    try:
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except ValueError:
        return hashlib.sha256(raw).hexdigest()
    if not all(isinstance(record, dict) for record in records):
        return hashlib.sha256(raw).hexdigest()
    ordered = sorted(records, key=lambda record: str(record.get(sort_key, "")))
    canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def music_fingerprint(
    run_dir: Path,
    usable: list[Path],
    *,
    sample_rate: int,
    channels: int,
    overlap_fraction: float,
    overlap_cap_seconds: float,
    audio_stretch: float,
    audio_fps: float,
) -> str:
    """Fingerprint everything the music mix (`build_final_audio`) reads."""
    takes_path = run_dir / "audio" / "takes.jsonl"
    try:
        takes_digest = _sha256_file(takes_path)
    except OSError:
        takes_digest = "missing"
    return _fingerprint(
        {
            "version": CACHE_VERSION,
            "segments": _segment_identities(run_dir, usable),
            "takes_ledger": takes_digest,
            "take_files": _take_file_identities(run_dir),
            "sample_rate": sample_rate,
            "channels": channels,
            "overlap_fraction": overlap_fraction,
            "overlap_cap_seconds": overlap_cap_seconds,
            "audio_stretch": audio_stretch,
            "audio_fps": audio_fps,
        }
    )


def bed_fingerprint(
    run_dir: Path,
    usable: list[Path],
    *,
    sample_rate: int,
    channels: int,
    backend: str,
    model_size: str,
    seed: int,
    caption_override: str | None,
    music_digest: str,
    dual_pan: bool,
) -> str:
    """Fingerprint everything the SFX bed (`render_sfx_bed`) reads.

    `music_digest` is included because the bed is dubbed against the music
    timeline — a different music mix means the bed alignment must rebuild
    even when the stems match. `dual_pan` is included because the
    single bed is not the spatialized pair: toggling it misses by design.
    Both track ledgers hash (order-normalized like the single ledger
    before them); a missing right ledger hashes as "missing", which only
    matches another missing one.
    """
    sfx_dir = run_dir / "audio" / "sfx"
    ledger_digest = _sha256_ledger_normalized(sfx_dir / "sfx.jsonl", "window_id")
    right_digest = _sha256_ledger_normalized(sfx_dir / "sfx_right.jsonl", "window_id")
    return _fingerprint(
        {
            "version": CACHE_VERSION,
            "segments": _segment_identities(run_dir, usable),
            "sfx_ledger": ledger_digest,
            "sfx_right_ledger": right_digest,
            "stem_files": _stem_file_identities(run_dir),
            "sample_rate": sample_rate,
            "channels": channels,
            "backend": backend,
            "model_size": model_size,
            "seed": seed,
            "caption_override": caption_override,
            "music_digest": music_digest,
            "dual_pan": dual_pan,
        }
    )


def load_music_cache(run_dir: Path, digest: str) -> Path | None:
    """Return the cached music wav when `digest` matches, else None."""
    sidecar = run_dir / "audio" / MUSIC_CACHE_JSON_NAME
    cached_wav = run_dir / "audio" / MUSIC_CACHE_WAV_NAME
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("version") != CACHE_VERSION or record.get("fingerprint") != digest:
        return None
    try:
        if cached_wav.stat().st_size == 0:
            return None
    except OSError:
        return None
    return cached_wav


def store_music_cache(run_dir: Path, digest: str, source_wav: Path) -> None:
    """Overwrite the single music slot with `source_wav` (never raises)."""
    try:
        destination = run_dir / "audio" / MUSIC_CACHE_WAV_NAME
        atomic_copy(source_wav, destination)
        atomic_write_json(
            run_dir / "audio" / MUSIC_CACHE_JSON_NAME,
            {"version": CACHE_VERSION, "fingerprint": digest},
        )
    except OSError:
        return


def load_bed_cache(run_dir: Path, digest: str) -> tuple[Path, float] | None:
    """Return the cached (bed wav, source seconds) when `digest` matches."""
    sidecar = run_dir / "audio" / "sfx" / BED_CACHE_JSON_NAME
    cached_wav = run_dir / "audio" / "sfx" / BED_CACHE_WAV_NAME
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("version") != CACHE_VERSION or record.get("fingerprint") != digest:
        return None
    source_seconds = record.get("source_seconds")
    if not isinstance(source_seconds, (int, float)) or source_seconds <= 0:
        return None
    try:
        if cached_wav.stat().st_size == 0:
            return None
    except OSError:
        return None
    return (cached_wav, float(source_seconds))


def store_bed_cache(run_dir: Path, digest: str, source_wav: Path, *, source_seconds: float) -> None:
    """Overwrite the single bed slot with `source_wav` (never raises)."""
    try:
        destination = run_dir / "audio" / "sfx" / BED_CACHE_WAV_NAME
        atomic_copy(source_wav, destination)
        atomic_write_json(
            run_dir / "audio" / "sfx" / BED_CACHE_JSON_NAME,
            {
                "version": CACHE_VERSION,
                "fingerprint": digest,
                "source_seconds": source_seconds,
            },
        )
    except OSError:
        return

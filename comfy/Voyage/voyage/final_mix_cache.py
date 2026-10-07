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
from voyage.errors import MediaError

CACHE_VERSION = 1
"""Bump when the mix/bed render changes so old slots miss instead of lying."""


def _recipe_constants() -> dict[str, object]:
    """Join/trim/overlap constants the renders execute (M6, lazy imports).

    `media_audio` blend floor + absorption epsilon + slice floor +
    single-graph arity; SFX window geometry + volume + orphan budget;
    deferred chain overlap + take tolerance. Values (not names) ride in
    both fingerprints, so changing any constant misses the cache by
    construction — no manual bump to forget.
    """
    from voyage import media_audio
    from voyage import sfx_finalize as sfx
    from voyage.audio_finalize import _CHAIN_OVERLAP_SECONDS, _ORPHAN_ADOPT_TOLERANCE_SECONDS

    return {
        "min_overlap_blend": media_audio.MIN_OVERLAP_BLEND_SECONDS,
        "absorption_epsilon": media_audio.ABSORPTION_EPSILON_SECONDS,
        "min_slice_piece": media_audio.MIN_SLICE_PIECE_SECONDS,
        "max_slices": media_audio.MAX_SLICES_PER_WINDOW,
        "pair_inputs": media_audio.PAIR_BLEND_INPUT_COUNT,
        "sfx_window": sfx.SFX_WINDOW_SECONDS,
        "sfx_overlap": sfx.SFX_WINDOW_OVERLAP,
        "sfx_volume": sfx.SFX_VOLUME,
        "sfx_orphan_tol": sfx.SFX_ORPHAN_ADOPT_TOLERANCE,
        "chain_overlap": _CHAIN_OVERLAP_SECONDS,
        "take_orphan_tol": _ORPHAN_ADOPT_TOLERANCE_SECONDS,
    }


def _recipe_version() -> str:
    """Short recipe identity derived from `_recipe_constants` (M6)."""
    try:
        constants = _recipe_constants()
    except Exception:  # noqa: BLE001 - recipe probe must never fail fingerprinting
        return "recipe-unknown"
    return _fingerprint(constants)[:16]


RECIPE_VERSION = "join-v1|trim-095-v1|overlap-010-05-v1|sfx-overlap-1.0-v1|sfx-vol-0.5-v1"
"""Recipe identity label for the mix/bed renders (M6, DESIGN §56).

Human-readable tag; the machine truth is `_recipe_constants()` values
inside both fingerprints (changing any constant misses by construction).
Bump this label alongside any recipe change so logs stay greppable —
the fingerprint, not this string, owns correctness.
"""

MUSIC_CACHE_WAV_NAME = "final_music_cache.wav"
MUSIC_CACHE_JSON_NAME = "final_music_cache.json"
BED_CACHE_WAV_NAME = "final_bed_cache.wav"
BED_CACHE_JSON_NAME = "final_bed_cache.json"


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
    """Take-file identities for the music fingerprint (M2: no `_src` files).

    Continuation sources (`take_NNNN_src.wav`) are unledgered build
    scratch pruned at ensure entry — including them would fork the
    fingerprint on every render (a new `_src` file per chained take)
    and miss the cache on no-change resumes. Only ledgered takes
    (`take_*.wav` minus `*_src*.wav` and `*.partial.wav` staging temps)
    participate.
    """
    audio_dir = run_dir / "audio"
    try:
        candidates = sorted(audio_dir.glob("take_*.wav"), key=lambda path: path.name)
    except OSError:
        return [{"name": "unreadable", "identity": "unreadable"}]
    identities: list[dict[str, str]] = []
    for candidate in candidates:
        if "_src" in candidate.name or candidate.name.endswith(".partial.wav"):
            continue
        try:
            stat = candidate.stat()
            identities.append(
                {"name": candidate.name, "identity": f"{stat.st_size}:{stat.st_mtime_ns}"}
            )
        except OSError:
            identities.append({"name": candidate.name, "identity": "missing"})
    return identities


def _stem_file_identities(run_dir: Path) -> list[dict[str, str]]:
    """Stem-file identities for the bed fingerprint (DESIGN §56 stage-skip).

    Crashed-render temps (`*.partial.wav`, written beside the stems so
    every backend's format inference keeps working) are excluded: a
    leftover staging file from a killed render must not fork the digest
    on its mere presence or absence — same exclusion `_take_file_identities`
    already applies to take staging temps. Digest-shape note: slots
    fingerprinted while a partial was on disk miss exactly once and
    rebuild (safe direction); runs without partials fingerprint
    byte-identically to before.
    """
    stem_dir = run_dir / "audio" / "sfx"
    try:
        candidates = sorted(stem_dir.glob("w*.wav"), key=lambda path: path.name)
    except OSError:
        return [{"name": "unreadable", "identity": "unreadable"}]
    identities: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate.name.endswith(".partial.wav"):
            continue
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
    mastering_enabled: bool = False,
) -> str:
    """Fingerprint everything the music mix (`build_final_audio`) reads.

    `mastering_enabled` is accepted and ignored (kept for caller
    compatibility): mastering applies after the cache hit at publish,
    every time, so toggling it never invalidates this slot. The slot
    always holds pre-master bytes — mastered bytes are never stored
    here.

    Digest-shape note (DESIGN §56 stage-skip): the `takes_ledger`
    component is the order-normalized sha (rows sorted by `take_id` via
    `_sha256_ledger_normalized`) instead of the raw-bytes sha, so
    identical coverage in any row order hits instead of rebuilding.
    Previously stored fingerprints therefore miss exactly once and
    rebuild (safe direction — a rebuild, never a false hit); a torn
    tail falls back to the raw-bytes hash, which matches the old value
    for the same bytes. Stem/take file identities stay mtime-based by
    design (content-hashing every take/stem is too slow at jango
    scale); `*.partial.wav` staging temps are excluded from the stem
    walk (see `_stem_file_identities`).
    """
    takes_path = run_dir / "audio" / "takes.jsonl"
    takes_digest = _sha256_ledger_normalized(takes_path, "take_id")
    del mastering_enabled
    try:
        recipe_constants: dict[str, object] = _recipe_constants()
    except Exception:  # noqa: BLE001 - recipe probe must never fail fingerprinting
        recipe_constants = {"recipe": RECIPE_VERSION}
    return _fingerprint(
        {
            "version": CACHE_VERSION,
            "recipe": RECIPE_VERSION,
            "recipe_constants": recipe_constants,
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
    mastering_enabled: bool = False,
) -> str:
    """Fingerprint everything the SFX bed (`render_sfx_bed`) reads.

    `music_digest` is included because the bed is dubbed against the music
    timeline — a different music mix means the bed alignment must rebuild
    even when the stems match. `dual_pan` is included because the
    single bed is not the spatialized pair: toggling it misses by design.
    Both track ledgers hash (order-normalized like the single ledger
    before them); a missing right ledger hashes as "missing", which only
    matches another missing one. `mastering_enabled` is accepted and
    ignored (same rule as the music slot); the slot itself always
    holds the pre-master bed — mastering applies after the cache
    hit, every time, and mastered bytes are never stored here. Stem
    file identities exclude `*.partial.wav` staging temps (see
    `_stem_file_identities`): a leftover from a killed render no longer
    forks the digest, at the cost of one miss for slots fingerprinted
    while a partial was present.
    """
    sfx_dir = run_dir / "audio" / "sfx"
    ledger_digest = _sha256_ledger_normalized(sfx_dir / "sfx.jsonl", "window_id")
    right_digest = _sha256_ledger_normalized(sfx_dir / "sfx_right.jsonl", "window_id")
    del mastering_enabled
    try:
        bed_recipe_constants: dict[str, object] = _recipe_constants()
    except Exception:  # noqa: BLE001 - recipe probe must never fail fingerprinting
        bed_recipe_constants = {"recipe": RECIPE_VERSION}
    return _fingerprint(
        {
            "version": CACHE_VERSION,
            "recipe": RECIPE_VERSION,
            "recipe_constants": bed_recipe_constants,
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


def _probed_wav_seconds(path: Path) -> float | None:
    """Probed WAV duration, None when unprobable (H5 hit-site gate helper).

    Lazy `media_audio` import (that module owns the probe + cache): a probe
    failure reads as a miss, never a finalize failure — the caller falls
    through to rebuild.
    """
    try:
        from voyage.media_audio import _audio_duration_seconds
    except ImportError:
        return None
    try:
        return _audio_duration_seconds(path)
    except (OSError, ValueError, MediaError):
        return None


def music_cache_hit_valid(
    cached_wav: Path, expected_seconds: float | None, tolerance: float = 0.6
) -> bool:
    """Whether a digest-matched music slot is safe to reuse (H5, pure gate).

    Mirrors the bed pattern: the fingerprint already covers inputs, but a
    stale slot (hand-copied run, truncated write that kept its size) would
    still digest-match. Probing the cached duration against the expected
    stretched timeline at the hit site catches it — past `tolerance`
    (the 0.6 s A/V budget) the caller falls through to rebuild. `None`
    expectation keeps the legacy digest-only hit.
    """
    if expected_seconds is None:
        return True
    probed = _probed_wav_seconds(cached_wav)
    if probed is None:
        return False
    return abs(probed - expected_seconds) <= tolerance


def load_music_cache(
    run_dir: Path, digest: str, *, expected_seconds: float | None = None
) -> Path | None:
    """Return the cached music wav when `digest` matches, else None.

    H5 music re-probe (bed-pattern mirror): when `expected_seconds` is
    given, the cached duration is probed against the expected stretched
    timeline and a disagreement reads as a miss (fall through to
    rebuild). Legacy callers omitting it keep the digest-only hit.
    """
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
    if not music_cache_hit_valid(cached_wav, expected_seconds):
        return None
    return cached_wav


def load_music_cache_with_seconds(
    run_dir: Path, digest: str, *, expected_seconds: float | None = None
) -> tuple[Path, float] | None:
    """Bed-pattern twin: `(cached wav, source seconds)` when valid, else None.

    Track A/C prefers this over `load_music_cache` — the sidecar
    `source_seconds` (written by `store_music_cache`) rides along so the
    caller need not re-probe for the dub stretch. Legacy slots without
    the key probe on the spot (same tolerance as the hit gate) and still
    hit; unprobable slots miss.
    """
    hit = load_music_cache(run_dir, digest, expected_seconds=expected_seconds)
    if hit is None:
        return None
    sidecar = run_dir / "audio" / MUSIC_CACHE_JSON_NAME
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    stored = record.get("source_seconds")
    if isinstance(stored, (int, float)) and stored > 0:
        if expected_seconds is not None and abs(float(stored) - expected_seconds) > 0.6:
            return None
        return (hit, float(stored))
    probed = _probed_wav_seconds(hit)
    if probed is None:
        return None
    if expected_seconds is not None and abs(probed - expected_seconds) > 0.6:
        return None
    return (hit, probed)


def store_music_cache(
    run_dir: Path, digest: str, source_wav: Path, *, source_seconds: float | None = None
) -> None:
    """Overwrite the single music slot with `source_wav` (never raises).

    H5: `source_seconds` (the stretched timeline the mix was built for)
    rides in the sidecar next to the fingerprint — the hit-site probe
    (`load_music_cache_with_seconds`) compares it without re-probing.
    Legacy callers omitting it write a slot without the key (still hit
    via the probe fallback). Callers prefer passing the probed mix
    seconds from `build_final_audio_with_metrics`.
    """
    try:
        probed_seconds = source_seconds
        if probed_seconds is None:
            probed_seconds = _probed_wav_seconds(source_wav)
        destination = run_dir / "audio" / MUSIC_CACHE_WAV_NAME
        atomic_copy(source_wav, destination)
        sidecar: dict[str, object] = {
            "version": CACHE_VERSION,
            "recipe": RECIPE_VERSION,
            "fingerprint": digest,
        }
        if probed_seconds is not None and probed_seconds > 0:
            sidecar["source_seconds"] = float(probed_seconds)
        atomic_write_json(run_dir / "audio" / MUSIC_CACHE_JSON_NAME, sidecar)
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
                "recipe": RECIPE_VERSION,
                "fingerprint": digest,
                "source_seconds": source_seconds,
            },
        )
    except OSError:
        return


def delete_audio_caches(run_dir: Path) -> int:
    """Delete the single-slot music/bed caches (M4 shrink helper, never raises).

    Track C's `configure shrink` calls this after `drop_sfx_beyond` so a
    shrunk timeline never reuses a stale mix/bed. Deletes the two wavs
    plus the two JSON sidecars best-effort and returns the deleted file
    count. Missing files count 0 — shrinking a run that never finalized
    is a no-op, not an error.
    """
    import contextlib

    deleted = 0
    candidates = (
        run_dir / "audio" / MUSIC_CACHE_WAV_NAME,
        run_dir / "audio" / MUSIC_CACHE_JSON_NAME,
        run_dir / "audio" / "sfx" / BED_CACHE_WAV_NAME,
        run_dir / "audio" / "sfx" / BED_CACHE_JSON_NAME,
    )
    for candidate in candidates:
        with contextlib.suppress(OSError):
            if candidate.is_file():
                candidate.unlink()
                deleted += 1
    return deleted

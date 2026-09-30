"""Issue 152: deadlock-class proof for a wide MANUAL-fade N-way join.

The live 31-segment incident deadlocked a single ffmpeg invocation chaining
31 `acrossfade` filters, so `test_final_blend_scale` pins every blend ffmpeg
call to at most 2 audio inputs (foreign file, not owned — untouched here).
`acrossfade` is one filter class; the manual recipe (`afade` out/in + `adelay`
+ `amix`, what `_blend_pair` already uses pairwise) is sample-passthrough
filters with no scheduler pathology per the `_blend_pair` docstring. This
module proves, on CPU ffmpeg via fixtures, whether ONE invocation over N
inputs with the manual recipe (a) completes under a timeout guard (no hang)
and (b) reproduces the pairwise fold on a small case.

Batch 12 proof outcome (2026-09-30, CPU ffmpeg, in-container): (a) PASSED —
N=8 completes in ~1 s under the 180 s guard with exact duration; (b) FAILED
— N=4 sizes/durations exact but bytes differ from byte 2304110 (max 65536
s32 units = 1 s16 LSB, mean ~5041, second-and-later overlaps only; N=2 is
byte-identical so the recipe matches and the gap is generational ordering).
The join therefore stays BLOCKED; the probe-memo fold stays.

No production change rides on this file either way: the probe-memo fold
stays, and relaxing the <=2-input pin belongs to that file's owner.
"""

from __future__ import annotations

import array
import subprocess
import sys
import wave
from pathlib import Path

import pytest

STEM_SECONDS = 4.0
OVERLAP_SECONDS = 1.0
WIDE_TIMEOUT_SECONDS = 180


def _sine(dest: Path, seconds: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    return dest


def _pairwise_fold(stems: list[Path], overlap: float, workdir: Path) -> Path:
    """The production fold shape (`render_sfx_bed` / `build_final_audio`)."""
    from voyage.media import _blend_fade_seconds, _blend_pair

    durations = [STEM_SECONDS] * len(stems)
    accum = stems[0]
    accum_seconds = durations[0]
    for index in range(1, len(stems)):
        step = workdir / f"pair_blend_{index:02d}.wav"
        fade = _blend_fade_seconds(accum_seconds, durations[index], overlap)
        _blend_pair(
            accum,
            stems[index],
            step,
            overlap,
            first_seconds=accum_seconds,
            second_seconds=durations[index],
        )
        accum_seconds = accum_seconds + durations[index] - fade
        accum = step
    return accum


def _wide_manual_join(
    stems: list[Path], durations: list[float], overlap: float, dest: Path
) -> Path:
    """ONE ffmpeg invocation over N inputs with chained manual fades.

    Fold arithmetic mirrors the production folds exactly (same
    `_blend_fade_seconds` per pair, same `%.3f` fades, same integer-ms
    delays, same `amix ... normalize=0` tail) — the only difference is
    topology: each stem decoded once instead of the accum re-encoded N-1
    times. Never `acrossfade` (the deadlocked filter class).
    """
    from voyage.errors import MediaError
    from voyage.media import _blend_fade_seconds

    count = len(stems)
    if count < 2:
        raise MediaError("wide join needs at least 2 stems")
    fades: list[float] = []
    starts = [0.0]
    accum = durations[0]
    for index in range(1, count):
        fade = _blend_fade_seconds(accum, durations[index], overlap)
        fades.append(fade)
        starts.append(accum - fade)
        accum = accum + durations[index] - fade
    chains: list[str] = []
    for index in range(count):
        chain = f"[{index}:a]"
        if index > 0:
            chain += f"afade=t=in:st=0:d={fades[index - 1]:.3f},"
        if index < count - 1:
            fout = fades[index]
            chain += f"afade=t=out:st={durations[index] - fout:.3f}:d={fout:.3f},"
        chain = chain.rstrip(",")
        delay_ms = int(round(starts[index] * 1000))
        if delay_ms:
            chain += f",adelay={delay_ms}:all=1"
        chain += f"[a{index}]"
        chains.append(chain)
    mixed = "".join(f"[a{index}]" for index in range(count))
    filter_graph = (
        ";".join(chains) + f";{mixed}amix=inputs={count}:duration=longest:"
        "dropout_transition=0:normalize=0[aout]"
    )
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for stem in stems:
        argv += ["-i", str(stem)]
    argv += ["-filter_complex", filter_graph, "-map", "[aout]", "-c:a", "pcm_s32le", str(dest)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=WIDE_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        raise MediaError(f"wide manual join failed: {proc.stderr[-2000:]}")
    return dest


def _read_mono_frames(path: Path) -> array.array[int]:
    """First-channel samples as a signed array (stdlib only, no numpy)."""
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
    samples: array.array[int] = array.array("i" if width == 4 else "h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    step = 2  # stereo fixture: keep channel 0 only
    return samples[::step]


def test_wide_manual_join_completes_without_hang(tmp_path: Path) -> None:
    """N=8 manual-fade inputs in ONE invocation finish under the guard."""
    from voyage.media import probe

    stems = [_sine(tmp_path / f"stem{i}.wav", STEM_SECONDS) for i in range(8)]
    dest = tmp_path / "wide.wav"
    _wide_manual_join(stems, [STEM_SECONDS] * 8, OVERLAP_SECONDS, dest)
    assert dest.exists() and dest.stat().st_size > 0
    expected = 8 * STEM_SECONDS - 7 * OVERLAP_SECONDS
    duration = float(probe(dest).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(expected, abs=0.15)


def test_wide_manual_join_matches_pairwise_fold(tmp_path: Path) -> None:
    """N=4: the single graph stays within measured closeness of the fold.

    Characterization, not TDD (batch 12 proof outcome): byte-parity FAILED
    (first diff at byte 2304110; sizes/durations exact), so the single-graph
    join stays BLOCKED and the probe-memo fold stays. Measured 2026-09-30
    on CPU ffmpeg: max 65536 s32 units (= 1 s16 LSB, ~-69 dBFS peak),
    mean ~5041 (~-91 dBFS), confined to second-and-later overlap regions;
    the N=2 single blend above is byte-identical, so the recipe matches and
    the divergence is generational ordering across chained blends. Bounds
    below carry ~2x headroom as a tripwire for filter-behavior drift.
    """
    from voyage.media import probe

    stems = [_sine(tmp_path / f"stem{i}.wav", STEM_SECONDS) for i in range(4)]
    pair = _pairwise_fold(stems, OVERLAP_SECONDS, tmp_path)
    wide = _wide_manual_join(stems, [STEM_SECONDS] * 4, OVERLAP_SECONDS, tmp_path / "wide.wav")
    pair_duration = float(probe(pair).get("format", {}).get("duration", 0.0))
    wide_duration = float(probe(wide).get("format", {}).get("duration", 0.0))
    assert wide_duration == pytest.approx(pair_duration, abs=0.05)
    assert pair.stat().st_size == wide.stat().st_size
    first = _read_mono_frames(pair)
    second = _read_mono_frames(wide)
    assert len(first) == len(second)
    diffs = [abs(one - other) for one, other in zip(first, second, strict=True)]
    assert max(diffs) <= 131072
    assert sum(diffs) / len(diffs) < 15000.0


def test_wide_manual_join_matches_single_blend_byte_for_byte(tmp_path: Path) -> None:
    """N=2: one manual blend equals `_blend_pair` exactly (recipe check)."""
    from voyage.media import _blend_pair

    stems = [_sine(tmp_path / f"stem{i}.wav", STEM_SECONDS) for i in range(2)]
    expected = tmp_path / "pair.wav"
    _blend_pair(
        stems[0],
        stems[1],
        expected,
        OVERLAP_SECONDS,
        first_seconds=STEM_SECONDS,
        second_seconds=STEM_SECONDS,
    )
    wide = _wide_manual_join(stems, [STEM_SECONDS] * 2, OVERLAP_SECONDS, tmp_path / "wide.wav")
    assert expected.stat().st_size == wide.stat().st_size
    assert expected.read_bytes() == wide.read_bytes()

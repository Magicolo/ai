"""Issue 152 parity-rescue research: pin the exact rounding mechanism.

Batch-12 proved the wide MANUAL-fade single graph completes without hanging
but is NOT bit-identical to the pairwise left-fold (generational ordering,
<=1 s16 LSB in post-first overlaps). This module nails the mechanism and
pins two bit-identical constructions WITHOUT changing production code (the
`test_finalize_fastpath` <=2-input pin still holds, so nothing lands yet):

- `afade` runs in its input's NATIVE sample format (ffmpeg 7.1.5 filter
  negotiation): s16-fed fades truncate every faded sample to the s16 grid,
  s32-fed fades keep full 32-bit precision, fltp-fed fades use float gains.
  The fold re-ingests s32 intermediates, so blends 2..N fade in s32 while
  the flat wide graph (fed s16 stems) fades every chain in s16. Same ideal
  gains, different truncation grids: <=1 s16 LSB (65536 s32 units) per faded
  sample, confined to second-and-later overlaps (the only regions whose
  accum passed through an s32 file). `amix`/`adelay` always run fltp (exact
  for grid values); the terminal fltp->s32 quantization is shared. No
  resampling occurs (converters show 48000->48000) and reruns are
  byte-identical, closing the aresample-precision and dither legs.
- Construction 1 (single-spawn rescue): chained pairwise stages with an
  `aformat=sample_fmts=s32` barrier per stage reproduces the production
  fold byte-for-byte (each stem decoded once, one spawn).
- Construction 2 (other direction): an s16-intermediate fold reproduces the
  flat wide graph byte-for-byte (both fade every stage in s16).

All fixtures are CPU-only sine/DC via lavfi; scratch probing lived in /tmp.
"""

from __future__ import annotations

import array
import subprocess
import sys
import wave
from pathlib import Path

STEM_SECONDS = 4.0
OVERLAP_SECONDS = 1.0
SAMPLE_RATE = 48000
S16_GRID = 65536
WIDE_TIMEOUT_SECONDS = 180


def _run(argv: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-1000:]
    return proc


def _sine(dest: Path, seconds: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
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
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    return dest


def _dc(dest: Path, frac: float, seconds: float = 4.0) -> Path:
    """Constant-level fixture; returns dest (level must be read back)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc={frac}:d={seconds}:s={SAMPLE_RATE}",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    return dest


def _read_s32_mono(path: Path) -> array.array[int]:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        assert handle.getsampwidth() == 4, path
        channels = handle.getnchannels()
    samples: array.array[int] = array.array("i", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples[::channels]


def _read_s16_mono(path: Path) -> array.array[int]:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        assert handle.getsampwidth() == 2, path
        channels = handle.getnchannels()
    samples: array.array[int] = array.array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples[::channels]


def _production_fold(stems: list[Path], workdir: Path, tag: str) -> Path:
    """The production topology: `_blend_pair` left-fold over s32 files."""
    from voyage.media import _blend_fade_seconds, _blend_pair

    accum = stems[0]
    accum_seconds = STEM_SECONDS
    for index in range(1, len(stems)):
        step = workdir / f"{tag}_{index:02d}.wav"
        _blend_pair(
            accum,
            stems[index],
            step,
            OVERLAP_SECONDS,
            first_seconds=accum_seconds,
            second_seconds=STEM_SECONDS,
        )
        accum_seconds += STEM_SECONDS - _blend_fade_seconds(
            accum_seconds, STEM_SECONDS, OVERLAP_SECONDS
        )
        accum = step
    return accum


def _fold_custom_intermediate(stems: list[Path], workdir: Path, tag: str, inter_codec: str) -> Path:
    """Same pairwise graph as production, selectable intermediate codec."""
    from voyage.media import _blend_fade_seconds

    accum = stems[0]
    accum_seconds = STEM_SECONDS
    for index in range(1, len(stems)):
        step = workdir / f"{tag}_{index:02d}.wav"
        fade = _blend_fade_seconds(accum_seconds, STEM_SECONDS, OVERLAP_SECONDS)
        fade_start = accum_seconds - fade
        delay_ms = int(round(fade_start * 1000))
        graph = (
            f"[0:a]afade=t=out:st={fade_start:.3f}:d={fade:.3f}[a0];"
            f"[1:a]afade=t=in:st=0:d={fade:.3f},adelay={delay_ms}:all=1[a1];"
            "[a0][a1]amix=inputs=2:duration=longest:"
            "dropout_transition=0:normalize=0[aout]"
        )
        codec = "pcm_s32le" if index == len(stems) - 1 else inter_codec
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(accum),
                "-i",
                str(stems[index]),
                "-filter_complex",
                graph,
                "-map",
                "[aout]",
                "-c:a",
                codec,
                str(step),
            ]
        )
        accum_seconds += STEM_SECONDS - fade
        accum = step
    return accum


def _wide_flat(stems: list[Path], dest: Path) -> Path:
    """ONE invocation over N inputs with chained manual fades (batch-12 shape)."""
    from voyage.errors import MediaError
    from voyage.media import _blend_fade_seconds

    count = len(stems)
    if count < 2:
        raise MediaError("wide join needs at least 2 stems")
    durations = [STEM_SECONDS] * count
    fades: list[float] = []
    starts = [0.0]
    accum = durations[0]
    for index in range(1, count):
        fade = _blend_fade_seconds(accum, durations[index], OVERLAP_SECONDS)
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
    graph = (
        ";".join(chains) + f";{mixed}amix=inputs={count}:duration=longest:"
        "dropout_transition=0:normalize=0[aout]"
    )
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for stem in stems:
        argv += ["-i", str(stem)]
    argv += [
        "-filter_complex",
        graph,
        "-map",
        "[aout]",
        "-c:a",
        "pcm_s32le",
        str(dest),
    ]
    _run(argv, timeout=WIDE_TIMEOUT_SECONDS)
    return dest


def _chained_staged_s32(stems: list[Path], dest: Path) -> Path:
    """ONE spawn, sequential pairwise stages, s32 barrier per stage.

    The barrier replays the fold's per-blend s32 quantization inside a
    single graph: each stage's `afade` negotiates s32 (as when fed the
    production s32 intermediates) instead of fltp.
    """
    from voyage.media import _blend_fade_seconds

    count = len(stems)
    durations = [STEM_SECONDS] * count
    fades: list[float] = []
    starts = [0.0]
    accum = durations[0]
    for index in range(1, count):
        fade = _blend_fade_seconds(accum, durations[index], OVERLAP_SECONDS)
        fades.append(fade)
        starts.append(accum - fade)
        accum = accum + durations[index] - fade
    parts: list[str] = []
    for index in range(1, count):
        if index == 1:
            left = f"[0:a]afade=t=out:st={starts[index]:.3f}:d={fades[index - 1]:.3f}[m{index}a]"
        else:
            left = (
                f"[m{index - 1}q]afade=t=out:st={starts[index]:.3f}:"
                f"d={fades[index - 1]:.3f}[m{index}a]"
            )
        right = (
            f"[{index}:a]afade=t=in:st=0:d={fades[index - 1]:.3f},"
            f"adelay={int(round(starts[index] * 1000))}:all=1[m{index}b]"
        )
        mix = (
            f"[m{index}a][m{index}b]amix=inputs=2:duration=longest:"
            f"dropout_transition=0:normalize=0[m{index}]"
        )
        parts += [left, right, mix]
        if index < count - 1:
            parts.append(f"[m{index}]aformat=sample_fmts=s32[m{index}q]")
    argv: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for stem in stems:
        argv += ["-i", str(stem)]
    argv += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        f"[m{count - 1}]",
        "-c:a",
        "pcm_s32le",
        str(dest),
    ]
    _run(argv, timeout=WIDE_TIMEOUT_SECONDS)
    return dest


def test_afade_fades_in_native_input_format_s16_truncates(tmp_path: Path) -> None:
    """s16-fed `afade` truncates every faded sample to the s16 grid.

    A float filter path would emit sample 1 at ~1.37x the DC s16 level;
    s16-integer math truncates levels below one fade step to zero. Any
    nonzero DC level below one fade step (|L| < 48000, always true for
    s16) discriminates the domains independent of the exact level.
    """
    src = _dc(tmp_path / "dc.wav", 0.6)
    level = _read_s16_mono(src)[0]
    assert abs(level) > 1000
    dest = tmp_path / "fade.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            "[0:a]afade=t=in:st=0:d=1.000[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "pcm_s32le",
            str(dest),
        ]
    )
    faded = _read_s32_mono(dest)
    ramp = faded[0 : 1 * SAMPLE_RATE]
    assert all(value % S16_GRID == 0 for value in ramp)
    assert faded[0] == 0
    assert faded[1] == 0
    assert faded[SAMPLE_RATE] == level * S16_GRID


def test_afade_s32_input_keeps_sub_lsb_precision(tmp_path: Path) -> None:
    """s32-fed `afade` does NOT collapse to the s16 grid (domain contrast).

    Same fade, same shape: an s16 input's fade region is 100% grid while
    an s32 input's is essentially never grid — the arithmetic domain
    follows the negotiated input format, not the filter.
    """
    src = _sine(tmp_path / "stem.wav", STEM_SECONDS)
    s32src = tmp_path / "stem_s32.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(src),
            "-c:a",
            "pcm_s32le",
            str(s32src),
        ]
    )
    s16fade = tmp_path / "fade_s16.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            "[0:a]afade=t=out:st=3.000:d=1.000[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "pcm_s32le",
            str(s16fade),
        ]
    )
    s32fade = tmp_path / "fade_s32.wav"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(s32src),
            "-filter_complex",
            "[0:a]afade=t=out:st=3.000:d=1.000[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "pcm_s32le",
            str(s32fade),
        ]
    )
    region = slice(3 * SAMPLE_RATE, 4 * SAMPLE_RATE)
    s16_vals = _read_s32_mono(s16fade)[region]
    s32_vals = _read_s32_mono(s32fade)[region]
    assert all(value % S16_GRID == 0 for value in s16_vals)
    grid_fraction = sum(1 for value in s32_vals if value % S16_GRID == 0) / len(s32_vals)
    assert grid_fraction < 0.5


def test_s16_intermediate_fold_matches_flat_wide_byte_for_byte(tmp_path: Path) -> None:
    """Construction 2: s16 intermediates reproduce the wide graph exactly.

    Both fade every stage in s16, so the truncation grids agree and the
    single terminal quantization lands identically.
    """
    stems = [_sine(tmp_path / f"stem{i}.wav", STEM_SECONDS) for i in range(4)]
    fold = _fold_custom_intermediate(stems, tmp_path, "s16fold", "pcm_s16le")
    wide = _wide_flat(stems, tmp_path / "wide.wav")
    assert fold.stat().st_size == wide.stat().st_size
    assert fold.read_bytes() == wide.read_bytes()


def test_staged_s32_single_graph_matches_production_fold(tmp_path: Path) -> None:
    """Construction 1 (rescue): one spawn replays the fold byte-for-byte.

    Chained pairwise stages with an s32 barrier per stage make each
    stage's `afade` negotiate s32 — the fold's per-blend quantization —
    while every stem is decoded once. NOT landed: the <=2-input pin still
    holds and the pin owner's 31-input-scale proof is still required.
    """
    stems = [_sine(tmp_path / f"stem{i}.wav", STEM_SECONDS) for i in range(4)]
    fold = _production_fold(stems, tmp_path, "prod")
    staged = _chained_staged_s32(stems, tmp_path / "staged.wav")
    assert staged.stat().st_size == fold.stat().st_size
    assert staged.read_bytes() == fold.read_bytes()


def test_production_fold_vs_flat_wide_bounded_divergence(tmp_path: Path) -> None:
    """Characterization of the blocked state: fold vs wide differ only by
    truncation-grid choice — bounded by 1 s16 LSB, confined to overlaps
    whose accum passed through an s32 file (second and later)."""
    stems = [_sine(tmp_path / f"stem{i}.wav", STEM_SECONDS) for i in range(4)]
    fold = _production_fold(stems, tmp_path, "prod")
    wide = _wide_flat(stems, tmp_path / "wide.wav")
    assert fold.stat().st_size == wide.stat().st_size
    first = _read_s32_mono(fold)
    second = _read_s32_mono(wide)
    assert len(first) == len(second)
    diffs = [abs(one - other) for one, other in zip(first, second, strict=True)]
    assert max(diffs) > 0
    assert max(diffs) <= S16_GRID
    assert sum(diffs) / len(diffs) < 15000.0
    second_overlap_start = 6 * SAMPLE_RATE
    assert all(diff == 0 for diff in diffs[:second_overlap_start])


def test_production_fold_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Reruns are byte-identical: no random dither stage exists anywhere
    in the blend path (closes the dither leg alongside the mechanism)."""
    from voyage.media import _blend_pair

    first = _sine(tmp_path / "a.wav", 2.0)
    second = _sine(tmp_path / "b.wav", 2.0)
    one = tmp_path / "one.wav"
    two = tmp_path / "two.wav"
    _blend_pair(first, second, one, 0.5, first_seconds=2.0, second_seconds=2.0)
    _blend_pair(first, second, two, 0.5, first_seconds=2.0, second_seconds=2.0)
    assert one.read_bytes() == two.read_bytes()

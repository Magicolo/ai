"""Beat grid for rhythm-locked segments (DESIGN §35).

Each committed segment spans an integer number of beats so cuts land on
the beat grid: ``beats_for_segment`` starts from ``beats_per_segment``
(4) and doubles (4 → 8 → 16 …) until the implied tempo reaches
``min_bpm`` (60). A 4 s LTXV segment yields exactly 4 beats = 60 BPM;
a 2 s segment yields 4 beats = 120 BPM; a 5.04 s cold-start segment
doubles to 8 beats ≈ 95 BPM.

ACE honors tempo as a style hint, not a sample-exact grid, and takes
chain on segment-aligned boundaries — so cuts land *approximately* on
beats. The math here is exact; the renderer is approximate. Pure logic,
no I/O, no torch.
"""

from __future__ import annotations

import math

MIN_BPM = 60.0
"""Below this a beat feels like separate events, not rhythm."""


def _require_finite(measured_value: float, field_label: str) -> float:
    """Reject nan/inf with the documented ValueError contract.

    Why a shared helper: bare round() internals leak confusing messages
    (nan) or the wrong exception class (OverflowError for inf), while an
    infinite segment silently yields an infinite take downstream (issue 068).
    """
    if not math.isfinite(measured_value):
        raise ValueError(f"{field_label} must be finite (got {measured_value})")
    return measured_value


def beats_for_segment(
    segment_seconds: float,
    base_beats: int = 4,
    min_bpm: float = MIN_BPM,
    max_bpm: float | None = None,
) -> tuple[int, float]:
    """Beats in this segment and the implied take BPM.

    Returns ``(beats, bpm)`` with ``bpm = beats * 60 / segment_seconds``
    and ``beats`` the smallest doubling of ``base_beats`` whose BPM
    reaches ``min_bpm``. Raises ValueError on non-positive durations.

    The ceiling is opt-in (issue 120): the committed-behavior pins in
    `test_rhythm.py` document the raw doubling math, so callers that feed
    a renderer with a tempo cap (ACE-Step rejects BPM > 300) pass
    `max_bpm` explicitly. A grid above the ceiling halves toward one beat
    (cuts stay on an integer grid, only coarser); a segment so short even
    one beat exceeds the cap raises ValueError naming `beats_per_segment`
    — a config error that must fail before any GPU work, not as a fatal
    payload error after the ACE load.
    """
    _require_finite(segment_seconds, "segment duration")
    _require_finite(min_bpm, "minimum tempo")
    if segment_seconds <= 0:
        raise ValueError(f"segment duration must be positive (got {segment_seconds})")
    if base_beats <= 0:
        raise ValueError(f"base beats must be positive (got {base_beats})")
    if min_bpm <= 0:
        raise ValueError(f"minimum tempo must be positive (got {min_bpm})")
    ceiling: float | None = None
    if max_bpm is not None:
        _require_finite(max_bpm, "maximum tempo")
        if max_bpm <= 0:
            raise ValueError(f"maximum tempo must be positive (got {max_bpm})")
        ceiling = max_bpm
    beats = base_beats
    bpm = beats * 60.0 / segment_seconds
    while bpm < min_bpm:
        beats *= 2
        bpm = beats * 60.0 / segment_seconds
    while ceiling is not None and bpm > ceiling and beats > 1:
        beats //= 2
        bpm = beats * 60.0 / segment_seconds
    if ceiling is not None and bpm > ceiling:
        raise ValueError(
            f"beats_per_segment={base_beats} implies {bpm:.0f} BPM for a "
            f"{segment_seconds:.3f} s segment, above the {ceiling:.0f} BPM "
            "render ceiling — lower beats_per_segment or lengthen segments"
        )
    return beats, bpm


def quantize_take_seconds(take_seconds: float, segment_seconds: float) -> float:
    """Snap a take length to a whole number of segments (min 1 segment).

    Takes chained on segment-aligned boundaries keep every take start —
    and therefore every beat-grid downbeat — on a segment boundary, so
    segment cuts stay on the grid across take joints. Raises ValueError
    on non-positive inputs.

    Tie rule (issue 121, pinned by `tests/test_beat_quantize_ties_121.py`
    and `tests/test_rhythm.py:66-67`): the snap uses `math.ceil`, so it
    never plans SHORT (`45 s / 18 s = 2.5 → 3 → 54 s`). Over-coverage is
    trimmed by the finalize slice walk, while the old round-half-to-even
    shortfall chained extra takes (extra GPU swaps) with no diagnostic.
    The supervisor's 094 clamp still contains any per-take overrun.
    """
    _require_finite(take_seconds, "take length")
    _require_finite(segment_seconds, "segment duration")
    if take_seconds <= 0:
        raise ValueError(f"take length must be positive (got {take_seconds})")
    if segment_seconds <= 0:
        raise ValueError(f"segment duration must be positive (got {segment_seconds})")
    multiples = max(1, math.ceil(take_seconds / segment_seconds))
    return multiples * segment_seconds


def segment_progress_info(
    segment_seconds: float,
    beats_per_segment: int = 4,
) -> dict[str, float]:
    """Progress-only beat/BPM numbers for one segment (issue 046).

    The supervisor needs these for display while `build_final_audio`
    re-derives the same timeline for the real mix — this helper keeps the
    display math in `audio.beat` (one owner) instead of a lazy mid-commit
    re-import at the call site. Raises ValueError on bad inputs, same
    contract as `beats_for_segment`.
    """
    beats, grid_bpm = beats_for_segment(segment_seconds, beats_per_segment)
    return {"beats": float(beats), "grid_bpm": grid_bpm}

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

MIN_BPM = 60.0
"""Below this a beat feels like separate events, not rhythm."""


def beats_for_segment(
    segment_seconds: float,
    base_beats: int = 4,
    min_bpm: float = MIN_BPM,
) -> tuple[int, float]:
    """Beats in this segment and the implied take BPM.

    Returns ``(beats, bpm)`` with ``bpm = beats * 60 / segment_seconds``
    and ``beats`` the smallest doubling of ``base_beats`` whose BPM
    reaches ``min_bpm``. Raises ValueError on non-positive durations.
    """
    if segment_seconds <= 0:
        raise ValueError(f"segment duration must be positive (got {segment_seconds})")
    if base_beats <= 0:
        raise ValueError(f"base beats must be positive (got {base_beats})")
    beats = base_beats
    bpm = beats * 60.0 / segment_seconds
    while bpm < min_bpm:
        beats *= 2
        bpm = beats * 60.0 / segment_seconds
    return beats, bpm


def quantize_take_seconds(take_seconds: float, segment_seconds: float) -> float:
    """Snap a take length to a whole number of segments (min 1 segment).

    Takes chained on segment-aligned boundaries keep every take start —
    and therefore every beat-grid downbeat — on a segment boundary, so
    segment cuts stay on the grid across take joints. Raises ValueError
    on non-positive inputs.
    """
    if take_seconds <= 0:
        raise ValueError(f"take length must be positive (got {take_seconds})")
    if segment_seconds <= 0:
        raise ValueError(f"segment duration must be positive (got {segment_seconds})")
    multiples = max(1, round(take_seconds / segment_seconds))
    return multiples * segment_seconds

"""Stage B: caption-similarity repaint gate (DESIGN §140).

A one-word LLM rewording of the music caption must not repaint the whole
unconsumed tail — the music evolution lands at the next take boundary
instead (chained take carries the new caption; keep serves the current
one). Only a genuinely different caption (Jaccard token-set similarity
below threshold) repaints. Threshold calibrated on the boba baseline:
consecutive reword-repaint ledger sims were 0.72–1.00 (suppressed at
0.5); the one defensible shift scored 0.444 (still repaints).
"""

from __future__ import annotations

from voyage.audio.planner import AudioPlanner, AudioTake
from voyage.config import AudioConfig

REWORDED = (
    "A blend of ambient electronic tones with deep, resonant bass, "
    "creating a sense of movement and transformation"
)
REWORDING = (
    "A continuous transformation of ambient electronic tones with deep, "
    "resonant bass, evolving in texture and intensity"
)


def _recorded(planner: AudioPlanner, caption: str = REWORDED) -> AudioTake:
    take = AudioTake(
        take_id="take_0000",
        path="/takes/take_0000.wav",
        caption=caption,
        seed=7,
        covers_from=0.0,
        duration=45.0,
        segment_index=0,
    )
    planner.record(take)
    return take


def test_similar_rewording_keeps_serving_current_take() -> None:
    """Jaccard ~0.7 rewording with margin coverage → keep, no repaint."""
    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, repaint_similarity_threshold=0.5)
    current = _recorded(planner)
    decision = planner.plan(video_time=4.0, caption=REWORDING, seed=7, segment_index=2)
    assert decision.action == "keep"
    assert decision.current == current


def test_dissimilar_caption_still_repaints() -> None:
    """Jaccard 0.0 caption change with unconsumed region → repaint."""
    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, repaint_similarity_threshold=0.5)
    current = _recorded(planner, caption="ambient drift")
    decision = planner.plan(video_time=10.0, caption="brighter pulse", seed=9, segment_index=5)
    assert decision.action == "repaint"
    assert decision.current == current
    assert decision.take is not None
    assert decision.take.covers_from == current.covers_from


def test_similar_caption_chains_with_new_caption_at_boundary() -> None:
    """Similar rewording inside the ahead window → chained render carrying the NEW caption."""
    planner = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, repaint_similarity_threshold=0.5)
    _recorded(planner)
    decision = planner.plan(video_time=30.0, caption=REWORDING, seed=7, segment_index=15)
    assert decision.action == "render"
    assert decision.take is not None
    assert decision.take.covers_from == 45.0
    assert decision.take.caption == REWORDING


def test_threshold_is_honored() -> None:
    """A strict threshold repaints the same pair; zero threshold never repaints."""
    strict = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, repaint_similarity_threshold=0.99)
    _recorded(strict)
    decision = strict.plan(video_time=4.0, caption=REWORDING, seed=7, segment_index=2)
    assert decision.action == "repaint"

    lax = AudioPlanner(take_seconds=45.0, ahead_seconds=20.0, repaint_similarity_threshold=0.0)
    _recorded(lax)
    decision = lax.plan(video_time=4.0, caption="brighter pulse", seed=7, segment_index=2)
    assert decision.action == "keep"


def test_config_default_threshold_is_calibrated_half() -> None:
    """AudioConfig carries the boba-calibrated default (0.5)."""
    assert AudioConfig().repaint_similarity_threshold == 0.5

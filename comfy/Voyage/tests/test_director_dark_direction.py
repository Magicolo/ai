"""Dark/experimental music direction pins (DESIGN §140 audio continuity).

Every generated music caption — LLM-decided or deterministic-fallback —
must carry dark/experimental musical language on top of the charter and
concept derivation. Explicit CLI `--music-caption` pins pass through
untouched (manual control); only generated captions are directed here.
"""

from voyage.director import DIRECTOR_SYSTEM_PROMPT, deterministic_decision


def test_system_prompt_directs_dark_experimental_music() -> None:
    assert "dark experimental touch" in DIRECTOR_SYSTEM_PROMPT
    assert "minor and modal harmony" in DIRECTOR_SYSTEM_PROMPT
    assert "sub-bass pressure" in DIRECTOR_SYSTEM_PROMPT


def test_deterministic_music_caption_is_dark_experimental() -> None:
    decision = deterministic_decision(0, "harbor", "harbor", "ESTABLISH", "misty charter")
    caption = decision.audio.music_caption
    assert "dark experimental" in caption
    assert "harbor" in caption
    assert "misty charter" in caption
    assert "sub-bass" in caption


def test_deterministic_music_caption_stays_bit_stable() -> None:
    first = deterministic_decision(1, "harbor", "harbor", "ESTABLISH", "misty charter")
    second = deterministic_decision(1, "harbor", "harbor", "ESTABLISH", "misty charter")
    assert first.audio.music_caption == second.audio.music_caption

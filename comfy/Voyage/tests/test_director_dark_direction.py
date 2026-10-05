"""Ambient dark music + ultra-high-definition video direction pins.

Every generated music caption — LLM-decided or deterministic-fallback —
must carry ambient slow dark musical language (morphing pads, held
chords, vast drones) with the dark/experimental touch on top of the
charter and concept derivation. Explicit CLI `--music-caption` pins
pass through untouched (manual control); only generated captions are
directed here. Every video stage must be directed ultra high
definition with a concrete continuous camera move (LTX-25-suited).
"""

from voyage.director import DIRECTOR_SYSTEM_PROMPT, deterministic_decision
from voyage.models import DEFAULT_MUSIC_STYLE


def test_default_music_style_is_ambient_slow_dark() -> None:
    assert "ambient dark experimental" in DEFAULT_MUSIC_STYLE
    assert "morphing pads" in DEFAULT_MUSIC_STYLE
    assert "held chords" in DEFAULT_MUSIC_STYLE
    assert "sub-bass" in DEFAULT_MUSIC_STYLE


def test_system_prompt_directs_dark_experimental_music() -> None:
    assert "dark experimental touch" in DIRECTOR_SYSTEM_PROMPT
    assert "minor and modal harmony" in DIRECTOR_SYSTEM_PROMPT
    assert "sub-bass pressure" in DIRECTOR_SYSTEM_PROMPT


def test_system_prompt_directs_ambient_slow_pads() -> None:
    assert "morphing pads" in DIRECTOR_SYSTEM_PROMPT
    assert "held chords" in DIRECTOR_SYSTEM_PROMPT


def test_system_prompt_directs_uhd_video_with_camera_move() -> None:
    assert "ultra high definition" in DIRECTOR_SYSTEM_PROMPT
    assert "camera move" in DIRECTOR_SYSTEM_PROMPT


def test_deterministic_music_caption_is_dark_experimental() -> None:
    decision = deterministic_decision(0, "harbor", "harbor", "ESTABLISH", "misty charter")
    caption = decision.audio.music_caption
    assert "dark experimental" in caption
    assert "harbor" in caption
    assert "misty charter" in caption
    assert "sub-bass" in caption
    assert "morphing pads" in caption
    assert "held chords" in caption


def test_deterministic_music_caption_stays_bit_stable() -> None:
    first = deterministic_decision(1, "harbor", "harbor", "ESTABLISH", "misty charter")
    second = deterministic_decision(1, "harbor", "harbor", "ESTABLISH", "misty charter")
    assert first.audio.music_caption == second.audio.music_caption

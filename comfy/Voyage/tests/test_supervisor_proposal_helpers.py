"""Agreement + split tests for the supervisor proposal-helper extraction (081).

The director-proposal pure helpers live in `voyage.supervisor_proposal`
(DESIGN §§73, 18.2); `voyage.supervisor` re-exports them so every
existing importer (`test_sfx_contract`, `test_three_captions`,
`cli_observe`, the commit path) keeps working. These tests pin the
split: same objects, same pure behavior.
"""

from __future__ import annotations

from pathlib import Path

import voyage.supervisor as supervisor_module
import voyage.supervisor_proposal as proposal_module
from voyage.supervisor_proposal import (
    _token_counts,
    effective_music_caption,
    effective_video_stages,
    previous_transition_captions,
)


def test_reexports_are_identical_objects() -> None:
    """Supervision facade re-exports the proposal helpers (no fork)."""
    assert supervisor_module.previous_transition_captions is previous_transition_captions
    assert supervisor_module.effective_music_caption is effective_music_caption
    assert supervisor_module.effective_video_stages is effective_video_stages
    assert supervisor_module._token_counts is _token_counts
    assert proposal_module.__name__ == "voyage.supervisor_proposal"


def test_music_caption_precedence() -> None:
    """Explicit pin wins, else director caption, else style fallback."""
    assert effective_music_caption("pin", "director", "style") == "pin"
    assert effective_music_caption(None, "director", "style") == "director"
    assert effective_music_caption("", "", "style") == "style"


def test_video_stages_precedence() -> None:
    """One-item explicit pin, else the director's stages (copied)."""
    assert effective_video_stages("red dune", ["a", "b"]) == ["red dune"]
    stages = ["a", "b"]
    assert effective_video_stages(None, stages) == ["a", "b"]
    assert effective_video_stages(None, stages) is not stages


def test_previous_captions_empty_for_segment_zero(tmp_path: Path) -> None:
    """Segment 0 has no predecessor — history read stays empty."""
    assert previous_transition_captions(tmp_path, 0) == ""
    assert previous_transition_captions(tmp_path, -1) == ""


def test_token_counts_rejects_untrusted_wire_data() -> None:
    """Missing/non-int/bool/negative token fields read as zero."""
    assert _token_counts({}) == {"prompt_tokens": 0, "completion_tokens": 0}
    assert _token_counts({"prompt_tokens": True, "completion_tokens": -3}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    assert _token_counts({"prompt_tokens": 12, "completion_tokens": 34}) == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
    }

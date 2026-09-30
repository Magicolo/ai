"""Director-proposal pure helpers for the supervisor (DESIGN §§73, 18.2).

Split from `voyage.supervisor` (issue 081): the commit path's
director-caption flow — previous-segment history, caption/stage
precedence, and LLM token telemetry — as importable pure functions with
no supervisor state. `voyage.supervisor` re-exports every name below so
existing importers keep working; new code imports from here directly.

- `previous_transition_captions`: best-effort drift-chain history
  (three caption families, §140 three-caption doctrine).
- `effective_music_caption` / `effective_video_stages`: explicit-pin
  precedence the planner and the console share (§18.2 staged plan).
- `_token_counts`: Stage A token telemetry on accepted/prefetched raws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voyage import paths
from voyage.models import EvolutionDecision
from voyage.segment_manifest import load_transition


def previous_transition_captions(run_dir: Path, number: int) -> str:
    """Previous segment's three caption families as director-prompt text.

    Best-effort history for the drift chain: reads segment number-1's
    committed transition.json and formats its video stages + music/SFX
    captions via format_previous_captions. Missing segment (voyage
    start), torn JSON, or legacy decisions without captions all yield ""
    so the director prompt is unchanged — a history read must never
    break a commit.
    """
    if number <= 0:
        return ""
    from voyage.director import format_previous_captions

    prev_id = paths.format_segment_id(number - 1)
    prev_dir = paths.segment_dir(run_dir, prev_id)
    raw = load_transition(prev_dir)
    if not raw:
        return ""
    try:
        decision = EvolutionDecision.model_validate(raw)
    except Exception:  # noqa: BLE001 — best-effort history read, never breaks a commit
        return ""
    return format_previous_captions(
        previous_video_stages=list(decision.video.stages),
        previous_music=decision.audio.music_caption,
        previous_sfx=decision.audio.sfx_caption,
    )


def effective_music_caption(
    explicit: str | None, decision_caption: str, style_fallback: str
) -> str:
    """Music caption precedence: explicit CLI pin, else the director's
    evolving caption, else the charter style fallback. Pure (pins the
    precedence the slow-loop planner and the console display share)."""
    return explicit or decision_caption or style_fallback


def effective_video_stages(explicit: str | None, stages: list[str]) -> list[str]:
    """Video stage precedence: a one-item explicit CLI pin, else the
    director's evolving stages. Pure (pins the substitution the prompt
    planner applies after the accept transaction)."""
    if explicit:
        return [explicit]
    return list(stages)


def _token_counts(raw: dict[str, Any]) -> dict[str, int]:
    """LLM token usage carried on an accepted/prefetched raw (Stage A telemetry).

    Worker decide replies report `prompt_tokens`/`completion_tokens`; older
    or deterministic payloads carry neither and read as zero. Non-int,
    bool or negative values are untrusted wire data and also read as zero.
    Pure so the accept loop and the prefetch consumer share one rule.
    """

    def _as_count(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    return {
        "prompt_tokens": _as_count(raw.get("prompt_tokens")),
        "completion_tokens": _as_count(raw.get("completion_tokens")),
    }

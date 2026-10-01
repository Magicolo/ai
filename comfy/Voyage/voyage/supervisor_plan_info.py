"""Commit-plan helpers for the supervisor (DESIGN §§73, 18.2).

Split from `voyage.supervisor` (issue 081): the commit path's
console-plan dict — decision + prompts shown before the render — as an
importable pure function with no supervisor state. `voyage.supervisor`
re-exports the name below and keeps `Supervisor._segment_plan_info` as a
one-line delegating wrapper so existing callers keep working; new code
imports from here directly.

- `segment_plan_info`: pure plan dict (frames fallback, caption
  precedence, seeds/cuts normalization, geometry/fps mirror).
"""

from __future__ import annotations

from typing import Any

from voyage.config import ProjectConfig
from voyage.models import EvolutionDecision
from voyage.supervisor_proposal import effective_music_caption


def segment_plan_info(
    config: ProjectConfig,
    number: int,
    segment_id: str,
    decision: EvolutionDecision,
    block_prompts: list[str],
    video_payload: dict[str, Any],
    num_blocks: int,
    prefetch_hit: bool,
    drift_hold: bool,
) -> dict[str, Any]:
    """Console plan dict: decision + prompts shown before the render."""
    from voyage.audio.acestep import MAX_BPM
    from voyage.audio.beat import beats_for_segment

    planned_frames = video_payload.get("frames", config.video.segment_frames)
    if not isinstance(planned_frames, int) or planned_frames <= 0:
        planned_frames = config.video.segment_frames
    planned_duration = planned_frames / config.video.fps
    caption = effective_music_caption(
        config.audio.music_caption,
        decision.audio.music_caption,
        config.audio.music_style,
    )
    energy = min(1.0, max(0.0, decision.audio.energy))
    beats, grid_bpm = beats_for_segment(
        planned_duration, config.audio.beats_per_segment, max_bpm=MAX_BPM
    )
    seeds = video_payload.get("seeds", [video_payload.get("seed", 0)])
    cuts = video_payload.get("scene_cuts", [])
    return {
        "number": number,
        "segment_id": segment_id,
        "destination": decision.destination_concept,
        "phase": decision.phase,
        "novelty_accepted": decision.novelty_accepted,
        "drift_hold": drift_hold,
        "prefetch_hit": prefetch_hit,
        "director_backend": config.director.backend,
        "video_backend": config.video.backend,
        "audio_backend": config.audio.backend,
        "geometry": f"{config.video.width}x{config.video.height}",
        "fps": config.video.fps,
        "planned_frames": planned_frames,
        "planned_duration": planned_duration,
        "blocks": num_blocks,
        "video_prompts": list(block_prompts),
        "video_seeds": list(seeds) if isinstance(seeds, list) else [seeds],
        "scene_cuts": list(cuts) if isinstance(cuts, list) else [],
        "transition_mechanism": decision.transition.mechanism,
        "transition_stages": list(decision.transition.intermediate_stages),
        "audio_caption": caption,
        "audio_energy": energy,
        "audio_bpm": grid_bpm,
        "audio_beats": beats,
        "audio_texture": decision.audio.texture,
        "audio_environment": list(decision.audio.environment),
        "audio_sfx_caption": decision.audio.sfx_caption,
        "notes": decision.notes,
    }

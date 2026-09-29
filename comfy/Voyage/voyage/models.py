"""Pydantic schemas for persistent state, RPC, and plans.

Backend-specific code must not leak into these types (DESIGN §82):
no tensors, no embedding vectors — artifact references only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

LifecycleStatus = Literal[
    "CREATED",
    "STARTING",
    "RUNNING",
    "PAUSE_REQUESTED",
    "PAUSED",
    "STOP_REQUESTED",
    "FINALIZING",
    "COMPLETE",
    "FAILED",
    "PAUSED_DISK_FULL",
]

TransitionPhase = Literal[
    "ESTABLISH",
    "DRIFT",
    "TRANSFORM",
    "DESTABILIZE",
    "EMERGE",
    "STABILIZE",
]


class ArtifactRef(BaseModel):
    path: str = Field(description="Path relative to the segment directory")
    sha256: str | None = None
    bytes: int | None = None


class PromptStage(BaseModel):
    stage: int
    block_start: int
    block_end: int
    prompt: str

    @model_validator(mode="after")
    def block_range_ordered(self) -> PromptStage:
        """Reject inverted ranges that break segment ordering.

        Why: an inverted range flows into directory names and ordering
        checks that assume block_end >= block_start (issue 091).
        """
        if self.block_end < self.block_start:
            raise ValueError(
                f"block_end must be >= block_start (got {self.block_start}..{self.block_end})"
            )
        return self


class PromptPlan(BaseModel):
    segment_id: str
    stages: list[PromptStage] = Field(default_factory=list)


class StyleSpec(BaseModel):
    """Immutable human-owned style charter (DESIGN §15)."""

    prompt: str
    motion_energy_min: float = 0.20
    motion_energy_max: float = 0.35
    visual_complexity_min: float = 0.30
    visual_complexity_max: float = 0.50
    semantic_drift_min: float = 0.12
    semantic_drift_max: float = 0.25
    # Floor for the drift-vs-segment-0 style proxy (§43): below this the
    # inspector flags style collapse. Provisional — recalibrate against
    # real footage in the Phase 5 E2E (same pass as the other bands).
    style_similarity_min: float = 0.60
    surrealism: float = 0.70
    transition_smoothness: float = 0.90

    @model_validator(mode="after")
    def bands_ordered(self) -> StyleSpec:
        """Reject inverted calibration bands that silently mis-flag.

        Why: the inspector compares measured metrics against these
        min/max pairs — an inverted band (min > max) would flag every
        segment or none, with no error at the comparison site.
        """
        for minimum_name, maximum_name in (
            ("motion_energy_min", "motion_energy_max"),
            ("visual_complexity_min", "visual_complexity_max"),
            ("semantic_drift_min", "semantic_drift_max"),
        ):
            minimum = getattr(self, minimum_name)
            maximum = getattr(self, maximum_name)
            if maximum < minimum:
                raise ValueError(
                    f"{maximum_name} must be >= {minimum_name} (got {minimum}..{maximum})"
                )
        return self


TransitionMechanism = Literal[
    "material_metamorphosis",
    "environmental_transformation",
    "scale_shift",
    "geometric_transformation",
    "physical_rule_change",
    "lighting_transformation",
    "perceptual_transformation",
    "hybrid",
]


class TransitionPlan(BaseModel):
    """Explicit narrative bridge between concepts (DESIGN §17)."""

    source_concept: str = ""
    destination_concept: str = ""
    mechanism: TransitionMechanism = "hybrid"
    transition_strength: float = 0.25
    estimated_duration_seconds: float = 64.0
    intermediate_stages: list[str] = Field(default_factory=list)
    major_transition: bool = False


class DirectorDestination(BaseModel):
    canonical_name: str
    summary: str = ""


class DirectorVideoPlan(BaseModel):
    """Ordered scene descriptions, current world → destination (DESIGN §19).

    The stages ARE the video caption family: each stage must carry
    concrete visual detail (motion, scenes, objects, characters, shots,
    angles) derived from the style charter + general prompt, and must
    drift gradually as the general prompt drifts — never jump.
    """

    stages: list[str] = Field(default_factory=list)


class DirectorAudioPlan(BaseModel):
    """Music + SFX caption families (DESIGN §19, three-caption doctrine).

    Both captions derive from the same style charter + general prompt as
    the video stages and evolve with it: a drifted concept yields drifted
    captions, a held concept yields stable captions. The slow-loop music
    planner keys repaints on `music_caption` only — `sfx_caption` drift
    never triggers a music take render.
    """

    music_caption: str = "slow ambient electronic composition"
    energy: float = 0.48
    tempo_bpm: int = 74
    texture: str = ""
    environment: list[str] = Field(default_factory=list)
    sfx_caption: str = ""


class DirectorNovelty(BaseModel):
    why_new: str = ""


class EvolutionDecision(BaseModel):
    """Director proposal, validated before anything else touches it (§§19, 74)."""

    decision_index: int
    destination: DirectorDestination
    transition: TransitionPlan = Field(default_factory=TransitionPlan)
    video: DirectorVideoPlan = Field(default_factory=DirectorVideoPlan)
    audio: DirectorAudioPlan = Field(default_factory=DirectorAudioPlan)
    novelty: DirectorNovelty = Field(default_factory=DirectorNovelty)
    phase: TransitionPhase = "ESTABLISH"
    novelty_accepted: bool = True
    notes: str = ""

    @property
    def destination_concept(self) -> str:
        return self.destination.canonical_name


class SegmentWorldState(BaseModel):
    segment_id: str
    current_concept: str
    destination_concept: str
    phase: TransitionPhase
    seed: int


class AudioPlan(BaseModel):
    segment_id: str
    music_style: str = "ambient electronic"
    energy: float = 0.5
    seed: int = 0
    # Slow-loop takes (§35) serving this segment, oldest first. Empty for
    # runs committed before the slow loop existed.
    take_ids: list[str] = Field(default_factory=list)


class RunState(BaseModel):
    schema_version: int = 1
    run_id: str
    status: LifecycleStatus = "CREATED"
    # Counters only ever advance from zero (validate_run relies on the
    # contiguous/non-negative invariant), so negatives are corrupt state,
    # not data — fail loud at the read_state boundary instead of letting
    # a torn write strand the run. fps is deliberately unguarded: the CLI
    # tolerates legacy fps=0 states, and audio_buffer_seconds takes
    # planner-computed floats whose sign this layer cannot judge.
    next_segment_number: int = Field(default=0, ge=0)
    committed_segments: int = Field(default=0, ge=0)
    timeline_frames: int = Field(default=0, ge=0)
    fps: int = 24
    current_concept: str = ""
    destination_concept: str = ""
    phase: TransitionPhase = "ESTABLISH"
    decision_index: int = Field(default=0, ge=0)
    video_checkpoint: dict[str, Any] = Field(default_factory=dict)
    audio_buffer_seconds: float = 0.0
    last_error: str | None = None


class WorkerRequest(BaseModel):
    id: str
    op: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = True


class WorkerResponse(BaseModel):
    id: str
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: WorkerErrorDetail | None = None

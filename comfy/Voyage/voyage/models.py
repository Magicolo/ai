"""Pydantic schemas for persistent state, RPC, and plans.

Backend-specific code must not leak into these types (DESIGN §82):
no tensors, no embedding vectors — artifact references only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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
    stages: list[str] = Field(default_factory=list)


class DirectorAudioPlan(BaseModel):
    music_caption: str = "slow ambient electronic composition"
    energy: float = 0.48
    tempo_bpm: int = 74
    texture: str = ""
    environment: list[str] = Field(default_factory=list)


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
    next_segment_number: int = 0
    committed_segments: int = 0
    timeline_frames: int = 0
    fps: int = 24
    current_concept: str = ""
    destination_concept: str = ""
    phase: TransitionPhase = "ESTABLISH"
    decision_index: int = 0
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

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


class EvolutionDecision(BaseModel):
    decision_index: int
    destination_concept: str
    phase: TransitionPhase
    novelty_accepted: bool = True
    notes: str = ""


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

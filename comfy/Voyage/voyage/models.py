"""Pydantic schemas for persistent state, RPC, and plans.

Backend-specific code must not leak into these types (DESIGN §82):
no tensors, no embedding vectors — artifact references only.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

#: Default music steering (§35): ambient dark experimental music — slow
#: morphing pads and held chords, vast powerful drones, weird textures.
DEFAULT_MUSIC_STYLE = (
    "ambient dark experimental music, slow morphing pads and held chords, "
    "vast powerful drones, weird slow-evolving textures, deep sub-bass"
)

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

    @model_validator(mode="after")
    def path_and_size_sane(self) -> ArtifactRef:
        """Reject empty paths and negative sizes (issue 119).

        Why: an empty path + negative byte count persist into segment
        metadata and bias every downstream consumer — fail the one bad
        parse loudly instead.
        """
        if not self.path.strip():
            raise ValueError("artifact path must be non-empty")
        if self.bytes is not None and self.bytes < 0:
            raise ValueError(f"artifact bytes must be >= 0 (got {self.bytes})")
        return self


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
        if self.block_start < 0:
            raise ValueError(f"block_start must be >= 0 (got {self.block_start})")
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
    # Ceilings for the two §43 metrics the feedback loop steers on
    # (issue 180): palette blowout and scene-cut instability. Defaults
    # match the director-context bands (`director.py`
    # `format_measured_context`), so the MEASURED flags and the steering
    # agree out of the box; overridable per charter like every other band.
    palette_distance_max: float = 0.30
    scene_boundary_strength_max: float = 0.30

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

    @model_validator(mode="after")
    def values_in_domain(self) -> StyleSpec:
        """Reject out-of-domain calibration values (issue 119).

        Why: the inspector compares 0..1 metrics against these bands — an
        ordered-but-out-of-domain band (e.g. `5.0..6.0`) reads BELOW
        forever, and `style_similarity_min=99` flags every segment as
        style collapse. The model layer is the single source everything
        else trusts, so fail here, not per segment. NaN fails the chained
        comparison automatically; inf fails the upper bound.
        """
        for field_name in (
            "motion_energy_min",
            "motion_energy_max",
            "visual_complexity_min",
            "visual_complexity_max",
            "semantic_drift_min",
            "semantic_drift_max",
            "style_similarity_min",
            "surrealism",
            "transition_smoothness",
            "palette_distance_max",
            "scene_boundary_strength_max",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1] (got {value})")
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

    @model_validator(mode="after")
    def values_in_domain(self) -> TransitionPlan:
        """Reject out-of-domain narrative knobs (issue 119).

        Why: a negative duration flows into timeline math and a wild
        strength into prompt staging — fail the parse, not the segment.
        """
        if not 0.0 <= self.transition_strength <= 1.0:
            raise ValueError(
                f"transition_strength must be within [0, 1] (got {self.transition_strength})"
            )
        if not math.isfinite(self.estimated_duration_seconds) or (
            self.estimated_duration_seconds < 0.0
        ):
            raise ValueError(
                "estimated_duration_seconds must be finite and >= 0 "
                f"(got {self.estimated_duration_seconds})"
            )
        return self


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

    @model_validator(mode="after")
    def values_in_domain(self) -> DirectorAudioPlan:
        """Reject out-of-domain music knobs (issue 119).

        Why: energy feeds `bpm_for_energy` and the fake worker's 0..1
        validator; a negative tempo flows toward ACE payloads — mirror the
        `AudioConfig.energy` 0..1 precedent at the contract layer.
        """
        if not 0.0 <= self.energy <= 1.0:
            raise ValueError(f"energy must be within [0, 1] (got {self.energy})")
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be positive (got {self.tempo_bpm})")
        return self


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
    music_style: str = DEFAULT_MUSIC_STYLE
    energy: float = 0.5
    seed: int = 0
    # Slow-loop takes (§35) serving this segment, oldest first. Empty for
    # runs committed before the slow loop existed.
    take_ids: list[str] = Field(default_factory=list)


class RunState(BaseModel):
    name: str
    status: LifecycleStatus = "CREATED"
    # Counters only ever advance from zero (validate_run relies on the
    # contiguous/non-negative invariant), so negatives are corrupt state,
    # not data — fail loud at the read_state boundary instead of letting
    # a torn write strand the run.
    next_segment_number: int = Field(default=0, ge=0)
    committed_segments: int = Field(default=0, ge=0)
    timeline_frames: int = Field(default=0, ge=0)
    current_concept: str = ""
    destination_concept: str = ""
    phase: TransitionPhase = "ESTABLISH"
    decision_index: int = Field(default=0, ge=0)


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

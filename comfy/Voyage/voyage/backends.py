"""Backend interface + spec-vocabulary adapter (DESIGN §§5.1, 14, 45-46).

The supervisor talks to *workers* over sync JSONL RPC (`generate_blocks`;
voyage/workers/loop.py, voyage/rpc.py); workers own backend objects.
Fake backends generate real tiny media with ffmpeg so the whole
persistence/validation/finalize path is exercised without GPUs.
Real LongLive / ACE-Step adapters implement the same interface in
their own worker environments later.

Stream C (config/interface duality): the merged spec sketches an async
`VideoBackend.generate_segment` with `segment_seconds` + `state_mode` +
per-backend `[video.*]` profiles, while the live wire is the sync
`generate_blocks` op and the live schema is
`VideoConfig(segment_frames, blocks_per_segment, quantization)`. The single
contract is `VideoBackendAdapter` below: spec vocabulary caller-side,
mapped onto the unchanged wire and schema. The adapter is sync on purpose
(the transport blocks on a select-deadline and DESIGN §46 forbids
concurrent GPU ops to one worker, so `async` adds no concurrency), and
per-backend profile blocks stay a follow-up schema migration.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, field_validator

from voyage.config import VideoConfig
from voyage.errors import ConfigurationError


class VideoBackend(Protocol):
    name: str

    def generate_segment(
        self,
        output_path: Path,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        fps: int,
        frames: int,
    ) -> dict[str, object]: ...


class AudioBackend(Protocol):
    name: str

    def generate_segment(
        self,
        output_path: Path,
        style: str,
        energy: float,
        seed: int,
        sample_rate: int,
        channels: int,
        duration_seconds: float,
    ) -> dict[str, object]: ...


StateMode = Literal["persistent_kv", "reconstructable_prefix", "independent_clip"]
"""Continuation-state vocabulary (DESIGN §5.1): how a backend resumes work."""

BACKEND_STATE_MODES: dict[str, StateMode] = {
    "fake": "independent_clip",
    "longlive2": "persistent_kv",
    "ltxv": "reconstructable_prefix",
    "causvid": "reconstructable_prefix",
}
"""Backend name → continuation mode (DESIGN §5.1 table; `fake` clips are stateless)."""

_STREAMING_BACKENDS = frozenset({"longlive2", "ltxv"})
"""Backends taking the multi-block payload (mirrors `supervisor.STREAMING_VIDEO_BACKENDS`)."""

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]
"""Worker call shape: `(op, payload) -> result` (matches `SubprocessWorker.call`)."""


def frames_for_segment_seconds(segment_seconds: float, fps: int) -> int:
    """Frames covering at least `segment_seconds` at `fps` (min 1).

    Rounds up so a duration request never runs short — the same philosophy
    as `cli.segments_for_duration` — with a 1e-9 epsilon so float dust on
    exact integers (e.g. 2.0 * 24) does not bill an extra frame.
    """
    if not math.isfinite(segment_seconds) or not segment_seconds > 0:
        raise ValueError(f"segment_seconds must be positive (got {segment_seconds!r})")
    if fps <= 0:
        raise ValueError(f"fps must be positive (got {fps!r})")
    return max(1, math.ceil(segment_seconds * fps - 1e-9))


def segment_seconds_for_frames(frames: int, fps: int) -> float:
    """Duration in seconds of `frames` at `fps` (reverse of the above)."""
    if frames <= 0:
        raise ValueError(f"frames must be positive (got {frames!r})")
    if fps <= 0:
        raise ValueError(f"fps must be positive (got {fps!r})")
    return frames / fps


class VideoSegmentRequest(BaseModel):
    """Caller-side segment vocabulary (DESIGN §5.1 target, adapted to sync).

    Geometry (`width`/`height`/`fps`) is authoritative per request; block
    structure (`blocks_per_segment`) comes from the stored `VideoConfig`
    held by the adapter. `state_mode` must match the backend's mode.
    """

    segment_id: str
    prompt: str
    seed: int
    width: int
    height: int
    fps: int
    segment_seconds: float
    state_mode: StateMode
    scene_cut: bool = False

    @field_validator("segment_id", "prompt")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("width", "height", "fps")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("segment_seconds")
    @classmethod
    def positive_seconds(cls, value: float) -> float:
        if not math.isfinite(value) or not value > 0:
            raise ValueError("must be positive")
        return value


class VideoSegmentResult(BaseModel):
    """Normalized segment outcome: worker actuals, never relabeled requests."""

    requested_frames: int
    returned_frames: int
    conditioning_frames: int
    novel_frames: int
    native_fps: int
    output_path: str
    backend: str
    state_mode: StateMode


class BackendCapabilities(BaseModel):
    """Supervisor-visible backend facts (spec `initialize -> capabilities`)."""

    backend: str
    state_mode: StateMode
    streaming: bool


class VideoBackendAdapter:
    """Spec vocabulary over the live `generate_blocks` wire op.

    Maps `VideoSegmentRequest` (segment_seconds/state_mode) onto the
    unchanged sync payloads — single-prompt form for `fake`, multi-block
    prompts/seeds/scene_cuts for streaming backends (mirroring
    `supervisor.commit_one_segment`) — and normalizes the worker result.
    Transport errors propagate untouched; the adapter never retries.
    """

    def __init__(
        self,
        transport: Transport,
        backend: str,
        video_config: VideoConfig,
        state_mode: StateMode | None = None,
    ) -> None:
        try:
            expected: StateMode = BACKEND_STATE_MODES[backend]
        except KeyError:
            known = ", ".join(sorted(BACKEND_STATE_MODES))
            raise ConfigurationError(
                f"unknown video backend {backend!r} (known: {known})"
            ) from None
        if state_mode is not None and state_mode != expected:
            raise ConfigurationError(
                f"backend {backend!r} uses state_mode {expected!r}, not {state_mode!r}"
            )
        self._transport = transport
        self._backend = backend
        self._video_config = video_config
        self._state_mode: StateMode = expected if state_mode is None else state_mode

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def state_mode(self) -> StateMode:
        return self._state_mode

    @property
    def streaming(self) -> bool:
        return self._backend in _STREAMING_BACKENDS

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self._backend, state_mode=self._state_mode, streaming=self.streaming
        )

    def requested_frames(self, request: VideoSegmentRequest) -> int:
        return frames_for_segment_seconds(request.segment_seconds, request.fps)

    def block_count(self) -> int:
        """Blocks per commit: config value for streaming backends, else 1."""
        if self.streaming:
            return max(1, self._video_config.blocks_per_segment)
        return 1

    def config_segment_seconds(self) -> float:
        """Stored config's segment length in seconds (frames/fps bridge)."""
        return segment_seconds_for_frames(self._video_config.segment_frames, self._video_config.fps)

    def build_payload(self, request: VideoSegmentRequest, output_path: Path) -> dict[str, Any]:
        """Pure mapping from spec request to the `generate_blocks` payload."""
        if request.state_mode != self._state_mode:
            raise ConfigurationError(
                f"backend {self._backend!r} uses state_mode {self._state_mode!r}, "
                f"not {request.state_mode!r}"
            )
        frames = self.requested_frames(request)
        payload: dict[str, Any] = {
            "segment_id": request.segment_id,
            "output_path": str(output_path),
            "width": request.width,
            "height": request.height,
            "fps": request.fps,
            "frames": frames,
        }
        if self.streaming:
            count = self.block_count()
            payload["prompts"] = [request.prompt] * count
            payload["seeds"] = [request.seed + index for index in range(count)]
            payload["scene_cuts"] = [request.scene_cut] + [False] * (count - 1)
        else:
            payload["prompt"] = request.prompt
            payload["seed"] = request.seed
        return payload

    def generate_segment(
        self, request: VideoSegmentRequest, output_path: Path
    ) -> VideoSegmentResult:
        """Build the payload, call the worker, normalize the result.

        Frame accounting mirrors the supervisor: worker-reported frames win,
        otherwise the request's frames stand in. Conditioning is 0 committed
        duplicates as built (ltxv chains via tail PNG, longlive appends to
        one stream — the proposal's prefix-overlap replay was not built).
        Worker-reported fps wins so a future 16 fps backend is never
        relabeled (TASK §19.5).
        """
        payload = self.build_payload(request, output_path)
        result = self._transport("generate_blocks", payload)
        requested = self.requested_frames(request)
        returned = requested
        native_fps = request.fps
        video_block = result.get("video")
        if isinstance(video_block, dict):
            reported_frames = video_block.get("frames")
            if isinstance(reported_frames, int) and reported_frames > 0:
                returned = reported_frames
            reported_fps = video_block.get("fps")
            if isinstance(reported_fps, int) and reported_fps > 0:
                native_fps = reported_fps
        return VideoSegmentResult(
            requested_frames=requested,
            returned_frames=returned,
            conditioning_frames=0,
            novel_frames=returned,
            native_fps=native_fps,
            output_path=str(output_path),
            backend=self._backend,
            state_mode=self._state_mode,
        )

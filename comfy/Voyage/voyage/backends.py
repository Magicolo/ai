"""Backend interface + spec-vocabulary adapter (DESIGN §§5.1, 14, 45-46).

The supervisor talks to *workers* over sync JSONL RPC (`generate_blocks`;
voyage/workers/loop.py, voyage/rpc.py); workers own backend objects.
Fake backends generate real tiny media with ffmpeg so the whole
persistence/validation/finalize path is exercised without GPUs.
Real ACE-Step adapters implement the same interface in
their own worker environments later.

Stream C (config/interface duality): the merged spec sketches an async
`VideoBackend.generate_segment` with `segment_seconds` + `state_mode` +
per-backend `[video.*]` profiles, while the live wire is the sync
`generate_blocks` op and the live schema is
`VideoConfig(segment_frames, blocks_per_segment, quantization)`. The single
contract is `VideoSegmentRequest` + `VideoBackendAdapter.generate_segment`
below: spec vocabulary caller-side, mapped onto the unchanged wire and
schema. The adapter is sync on purpose (the transport blocks on a
select-deadline and DESIGN §46 forbids concurrent GPU ops to one worker,
so `async` adds no concurrency), and per-backend profile blocks stay a
follow-up schema migration. The streaming-vs-fake payload fork lives
inside the adapter (build_payload) so there is exactly one copy; the
supervisor wiring hook is transport_from_restarting_call (see its
docstring — the supervisor track owns the rewire, issue 023).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, field_validator

from voyage.config import (
    BACKEND_REGISTRY,
    StateMode,
    VideoBackendName,
    VideoConfig,
)
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


BACKEND_STATE_MODES: dict[VideoBackendName, StateMode] = {
    name: record.state_mode for name, record in BACKEND_REGISTRY.items()
}
"""Backend name → continuation mode (DESIGN §5.1 table; `fake` clips are stateless).

Derived view of config.BACKEND_REGISTRY (issue 022) — kept under this
name for the adapter and existing tests, never a second source."""

_STREAMING_BACKENDS: frozenset[VideoBackendName] = frozenset(
    name for name, record in BACKEND_REGISTRY.items() if record.streaming
)
"""Backends taking the multi-block payload (mirrors `supervisor.STREAMING_VIDEO_BACKENDS`).

Derived from config.BACKEND_REGISTRY (issue 022): the supervisor track
owns unifying this with supervisor.STREAMING_VIDEO_BACKENDS.
"""

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]
"""Worker call shape: `(op, payload) -> result` (matches `SubprocessWorker.call`)."""

FLOAT_DUST_EPSILON = 1e-9
"""Epsilon keeping exact second boundaries on their frame count.

`segment_seconds * fps` for exact durations (e.g. 2.0 * 24) can land a dust
above the integer in float64; without the epsilon `ceil` bills one frame too
many. The value only absorbs representation dust — real fractional requests
still round up so a duration never runs short.
"""


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
    return max(1, math.ceil(segment_seconds * fps - FLOAT_DUST_EPSILON))


def segment_seconds_for_frames(frames: int, fps: int) -> float:
    """Duration in seconds of `frames` at `fps` (reverse of the above).

    The stored-schema bridge: `VideoConfig` persists `segment_frames` while
    callers speak `segment_seconds`, and this direction never rounds — it is
    the exact quotient the forward mapping rounds up to.
    """
    if frames <= 0:
        raise ValueError(f"frames must be positive (got {frames!r})")
    if fps <= 0:
        raise ValueError(f"fps must be positive (got {fps!r})")
    return frames / fps


class VideoSegmentRequest(BaseModel):
    """Caller-side segment vocabulary (DESIGN §5.1 target, adapted to sync).

    THE segment contract (issue 023): the supervisor builds one of these
    per commit (see VideoBackendAdapter.request_from_config) and the
    adapter maps it onto the wire. Geometry (`width`/`height`/`fps`) is
    authoritative per request; block structure (`blocks_per_segment`)
    comes from the stored `VideoConfig` held by the adapter. `state_mode`
    must match the backend's mode. Streaming backends take per-block
    prompt/seed sequences when blocks differ (the staged prompt plan);
    omit them to repeat one prompt/seed across blocks.
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
    block_prompts: list[str] | None = None
    block_seeds: list[int] | None = None

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
    backend: VideoBackendName
    state_mode: StateMode


class BackendCapabilities(BaseModel):
    """Supervisor-visible backend facts (spec `initialize -> capabilities`)."""

    backend: VideoBackendName
    state_mode: StateMode
    streaming: bool


class VideoBackendAdapter:
    """Spec vocabulary over the live `generate_blocks` wire op.

    THE video dispatch contract (issue 023): callers build a
    VideoSegmentRequest (see request_from_config) and call
    generate_segment with a Transport. The adapter maps the request onto
    the unchanged sync payloads — single-prompt form for `fake`,
    multi-block prompts/seeds/scene_cuts for streaming backends — and
    normalizes the worker result. Transport errors propagate untouched;
    the adapter never retries (retries stay in the supervisor's
    _call_with_restart; see transport_from_restarting_call).
    """

    def __init__(
        self,
        transport: Transport,
        backend: VideoBackendName,
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
        self._backend: VideoBackendName = backend
        self._video_config = video_config
        self._state_mode: StateMode = expected if state_mode is None else state_mode

    @staticmethod
    def request_from_config(
        video_config: VideoConfig,
        *,
        segment_id: str,
        prompt: str,
        seed: int,
        scene_cut: bool = False,
        segment_seconds: float | None = None,
        block_prompts: list[str] | None = None,
        block_seeds: list[int] | None = None,
    ) -> VideoSegmentRequest:
        """Build the contract request from the stored video config.

        Geometry and the seconds bridge (segment_frames/fps) come from the
        config; per-block prompt/seed sequences ride along when the caller
        stages them (the supervisor's staged prompt plan maps here).
        Unknown backends raise ConfigurationError (unreachable through a
        validated VideoConfig — the Literal rejects them first — but open
        to unvalidated construction).
        """
        try:
            state_mode: StateMode = BACKEND_STATE_MODES[video_config.backend]
        except KeyError:
            known = ", ".join(sorted(BACKEND_STATE_MODES))
            raise ConfigurationError(
                f"unknown video backend {video_config.backend!r} (known: {known})"
            ) from None
        seconds = (
            segment_seconds
            if segment_seconds is not None
            else segment_seconds_for_frames(video_config.segment_frames, video_config.fps)
        )
        return VideoSegmentRequest(
            segment_id=segment_id,
            prompt=prompt,
            seed=seed,
            width=video_config.width,
            height=video_config.height,
            fps=video_config.fps,
            segment_seconds=seconds,
            state_mode=state_mode,
            scene_cut=scene_cut,
            block_prompts=list(block_prompts) if block_prompts is not None else None,
            block_seeds=list(block_seeds) if block_seeds is not None else None,
        )

    @property
    def backend(self) -> VideoBackendName:
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
        """Frames the request covers at its fps (forward seconds bridge)."""
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
        """Pure mapping from spec request to the `generate_blocks` payload.

        The streaming-vs-fake fork lives here and only here (issue 023):
        streaming backends get the multi-block prompts/seeds/scene_cuts
        lists (per-block sequences when the request stages them, else one
        prompt/seed repeated), fake gets the single-prompt form.
        """
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
            prompts = (
                list(request.block_prompts)
                if request.block_prompts is not None
                else [request.prompt] * count
            )
            seeds = (
                list(request.block_seeds)
                if request.block_seeds is not None
                else [request.seed + index for index in range(count)]
            )
            if len(prompts) != count:
                raise ConfigurationError(
                    f"backend {self._backend!r} commits {count} blocks, "
                    f"but got {len(prompts)} block prompts"
                )
            if len(seeds) != count:
                raise ConfigurationError(
                    f"backend {self._backend!r} commits {count} blocks, "
                    f"but got {len(seeds)} block seeds"
                )
            payload["prompts"] = prompts
            payload["seeds"] = seeds
            payload["scene_cuts"] = [request.scene_cut] + [False] * (count - 1)
        else:
            if request.block_prompts is not None or request.block_seeds is not None:
                raise ConfigurationError(
                    f"backend {self._backend!r} is not streaming: "
                    "block_prompts/block_seeds do not apply"
                )
            payload["prompt"] = request.prompt
            payload["seed"] = request.seed
        return payload

    def generate_segment(
        self, request: VideoSegmentRequest, output_path: Path
    ) -> VideoSegmentResult:
        """Build the payload, call the worker, normalize the result.

        Frame accounting mirrors the supervisor: worker-reported frames win,
        otherwise the request's frames stand in. When the worker reports
        explicit novel/conditioning counts (causvid's 72-novel accounting),
        those win; otherwise conditioning is 0 committed duplicates as built
        (ltxv chains via tail PNG — the proposal's prefix-overlap replay
        was not built).
        Worker-reported fps wins so a future 16 fps backend is never
        relabeled (TASK §19.5).
        """
        payload = self.build_payload(request, output_path)
        result = self._transport("generate_blocks", payload)
        requested = self.requested_frames(request)
        returned = requested
        conditioning = 0
        novel: int | None = None
        native_fps = request.fps
        video_block = result.get("video")
        if isinstance(video_block, dict):
            reported_frames = video_block.get("frames")
            if isinstance(reported_frames, int) and reported_frames > 0:
                returned = reported_frames
            reported_conditioning = video_block.get("conditioning_frames")
            if isinstance(reported_conditioning, int) and reported_conditioning >= 0:
                conditioning = reported_conditioning
            reported_novel = video_block.get("novel_frames")
            if isinstance(reported_novel, int) and reported_novel > 0:
                novel = reported_novel
            reported_fps = video_block.get("fps")
            if isinstance(reported_fps, int) and reported_fps > 0:
                native_fps = reported_fps
        return VideoSegmentResult(
            requested_frames=requested,
            returned_frames=returned,
            conditioning_frames=conditioning,
            novel_frames=novel if novel is not None else returned,
            native_fps=native_fps,
            output_path=str(output_path),
            backend=self._backend,
            state_mode=self._state_mode,
        )


def transport_from_restarting_call(
    restarting_call: Callable[..., dict[str, Any]],
    worker: object,
    worker_name: str,
    segment_id: str,
    restart_hook: Callable[[str], None] | None = None,
) -> Transport:
    """Adapt a supervisor-style restarting call to the Transport shape.

    Injection point for the supervisor wiring (issue 023): pass the bound
    `Supervisor._call_with_restart`, the video worker, `"video"`, the
    segment id, and the resume hook for streaming backends
    (`supervisor._resume_video_worker`, else None). The adapter then owns
    payload + frame accounting while retries, restarts, and resume stay in
    the supervisor — the adapter itself never retries.
    """

    def transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        return restarting_call(
            worker, worker_name, segment_id, operation, payload, restart_hook=restart_hook
        )

    return transport

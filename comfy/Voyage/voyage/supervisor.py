"""Supervisor: lifecycle state machine + transactional segment commit.

Owns lifecycle and commit state (DESIGN §73). The director proposes,
the supervisor validates and commits. One segment commit:

  1. director decide (via worker) → EvolutionDecision
  2. novelty check → accept/reject (rejections recorded, never deleted)
  3. prompt plan → video generate_blocks → audio generate_audio
  4. validate media → write metadata (.partial + fsync + rename)
  5. checksums → DONE (.partial + fsync + rename) → state.json update
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from voyage import paths
from voyage.atomic import atomic_write_bytes, atomic_write_json
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig
from voyage.errors import (
    ConfigurationError,
    DiskSpaceError,
    FatalWorkerError,
    MediaError,
    RecoverableWorkerError,
    VoyageError,
)
from voyage.media import validate_audio, validate_video
from voyage.models import AudioPlan, EvolutionDecision, SegmentWorldState
from voyage.persistence import read_state, write_state
from voyage.prompts import build_prompt_plan
from voyage.rpc import SubprocessWorker
from voyage.seeds import audio_seed, video_seed


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_free_space(run_dir: Path, min_free_gib: float) -> float:
    free_gib = shutil.disk_usage(run_dir).free / (1024**3)
    if free_gib < min_free_gib:
        raise DiskSpaceError(f"free space {free_gib:.1f} GiB below reserve {min_free_gib:.1f} GiB")
    return free_gib


VIDEO_WORKER_MODULES = {
    "fake": "voyage.workers.video",
    "longlive2": "voyage.workers.video_longlive",
}
"""Backend name → worker module. longlive2 only exists in the CUDA image."""


def video_worker_module(backend: str) -> str:
    try:
        return VIDEO_WORKER_MODULES[backend]
    except KeyError:
        raise ConfigurationError(
            f"unknown video backend {backend!r} (known: {sorted(VIDEO_WORKER_MODULES)})"
        ) from None


class Supervisor:
    def __init__(self, run_dir: Path, config: ProjectConfig) -> None:
        self._run_dir = run_dir
        self._config = config
        self._logs = run_dir / paths.LOGS_DIRNAME
        video_module = video_worker_module(config.video.backend)
        video_init: dict[str, Any] = {}
        if config.video.backend == "longlive2":
            video_init = {
                "models_dir": config.video.models_dir,
                "device": config.video.device,
                "latent_shape": list(config.video.latent_shape),
            }
        self._video = SubprocessWorker(
            video_module,
            run_dir,
            self._logs / "video-worker.log",
            init_payload=video_init,
        )
        self._audio = SubprocessWorker(
            "voyage.workers.audio", run_dir, self._logs / "audio-worker.log"
        )
        self._director = SubprocessWorker(
            "voyage.workers.director", run_dir, self._logs / "director-worker.log"
        )
        self._workers_running = False
        self._stop_flag = False

    def request_stop(self) -> None:
        """Ask the run loop to exit after the current segment (SIGINT path)."""
        self._stop_flag = True

    def inject_worker_crash(self, worker_name: str) -> int:
        """SIGKILL one worker without cleanup (crash-injection hook, §69).

        Leaves broken pipes behind on purpose: the next RPC must raise
        RecoverableWorkerError so the restart path engages. Returns the
        killed pid. Reserved for tests/chaos — never used by the run loop.
        """
        workers = {
            "video": self._video,
            "audio": self._audio,
            "director": self._director,
        }
        worker = workers[worker_name]
        pid = worker.pid
        if pid is None:
            raise FatalWorkerError(f"worker {worker_name} is not running")
        os.kill(pid, signal.SIGKILL)
        return pid

    def start_workers(self) -> None:
        self._logs.mkdir(parents=True, exist_ok=True)
        self._video.start()
        self._audio.start()
        self._director.start()
        self._workers_running = True

    def stop_workers(self) -> None:
        self._video.stop()
        self._audio.stop()
        self._director.stop()
        self._workers_running = False

    def _log_metric(self, event: dict[str, object]) -> None:
        line = json.dumps({"ts": time.time(), **event})
        metrics_path = self._logs / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _call_with_restart(
        self,
        worker: SubprocessWorker,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """One RPC call with a single worker restart on recoverable failure.

        A second consecutive failure is recorded and re-raised: the run loop
        decides (fatal → FAILED state, anything else → last_error + abort).
        """
        try:
            return worker.call(op, dict(payload))
        except RecoverableWorkerError as exc:
            self._log_metric(
                {
                    "event": "worker_restart",
                    "worker": worker_name,
                    "op": op,
                    "segment_id": segment_id,
                    "reason": str(exc),
                }
            )
            worker.restart()
            return worker.call(op, dict(payload))

    def _pause_requested(self) -> bool:
        """Honor an external `voyage pause`: transition to PAUSED and exit."""
        state = read_state(self._run_dir)
        if state.status != "PAUSE_REQUESTED":
            return False
        state.status = "PAUSED"
        write_state(self._run_dir, state)
        return True

    def _stop_requested(self) -> bool:
        if self._stop_flag:
            return True
        return read_state(self._run_dir).status == "STOP_REQUESTED"

    def run_segments(self, count: int | None) -> list[str]:
        """Generate segments until `count` commits or a pause/stop arrives.

        `count=None` runs indefinitely (the autonomous voyage): each loop
        iteration re-reads state.json so `voyage pause` / `voyage stop` from
        another process — or SIGINT via `request_stop()` — takes effect at
        the next segment boundary. Returns committed segment ids.
        """
        self.start_workers()
        try:
            state = read_state(self._run_dir)
            if state.status in ("PAUSE_REQUESTED", "STOP_REQUESTED"):
                # A request that arrived before startup wins over RUNNING.
                if state.status == "PAUSE_REQUESTED":
                    state.status = "PAUSED"
                    write_state(self._run_dir, state)
                return []
            state.status = "RUNNING"
            write_state(self._run_dir, state)
            committed: list[str] = []
            stopped = False
            while count is None or len(committed) < count:
                if self._stop_requested():
                    stopped = True
                    break
                if self._pause_requested():
                    break
                try:
                    segment_id = self.commit_one_segment()
                except VoyageError as exc:
                    failed = read_state(self._run_dir)
                    failed.last_error = str(exc)
                    if isinstance(exc, FatalWorkerError):
                        failed.status = "FAILED"
                    write_state(self._run_dir, failed)
                    self._log_metric({"event": "segment_commit_failed", "error": str(exc)})
                    raise
                committed.append(segment_id)
            if stopped and self._stop_flag and not self._stop_requested_via_file():
                # SIGINT path: rest as PAUSED so `voyage run` resumes cleanly.
                resting = read_state(self._run_dir)
                resting.status = "PAUSED"
                write_state(self._run_dir, resting)
            elif not self._stop_requested() and not self._pause_requested():
                # Finite batch completed without external requests.
                resting = read_state(self._run_dir)
                resting.status = "PAUSED"
                write_state(self._run_dir, resting)
            return committed
        finally:
            self.stop_workers()

    def _stop_requested_via_file(self) -> bool:
        return read_state(self._run_dir).status == "STOP_REQUESTED"

    def commit_one_segment(self) -> str:
        if not self._workers_running:
            raise FatalWorkerError("commit_one_segment requires start_workers() first")
        started = time.monotonic()
        config = self._config
        state = read_state(self._run_dir)
        check_free_space(self._run_dir, config.min_free_space_gib)

        number = state.next_segment_number
        segment_id = paths.format_segment_id(number)
        segment = paths.segment_dir(self._run_dir, segment_id)
        segment.mkdir(parents=True, exist_ok=True)

        # 1. Director proposal (validated schema; never writes state itself).
        raw = self._call_with_restart(
            self._director,
            "director",
            segment_id,
            "decide",
            {
                "decision_index": state.decision_index,
                "current_concept": state.current_concept,
                "destination_concept": state.destination_concept,
                "phase": state.phase,
                "style": config.style,
            },
        )
        decision = EvolutionDecision.model_validate(raw)

        # 2. Novelty check (rejections recorded in immutable history).
        store = ConceptStore(self._run_dir / paths.CONCEPTS_FILENAME)
        _record, _score = store.propose(decision.destination_concept)

        # 3. Prompt plan + media generation.
        prompt_plan = build_prompt_plan(
            segment_id,
            style=config.style,
            concept=decision.destination_concept,
            phase=decision.phase,
            block_starts=[0],
            block_ends=[0],
        )
        video_out = segment / "video.mp4"
        audio_out = segment / "audio.wav"
        video_result = self._call_with_restart(
            self._video,
            "video",
            segment_id,
            "generate_blocks",
            {
                "segment_id": segment_id,
                "prompt": prompt_plan.stages[0].prompt,
                "seed": video_seed(config.seed, number, 0),
                "output_path": str(video_out),
                "width": config.video.width,
                "height": config.video.height,
                "fps": config.video.fps,
                "frames": config.video.segment_frames,
            },
        )
        # Truthful frame accounting: the worker reports what it rendered
        # (longlive's decoded count depends on the VAE chunking, not the
        # request), so the timeline always matches reality.
        frames = config.video.segment_frames
        video_block = video_result.get("video")
        if isinstance(video_block, dict):
            reported = video_block.get("frames")
            if isinstance(reported, int) and reported > 0:
                frames = reported
        duration = frames / config.video.fps
        audio_plan = AudioPlan(
            segment_id=segment_id,
            music_style=config.audio.music_style,
            energy=config.audio.energy,
            seed=audio_seed(config.seed, number, 0),
        )
        self._call_with_restart(
            self._audio,
            "audio",
            segment_id,
            "generate_audio",
            {
                "segment_id": segment_id,
                "style": audio_plan.music_style,
                "energy": audio_plan.energy,
                "seed": audio_plan.seed,
                "output_path": str(audio_out),
                "sample_rate": config.audio.sample_rate,
                "channels": config.audio.channels,
                "duration_seconds": duration,
            },
        )

        # 4. Validate before anything claims the segment is committed.
        video_info = validate_video(
            video_out, config.video.width, config.video.height, config.video.fps
        )
        audio_info = validate_audio(audio_out, config.audio.sample_rate, config.audio.channels)
        if abs(float(video_info["duration"]) - duration) > 0.6:
            raise MediaError(f"segment {segment_id} A/V duration drift")

        # 5. Metadata → checksums → DONE → state. No state file may claim
        # the segment is committed until artifacts are valid and durable.
        world = SegmentWorldState(
            segment_id=segment_id,
            current_concept=state.current_concept,
            destination_concept=decision.destination_concept,
            phase=decision.phase,
            seed=config.seed,
        )
        atomic_write_json(segment / "world_state.json", world.model_dump())
        atomic_write_json(segment / "transition.json", decision.model_dump())
        atomic_write_json(segment / "prompt_plan.json", prompt_plan.model_dump())
        atomic_write_json(segment / "audio_state.json", audio_plan.model_dump())
        atomic_write_json(
            segment / "metrics.json",
            {"video": video_info, "audio": audio_info, "frames": frames},
        )
        atomic_write_json(
            segment / "sha256.json",
            {"video.mp4": sha256_file(video_out), "audio.wav": sha256_file(audio_out)},
        )
        done_partial = segment / "DONE.partial"
        atomic_write_bytes(done_partial, b"")
        done_partial.replace(segment / paths.DONE_MARKER)

        # 6. Supervisor-owned state advance (single writer).
        fresh = read_state(self._run_dir)
        fresh.next_segment_number = number + 1
        fresh.committed_segments += 1
        fresh.timeline_frames += frames
        fresh.current_concept = decision.destination_concept
        fresh.destination_concept = decision.destination_concept
        fresh.phase = decision.phase
        fresh.decision_index = state.decision_index + 1
        fresh.audio_buffer_seconds = duration
        fresh.last_error = None
        write_state(self._run_dir, fresh)
        self._log_metric(
            {
                "event": "segment_committed",
                "segment_id": segment_id,
                "frames": frames,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        return segment_id

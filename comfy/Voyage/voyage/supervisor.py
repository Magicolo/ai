"""Supervisor: lifecycle state machine + transactional segment commit.

Owns lifecycle and commit state (DESIGN §73). The director proposes,
the supervisor validates and commits. One segment commit:

  1. director decide (via worker) → EvolutionDecision (schema-validated)
  2. novelty check (bounded retries, then deterministic fallback) → accept/reject
  3. style check (code-level, §18.1) → staged prompt plan (§18.2)
  4. video generate_blocks → audio generate_audio
  5. validate media → write metadata (.partial + fsync + rename)
  6. checksums → DONE (.partial + fsync + rename) → state.json update
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from voyage import paths
from voyage.atomic import atomic_write_bytes, atomic_write_json, fsync_dir
from voyage.audio.planner import TAKES_FILENAME, AudioPlanner, append_take, load_takes
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig
from voyage.director import (
    DeterministicDirector,
    director_input_from_state,
    format_measured_context,
)
from voyage.errors import (
    ConfigurationError,
    DiskSpaceError,
    FatalWorkerError,
    MediaError,
    ProposalRejected,
    RecoverableWorkerError,
    VoyageError,
)
from voyage.logrotate import append_line
from voyage.media import (
    AV_ALIGNMENT_TOLERANCE_SECONDS,
    assemble_segment_audio,
    check_free_space,
    probe,
    run_capture,
    slice_take,
    validate_audio,
    validate_video,
)
from voyage.models import AudioPlan, EvolutionDecision, SegmentWorldState, StyleSpec
from voyage.persistence import read_state, write_state
from voyage.prompts import (
    apply_feedback_amendments,
    build_staged_prompt_plan,
    check_prompt_against_style,
    feedback_amendments,
)
from voyage.rpc import SubprocessWorker
from voyage.seeds import audio_seed, video_seed
from voyage.vision.metrics import (
    Histogram,
    frame_histogram,
    sample_frames,
    summarize_segment,
)


def sha256_file(path: Path) -> str:
    """Chunked SHA-256 (constant memory — takes can be multi-GB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


VIDEO_WORKER_MODULES = {
    "fake": "voyage.workers.video",
    "longlive2": "voyage.workers.video_longlive",
    "ltxv": "voyage.workers.video_ltxv",
}
"""Backend name → worker module. longlive2/ltxv only exist in the CUDA image."""

STREAMING_VIDEO_BACKENDS = ("longlive2", "ltxv")
"""Backends whose worker holds a resident session across blocks/segments.

These get the multi-block prompts/seeds payload, the resume-hook restart
path, and the acestep audio GPU swap (their DiT is GPU-resident, so audio
must evict + rebuild around takes). Fake renders statelessly per segment.
"""

AUDIO_WORKER_MODULES = {
    "fake": "voyage.workers.audio",
    "acestep": "voyage.workers.audio_acestep",
}
"""Backend name → worker module. acestep only exists in the GPU image."""


def audio_worker_module(backend: str) -> str:
    try:
        return AUDIO_WORKER_MODULES[backend]
    except KeyError:
        raise ConfigurationError(
            f"unknown audio backend {backend!r} (known: {sorted(AUDIO_WORKER_MODULES)})"
        ) from None


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
        if config.video.backend in STREAMING_VIDEO_BACKENDS:
            video_init = {
                "models_dir": config.video.models_dir,
                "device": config.video.device,
            }
        if config.video.backend == "longlive2":
            video_init.update(
                {
                    "latent_shape": list(config.video.latent_shape),
                    "quantization": config.video.quantization,
                    "local_attn_size": config.video.local_attn_size,
                }
            )
        self._video = SubprocessWorker(
            video_module,
            run_dir,
            self._logs / "video-worker.log",
            init_payload=video_init,
            timeout=config.voyage.rpc_timeout_seconds,
        )
        audio_module = audio_worker_module(config.audio.backend)
        audio_init: dict[str, Any] = {}
        if config.audio.backend == "acestep":
            audio_init = {
                "models_dir": config.audio.models_dir,
                "device": config.audio.device,
            }
        self._audio = SubprocessWorker(
            audio_module,
            run_dir,
            self._logs / "audio-worker.log",
            init_payload=audio_init,
            timeout=config.voyage.rpc_timeout_seconds,
        )
        self._director = SubprocessWorker(
            "voyage.workers.director",
            run_dir,
            self._logs / "director-worker.log",
            timeout=config.voyage.rpc_timeout_seconds,
        )
        self._workers_running = False
        self._stop_flag = False
        # Parallel director prefetch (§20): while segment N renders video
        # (GPU) + audio, one background thread pre-computes the raw LLM
        # proposal for N+1 on CPU. The commit path still owns validation →
        # novelty → style → accept, so a stale/slow proposal degrades to a
        # synchronous decide — never corrupt state. Keyed by target segment
        # number; results for any other number are discarded.
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_target: int | None = None
        self._prefetch_future: Future[dict[str, Any] | None] | None = None
        # Restarts used per worker since the current run started (Phase 6
        # slice B budget). Reset by run_segments; direct commit_one_segment
        # callers share the counters for the supervisor's lifetime.
        self._restarts: dict[str, int] = {}

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
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

    def stop_workers(self) -> None:
        self._video.stop()
        self._audio.stop()
        self._director.stop()
        self._workers_running = False
        executor, self._prefetch_executor = self._prefetch_executor, None
        self._prefetch_target = None
        self._prefetch_future = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _log_metric(self, event: dict[str, object]) -> None:
        line = json.dumps({"ts": time.time(), "run_id": self._config.run_id, **event})
        append_line(self._logs / "metrics.jsonl", line)

    def _call_with_restart(
        self,
        worker: SubprocessWorker,
        worker_name: str,
        segment_id: str,
        op: str,
        payload: dict[str, object],
        restart_hook: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        """One RPC call with worker restarts on recoverable failure.

        At most `config.voyage.max_worker_restarts` restarts per worker per
        run; past that the circuit breaker opens (FatalWorkerError → the
        run rests at FAILED). `restart_hook` (e.g. video recovery-tape
        resume) runs after the restart replays `init`, before the retried
        call — and shares the same budget when it retries internally.
        """
        budget = self._config.voyage.max_worker_restarts
        while True:
            try:
                return worker.call(op, dict(payload))
            except RecoverableWorkerError as exc:
                used = self._restarts.get(worker_name, 0)
                if used >= budget:
                    self._log_metric(
                        {
                            "event": "circuit_breaker_open",
                            "worker": worker_name,
                            "op": op,
                            "segment_id": segment_id,
                            "restarts_used": used,
                            "budget": budget,
                            "reason": str(exc),
                        }
                    )
                    raise FatalWorkerError(
                        f"circuit breaker open for {worker_name}/{op}: "
                        f"{used} restarts exhausted ({exc})"
                    ) from exc
                self._restarts[worker_name] = used + 1
                self._log_metric(
                    {
                        "event": "worker_restart",
                        "worker": worker_name,
                        "op": op,
                        "segment_id": segment_id,
                        "attempt": used + 1,
                        "budget": budget,
                        "reason": str(exc),
                    }
                )
                worker.restart()
                if restart_hook is not None:
                    restart_hook(segment_id)
                # Loop: the retried call runs at the top. A FatalWorkerError
                # from the hook (e.g. its own budget exhausted) is not
                # Recoverable, so it propagates without further retries.

    def _latest_recovery_tape(self) -> Path | None:
        """Newest committed segment's recovery.pt, or None (DESIGN §27)."""
        segments_root = self._run_dir / paths.SEGMENTS_DIRNAME
        if not segments_root.exists():
            return None
        for segment in sorted(
            (p for p in segments_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        ):
            if (segment / paths.DONE_MARKER).exists():
                tape = segment / "recovery.pt"
                if tape.exists():
                    return tape
        return None

    def _resume_video_worker(self, segment_id: str) -> None:
        """Rebuild video causal context from the latest tape (DESIGN §27.1).

        Runs after a video worker restart: replays the tail latents at clean
        timestep=0 so the retried block continues the stream instead of
        starting a fresh one. No tape (first segment) → fresh stream is
        correct — nothing to resume. The resume call itself retries through
        the shared restart budget, so a resume failure gets a second chance
        instead of aborting the segment at once.
        """
        tape = self._latest_recovery_tape()
        if tape is None:
            return
        result = self._call_with_restart(
            self._video, "video", segment_id, "resume", {"recovery_path": str(tape)}
        )
        self._log_metric({"event": "video_resumed", "tape": str(tape), **result})

    def _sample_gauges(self, segment_id: str) -> None:
        """Best-effort resource snapshot after a commit (Phase 6 slice E).

        Never fails the commit: any probe error degrades to a skipped
        event. Worker health is polled directly (no restart) so a sick
        worker shows up as missing fields, not a recovery.
        """
        try:
            import resource
            import shutil

            gauges: dict[str, object] = {
                "event": "resource_gauges",
                "segment_id": segment_id,
                "disk_free_gib": round(shutil.disk_usage(self._run_dir).free / 1024**3, 2),
                "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            }
            for name, worker in (
                ("video", self._video),
                ("audio", self._audio),
                ("director", self._director),
            ):
                try:
                    health = worker.call("health", {})
                except VoyageError:
                    continue
                for key in ("vram_free_gib", "vram_total_gib"):
                    value = health.get(key)
                    if isinstance(value, (int, float)):
                        gauges[f"{name}_{key}"] = value
            self._log_metric(gauges)
        except Exception:
            pass

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
            self._restarts = {}
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
                    if isinstance(exc, DiskSpaceError):
                        # Free the operator to clear space and `run` again:
                        # the next commit precheck re-pauses if still full.
                        failed.status = "PAUSED_DISK_FULL"
                    else:
                        # Any abort leaves FAILED — resting at RUNNING after
                        # a twice-failed recoverable op misled operators.
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

    def _prefetch_decide_for_next(
        self,
        config: ProjectConfig,
        number: int,
        decision: EvolutionDecision,
        store: ConceptStore,
        style_spec: StyleSpec,
    ) -> None:
        """Submit a background raw LLM proposal for segment number+1.

        The speculative next-state derives entirely from the just-accepted
        decision (current/destination = its destination, index + 1), so no
        commit output is needed. Only the raw worker reply is prefetched —
        validation, novelty, style and store writes stay on the commit
        path. Failures (including worker restarts, which stay synchronous)
        degrade to None: the next commit decides synchronously.
        """
        executor = self._prefetch_executor
        if executor is None or not self._workers_running:
            return
        if self._prefetch_future is not None and not self._prefetch_future.done():
            return
        target = number + 1
        self._prefetch_target = target
        spec_state = SimpleNamespace(
            decision_index=decision.decision_index + 1,
            phase=decision.phase,
            current_concept=decision.destination_concept,
            destination_concept=decision.destination_concept,
            committed_segments=0,
            timeline_frames=0,
        )
        try:
            payload = self._decide_payload(config, spec_state, store, style_spec)
        except VoyageError:
            self._prefetch_target = None
            return

        def _call() -> dict[str, Any] | None:
            try:
                return self._director.call("decide", payload)
            except VoyageError:
                return None

        self._prefetch_future = executor.submit(_call)

    def _take_prefetch(self, number: int, segment_id: str) -> dict[str, Any] | None:
        """Consume the prefetched raw proposal when it targets this segment.

        Hit = future done with a dict result (used as the accept loop's
        first candidate, still fully validated). Anything else is a miss:
        the next commit decides synchronously. Stale targets are dropped.
        """
        future, target = self._prefetch_future, self._prefetch_target
        self._prefetch_future = None
        self._prefetch_target = None
        if future is None or target != number:
            self._log_metric({"event": "director_prefetch_miss", "segment_id": segment_id})
            return None
        if not future.done():
            self._log_metric({"event": "director_prefetch_miss", "segment_id": segment_id})
            return None
        try:
            raw = future.result()
        except Exception:  # noqa: BLE001 — prefetch must never break a commit
            self._log_metric({"event": "director_prefetch_miss", "segment_id": segment_id})
            return None
        if not isinstance(raw, dict):
            self._log_metric({"event": "director_prefetch_miss", "segment_id": segment_id})
            return None
        self._log_metric({"event": "director_prefetch_hit", "segment_id": segment_id})
        return raw

    def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Embed via the director worker; None when unavailable (fallback)."""
        try:
            result = self._director.call("embed", {"texts": texts})
        except VoyageError:
            return None
        vectors = result.get("vectors")
        if not isinstance(vectors, list):
            return None
        cleaned: list[list[float]] = []
        for row in vectors:
            if not isinstance(row, list):
                return None
            cleaned.append([float(value) for value in row])
        return cleaned

    def _decide_payload(
        self,
        config: ProjectConfig,
        state: Any,
        store: ConceptStore,
        style_spec: StyleSpec,
        retry_feedback: str = "",
        measured_context: str = "",
    ) -> dict[str, Any]:
        history = store.history_texts()
        payload = director_input_from_state(
            state,
            style_charter=style_spec.prompt,
            recent_summary="; ".join(history[-5:]) if history else "(no concepts yet)",
            forbidden_summary=(
                "; ".join(history[-20:])
                if not config.voyage.allow_concept_revisit and history
                else "(revisits allowed)"
            ),
            audio_state=(f"style={config.audio.music_style} energy={config.audio.energy}"),
            measured_context=measured_context,
        )
        payload.update(
            {
                "decision_index": state.decision_index,
                "phase": state.phase,
                # Flat fields for the deterministic backend (backward compat).
                "current_concept": state.current_concept,
                "destination_concept": state.destination_concept,
                "style": style_spec.prompt,
                "backend": config.director.backend,
                "model_id": config.director.model_id,
                "temperature": config.director.temperature,
                "max_new_tokens": config.director.max_new_tokens,
                "enable_thinking": config.director.enable_thinking,
                "retry_feedback": retry_feedback,
            }
        )
        return payload

    def _accept_director_decision(
        self,
        config: ProjectConfig,
        state: Any,
        store: ConceptStore,
        style_spec: StyleSpec,
        segment_id: str,
        measured_context: str = "",
        amendments: list[str] | None = None,
        prefetched_raw: dict[str, Any] | None = None,
    ) -> EvolutionDecision:
        """§74 transaction: validate → novelty → style → accept.

        Bounded retries with rejection feedback; exhaustion falls back to
        the local deterministic director. Every rejection is recorded in
        the immutable concept history. When the experimental visual
        inspector measured the previous segment, its §43 amendments are
        applied to each stage post-validation, pre-style-check — amended
        text still passes ProposalRejected, so the charter always wins.

        Drift cadence: only every Nth segment (config
        drift_every_n_segments, 1 = drift each segment) consults the LLM;
        other segments hold the current concept via the deterministic
        director (recorded, novelty_accepted=False).
        """
        drift_every = max(1, config.voyage.drift_every_n_segments)
        if state.next_segment_number % drift_every != 0:
            hold = DeterministicDirector(style_spec.prompt).propose(
                decision_index=state.decision_index,
                current_concept=state.current_concept,
                destination_concept=state.destination_concept,
                phase=state.phase,
            )
            store.append(
                hold.destination_concept,
                accepted=True,
                summary=f"drift cadence hold (every {drift_every})",
                segment=state.next_segment_number,
            )
            hold.novelty_accepted = False
            self._log_metric(
                {"event": "drift_hold", "segment_id": segment_id, "every": drift_every}
            )
            return hold
        max_attempts = max(1, config.voyage.novelty_max_attempts)
        feedback = ""
        last_score = 0.0
        prefetch_pending = prefetched_raw is not None and not amendments
        for _attempt in range(max_attempts):
            if prefetch_pending:
                # First candidate comes from the parallel prefetch window
                # (still fully validated below — a stale proposal just
                # burns one attempt, then the loop calls the worker live).
                prefetch_pending = False
                raw: dict[str, Any] = prefetched_raw or {}
            else:
                raw = self._call_with_restart(
                    self._director,
                    "director",
                    segment_id,
                    "decide",
                    self._decide_payload(
                        config, state, store, style_spec, feedback, measured_context
                    ),
                )
            try:
                decision = EvolutionDecision.model_validate(raw)
            except Exception as exc:
                feedback = f"previous output failed schema validation: {exc}"
                continue
            if not decision.video.stages:
                feedback = "previous output had no video stages; provide 3-5."
                continue
            if amendments:
                decision.video.stages = [
                    apply_feedback_amendments(stage_text, amendments)
                    for stage_text in decision.video.stages
                ]
            try:
                for stage_text in decision.video.stages:
                    check_prompt_against_style(stage_text, style_spec)
            except ProposalRejected as exc:
                store.append(
                    decision.destination_concept,
                    accepted=False,
                    summary=f"style-policy rejection: {exc}",
                    segment=state.next_segment_number,
                )
                feedback = f"style-policy rejection: {exc}"
                continue
            vectors = self._embed_texts([decision.destination_concept])
            vector = vectors[0] if vectors else None
            accepted, last_score = store.check_novel(decision.destination_concept, vector)
            if not accepted and not config.voyage.allow_concept_revisit:
                store.append(
                    decision.destination_concept,
                    accepted=False,
                    summary=decision.destination.summary,
                    vector=vector,
                    segment=state.next_segment_number,
                )
                feedback = (
                    "novelty rejection: concept too similar to history "
                    f"(similarity {last_score:.3f}); propose a different world. "
                    f"Director's novelty claim: {decision.novelty.why_new}"
                )
                continue
            record = store.append(
                decision.destination_concept,
                accepted=True,
                summary=decision.destination.summary,
                vector=vector,
                segment=state.next_segment_number,
            )
            decision.novelty_accepted = True
            suffix = (
                f"novelty similarity {last_score:.3f} "
                f"(embeddings {'on' if vector is not None else 'fallback'}) "
                f"record {record.id}"
            )
            decision.notes = f"{decision.notes} | {suffix}" if decision.notes else suffix
            return decision
        fallback = DeterministicDirector(style_spec.prompt).propose(
            decision_index=state.decision_index,
            current_concept=state.current_concept,
            destination_concept=state.destination_concept,
            phase=state.phase,
        )
        store.append(
            fallback.destination_concept,
            accepted=True,
            summary="deterministic fallback after exhausted retries",
            segment=state.next_segment_number,
        )
        fallback.novelty_accepted = False
        return fallback

    def _with_audio_gpu(
        self,
        segment_id: str,
        audio_payload: dict[str, object],
        recovery_path: str | None,
    ) -> dict[str, object]:
        """Render one music take with the GPU to itself (§40).

        The video session is evicted first, the (lazily loading) ACE stack
        renders, then audio is evicted and video rebuilds from the latest
        tape. Fake backends answer the same ops as no-ops, so the swap
        only happens for the acestep + resident-session video pair — every
        other pairing renders without touching video residency.
        """
        swap = (
            self._config.audio.backend == "acestep"
            and self._config.video.backend in STREAMING_VIDEO_BACKENDS
        )
        if swap:
            self._call_with_restart(self._video, "video", segment_id, "evict_gpu", {})
        try:
            return self._call_with_restart(
                self._audio, "audio", segment_id, "generate_audio", audio_payload
            )
        finally:
            if swap:
                self._call_with_restart(self._audio, "audio", segment_id, "evict_gpu", {})
                if recovery_path is None:
                    raise MediaError(f"segment {segment_id}: no recovery tape for video rebuild")
                self._call_with_restart(
                    self._video,
                    "video",
                    segment_id,
                    "rebuild",
                    {"recovery_path": recovery_path},
                )

    def _ensure_audio_coverage(
        self,
        config: ProjectConfig,
        number: int,
        segment_id: str,
        segment: Path,
        video_time: float,
        duration: float,
        decision: EvolutionDecision,
        recovery_tape: str | None,
    ) -> tuple[AudioPlan, float]:
        """Render takes when coverage runs low, then slice/assemble (§35).

        Returns the segment's AudioPlan plus the seconds of music coverage
        remaining ahead of the new segment end (drives audio_buffer_seconds).
        Take files are immutable and versioned under `<run>/audio/`; the
        ledger (`takes.jsonl`) is the truth the next commit plans against.
        """
        audio_cfg = config.audio
        audio_dir = self._run_dir / "audio"
        ledger = audio_dir / TAKES_FILENAME
        planner = AudioPlanner(
            take_seconds=audio_cfg.take_seconds,
            ahead_seconds=audio_cfg.ahead_seconds,
            takes=load_takes(ledger),
            segment_seconds=duration,
        )
        caption = decision.audio.music_caption or audio_cfg.music_style
        energy = min(1.0, max(0.0, decision.audio.energy))
        seed = audio_seed(config.seed, number, len(planner.takes))
        # Beat grid: the take BPM derives from this segment's duration so
        # cuts land on beats (adaptive k: 4 → 8 → 16 … until BPM >= 60).
        # ACE treats tempo as a hint, so alignment is approximate.
        from voyage.audio.beat import beats_for_segment

        beats, grid_bpm = beats_for_segment(duration, audio_cfg.beats_per_segment)
        take_bpm = int(round(grid_bpm))
        plan = planner.plan(video_time, caption, seed, number)
        if plan.action in ("render", "repaint") and plan.take is not None:
            take = plan.take
            take.bpm = float(take_bpm)
            take_file = audio_dir / f"{take.take_id}.wav"
            payload: dict[str, object] = {
                "segment_id": segment_id,
                "style": caption,
                "energy": energy,
                "seed": take.seed,
                "output_path": str(take_file),
                "sample_rate": audio_cfg.sample_rate,
                "channels": audio_cfg.channels,
                "duration_seconds": take.duration,
                "bpm": take_bpm,
            }
            if plan.action == "repaint" and plan.current is not None:
                current = plan.current
                payload["task_type"] = "repaint"
                payload["reference_audio"] = current.path
                payload["repaint_start"] = video_time - current.covers_from
                payload["repaint_end"] = current.duration
            self._with_audio_gpu(segment_id, payload, recovery_tape)
            take.path = str(take_file)
            planner.record(take)
            append_take(ledger, take)
            self._log_metric(
                {
                    "event": "take_rendered",
                    "segment_id": segment_id,
                    "take_id": take.take_id,
                    "action": plan.action,
                    "reason": plan.reason,
                    "beats": beats,
                    "bpm": take_bpm,
                }
            )
        # Slice the takes covering [video_time, video_time + duration).
        # A take boundary inside the segment yields two slices joined with
        # a crossfade; the common case is exactly one slice.
        slices: list[Path] = []
        take_ids: list[str] = []
        cursor = video_time
        end = video_time + duration
        index = 0
        while cursor < end - 1e-6:
            serving = planner.take_for_time(cursor)
            if serving is None or not serving.path:
                raise MediaError(f"segment {segment_id}: audio gap at {cursor:.2f}s")
            piece = min(serving.covers_until(), end) - cursor
            slice_path = segment / f"slice_{index:02d}.wav"
            slice_take(
                Path(serving.path),
                cursor - serving.covers_from,
                piece,
                slice_path,
                audio_cfg.sample_rate,
                audio_cfg.channels,
            )
            slices.append(slice_path)
            if serving.take_id not in take_ids:
                take_ids.append(serving.take_id)
            cursor += piece
            index += 1
        audio_out = segment / "audio.wav"
        assemble_segment_audio(slices, audio_out, audio_cfg.crossfade_seconds)
        ahead = planner.coverage_until() - end
        audio_plan = AudioPlan(
            segment_id=segment_id,
            music_style=caption,
            energy=energy,
            seed=seed,
            take_ids=take_ids,
        )
        return audio_plan, max(ahead, 0.0)

    def _inspect_previous_segment(
        self, config: ProjectConfig, number: int, style_spec: StyleSpec
    ) -> tuple[str, list[str]]:
        """Piggyback inspect of the previous segment (DESIGN §44, experimental).

        Ordered and synchronous at the next commit — no threads: the
        inspector samples the previous segment's committed video, merges a
        `visual` section into its metrics.json, and returns the MEASURED
        director context plus any §43 prompt amendments. Disabled by
        default; any failure degrades to ('', []) with an inspect_skipped
        metric so a slow or missing inspector never stops a healthy voyage.
        """
        if not config.experimental.visual_inspector or number == 0:
            return "", []
        try:
            return self._run_previous_inspect(number, style_spec)
        except Exception as exc:  # noqa: BLE001 — inspector never breaks a commit
            self._log_metric({"event": "inspect_skipped", "error": str(exc)})
            return "", []

    def _run_previous_inspect(self, number: int, style_spec: StyleSpec) -> tuple[str, list[str]]:
        """Inspect helper; raises on any failure (caller converts to skip)."""
        prev_id = paths.format_segment_id(number - 1)
        prev_dir = paths.segment_dir(self._run_dir, prev_id)
        prev_video = prev_dir / "video.mp4"
        frames = sample_frames(prev_video, 5)
        reference: Histogram | None
        if number == 1:
            reference = frame_histogram(frames[len(frames) // 2])
        else:
            seg0_video = paths.segment_dir(self._run_dir, paths.format_segment_id(0)) / "video.mp4"
            reference = frame_histogram(sample_frames(seg0_video, 1)[0])
        summary = summarize_segment(frames, reference)
        scene_summary = self._inspect_frame_view(prev_dir, prev_video)
        amendments = feedback_amendments(summary, style_spec)
        visual = {
            "metrics": summary,
            "scene_summary": scene_summary,
            "inspected": bool(scene_summary),
            "amendments": amendments,
        }
        try:
            existing = json.loads((prev_dir / "metrics.json").read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                atomic_write_json(prev_dir / "metrics.json", {**existing, "visual": visual})
        except OSError:
            pass
        self._log_metric({"event": "segment_inspected", "segment_id": prev_id, **summary})
        return format_measured_context(style_spec, summary), amendments

    def _inspect_frame_view(self, prev_dir: Path, prev_video: Path) -> str:
        """Single middle-frame VLM read; '' when the inspector is unavailable."""
        try:
            info = probe(prev_video).get("format", {})
            duration = float(info.get("duration", 0.0) or 0.0) if isinstance(info, dict) else 0.0
            frame_path = prev_dir / "inspect_frame.png"
            proc = run_capture(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-ss",
                    f"{max(duration / 2.0, 0.0):.6f}",
                    "-i",
                    str(prev_video),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ]
            )
            if proc.returncode != 0:
                return ""
            result = self._director.call(
                "inspect",
                {
                    "frame_path": str(frame_path),
                    "model_id": self._config.director.inspector_model_id,
                },
            )
        except VoyageError:
            return ""
        if not isinstance(result, dict) or not result.get("inspected"):
            return ""
        scene = result.get("scene_summary")
        return str(scene) if isinstance(scene, str) else ""

    def commit_one_segment(self) -> str:
        if not self._workers_running:
            raise FatalWorkerError("commit_one_segment requires start_workers() first")
        started = time.monotonic()
        stage_seconds: dict[str, float] = {}
        config = self._config
        state = read_state(self._run_dir)
        check_free_space(self._run_dir, config.min_free_space_gib)

        number = state.next_segment_number
        segment_id = paths.format_segment_id(number)
        segment = paths.segment_dir(self._run_dir, segment_id)
        segment.mkdir(parents=True, exist_ok=True)

        # 1. Director proposal (validated schema; never writes state itself).
        # §74 proposal transaction: validate → novelty → style → accept.
        style_spec = StyleSpec(prompt=config.style)
        store = ConceptStore(
            self._run_dir / "novelty",
            similarity_threshold=config.voyage.novelty_threshold,
            legacy_path=self._run_dir / paths.CONCEPTS_FILENAME,
        )
        # 1b. Piggyback inspect of the previous segment (§44, experimental):
        # ordered, synchronous, never blocking the commit on failure.
        inspect_started = time.monotonic()
        measured_context, amendments = self._inspect_previous_segment(config, number, style_spec)
        stage_seconds["inspect"] = round(time.monotonic() - inspect_started, 3)
        # Prefetched raw proposal (computed during the previous segment's
        # render window). Usable only without fresh inspect amendments —
        # those postdate the prefetch payload.
        prefetched_raw = self._take_prefetch(number, segment_id)
        if amendments:
            prefetched_raw = None
        director_started = time.monotonic()
        decision = self._accept_director_decision(
            config,
            state,
            store,
            style_spec,
            segment_id,
            measured_context,
            amendments,
            prefetched_raw=prefetched_raw,
        )
        stage_seconds["director"] = round(time.monotonic() - director_started, 3)
        # Prefetch the next segment's raw proposal while this one renders
        # (CPU director vs GPU video — no contention by construction).
        self._prefetch_decide_for_next(config, number, decision, store, style_spec)

        # 3. Staged prompt plan (§18.2) + media generation.
        video_started = time.monotonic()
        streaming = config.video.backend in STREAMING_VIDEO_BACKENDS
        num_blocks = config.video.blocks_per_segment if streaming else 1
        prompt_plan = build_staged_prompt_plan(
            segment_id,
            style_spec,
            stage_texts=list(decision.video.stages),
            transition_texts=list(decision.transition.intermediate_stages),
            num_blocks=num_blocks,
            blocks_per_stage=config.voyage.blocks_per_prompt_stage,
        )
        # Map each block to its stage prompt.
        block_prompts = []
        for block in range(num_blocks):
            stage = next(
                stage
                for stage in prompt_plan.stages
                if stage.block_start <= block <= stage.block_end
            )
            block_prompts.append(stage.prompt)
        video_out = segment / "video.mp4"
        audio_out = segment / "audio.wav"
        video_payload: dict[str, Any] = {
            "segment_id": segment_id,
            "output_path": str(video_out),
            "width": config.video.width,
            "height": config.video.height,
            "fps": config.video.fps,
            "frames": config.video.segment_frames,
        }
        if streaming:
            # Resident-session backends take N blocks per commit. longlive2
            # appends to one stream (only the first seed starts the noise
            # RNG, §22.5; scene_cut re-pins the sink); ltxv chains
            # text-to-video + tail-conditioned extensions (scene_cut forces
            # a fresh start). Per-block seeds ride the same protocol shape.
            video_payload["prompts"] = list(block_prompts)
            video_payload["seeds"] = [
                video_seed(config.seed, number, block) for block in range(num_blocks)
            ]
            video_payload["scene_cuts"] = [state.destination_concept != state.current_concept] + [
                False
            ] * (num_blocks - 1)
        else:
            video_payload["prompt"] = prompt_plan.stages[0].prompt
            video_payload["seed"] = video_seed(config.seed, number, 0)
        video_result = self._call_with_restart(
            self._video,
            "video",
            segment_id,
            "generate_blocks",
            video_payload,
            restart_hook=self._resume_video_worker if streaming else None,
        )
        # Truthful frame accounting: the worker reports what it rendered
        # (longlive's decoded count depends on the VAE chunking, not the
        # request), so the timeline always matches reality.
        frames = config.video.segment_frames
        video_block = video_result.get("video")
        recovery_tape: str | None = None
        if isinstance(video_block, dict):
            reported = video_block.get("frames")
            if isinstance(reported, int) and reported > 0:
                frames = reported
            tape = video_block.get("recovery_path")
            if isinstance(tape, str):
                recovery_tape = tape
        stage_seconds["video"] = round(time.monotonic() - video_started, 3)
        duration = frames / config.video.fps
        video_time = state.timeline_frames / config.video.fps
        audio_started = time.monotonic()
        audio_plan, audio_ahead = self._ensure_audio_coverage(
            config,
            number,
            segment_id,
            segment,
            video_time,
            duration,
            decision,
            recovery_tape,
        )
        stage_seconds["audio"] = round(time.monotonic() - audio_started, 3)

        # 4. Validate before anything claims the segment is committed.
        validate_started = time.monotonic()
        video_info = validate_video(
            video_out, config.video.width, config.video.height, config.video.fps
        )
        audio_info = validate_audio(audio_out, config.audio.sample_rate, config.audio.channels)
        if abs(float(video_info["duration"]) - duration) > AV_ALIGNMENT_TOLERANCE_SECONDS:
            raise MediaError(f"segment {segment_id} A/V duration drift")
        stage_seconds["validate"] = round(time.monotonic() - validate_started, 3)

        # 5. Metadata → checksums → DONE → state. No state file may claim
        # the segment is committed until artifacts are valid and durable.
        commit_started = time.monotonic()
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
            {
                "video": video_info,
                "audio": audio_info,
                "frames": frames,
                # §23: RoPE mode is a first-class record — never change it
                # silently across resume; compare on recovery.
                "use_relative_rope": config.video.backend == "longlive2",
                # Backend identity pins every segment to the renderer that
                # produced it (tapes never resume across backends — the
                # worker rejects foreign profiles loudly).
                "video_backend": config.video.backend,
                "blocks": num_blocks,
                "recovery_tape": recovery_tape,
            },
        )
        atomic_write_json(
            segment / "sha256.json",
            {"video.mp4": sha256_file(video_out), "audio.wav": sha256_file(audio_out)},
        )
        done_partial = segment / "DONE.partial"
        atomic_write_bytes(done_partial, b"")
        done_partial.replace(segment / paths.DONE_MARKER)
        fsync_dir(segment)

        # 6. Supervisor-owned state advance (single writer).
        fresh = read_state(self._run_dir)
        fresh.next_segment_number = number + 1
        fresh.committed_segments += 1
        fresh.timeline_frames += frames
        fresh.current_concept = decision.destination_concept
        fresh.destination_concept = decision.destination_concept
        fresh.phase = decision.phase
        fresh.decision_index = state.decision_index + 1
        fresh.audio_buffer_seconds = audio_ahead
        fresh.last_error = None
        write_state(self._run_dir, fresh)
        stage_seconds["commit"] = round(time.monotonic() - commit_started, 3)
        self._log_metric(
            {
                "event": "segment_committed",
                "segment_id": segment_id,
                "frames": frames,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stages": stage_seconds,
            }
        )
        self._sample_gauges(segment_id)
        return segment_id

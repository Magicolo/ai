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

import errno
import fcntl
import json
import math
import os
import signal
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

from voyage import paths
from voyage.atomic import atomic_write_bytes, atomic_write_json
from voyage.audio.planner import TAKES_FILENAME, AudioPlanner, append_take, load_takes
from voyage.backends import VideoBackendAdapter, transport_from_restarting_call
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig
from voyage.console import SegmentProgress
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
    StateError,
    VoyageError,
)
from voyage.hashing import sha256_file as sha256_file  # re-export (issue 021, cf. cli.py)
from voyage.logrotate import append_line, rotate_worker_logs
from voyage.media import (
    AV_ALIGNMENT_TOLERANCE_SECONDS,
    assemble_segment_audio,
    check_av_alignment,
    check_free_space,
    probe,
    probed_take_seconds,
    run_capture,
    slice_take,
    validate_audio,
    validate_video,
)
from voyage.models import (
    AudioPlan,
    EvolutionDecision,
    PromptPlan,
    RunState,
    SegmentWorldState,
    StyleSpec,
)
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

VIDEO_WORKER_MODULES = {
    "fake": "voyage.workers.video",
    "longlive2": "voyage.workers.video_longlive",
    "ltxv": "voyage.workers.video_ltxv",
    "causvid": "voyage.workers.video_causvid",
}
"""Backend name → worker module. longlive2/ltxv/causvid only exist in the CUDA image."""

STREAMING_VIDEO_BACKENDS = ("longlive2", "ltxv", "causvid")
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

#: Seconds a best-effort gauge probe may take per worker (issue 017).
#: Gauges are observability, not correctness — at 5 s the three health
#: probes add at most ~15 s per commit instead of ~30 min at the 600 s
#: RPC default.
GAUGE_TIMEOUT_SECONDS = 5.0

#: Seconds a novelty embedding call may take before the token-set fallback
#: engages (issue 017). Generous on purpose: a slow director must degrade
#: to fallback instead of stalling the commit, while a healthy director
#: must never be cut off so early that embeddings silently disable.
EMBED_TIMEOUT_SECONDS = 60.0

#: Seconds the speculative director prefetch may take (issue 030). The
#: prefetch runs the full `decide` LLM call, so it needs an LLM-class
#: budget like EMBED_TIMEOUT_SECONDS — but it must never inherit the
#: 600 s RPC default: a best-effort thread must resolve quickly enough
#: that interpreter exit never waits ten minutes behind a wedged call.
PREFETCH_TIMEOUT_SECONDS = 60.0

#: Seconds `stop_workers` waits for an in-flight prefetch to finish
#: (issue 030). Best-effort drain only: expiry abandons the thread to
#: its own PREFETCH_TIMEOUT_SECONDS instead of stalling shutdown.
PREFETCH_SHUTDOWN_DRAIN_SECONDS = 2.0

#: Sample resource gauges every K segments (issue 017). 1 keeps the
#: per-segment cadence the benchmark/soak readers expect; raise it to
#: thin out probe traffic on long runs (a TOML knob needs config.py,
#: owned by another track — this constant is the option meanwhile).
RESOURCE_GAUGE_INTERVAL_SEGMENTS = 1

#: How far a worker-reported frame count may exceed the configured
#: segment size before it reads as corruption, not reality (issue 006):
#: `1..10 * segment_frames`. Beyond that the audio-coverage loop would
#: slice thousands of pieces and the timeline would corrupt.
REPORTED_FRAMES_SLACK = 10

#: Smallest legitimate take-slice piece, seconds (issue 104). Takes render
#: at >= 1 s and joints land on segment boundaries, so a sub-50 ms piece
#: is never real music coverage — only a degenerate ledger sliver. Must
#: match `voyage.media.MIN_SLICE_PIECE_SECONDS` (duplicated, not imported:
#: import direction is supervisor → media for functions, and a constant
#: import would still couple the two walk bounds textually — keep both
#: comments in sync instead).
MIN_SLICE_PIECE_SECONDS = 0.05

#: Bound on take slices per commit walk (issue 104). The common case is one
#: slice (two at a take joint); 128 caps ffmpeg spawns when the ledger
#: degrades. Mirrors `voyage.media.MAX_SLICES_PER_WINDOW` (same rationale
#: as above — keep both comments in sync).
MAX_SLICES_PER_SEGMENT = 128

#: Segment JSON artifacts the checksum manifest covers alongside media
#: (issue 095). Mirrors `voyage.media.METADATA_CHECKSUM_ARTIFACTS` so the
#: writer and both verifiers hash the same set without an import cycle
#: (media never imports supervisor).
METADATA_CHECKSUM_ARTIFACTS = (
    "metrics.json",
    "transition.json",
    "prompt_plan.json",
    "audio_state.json",
    "world_state.json",
)


class ProposedSegment(NamedTuple):
    """Director proposal + staged prompt plan for one commit (issue 020).

    Pure proposal: no media rendered, no state advanced. Built by
    `_propose_segment`, consumed by `_render_video` / `_commit_segment`.
    """

    decision: EvolutionDecision
    prompt_plan: PromptPlan
    block_prompts: list[str]
    num_blocks: int
    prefetch_hit: bool
    drift_hold: bool


class RenderedVideo(NamedTuple):
    """Video outcome of one commit (issue 020).

    `frames` is the worker-reported count after the issue-006 ceiling
    gate (never the raw report); `video_time` is the timeline offset the
    audio coverage starts from; `recovery_tape` is the validated absolute
    wire path (None when the worker reported none).
    """

    frames: int
    duration: float
    video_time: float
    recovery_tape: str | None


class CoveredAudio(NamedTuple):
    """Audio outcome of one commit (issue 020).

    The segment's AudioPlan plus the seconds of music coverage remaining
    ahead of the new segment end (drives audio_buffer_seconds) and the
    planner action/reason (surfaced in console summaries).
    """

    audio_plan: AudioPlan
    audio_ahead: float
    take_action: str
    take_reason: str


def summarize_prefetch_outcome(events: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate director-prefetch hit/miss events (issue 033).

    Pure reader over metrics/log events — the commit path already emits
    `director_prefetch_hit/miss` per segment; this turns them into the hit
    rate the soak report needs before any prefetch restructuring is
    considered (candidate 3: measure first). `prefetch_hit_rate` is None
    with no prefetch events (never 0/0). Lives beside the emitter (not in
    the CLI) so the aggregation and the event names cannot drift apart;
    the soak report renders the returned mapping as-is.
    """
    hits = sum(1 for event in events if event.get("event") == "director_prefetch_hit")
    misses = sum(1 for event in events if event.get("event") == "director_prefetch_miss")
    total = hits + misses
    return {
        "prefetch_hits": hits,
        "prefetch_misses": misses,
        "prefetch_hit_rate": (hits / total) if total else None,
    }


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
    transition_path = paths.segment_dir(run_dir, prev_id) / "transition.json"
    try:
        raw = json.loads(transition_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(raw, dict):
        return ""
    try:
        decision = EvolutionDecision.model_validate(raw)
    except Exception:
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


class Supervisor:
    def __init__(
        self,
        run_dir: Path,
        config: ProjectConfig,
        progress: SegmentProgress | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._config = config
        # Optional console progress sink (None = silent; tests use None).
        self._progress = progress
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
            init_payload={
                "models_dir": config.video.models_dir,
                "device": config.director.device,
            },
            timeout=config.voyage.rpc_timeout_seconds,
            # Unified image: the director runs in its own CUDA venv so the
            # Qwen decider serves from the second GPU; unset (slim image,
            # tests) falls back to the supervisor interpreter.
            executable=os.environ.get("VOYAGE_DIRECTOR_PYTHON"),
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

    @contextmanager
    def _held_run_lock(self) -> Iterator[None]:
        """Single-writer run lock, held for one commit (issue 004).

        `fcntl.flock(LOCK_EX | LOCK_NB)` on `<run>/state.json.lock`: the
        second supervisor fails fast with FatalWorkerError (naming the
        holder pid) instead of interleaving media + state writes. The lock
        dies with the process, so no stale-lock recovery exists by design.
        """
        lock_path = self._run_dir / "state.json.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        acquired = False
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    # Not contention (EBADF/EINVAL/ENOLCK, …) — a
                    # programming or environment error. Never misreport it
                    # as "locked by pid" (issue 004): it propagates raw so
                    # the real cause stays visible.
                    raise
                holder = self._read_lock_holder(lock_path)
                raise FatalWorkerError(
                    f"run {self._run_dir} is locked by pid {holder}; "
                    "refusing a second concurrent writer"
                ) from exc
            acquired = True
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.ftruncate(lock_fd, 0)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
            yield
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
            if acquired:
                # Best-effort tidy (issue 004): remove the rendezvous file
                # only while it still names this process — a successor that
                # already acquired rewrote the pid, and its file must
                # survive. Never raises: lock hygiene must not fail a
                # commit. Residual: a contender arriving between this read
                # and the unlink still splits onto a fresh inode (TOCTOU,
                # documented in 004) — the lock itself stays correct.
                try:
                    if lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                        lock_path.unlink()
                except OSError:
                    pass

    def _read_lock_holder(self, lock_path: Path) -> str:
        """Pid recorded by the lock holder, or 'unknown' (best-effort).

        Staleness-honest (issue 004): the lock dies with its holder, so a
        recorded pid for a dead process is residue from a previous run —
        the live holder simply has not written its pid yet
        (write-after-acquire). Report 'unknown' rather than naming a dead
        process; EPERM (alive but unsignalable) still names the pid.
        """
        try:
            text = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "unknown"
        if not text:
            return "unknown"
        try:
            pid = int(text, 10)
        except ValueError:
            return "unknown"
        if pid <= 0:
            return "unknown"
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "unknown"
        except PermissionError:
            return text
        except OSError:
            return "unknown"
        return text

    def _stored_relative(self, absolute_path: Path) -> str:
        """Persist `absolute_path` run-relative when possible (issue 016).

        Wire payloads stay absolute (workers need real paths); what lands
        in takes.jsonl / metrics.json is relative POSIX, so `mv` of a run
        keeps every reference valid. Resolution is the shared
        `paths.resolve_stored_path` convention (consumer side owns it).
        """
        try:
            return str(absolute_path.relative_to(self._run_dir))
        except ValueError:
            return str(absolute_path)

    def _checked_tape_path(self, tape: str, segment_id: str) -> str:
        """Validate a worker-reported recovery path (issues 006, 016).

        Returns the resolved absolute wire path. Anything escaping the run
        dir, pointing at a non-regular file (missing, directory, socket,
        fifo, …), or hiding behind a symlink fails fast with MediaError
        instead of burning restart budget on doomed resume/rebuild calls.
        `resolve()` first so `segments/evil.pt -> /etc/passwd` cannot pass
        the lexical gate; `is_file()` (not `exists()`) so directories fail
        here with MediaError instead of IsADirectoryError downstream.
        Residual TOCTOU (swap between this check and the worker's use) is
        documented in 016 — full elimination needs an fd-passing design.
        """
        candidate = Path(tape)
        if not candidate.is_absolute():
            candidate = self._run_dir / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            raise MediaError(
                f"segment {segment_id}: worker recovery path is unresolvable: {tape!r}"
            ) from exc
        try:
            resolved.relative_to(self._run_dir.resolve())
        except ValueError:
            raise MediaError(
                f"segment {segment_id}: worker recovery path escapes the run dir: {tape!r}"
            ) from None
        if not resolved.is_file():
            raise MediaError(
                f"segment {segment_id}: worker recovery path is not a regular file: {candidate}"
            )
        return str(resolved)

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
        """Start video/audio/director workers (issue 012).

        Exception-safe: if one start raises (e.g. a model-load init
        failure), already-started workers are stopped in reverse order
        before re-raising, so a partial start never orphans GPU residents
        (video DiT ~6-14 GiB, ACE ~5 GiB). A stop failure during the
        unwind must not mask the original start error.
        """
        self._logs.mkdir(parents=True, exist_ok=True)
        started: list[SubprocessWorker] = []
        try:
            for worker in (self._video, self._audio, self._director):
                worker.start()
                started.append(worker)
        except Exception:
            for worker in reversed(started):
                try:
                    worker.stop()
                except Exception:
                    pass
            raise
        self._workers_running = True
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

    def stop_workers(self) -> None:
        self._video.stop()
        self._audio.stop()
        self._director.stop()
        self._workers_running = False
        executor, self._prefetch_executor = self._prefetch_executor, None
        future, self._prefetch_future = self._prefetch_future, None
        self._prefetch_target = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if future is not None:
            # Best-effort drain (issue 030): a pending future cancels; a
            # running (wedged) one is abandoned after a short join so
            # shutdown never stalls — its own PREFETCH_TIMEOUT_SECONDS
            # still bounds the orphaned thread, and the next synchronous
            # decide resyncs the pipe via the issue-137 stale-line loop.
            try:
                future.cancel()
                future.result(timeout=PREFETCH_SHUTDOWN_DRAIN_SECONDS)
            except Exception:
                pass

    def _stage(self, label: str, detail: str = "") -> AbstractContextManager[Any]:
        """Progress spinner around one commit stage (no-op when silent)."""
        if self._progress is None:
            return nullcontext()
        return self._progress.stage(label, detail)

    def _log_metric(self, event: dict[str, object]) -> None:
        line = json.dumps({"ts": time.time(), "run_id": self._config.run_id, **event})
        append_line(self._logs / "metrics.jsonl", line)

    def _rotate_worker_logs(self) -> None:
        """Mid-run worker-log rotation, one cadence tick (issue 056).

        Called once per committed segment (both the render path and the
        orphan-adoption path): copytruncate-rolls any worker log past
        `MAX_WORKER_LOG_BYTES` or a day boundary, in place, so the live
        stderr handles keep writing to the live file. Best-effort by
        design — the helper never raises, and rotation must never fail
        a commit.
        """
        rotate_worker_logs(self._logs)

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
                try:
                    # The restart itself replays `init` over RPC
                    # (rpc.restart = stop + start), so it can raise the same
                    # RecoverableWorkerError the budget gates on (bad
                    # weights, wedged binary, init OOM). Route it through
                    # the same accounting instead of escaping raw (issue
                    # 014): the failed restart consumes an attempt, emits
                    # `worker_restart_failed`, and re-enters the gate, so
                    # the terminal error is still FatalWorkerError with a
                    # `circuit_breaker_open` event.
                    worker.restart()
                    if restart_hook is not None:
                        restart_hook(segment_id)
                except RecoverableWorkerError as restart_exc:
                    # A Recoverable failure from the restart or the resume
                    # hook is a second failure on the same attempt, not a
                    # free retry. A FatalWorkerError from the hook still
                    # propagates without further retries (not Recoverable).
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
                                "reason": str(restart_exc),
                            }
                        )
                        raise FatalWorkerError(
                            f"circuit breaker open for {worker_name}/{op}: "
                            f"{used} restarts exhausted ({restart_exc})"
                        ) from restart_exc
                    self._restarts[worker_name] = used + 1
                    self._log_metric(
                        {
                            "event": "worker_restart_failed",
                            "worker": worker_name,
                            "op": op,
                            "segment_id": segment_id,
                            "attempt": used + 1,
                            "budget": budget,
                            "reason": str(restart_exc),
                        }
                    )
                    continue
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
                # Discovery-side mirror of `_checked_tape_path` (issue
                # 016): a planted symlink or directory at this
                # supervisor-built path must not reach the worker — skip it
                # (resume degrades to a fresh stream) instead of crashing
                # downstream with IsADirectoryError or leaking an outside
                # file into the resume call. The skip is metric-visible so
                # a poisoned segment never hides silently.
                try:
                    resolved_tape = tape.resolve()
                    resolved_tape.relative_to(self._run_dir.resolve())
                except (OSError, RuntimeError, ValueError):
                    self._log_metric(
                        {
                            "event": "recovery_tape_skipped",
                            "segment_id": segment.name,
                            "reason": "escapes the run dir",
                        }
                    )
                    continue
                if not resolved_tape.is_file():
                    self._log_metric(
                        {
                            "event": "recovery_tape_skipped",
                            "segment_id": segment.name,
                            "reason": "not a regular file",
                        }
                    )
                    continue
                try:
                    empty_tape = resolved_tape.stat().st_size == 0
                except OSError:
                    empty_tape = True
                if empty_tape:
                    # Torn write (crash between torch.save and DONE, issue
                    # 139): a 0-byte tape deserializes nowhere, so skip to
                    # the next-newest tape instead of burning the shared
                    # restart budget on a knowably bad input. An
                    # un-stat-able tape reads as torn for the same reason.
                    self._log_metric(
                        {
                            "event": "recovery_tape_skipped",
                            "segment_id": segment.name,
                            "reason": "empty file (torn write)",
                        }
                    )
                    continue
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

        Never fails the commit — and never stalls it either (issue 017):
        every probe carries a short timeout, so a sick worker shows up as
        missing fields while adding at most GAUGE_TIMEOUT_SECONDS per
        worker. Sampled every RESOURCE_GAUGE_INTERVAL_SEGMENTS segments.
        """
        try:
            interval = max(1, RESOURCE_GAUGE_INTERVAL_SEGMENTS)
            if int(segment_id) % interval != 0:
                return
        except ValueError:
            pass
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
                    health = worker.call("health", {}, timeout=GAUGE_TIMEOUT_SECONDS)
                except Exception:
                    # Best-effort means best-effort: a sick worker — or a
                    # test double without a timeout kwarg — shows up as
                    # missing fields, never as a failed commit.
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
        try:
            self.start_workers()
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
                except Exception as exc:
                    # Belt-and-braces (issue 002): anything that is not a
                    # VoyageError — torn JSON, pydantic ValidationError, a
                    # ZeroDivisionError from media probing — still rests the
                    # run at FAILED instead of stranding RUNNING, and
                    # re-raises as FatalWorkerError so callers branch on
                    # class, never on message.
                    try:
                        failed = read_state(self._run_dir)
                    except VoyageError:
                        self._log_metric(
                            {
                                "event": "segment_commit_failed",
                                "error": f"unreadable state after {type(exc).__name__}: {exc}",
                            }
                        )
                        raise FatalWorkerError(
                            f"segment commit failed with {type(exc).__name__}: {exc} "
                            "(state.json unreadable)"
                        ) from exc
                    failed.last_error = f"{type(exc).__name__}: {exc}"
                    failed.status = "FAILED"
                    write_state(self._run_dir, failed)
                    self._log_metric(
                        {
                            "event": "segment_commit_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise FatalWorkerError(
                        f"segment commit failed with {type(exc).__name__}: {exc}"
                    ) from exc
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
            from voyage.director import format_previous_captions

            payload = self._decide_payload(
                config,
                spec_state,
                store,
                style_spec,
                previous_captions=format_previous_captions(
                    previous_video_stages=list(decision.video.stages),
                    previous_music=decision.audio.music_caption,
                    previous_sfx=decision.audio.sfx_caption,
                ),
            )
        except VoyageError:
            self._prefetch_target = None
            return

        def _call() -> dict[str, Any] | None:
            try:
                return self._director.call("decide", payload, timeout=PREFETCH_TIMEOUT_SECONDS)
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
        """Embed via the director worker; None when unavailable (fallback).

        Bounded by EMBED_TIMEOUT_SECONDS (issue 017): a wedged director
        degrades to the token-set fallback instead of stalling the commit.
        Hostile/non-finite worker vectors (issue 103) degrade the same way.
        """
        try:
            result = self._director.call("embed", {"texts": texts}, timeout=EMBED_TIMEOUT_SECONDS)
        except VoyageError:
            return None
        vectors = result.get("vectors")
        if not isinstance(vectors, list):
            return None
        cleaned: list[list[float]] = []
        try:
            for row in vectors:
                if not isinstance(row, list):
                    return None
                values = [float(value) for value in row]
                if not all(math.isfinite(value) for value in values):
                    return None
                cleaned.append(values)
        except (ValueError, TypeError):
            return None
        return cleaned

    def _decide_payload(
        self,
        config: ProjectConfig,
        state: Any,
        store: ConceptStore,
        style_spec: StyleSpec,
        retry_feedback: str = "",
        measured_context: str = "",
        previous_captions: str | None = None,
    ) -> dict[str, Any]:
        history = store.history_texts()
        if previous_captions is None:
            number = getattr(state, "next_segment_number", None)
            try:
                previous_captions = (
                    previous_transition_captions(self._run_dir, int(number))
                    if isinstance(number, int)
                    else ""
                )
            except (OSError, ValueError):
                previous_captions = ""
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
            previous_captions=previous_captions,
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
                "device": config.director.device,
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

        Teardown never masks the primary failure (issue 010): when the
        take render raises, evict/rebuild run best-effort (failures land
        as `audio_swap_teardown_error` metrics) and the original exception
        propagates — including the no-tape case, where the video is left
        evicted and marked explicitly instead of stranding the next
        segment on a fresh stream with no error.
        """
        swap = (
            self._config.audio.backend == "acestep"
            and self._config.video.backend in STREAMING_VIDEO_BACKENDS
        )
        if swap:
            self._call_with_restart(self._video, "video", segment_id, "evict_gpu", {})
        primary_error: BaseException | None = None
        audio_result: dict[str, object] | None = None
        try:
            audio_result = self._call_with_restart(
                self._audio, "audio", segment_id, "generate_audio", audio_payload
            )
        except BaseException as exc:
            primary_error = exc
        if primary_error is not None:
            if swap:
                self._best_effort_audio_teardown(segment_id, recovery_path)
            raise primary_error
        assert audio_result is not None  # no exception means a result arrived
        if swap:
            evict_error: Exception | None = None
            try:
                self._call_with_restart(self._audio, "audio", segment_id, "evict_gpu", {})
            except Exception as exc:
                evict_error = exc
                self._log_metric(
                    {
                        "event": "audio_swap_teardown_error",
                        "segment_id": segment_id,
                        "phase": "audio_evict",
                        "error": str(exc),
                    }
                )
            if recovery_path is None:
                if evict_error is not None:
                    raise evict_error
                raise MediaError(f"segment {segment_id}: no recovery tape for video rebuild")
            # Rebuild runs even when the audio evict failed, so the stream
            # is never stranded evicted after a rendered take.
            try:
                self._call_with_restart(
                    self._video,
                    "video",
                    segment_id,
                    "rebuild",
                    {"recovery_path": recovery_path},
                )
            except Exception as exc:
                self._log_metric(
                    {
                        "event": "audio_swap_teardown_error",
                        "segment_id": segment_id,
                        "phase": "video_rebuild",
                        "error": str(exc),
                    }
                )
                raise
            if evict_error is not None:
                raise evict_error
        return audio_result

    def _best_effort_audio_teardown(self, segment_id: str, recovery_path: str | None) -> None:
        """Post-failure GPU-swap teardown: log, never raise (issue 010).

        A primary failure is already in flight (it propagates from
        `_with_audio_gpu`), so every teardown step reports through
        `audio_swap_teardown_error` metrics instead of replacing the
        cause. Without a tape the video stays evicted — recorded as
        `video_left_evicted` so the seam is explicit, and the run still
        rests at FAILED via the propagating primary.
        """
        try:
            self._call_with_restart(self._audio, "audio", segment_id, "evict_gpu", {})
        except Exception as exc:
            self._log_metric(
                {
                    "event": "audio_swap_teardown_error",
                    "segment_id": segment_id,
                    "phase": "audio_evict",
                    "error": str(exc),
                }
            )
        if recovery_path is not None:
            try:
                self._call_with_restart(
                    self._video,
                    "video",
                    segment_id,
                    "rebuild",
                    {"recovery_path": recovery_path},
                )
            except Exception as exc:
                self._log_metric(
                    {
                        "event": "audio_swap_teardown_error",
                        "segment_id": segment_id,
                        "phase": "video_rebuild",
                        "error": str(exc),
                    }
                )
        else:
            self._log_metric(
                {
                    "event": "video_left_evicted",
                    "segment_id": segment_id,
                    "reason": "no recovery tape for video rebuild after audio failure",
                }
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
    ) -> tuple[AudioPlan, float, str, str]:
        """Render takes when coverage runs low, then slice/assemble (§35).

        Returns the segment's AudioPlan, the seconds of music coverage
        remaining ahead of the new segment end (drives
        audio_buffer_seconds), plus the planner action/reason
        (render/repaint/keep — surfaced in console summaries). Take files
        are immutable and versioned under `<run>/audio/`; the ledger
        (`takes.jsonl`) is the truth the next commit plans against.
        """
        audio_cfg = config.audio
        audio_dir = self._run_dir / "audio"
        ledger = audio_dir / TAKES_FILENAME
        try:
            takes = load_takes(ledger)
        except Exception as exc:
            # Torn ledger (SIGKILL mid-append) or hand-edit corruption must
            # read as StateError (issue 002) — never a bare JSONDecodeError
            # that escapes the commit boundary and strands RUNNING.
            raise StateError(f"segment {segment_id}: corrupt takes ledger {ledger}: {exc}") from exc
        planner = AudioPlanner(
            take_seconds=audio_cfg.take_seconds,
            ahead_seconds=audio_cfg.ahead_seconds,
            takes=takes,
            segment_seconds=duration,
        )
        caption = effective_music_caption(
            audio_cfg.music_caption, decision.audio.music_caption, audio_cfg.music_style
        )
        energy = min(1.0, max(0.0, decision.audio.energy))
        seed = audio_seed(config.seed, number, len(planner.takes))
        # Beat grid: the take BPM derives from this segment's duration so
        # cuts land on beats (adaptive k: 4 → 8 → 16 … until BPM >= 60).
        # ACE treats tempo as a hint, so alignment is approximate.
        from voyage.audio.beat import beats_for_segment

        beats, grid_bpm = beats_for_segment(duration, audio_cfg.beats_per_segment)
        take_bpm = int(round(grid_bpm))
        # A clamped-short take (issue 094 below) can leave this segment
        # uncovered — re-plan boundedly so coverage extends with another
        # chained take instead of erroring at slice time. Three attempts
        # bound GPU spend; the slice walk still fails loud on a true gap.
        # `plan` is primed before the loop (a second identical call opens
        # the first iteration) so the tail return below stays bound even
        # when every iteration takes the keep path.
        plan = planner.plan(video_time, caption, seed, number)
        for _coverage_attempt in range(3):
            seed = audio_seed(config.seed, number, len(planner.takes))
            plan = planner.plan(video_time, caption, seed, number)
            if plan.action not in ("render", "repaint") or plan.take is None:
                break
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
                # Absolute wire path via the shared 016 convention (the
                # ledger stores run-relative; the worker needs a real path).
                payload["reference_audio"] = str(current.resolved_path(self._run_dir))
                payload["repaint_start"] = video_time - current.covers_from
                payload["repaint_end"] = current.duration
            self._with_audio_gpu(segment_id, payload, recovery_tape)
            # Issue 094: ACE renders are not sample-exact vs the request —
            # clamp the ledger to the file so coverage math follows reality
            # instead of overstating it (a fully-past-EOF slice later
            # degrades the whole finalize to the hard-splice fallback).
            take_file_seconds = probed_take_seconds(take_file)
            take_shortfall = max(take.duration - take_file_seconds, 0.0)
            if take_shortfall > 1e-3:
                take.duration = take_file_seconds
            take.path = self._stored_relative(take_file)
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
                    "take_file_seconds": take_file_seconds,
                    "take_short_seconds": take_shortfall,
                }
            )
            if planner.coverage_until() >= video_time + duration - 1e-6:
                break
        # Slice the takes covering [video_time, video_time + duration).
        # A take boundary inside the segment yields two slices joined with
        # a crossfade; the common case is exactly one slice. Both guards
        # below are issue 104: without them a degenerate ledger (1 ms
        # takes, stagnant coverage) spawns thousands of ffmpeg processes
        # that the 0.6 s A/V gate only catches after the damage.
        slices: list[Path] = []
        take_ids: list[str] = []
        cursor = video_time
        end = video_time + duration
        index = 0
        while cursor < end - 1e-6:
            if index >= MAX_SLICES_PER_SEGMENT:
                raise MediaError(
                    f"segment {segment_id}: audio slice walk exceeded "
                    f"{MAX_SLICES_PER_SEGMENT} slices — corrupt takes ledger"
                )
            serving = planner.take_for_time(cursor)
            if serving is None or not serving.path:
                raise MediaError(f"segment {segment_id}: audio gap at {cursor:.2f}s")
            piece = min(serving.covers_until(), end) - cursor
            if piece < MIN_SLICE_PIECE_SECONDS:
                raise MediaError(
                    f"segment {segment_id}: degenerate take slice "
                    f"({piece:.6f}s at {cursor:.2f}s) — corrupt takes ledger"
                )
            slice_path = segment / f"slice_{index:02d}.wav"
            slice_take(
                # Shared 016 convention: run-relative ledger entries
                # resolve under the current run dir; legacy absolute
                # entries are used as-is while they exist.
                serving.resolved_path(self._run_dir),
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
        return audio_plan, max(ahead, 0.0), plan.action, plan.reason

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
                try:
                    recorded = json.loads((prev_dir / "sha256.json").read_text(encoding="utf-8"))
                    if isinstance(recorded, dict) and "metrics.json" in recorded:
                        recorded["metrics.json"] = sha256_file(prev_dir / "metrics.json")
                        atomic_write_json(prev_dir / "sha256.json", recorded)
                except (OSError, ValueError):
                    pass
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

    def _segment_plan_info(
        self,
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
        beats, grid_bpm = beats_for_segment(planned_duration, config.audio.beats_per_segment)
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

    def _propose_segment(
        self,
        config: ProjectConfig,
        state: RunState,
        number: int,
        segment_id: str,
        style_spec: StyleSpec,
        stage_seconds: dict[str, float],
    ) -> ProposedSegment:
        """Director proposal + staged prompt plan for one commit (issue 020).

        Accepts the next EvolutionDecision (§74 transaction: validate →
        novelty → style → accept), folds in the previous segment's visual
        inspect, and maps the accepted stages onto per-block prompts
        (§18.2). Records the `inspect` + `director` stage timings. Pure
        proposal: no media rendered, no state advanced.
        """
        try:
            store = ConceptStore(
                self._run_dir / "novelty",
                similarity_threshold=config.voyage.novelty_threshold,
                legacy_path=self._run_dir / paths.CONCEPTS_FILENAME,
            )
        except Exception as exc:
            # Corrupt concept history must read as StateError (issue 002)
            # — never a bare pydantic ValidationError that escapes the
            # commit boundary and strands the run at RUNNING.
            raise StateError(f"segment {segment_id}: corrupt concept history: {exc}") from exc
        # 1b. Piggyback inspect of the previous segment (§44, experimental):
        # ordered, synchronous, never blocking the commit on failure.
        inspect_started = time.monotonic()
        with self._stage("inspect", "previous segment"):
            measured_context, amendments = self._inspect_previous_segment(
                config, number, style_spec
            )
        stage_seconds["inspect"] = round(time.monotonic() - inspect_started, 3)
        # Prefetched raw proposal (computed during the previous segment's
        # render window). Usable only without fresh inspect amendments —
        # those postdate the prefetch payload.
        prefetched_raw = self._take_prefetch(number, segment_id)
        if amendments:
            prefetched_raw = None
        prefetch_hit = prefetched_raw is not None
        drift_every = max(1, config.voyage.drift_every_n_segments)
        drift_hold = number % drift_every != 0
        director_started = time.monotonic()
        with self._stage("director", config.director.backend):
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
        # Explicit video-caption pin (CLI --video-caption): the staged
        # prompt uses it instead of the director's evolving stages (no
        # drift for this family), still style-checked against the
        # charter — a charter-violating pin fails the commit loudly
        # instead of rendering off-charter. The decision record keeps
        # the director's stages (drift chain stays director-pure);
        # prompt_plan.json keeps what actually rendered.
        if config.video.video_caption:
            check_prompt_against_style(config.video.video_caption, style_spec)
        # Prefetch the next segment's raw proposal while this one renders
        # (CPU director vs GPU video — no contention by construction).
        self._prefetch_decide_for_next(config, number, decision, store, style_spec)

        # 3. Staged prompt plan (§18.2) + media generation.
        streaming = config.video.backend in STREAMING_VIDEO_BACKENDS
        num_blocks = config.video.blocks_per_segment if streaming else 1
        prompt_plan = build_staged_prompt_plan(
            segment_id,
            style_spec,
            stage_texts=effective_video_stages(
                config.video.video_caption, list(decision.video.stages)
            ),
            transition_texts=list(decision.transition.intermediate_stages),
            num_blocks=num_blocks,
            blocks_per_stage=config.voyage.blocks_per_prompt_stage,
        )
        # Map each block to its stage prompt.
        block_prompts: list[str] = []
        for block in range(num_blocks):
            stage = next(
                stage
                for stage in prompt_plan.stages
                if stage.block_start <= block <= stage.block_end
            )
            block_prompts.append(stage.prompt)
        return ProposedSegment(
            decision=decision,
            prompt_plan=prompt_plan,
            block_prompts=block_prompts,
            num_blocks=num_blocks,
            prefetch_hit=prefetch_hit,
            drift_hold=drift_hold,
        )

    def _render_video(
        self,
        config: ProjectConfig,
        state: RunState,
        number: int,
        segment_id: str,
        segment: Path,
        proposed: ProposedSegment,
        stage_seconds: dict[str, float],
    ) -> RenderedVideo:
        """Video render through the backend adapter (issues 020, 023).

        The commit path builds a `VideoSegmentRequest` via the adapter's
        `request_from_config` (geometry + seconds bridge from the stored
        config; the staged §18.2 per-block prompts/seeds ride along for
        streaming backends) and calls `VideoBackendAdapter.generate_segment`
        (issue 023): the adapter owns the single-vs-multi-block payload fork
        and the worker-result normalization, so the supervisor no longer
        duplicates either. Transport is the backends-track hook
        `transport_from_restarting_call` over the restart-guarded
        `_call_with_restart` (with the video resume hook for
        resident-session backends), so retries keep their budget and circuit
        breaker — the adapter itself never retries. Records the `video`
        timing.

        Two supervisor-side gates stay because the adapter normalizes
        leniently by design: the raw worker report is re-gated with the
        original issue-006 predicate (implausible-frame ceiling +
        recovery-tape confinement) before the normalized count is trusted.
        """
        streaming = config.video.backend in STREAMING_VIDEO_BACKENDS
        video_started = time.monotonic()
        request = VideoBackendAdapter.request_from_config(
            config.video,
            segment_id=segment_id,
            prompt=proposed.prompt_plan.stages[0].prompt,
            seed=video_seed(config.seed, number, 0),
            scene_cut=state.destination_concept != state.current_concept,
            block_prompts=list(proposed.block_prompts) if streaming else None,
            block_seeds=(
                [video_seed(config.seed, number, block) for block in range(proposed.num_blocks)]
                if streaming
                else None
            ),
        )
        video_out = segment / "video.mp4"
        raw_result: dict[str, dict[str, object]] = {}
        base_transport = transport_from_restarting_call(
            self._call_with_restart,
            self._video,
            "video",
            segment_id,
            restart_hook=self._resume_video_worker if streaming else None,
        )

        def _transport(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            result = base_transport(operation, payload)
            raw_result["result"] = result
            return result

        adapter = VideoBackendAdapter(_transport, config.video.backend, config.video)
        display_payload = adapter.build_payload(request, video_out)
        if self._progress is not None:
            self._progress.segment_plan(
                self._segment_plan_info(
                    config,
                    number,
                    segment_id,
                    proposed.decision,
                    proposed.block_prompts,
                    display_payload,
                    proposed.num_blocks,
                    proposed.prefetch_hit,
                    proposed.drift_hold,
                )
            )
        with self._stage(
            "video",
            f"{config.video.backend} {config.video.width}x{config.video.height}",
        ):
            segment_result = adapter.generate_segment(request, video_out)
        # Truthful frame accounting: the worker reports what it rendered
        # (longlive's decoded count depends on the VAE chunking, not the
        # request), so the timeline always matches reality. Reports are
        # clamped (issue 006): an unbounded count would send the
        # audio-coverage loop slicing thousands of pieces and corrupt the
        # timeline, and a foreign tape would burn restart budget on doomed
        # resume/rebuild calls. The adapter normalizes leniently, so the
        # raw report is re-gated here with the original predicate before
        # the normalized count is trusted.
        video_block = raw_result["result"].get("video")
        recovery_tape: str | None = None
        if isinstance(video_block, dict):
            if "frames" in video_block:
                reported = video_block["frames"]
                ceiling = REPORTED_FRAMES_SLACK * config.video.segment_frames
                if (
                    isinstance(reported, bool)
                    or not isinstance(reported, int)
                    or not 1 <= reported <= ceiling
                ):
                    raise MediaError(
                        f"segment {segment_id}: worker reported implausible "
                        f"frames {reported!r} (expected an int within 1..{ceiling})"
                    )
            tape = video_block.get("recovery_path")
            if isinstance(tape, str) and tape:
                recovery_tape = self._checked_tape_path(tape, segment_id)
        stage_seconds["video"] = round(time.monotonic() - video_started, 3)
        frames = segment_result.returned_frames
        duration = frames / config.video.fps
        video_time = state.timeline_frames / config.video.fps
        return RenderedVideo(
            frames=frames,
            duration=duration,
            video_time=video_time,
            recovery_tape=recovery_tape,
        )

    def _cover_audio(
        self,
        config: ProjectConfig,
        number: int,
        segment_id: str,
        segment: Path,
        video_time: float,
        duration: float,
        decision: EvolutionDecision,
        recovery_tape: str | None,
        stage_seconds: dict[str, float],
    ) -> CoveredAudio:
        """Music takes + slice/assemble for one commit (DESIGN §35, issue 020).

        Renders takes when coverage runs low, then slices/assembles the
        segment audio. Records the `audio` timing. Delegates to
        `_ensure_audio_coverage` — this seam exists so the commit
        orchestration reads as four stages.
        """
        audio_started = time.monotonic()
        with self._stage("audio", config.audio.backend):
            audio_plan, audio_ahead, take_action, take_reason = self._ensure_audio_coverage(
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
        return CoveredAudio(
            audio_plan=audio_plan,
            audio_ahead=audio_ahead,
            take_action=take_action,
            take_reason=take_reason,
        )

    def _write_state_preserving_control_plane(self, fresh: RunState) -> None:
        """Write back commit state without clobbering stop/pause (issue 099).

        Shared by the render path (`_commit_segment`) and the orphan
        adoption path (`_adopt_unaccounted_segment`, issue 013): `voyage
        stop` / `voyage pause` write state.json without the run lock, so a
        request that landed after the `fresh` read — e.g. during the
        seconds-long checksum passes — would otherwise be clobbered by
        this write-back. The request wins; the run loop honors it at the
        next segment boundary.
        """
        try:
            live_status = read_state(self._run_dir).status
        except VoyageError:
            live_status = fresh.status
        if live_status in ("STOP_REQUESTED", "PAUSE_REQUESTED"):
            fresh.status = live_status
        write_state(self._run_dir, fresh)

    def _adopt_unaccounted_segment(
        self, state: RunState, number: int, segment_id: str, segment: Path
    ) -> str:
        """Adopt a DONE-but-unaccounted segment (issue 013).

        Crash window: DONE went durable in `_commit_segment` but the
        state.json advance never landed (SIGKILL/OOM between the two
        writes), so the retry meets the same segment number with DONE
        already present. Re-rendering over it would silently destroy the
        first render and its provenance — adopt instead: verify the
        recorded checksums over the existing media, then advance the
        counters from the orphan's own metadata without touching a single
        media byte. Anything unverifiable (missing/torn metadata, checksum
        mismatch) refuses loudly with MediaError: the operator inspects or
        removes the segment manually. There is deliberately no --force
        overwrite on this path — silent media replacement is what 013
        eliminates.
        """
        video_out = segment / "video.mp4"
        audio_out = segment / "audio.wav"
        try:
            recorded = json.loads((segment / "sha256.json").read_text(encoding="utf-8"))
            metrics_raw = json.loads((segment / "metrics.json").read_text(encoding="utf-8"))
            world_state = SegmentWorldState.model_validate(
                json.loads((segment / "world_state.json").read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            raise MediaError(
                f"segment {segment_id}: DONE exists but orphan metadata is unreadable "
                f"({exc}); refusing to re-render over it — inspect or remove "
                f"{segment} manually"
            ) from exc
        if (
            not isinstance(recorded, dict)
            or not isinstance(recorded.get("video.mp4"), str)
            or not isinstance(recorded.get("audio.wav"), str)
        ):
            raise MediaError(
                f"segment {segment_id}: DONE exists but sha256.json is malformed; "
                f"refusing to re-render over it — inspect or remove {segment} manually"
            )
        for name, media_path in (("video.mp4", video_out), ("audio.wav", audio_out)):
            try:
                actual = sha256_file(media_path)
            except OSError as exc:
                raise MediaError(
                    f"segment {segment_id}: DONE exists but {name} is missing "
                    f"({exc}); refusing to re-render over it — inspect or remove "
                    f"{segment} manually"
                ) from exc
            if actual != recorded[name]:
                raise MediaError(
                    f"segment {segment_id}: DONE exists but {name} fails checksum "
                    "verification; refusing to re-render over it — inspect or "
                    f"remove {segment} manually"
                )
        for name in METADATA_CHECKSUM_ARTIFACTS:
            recorded_entry = recorded.get(name)
            if not isinstance(recorded_entry, str) or not recorded_entry:
                continue  # legacy manifest: media only means "not covered"
            try:
                actual_entry = sha256_file(segment / name)
            except OSError as exc:
                raise MediaError(
                    f"segment {segment_id}: DONE exists but {name} is missing "
                    f"({exc}); refusing to re-render over it — inspect or remove "
                    f"{segment} manually"
                ) from exc
            if actual_entry != recorded_entry:
                raise MediaError(
                    f"segment {segment_id}: DONE exists but {name} fails checksum "
                    "verification; refusing to re-render over it — inspect or "
                    f"remove {segment} manually"
                )
        frames = metrics_raw.get("frames") if isinstance(metrics_raw, dict) else None
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise MediaError(
                f"segment {segment_id}: DONE exists but metrics.json carries no "
                f"usable frame count; refusing to re-render over it — inspect or "
                f"remove {segment} manually"
            )
        # Issue 006 (adoption-side mirror of the `_render_video` ceiling):
        # the orphan's frame count drives `timeline_frames` directly, so an
        # absurd stored count corrupts the timeline exactly like an absurd
        # live worker report.
        ceiling = REPORTED_FRAMES_SLACK * self._config.video.segment_frames
        if not 1 <= frames <= ceiling:
            raise MediaError(
                f"segment {segment_id}: worker reported implausible "
                f"frames {frames!r} (expected an int within 1..{ceiling})"
            )
        # Issue 003 (adoption-side mirror of the `_commit_segment` gate):
        # a DONE orphan whose media drifted past the A/V budget must not
        # adopt silently — it would commit unfinalizable media exactly
        # like the pre-fix commit path did.
        video_info = validate_video(
            video_out,
            self._config.video.width,
            self._config.video.height,
            self._config.video.fps,
        )
        audio_info = validate_audio(
            audio_out, self._config.audio.sample_rate, self._config.audio.channels
        )
        check_av_alignment(float(video_info["duration"]), float(audio_info["duration"]), segment_id)
        # The takes ledger stays the truth for audio planning (see
        # `_ensure_audio_coverage`); the buffer gauge keeps its pre-crash
        # value — the next commit plans from the ledger, so worst case is
        # an extra take render, never silence.
        fresh = read_state(self._run_dir)
        fresh.next_segment_number = number + 1
        fresh.committed_segments += 1
        fresh.timeline_frames += frames
        fresh.current_concept = world_state.current_concept
        fresh.destination_concept = world_state.destination_concept
        fresh.phase = world_state.phase
        fresh.decision_index = state.decision_index + 1
        fresh.last_error = None
        self._write_state_preserving_control_plane(fresh)
        self._log_metric(
            {
                "event": "segment_adopted",
                "segment_id": segment_id,
                "frames": frames,
                "reason": "done_before_state",
            }
        )
        self._rotate_worker_logs()
        return segment_id

    def _commit_segment(
        self,
        config: ProjectConfig,
        state: RunState,
        number: int,
        segment_id: str,
        segment: Path,
        proposed: ProposedSegment,
        rendered: RenderedVideo,
        covered: CoveredAudio,
        stage_seconds: dict[str, float],
        started: float,
    ) -> str:
        """Validate + metadata + DONE + state advance (issue 020).

        Records the `validate` + `commit` stage timings, emits the
        `segment_committed` metric and the progress summary. `started` is
        the commit's monotonic start (drives the elapsed metric).
        """
        video_out = segment / "video.mp4"
        audio_out = segment / "audio.wav"
        decision = proposed.decision
        frames = rendered.frames
        duration = rendered.duration
        # 4. Validate before anything claims the segment is committed.
        validate_started = time.monotonic()
        with self._stage("validate", "media checks"):
            video_info = validate_video(
                video_out, config.video.width, config.video.height, config.video.fps
            )
            audio_info = validate_audio(audio_out, config.audio.sample_rate, config.audio.channels)
        if abs(float(video_info["duration"]) - duration) > AV_ALIGNMENT_TOLERANCE_SECONDS:
            raise MediaError(f"segment {segment_id} A/V duration drift")
        # Issue 003: video-vs-audio alignment at commit (validate/finalize
        # already enforce it — the commit gate was strictly weaker).
        av_drift = check_av_alignment(
            float(video_info["duration"]), float(audio_info["duration"]), segment_id
        )
        stage_seconds["validate"] = round(time.monotonic() - validate_started, 3)

        # 5. Metadata → checksums → DONE → state. No state file may claim
        # the segment is committed until artifacts are valid and durable.
        commit_started = time.monotonic()
        with self._stage("commit", "metadata + checksums + DONE"):
            world = SegmentWorldState(
                segment_id=segment_id,
                current_concept=state.current_concept,
                destination_concept=decision.destination_concept,
                phase=decision.phase,
                seed=config.seed,
            )
            atomic_write_json(segment / "world_state.json", world.model_dump())
            atomic_write_json(segment / "transition.json", decision.model_dump())
            atomic_write_json(segment / "prompt_plan.json", proposed.prompt_plan.model_dump())
            atomic_write_json(segment / "audio_state.json", covered.audio_plan.model_dump())
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
                    "blocks": proposed.num_blocks,
                    # Run-relative on disk (issue 016); the wire stays
                    # absolute (`recovery_tape` above) — resolved back via
                    # `_resolve_stored_path` at use.
                    "recovery_tape": (
                        self._stored_relative(Path(rendered.recovery_tape))
                        if rendered.recovery_tape is not None
                        else None
                    ),
                },
            )
            atomic_write_json(
                segment / "sha256.json",
                {
                    "video.mp4": sha256_file(video_out),
                    "audio.wav": sha256_file(audio_out),
                    **{name: sha256_file(segment / name) for name in METADATA_CHECKSUM_ARTIFACTS},
                },
            )
            # Single-step DONE (issue 058): one atomic write straight to
            # DONE. The old two-step (write DONE.partial, then rename left
            # a visible DONE.partial window where a concurrent validate
            # reported a spurious orphan on a healthy in-flight commit.
            atomic_write_bytes(segment / paths.DONE_MARKER, b"")

        # 6. Supervisor-owned state advance (single writer).
        fresh = read_state(self._run_dir)
        fresh.next_segment_number = number + 1
        fresh.committed_segments += 1
        fresh.timeline_frames += frames
        fresh.current_concept = decision.destination_concept
        fresh.destination_concept = decision.destination_concept
        fresh.phase = decision.phase
        fresh.decision_index = state.decision_index + 1
        fresh.audio_buffer_seconds = covered.audio_ahead
        fresh.last_error = None
        # Control-plane compare-and-swap (issue 099) lives in the shared
        # helper so the orphan adoption path (issue 013) honors stop/pause
        # identically — see `_write_state_preserving_control_plane`.
        self._write_state_preserving_control_plane(fresh)
        stage_seconds["commit"] = round(time.monotonic() - commit_started, 3)
        elapsed = round(time.monotonic() - started, 3)
        self._log_metric(
            {
                "event": "segment_committed",
                "segment_id": segment_id,
                "frames": frames,
                "av_drift_seconds": av_drift,
                "elapsed_seconds": elapsed,
                "stages": stage_seconds,
            }
        )
        self._sample_gauges(segment_id)
        self._rotate_worker_logs()
        if self._progress is not None:
            from voyage.audio.beat import beats_for_segment

            beats, grid_bpm = beats_for_segment(duration, config.audio.beats_per_segment)
            self._progress.segment_done(
                {
                    "number": number,
                    "segment_id": segment_id,
                    "frames": frames,
                    "duration": duration,
                    "take_ids": list(covered.audio_plan.take_ids),
                    "take_action": covered.take_action,
                    "take_reason": covered.take_reason,
                    "beats": beats,
                    "bpm": grid_bpm,
                    "video_backend": config.video.backend,
                    "overlap_fraction": config.audio.final_overlap_fraction,
                    "overlap_cap_seconds": config.audio.final_overlap_cap_seconds,
                    "stage_seconds": dict(stage_seconds),
                    "elapsed": elapsed,
                    "prefetch_hit": proposed.prefetch_hit,
                }
            )
        return segment_id

    def commit_one_segment(self) -> str:
        """Commit one segment; fails fast when another writer holds the run."""
        if not self._workers_running:
            raise FatalWorkerError("commit_one_segment requires start_workers() first")
        # Single writer per run dir (issue 004): number allocation, media
        # writes and the state advance are one critical section, so a
        # second `voyage run` on the same dir exits loudly instead of
        # interleaving segments and double-counting state.
        with self._held_run_lock():
            return self._commit_one_segment_locked()

    def _commit_one_segment_locked(self) -> str:
        """Segment commit body; the caller holds `_held_run_lock`.

        Sequencing only — the work lives in `_propose_segment` (director
        + prompt plan), `_render_video` (adapter video call),
        `_cover_audio` (music takes + slice/assemble) and
        `_commit_segment` (validate + metadata + DONE + state advance),
        each independently testable (issue 020).
        """
        started = time.monotonic()
        stage_seconds: dict[str, float] = {}
        config = self._config
        state = read_state(self._run_dir)
        check_free_space(self._run_dir, config.min_free_space_gib)

        number = state.next_segment_number
        segment_id = paths.format_segment_id(number)
        segment = paths.segment_dir(self._run_dir, segment_id)
        segment.mkdir(parents=True, exist_ok=True)
        if (segment / paths.DONE_MARKER).exists():
            # Crash-window orphan (issue 013): DONE went durable but the
            # state.json advance never landed, so this retry meets the same
            # number with DONE already present. Never re-render over a
            # previous render — adopt after checksum verification, else
            # refuse loudly. Exception: an artifact-free DONE dir (DONE is
            # written last, so the real crash window always leaves full
            # media + metadata alongside it) holds no render to protect —
            # fall through to a fresh render, metric-visible.
            if sorted(child.name for child in segment.iterdir()) == [paths.DONE_MARKER]:
                self._log_metric(
                    {
                        "event": "segment_reclaimed",
                        "segment_id": segment_id,
                        "reason": "done_without_artifacts",
                    }
                )
            else:
                return self._adopt_unaccounted_segment(state, number, segment_id, segment)
        if self._progress is not None:
            self._progress.segment_start(number, segment_id)

        # 1. Director proposal (validated schema; never writes state itself).
        # §74 proposal transaction: validate → novelty → style → accept.
        style_spec = StyleSpec(prompt=config.style)
        proposed = self._propose_segment(
            config, state, number, segment_id, style_spec, stage_seconds
        )
        rendered = self._render_video(
            config, state, number, segment_id, segment, proposed, stage_seconds
        )
        covered = self._cover_audio(
            config,
            number,
            segment_id,
            segment,
            rendered.video_time,
            rendered.duration,
            proposed.decision,
            rendered.recovery_tape,
            stage_seconds,
        )
        return self._commit_segment(
            config,
            state,
            number,
            segment_id,
            segment,
            proposed,
            rendered,
            covered,
            stage_seconds,
            started,
        )

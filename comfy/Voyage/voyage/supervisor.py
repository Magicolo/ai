"""Supervisor: lifecycle state machine + transactional segment commit.

Owns lifecycle and commit state (DESIGN §73). The director proposes,
the supervisor validates and commits. One segment commit:

  1. director decide (via worker) → EvolutionDecision (schema-validated)
  2. style check (code-level, §18.1) → novelty score (recorded, never rejects)
  3. staged prompt plan (§18.2)
4. video generate_blocks → deferred-audio cover (no audio.wav at commit;
   ACE music renders at finalize)
5. validate media → write metadata (.partial + fsync + rename)
  6. checksums → DONE (.partial + fsync + rename) → state.json update
"""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import queue
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from voyage import llama_server, paths, prompt_enhancer
from voyage.atomic import JsonValue, atomic_write_bytes
from voyage.audio_finalize import (
    deferred_tail_frames,
    derive_conditioning_tail,
)
from voyage.backends import (
    VideoBackendAdapter,
    VideoSegmentResult,
    transport_from_restarting_call,
)
from voyage.concepts import ConceptStore
from voyage.config import ProjectConfig
from voyage.console import SegmentProgress
from voyage.director import (
    REVISITS_ALLOWED_SENTINEL,
    DeterministicDirector,
    director_input_from_state,
    format_measured_context,
)
from voyage.errors import (
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
    check_free_space,
)
from voyage.media import (
    validate_video as validate_video,
)
from voyage.models import (
    AudioPlan,
    EvolutionDecision,
    RunState,
    SegmentWorldState,
    StyleSpec,
)
from voyage.motion_sense import sense_motion
from voyage.persistence import read_state, write_state
from voyage.prompts import (
    apply_feedback_amendments,
    build_staged_prompt_plan,
    check_prompt_against_style,
    feedback_amendments,
)
from voyage.rpc import SubprocessWorker
from voyage.seeds import video_seed
from voyage.segment_manifest import (
    build_segment_manifest,
    load_segment_manifest,
    write_segment_manifest,
)
from voyage.supervisor_commit_types import (
    CoveredAudio as CoveredAudio,
)
from voyage.supervisor_commit_types import (
    ProposedSegment as ProposedSegment,
)
from voyage.supervisor_commit_types import (
    RenderedVideo as RenderedVideo,
)
from voyage.supervisor_lock import (
    read_lock_holder as read_lock_holder,
)
from voyage.supervisor_plan_info import (
    segment_plan_info as segment_plan_info,
)
from voyage.supervisor_prefetch import (
    summarize_prefetch_outcome as summarize_prefetch_outcome,
)
from voyage.supervisor_proposal import (
    _token_counts as _token_counts,
)
from voyage.supervisor_proposal import (
    effective_music_caption as effective_music_caption,
)
from voyage.supervisor_proposal import (
    effective_video_stages as effective_video_stages,
)
from voyage.supervisor_proposal import (
    previous_transition_captions as previous_transition_captions,
)
from voyage.supervisor_routing import (
    AUDIO_WORKER_MODULES as AUDIO_WORKER_MODULES,
)
from voyage.supervisor_routing import (
    STREAMING_VIDEO_BACKENDS as STREAMING_VIDEO_BACKENDS,
)
from voyage.supervisor_routing import (
    VIDEO_WORKER_MODULES as VIDEO_WORKER_MODULES,
)
from voyage.supervisor_routing import (
    audio_worker_module as audio_worker_module,
)
from voyage.supervisor_routing import (
    video_worker_module as video_worker_module,
)
from voyage.supervisor_tape import (
    tape_tail_sha_matches as tape_tail_sha_matches,
)

# Backend routing lives in `voyage.supervisor_routing`
# (issue 081; re-exported at the top so existing importers keep working).


def point_temp_at_run_scratch(scratch: Path) -> None:
    """Point all default-tempfile users at the run scratch (boba /tmp-quota incident).

    Explicit `scratch_dir` init fields cover the big session dirs, but
    bench harnesses, `TemporaryDirectory` calls without `dir=`, and
    third-party libs (torch, diffusers, PIL) all resolve through
    `tempfile` defaults — which is the host /tmp tmpfs (31G, usrquota)
    that a 12-segment LTX25 run exhausted mid-mux. Setting `TMPDIR`
    redirects child processes (workers inherit the supervisor env via
    `rpc._spawn_env`) while `tempfile.tempdir` redirects this process
    (overriding any cached /tmp value). Called in `start_workers` —
    mere construction stays side-effect-free for tests.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)


def _director_init_payload(config: ProjectConfig) -> dict[str, Any]:
    """Director worker init payload: models_dir + device, plus the sidecar
    endpoint when the llama backend is active (DESIGN §140 llama entry).

    The worker's `handle_init` records `llama_endpoint` (str arms the HTTP
    branch, None/absent keeps the AWQ path), so the key is sent only for
    the llama backend — every other backend's payload stays byte-identical.
    """
    payload: dict[str, Any] = {
        "models_dir": config.video.models_dir,
        "device": config.director.device,
    }
    if config.director.backend == "llama":
        payload["llama_endpoint"] = config.director.llama_endpoint
    return payload


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

#: Seconds the generation pump waits for the next pre-warm event before
#: re-checking the video worker thread. Small enough that the bar feels
#: live, large enough that an idle render does not spin the main thread
#: (a video render runs minutes; a 0.2 s tick is ~10 wakeups per second
#: of idle at most, each a single queue poll).
_PREWARM_PUMP_TICK_SECONDS = 0.2

#: Video backends whose resident session cannot survive continuation
#: blocks (issue 198: ltx25 OOMs in GGUF dequant on the second block in
#: the same process while a fresh process rebuilt from the recovery tape
#: renders fine). These get a proactive restart + tape resume between
#: commits; everyone else keeps the resident session (no reload tax).
VIDEO_BACKENDS_NEEDING_FRESH_SESSION = frozenset({"ltx25"})

#: Sample resource gauges every K segments (issue 017). 1 keeps the
#: per-segment cadence the benchmark/soak readers expect; raise it to
#: thin out probe traffic on long runs (a TOML knob needs config.py,
#: owned by another track — this constant is the option meanwhile).
RESOURCE_GAUGE_INTERVAL_SEGMENTS = 1

#: Force a fresh visual start every K segments (scene_cut=True): the
#: worker drops its tail and renders fresh, causing a strong change in
#: visual content. `number` is zero-based, so `(number + 1) % K == 0`
#: cuts on human segments 3, 6, 9, ... Segments in between keep
#: always-continue (scene_cut=False; fresh only when the worker has no
#: tail — first segment / missing tail). K itself is configurable
#: (`voyage.scene_cut_every_n_segments`, default 3) — this constant is
#: the code fallback when a config does not carry the knob.
SCENE_CUT_EVERY_N_SEGMENTS = 3


def scene_cut_for_segment(number: int, every_n: int) -> bool:
    """Whether zero-based segment `number` takes the periodic scene cut.

    Pure cadence predicate behind the `_render_video` usage below (and
    its tests): `(number + 1) % every_n == 0`, so every_n=3 cuts human
    segments 3, 6, 9, ... A non-positive cadence never cuts (defensive —
    the config validator already rejects those, but a hand-built config
    must not ZeroDivisionError a render).
    """
    if every_n <= 0:
        return False
    return (number + 1) % every_n == 0


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


# Commit-pipeline types live in `voyage.supervisor_commit_types`
# (issue 081; re-exported at the top so existing importers keep working).


# Prefetch-outcome aggregation lives in `voyage.supervisor_prefetch`
# (issue 081; re-exported at the top so existing importers keep working).


# Backend-routing resolvers live in `voyage.supervisor_routing`
# (issue 081; re-exported at the top so existing importers keep working).


# Director-proposal pure helpers live in `voyage.supervisor_proposal`
# (issue 081; re-exported at the top so existing importers keep working).


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
        # Generation scratch (boba /tmp-quota incident): every temp file
        # this run produces lives under `run_dir/tmp/`, never on the host
        # /tmp tmpfs. Workers that need session dirs get the path
        # explicitly (`scratch_dir` init field); everything else — bench
        # harnesses, third-party libs — follows the TMPDIR backstop set
        # in `start_workers`.
        self._scratch_dir = paths.ensure_scratch_dir(run_dir)
        video_module = video_worker_module(config.video.backend)
        video_init: dict[str, Any] = {}
        if config.video.backend in STREAMING_VIDEO_BACKENDS:
            video_init = {
                "models_dir": config.video.models_dir,
                "device": config.video.device,
                "scratch_dir": str(self._scratch_dir),
            }
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
                "scratch_dir": str(self._scratch_dir),
            }
        self._audio = SubprocessWorker(
            audio_module,
            run_dir,
            self._logs / "audio-worker.log",
            init_payload=audio_init,
            timeout=config.voyage.rpc_timeout_seconds,
            # ACE venv (DESIGN §140 audio continuity): the ACE stack is
            # isolated from the LTX freeze (transformers pin); unset (video
            # image, tests) falls back to the supervisor interpreter.
            executable=os.environ.get("VOYAGE_ACESTEP_PYTHON"),
        )
        self._director = SubprocessWorker(
            "voyage.workers.director",
            run_dir,
            self._logs / "director-worker.log",
            init_payload=_director_init_payload(config),
            timeout=config.voyage.rpc_timeout_seconds,
            # Unified image: the director runs in its own CUDA venv so the
            # Qwen decider serves from the second GPU; unset (slim image,
            # tests) falls back to the supervisor interpreter.
            executable=os.environ.get("VOYAGE_DIRECTOR_PYTHON"),
        )
        # Loopback llama-server sidecar (DESIGN §140 llama entry): process
        # handle when the llama backend is active, else None forever — the
        # AWQ path never touches it.
        self._llama_sidecar: llama_server.LlamaSidecar | None = None
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
        self._prefetch_submitted_at: float | None = None
        # Background model-pass pre-warm (DESIGN §140): while segment N+1
        # renders video (cuda:0) + audio, one thread upscales + interpolates
        # already-committed segments on cuda:1 whenever the director is
        # idle — finalize then only drains + concats + encodes. Ledgered
        # chunks are skipped, so a resumed generation resumes pre-warm
        # trivially, and finalize polls to completion over whatever is
        # already done. Never fails a commit (best-effort by design).
        self._background: Any | None = None
        # Last reported pre-warm ledger as (driver, upscale, interp): the
        # post-commit report announces only newly ledgered chunks, and the
        # driver identity resets the baseline across worker restarts (a
        # fresh driver starts its totals at zero).
        self._reported_prewarm: tuple[Any, int, int, int, int, float, float] = (
            None,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
        )
        # Per-leg pre-warm bars for the run loop (source frames ledgered
        # vs committed, one bar per leg). Opened lazily when a leg first
        # reports frames; closed after every post-commit report so no
        # Live display is held open across segment stages. The background
        # thread never touches them — pump/report advance them on the
        # main thread. Held as entered contexts plus their trackers.
        self._prewarm_bars: dict[str, tuple[Any, Any]] = {}
        # Latest background pass already announced on the verbose sweep
        # line (object identity — the driver only ever replaces it).
        self._reported_prewarm_result: Any = None
        # Live pre-warm event queue (generation fork video ∥ model-pass):
        # the background driver was constructed with queue-appending
        # callbacks (see `_start_background_prewarm`), so poller
        # per-chunk/per-frames events land here on the pre-warm thread and
        # the main thread drains them into the persistent model-pass bar
        # while the video render blocks (pump in
        # `_generate_segment_with_live_prewarm`). Unbounded — entries are
        # small tuples and each render drains fully; the post-commit
        # report discards stragglers after advancing the ledger delta.
        self._prewarm_queue: queue.Queue[tuple[Any, ...]] = queue.Queue()
        # Pumped-since-report frame counts, subtracted from the next
        # ledger delta so live-advanced frames are never counted twice.
        self._pumped_upscale_frames = 0
        self._pumped_interp_frames = 0
        # Post-commit motion sense (cheap tier): pixel-delta energy of the
        # just-committed segment, steering the NEXT proposal via the
        # measured_context/amendments path. Never gates a commit — a frozen
        # segment still commits, the next prompt just pushes motion harder.
        # Empty dict = unknown (first segment, sense failure), never a
        # deviation. Written after the state advance so sensing can never
        # strand a commit; read at the next propose.
        self._last_motion: dict[str, float] = {}
        # Commit→propose gap ledger (Stage A telemetry): each gap-phase
        # site adds its wall ms here; `_propose_segment` emits the
        # `gap_breakdown` metric and resets. Keys are fixed so the metric
        # shape is stable even when a phase does not run.
        self._gap_ms: dict[str, float] = {
            "gauges_ms": 0.0,
            "rotate_ms": 0.0,
            "control_ms": 0.0,
            "lock_ms": 0.0,
            "precheck_ms": 0.0,
            "concept_store_ms": 0.0,
        }
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
            lock_started = time.monotonic()
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
            self._gap_ms["lock_ms"] += (time.monotonic() - lock_started) * 1000.0
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
        """Pid recorded by the lock holder (logic lives in supervisor_lock)."""
        return read_lock_holder(lock_path)

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
        """Validate a worker-reported recovery path (issues 006, 016, 171).

        Returns the resolved absolute wire path. Anything escaping the run
        dir, pointing at a non-regular file (missing, directory, socket,
        fifo, …), hiding behind a symlink, or implausibly large (issue 171:
        legitimate tapes are MBs — a GB-scale `.pt` OOMs `torch.load` and
        burns the restart budget) fails fast with MediaError instead of
        burning restart budget on doomed resume/rebuild calls.
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
        from voyage.workers.video_common import MAX_RECOVERY_TAPE_BYTES

        try:
            tape_bytes = resolved.stat().st_size
        except OSError as exc:
            raise MediaError(
                f"segment {segment_id}: worker recovery path is unreadable: {candidate}"
            ) from exc
        if tape_bytes > MAX_RECOVERY_TAPE_BYTES:
            raise MediaError(
                f"segment {segment_id}: worker recovery path has implausible tape size "
                f"{tape_bytes} bytes (>{MAX_RECOVERY_TAPE_BYTES}): {candidate}"
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

    def _llama_backend_active(self) -> bool:
        """Whether this run drives the director through the sidecar."""
        return self._config.director.backend == "llama"

    def _start_llama_sidecar(self) -> None:
        """Start the loopback sidecar before the director initializes.

        No-op unless the llama backend is active (or already started).
        The child is masked to the director device via CUDA_VISIBLE_DEVICES
        (default cuda:1 = the 2060), so the server can never straddle the
        video card. A readiness failure raises `FatalWorkerError`: the run
        aborts instead of silently falling back to AWQ, which would corrupt
        the experiment with mixed-backend directives.
        """
        if not self._llama_backend_active() or self._llama_sidecar is not None:
            return
        try:
            port = llama_server.port_for_endpoint(self._config.director.llama_endpoint)
            self._llama_sidecar = llama_server.start(
                self._config.video.models_dir,
                port=port,
                visible_devices=llama_server.visible_devices_for(self._config.director.device),
            )
        except llama_server.LlamaServerError as exc:
            raise FatalWorkerError(f"llama sidecar failed to start: {exc}") from exc

    def _stop_llama_sidecar(self) -> None:
        """Stop the sidecar if running (shutdown unwind is unconditional)."""
        sidecar, self._llama_sidecar = self._llama_sidecar, None
        llama_server.stop(sidecar)

    def start_workers(self) -> None:
        """Start video/audio/director workers (issue 012).

        Exception-safe: if one start raises (e.g. a model-load init
        failure), already-started workers are stopped in reverse order
        before re-raising, so a partial start never orphans GPU residents
        (video DiT ~6-14 GiB, ACE ~5 GiB). A stop failure during the
        unwind must not mask the original start error.
        """
        point_temp_at_run_scratch(self._scratch_dir)
        self._logs.mkdir(parents=True, exist_ok=True)
        started: list[SubprocessWorker] = []
        try:
            with self._stage("start workers", "video/audio/director"):
                for worker in (self._video, self._audio, self._director):
                    if worker is self._director:
                        # Sidecar readiness gates the director init: the worker
                        # must never initialize against a dead server.
                        self._start_llama_sidecar()
                    worker.start()
                    started.append(worker)
        except Exception:
            self._stop_llama_sidecar()
            for worker in reversed(started):
                try:
                    worker.stop()
                except Exception:
                    pass
            raise
        self._workers_running = True
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._start_background_prewarm()

    def stop_workers(self) -> None:
        self._stop_background_prewarm()
        self._video.stop()
        self._audio.stop()
        try:
            self._director.stop()
        finally:
            # The sidecar must never outlive the director, even when the
            # director stop itself fails — otherwise the next run inherits
            # a stale server on the fixed port.
            self._stop_llama_sidecar()
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

    def _start_background_prewarm(self) -> None:
        """Start the background model-pass pre-warm (best-effort, never raises).

        The driver runs the finalize pollers over committed segments on the
        augment device whenever the director is idle (no prefetch in flight,
        probe unblocked — director wins every contention on `cuda:1`). A
        start failure (or a config with the knob off / legs absent) just
        leaves pre-warm inactive: finalize still polls to completion.
        """
        try:
            from voyage.augment_background import BackgroundPrewarm

            if self._background is not None:
                return
            driver = BackgroundPrewarm(
                self._run_dir,
                self._config,
                idle_fn=lambda: (
                    not self._prefetch_in_flight() and not self._director_probe_blocked()
                ),
                on_upscale_chunk=lambda segment, index, total: self._enqueue_prewarm_event(
                    ("upscale_chunk", segment, index, total)
                ),
                on_upscale_frames=lambda segment, frames: self._enqueue_prewarm_event(
                    ("upscale_frames", segment, frames)
                ),
                on_interp_chunk=lambda segment, index, total: self._enqueue_prewarm_event(
                    ("interp_chunk", segment, index, total)
                ),
                on_interp_frames=lambda segment, frames: self._enqueue_prewarm_event(
                    ("interp_frames", segment, frames)
                ),
            )
            driver.start()
            self._background = driver
        except Exception:
            self._background = None

    def _stop_background_prewarm(self) -> None:
        """Stop the background pre-warm (best-effort, never raises)."""
        driver, self._background = self._background, None
        if driver is None:
            return
        try:
            driver.stop()
        except Exception:
            pass

    def _notify_background_committed(self) -> None:
        """Wake the pre-warm thread after a commit (best-effort, never raises)."""
        driver = self._background
        if driver is None:
            return
        try:
            driver.notify_committed()
        except Exception:
            pass

    def _model_pass_demanded(self) -> bool:
        """Whether the run wants model-pass work (bar/report worth opening)."""
        try:
            augment = self._config.augment
            return bool(augment.upscale > 1 or augment.interpolate > 1)
        except Exception:  # noqa: BLE001 - config shape is adopt-path tolerant
            return False

    def _report_background_prewarm(self) -> None:
        """One console line for newly pre-warm-ledgered frames (best-effort).

        The background thread never touches display code — it only
        accumulates `ledgered_frames()`, and this post-commit call (main
        thread) announces the delta since the last report, advances one
        per-leg bar per leg (`upscale frames`, `interpolate frames`),
        and (verbose only) logs the latest sweep breakdown. Silent when
        nothing new ledgered, so the log stays clean on idle passes.
        Frame counts are source frames per leg. Bars close before return
        so no Live is held across segment stages.
        """
        driver = self._background
        progress = self._progress
        if driver is None or progress is None:
            return
        try:
            reader: Any = getattr(driver, "ledgered_frames", None)
            if callable(reader):
                values: Any = reader()
                _passes, _up, _ip = int(values[0]), int(values[1]), int(values[2])
                upf, ipf = int(values[3]), int(values[4])
                ups, ips = float(values[5]), float(values[6])
            else:  # legacy drivers expose chunk totals only
                totals = driver.ledgered_totals()
                _passes, _up, _ip = int(totals[0]), int(totals[1]), int(totals[2])
                upf, ipf, ups, ips = 0, 0, 0.0, 0.0
        except Exception:
            return
        baseline = self._reported_prewarm
        if len(baseline) == 7:
            seen_driver, _seen_up, _seen_ip, seen_upf, seen_ipf, seen_ups, seen_ips = baseline
        else:  # legacy 3-tuple shape: force a re-baseline below
            seen_driver, seen_upf, seen_ipf, seen_ups, seen_ips = None, 0, 0, 0.0, 0.0
        upf, ipf, ups, ips = int(upf), int(ipf), float(ups), float(ips)
        seen_upf, seen_ipf, seen_ups, seen_ips = (
            int(seen_upf),
            int(seen_ipf),
            float(seen_ups),
            float(seen_ips),
        )
        if driver is not seen_driver:
            seen_upf, seen_ipf, seen_ups, seen_ips = 0, 0, 0.0, 0.0
        self._reported_prewarm = (driver, _up, _ip, upf, ipf, ups, ips)
        latest = getattr(driver, "last_result", None)
        if (
            latest is not None
            and latest is not getattr(self, "_reported_prewarm_result", None)
            and int(getattr(latest, "seams_done", 0) or 0) > 0
        ):
            progress.note(
                f"pre-warm seams: {int(getattr(latest, 'seams_done', 0) or 0)} rendered early"
            )
        new_upf, new_ipf = upf - seen_upf, ipf - seen_ipf
        new_ups, new_ips = ups - seen_ups, ips - seen_ips
        # Frames already advanced live during the render (pump) must not
        # advance twice: the bar moves only by the unreported remainder,
        # while the note keeps the full ledger delta (ledger truth).
        # `getattr` guards keep doubles built without `__init__` (which
        # never pump) working — live instances always carry the counters.
        pumped_up = int(getattr(self, "_pumped_upscale_frames", 0) or 0)
        pumped_ip = int(getattr(self, "_pumped_interp_frames", 0) or 0)
        if hasattr(self, "_pumped_upscale_frames"):
            self._pumped_upscale_frames = 0
        if hasattr(self, "_pumped_interp_frames"):
            self._pumped_interp_frames = 0
        if hasattr(self, "_prewarm_queue"):
            self._clear_prewarm_queue()
        unreported_up = max(0, new_upf - pumped_up)
        unreported_ip = max(0, new_ipf - pumped_ip)
        if new_upf > 0 or new_ipf > 0:
            try:
                total = read_state(self._run_dir).timeline_frames
            except Exception:  # noqa: BLE001 - unreadable state still reports deltas
                total = None
            self._advance_prewarm_leg_bar("upscale", unreported_up, total)
            self._advance_prewarm_leg_bar("interp", unreported_ip, total)
            if isinstance(total, int) and not isinstance(total, bool):
                scope = f" (total {upf}/{ipf}f of {total}f committed)"
            else:
                scope = f" (total {upf}/{ipf}f)"
            parts = []
            if new_upf > 0:
                parts.append(f"+{new_upf}f upscale in {new_ups:.1f}s")
            if new_ipf > 0:
                parts.append(f"+{new_ipf}f interp in {new_ips:.1f}s")
            progress.note(f"pre-warm ledgered {', '.join(parts)}{scope}")
            last = getattr(driver, "last_result", None)
            if (
                last is not None
                and last is not getattr(self, "_reported_prewarm_result", None)
                and bool(getattr(progress, "verbose", False))
            ):
                progress.note(
                    f"pre-warm sweep: {last.segments_seen} segments, "
                    f"upscale {last.upscale_frames_done}f in {last.upscale_seconds:.1f}s, "
                    f"interp {last.interp_frames_done}f in {last.interp_seconds:.1f}s"
                )
            self._reported_prewarm_result = last
        else:
            # No new frames, but the pass may still have news: a held-back
            # sweep (low VRAM, director busy) surfaces once as a compact
            # note instead of staying silent. Moot passes (result None)
            # stay silent by design.
            last = getattr(driver, "last_result", None)
            if last is not None and last is not getattr(self, "_reported_prewarm_result", None):
                # Annotated assignment: this mypy pins the 3-arg `getattr`
                # default against the condition type when the call sits
                # directly in the boolean chain, so read the reason out
                # first (probes: bare-chain form fails, assigned form is
                # clean under `mypy --strict`).
                skip_reason: str = getattr(cast(Any, last), "skip_reason", "")
                if skip_reason:
                    progress.note(f"pre-warm held back: {skip_reason}")
            self._reported_prewarm_result = last
        # Never hold a bar Live across segment stages: per-segment bars
        # close here (finish lines print), and the next segment reopens
        # them lazily on its first pump/report event.
        self._close_model_pass_bar()

    def _advance_prewarm_leg_bar(self, leg: str, new_frames: int, total: int | None) -> None:
        """Advance one per-leg pre-warm bar, opening it lazily.

        `leg` is `upscale` (`upscale frames`) or `interp`
        (`interpolate frames`). Each leg counts its own source frames
        against committed frames, so upscale and interp never share one
        X/Y total. Dict keys use the display spelling (`upscale` /
        `interpolate`). No-op when augmentation is not demanded.
        """
        progress = self._progress
        if progress is None or not self._model_pass_demanded():
            return
        if leg == "upscale":
            label = "upscale frames"
            key = "upscale"
        elif leg == "interp":
            label = "interpolate frames"
            key = "interpolate"
        else:
            return
        try:
            bars = getattr(self, "_prewarm_bars", None)
            if not isinstance(bars, dict):
                bars = {}
                self._prewarm_bars = bars
            entry = bars.get(key)
            if entry is None:
                bar_cm = progress.bar(label)
                tracker = bar_cm.__enter__()
                bars[key] = (bar_cm, tracker)
            else:
                _bar_cm, tracker = entry
            if isinstance(total, int) and not isinstance(total, bool) and tracker is not None:
                tracker.set_total(total)
            if new_frames > 0 and tracker is not None:
                tracker.update(new_frames)
        except Exception:  # noqa: BLE001 - display must never fail a commit
            bars = getattr(self, "_prewarm_bars", None)
            if isinstance(bars, dict):
                bars.pop(key, None)

    def _close_model_pass_bar(self) -> None:
        """Close all per-leg pre-warm bars (best-effort, idempotent)."""
        bars = getattr(self, "_prewarm_bars", None)
        if isinstance(bars, dict):
            entries = list(bars.items())
            bars.clear()
        else:
            entries = []
            self._prewarm_bars = {}
        for _leg, (bar_cm, _tracker) in entries:
            if bar_cm is None:
                continue
            try:
                bar_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - display must never fail teardown
                pass

    def _enqueue_prewarm_event(self, event: tuple[Any, ...]) -> None:
        """Append one background pre-warm event (pre-warm thread, never raises).

        The queue is unbounded so `put_nowait` cannot fail in practice;
        the guard keeps a display-path hiccup from ever failing a chunk.
        The main thread drains these into the persistent model-pass bar
        while the video render blocks.
        """
        try:
            self._prewarm_queue.put_nowait(event)
        except Exception:  # noqa: BLE001 - best-effort live display only
            pass

    def _drain_prewarm_queue(self, *, block: bool) -> None:
        """Drain queued pre-warm events into the model-pass bar (main thread).

        Blocking mode waits up to one pump tick for the next event (the
        caller re-checks the video thread between drains); non-blocking
        mode sweeps stragglers after the render finishes. Frame events
        advance the bar live; chunk lifecycle events carry no frame
        counts — the pass-end ledger delta stays their source of truth.
        """
        while True:
            try:
                event = self._prewarm_queue.get(
                    block=block, timeout=_PREWARM_PUMP_TICK_SECONDS if block else 0
                )
            except queue.Empty:
                return
            block = False
            try:
                kind = event[0] if event else ""
                if kind == "upscale_frames":
                    frames = int(event[2])
                    if frames > 0:
                        self._pumped_upscale_frames += frames
                        self._advance_prewarm_leg_bar("upscale", frames, None)
                elif kind == "interp_frames":
                    frames = int(event[2])
                    if frames > 0:
                        self._pumped_interp_frames += frames
                        self._advance_prewarm_leg_bar("interp", frames, None)
            except Exception:  # noqa: BLE001 - display must never fail a render
                pass

    def _clear_prewarm_queue(self) -> None:
        """Discard queued pre-warm events without advancing (main thread).

        The post-commit report advances the full ledger delta, which
        already covers any still-queued stragglers — dropping them here
        keeps the next render's pump from counting them twice. Never
        raises (display hygiene only).
        """
        try:
            while True:
                self._prewarm_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001 - display must never fail a commit
            pass

    def _generate_segment_with_live_prewarm(
        self, adapter: VideoBackendAdapter, request: Any, video_out: Path
    ) -> VideoSegmentResult:
        """Run the video render with live model-pass progress (main thread).

        The render itself moves to a worker thread while this thread
        drains the pre-warm event queue into the persistent model-pass
        bar — so the console shows the video stage spinner plus a live
        upscale/interp frame count instead of a frozen spinner. No new
        Live display is opened (the bar owns the only bar Live, exactly
        as in the post-commit path), and the worker result or exception
        is re-surfaced after the join, so behavior without an active
        pre-warm is byte-identical to the direct call. Falls back to the
        direct call when progress or the background driver is absent (or
        its thread died) — silent/library runs never pay for the fork.
        """
        driver = self._background
        if self._progress is None or driver is None or not driver.is_alive():
            return adapter.generate_segment(request, video_out)
        outcome: dict[str, Any] = {}

        def _render() -> None:
            try:
                outcome["result"] = adapter.generate_segment(request, video_out)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                outcome["error"] = exc

        worker = threading.Thread(target=_render, name="voyage-video-render", daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                self._drain_prewarm_queue(block=True)
            self._drain_prewarm_queue(block=False)
        finally:
            worker.join()
        if "error" in outcome:
            raise outcome["error"]
        return cast(VideoSegmentResult, outcome["result"])

    def _stage(self, label: str, detail: str = "") -> AbstractContextManager[Any]:
        """Progress spinner around one commit stage (no-op when silent)."""
        if self._progress is None:
            return nullcontext()
        return self._progress.stage(label, detail)

    def _log_metric(self, event: dict[str, object]) -> None:
        line = json.dumps({"ts": time.time(), "run_id": self._config.name, **event})
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
                return cast(
                    dict[str, object],
                    worker.call(op, cast(dict[str, JsonValue], dict(payload))),
                )
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
                    tape_bytes = resolved_tape.stat().st_size
                except OSError:
                    tape_bytes = 0
                if tape_bytes == 0:
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
                from voyage.workers.video_common import MAX_RECOVERY_TAPE_BYTES

                if tape_bytes > MAX_RECOVERY_TAPE_BYTES:
                    # Implausible size (issue 171): a GB-scale `.pt` OOMs
                    # `torch.load` and burns the restart budget — skip to
                    # the next-newest tape, metric-visible.
                    self._log_metric(
                        {
                            "event": "recovery_tape_skipped",
                            "segment_id": segment.name,
                            "reason": f"implausible size {tape_bytes} bytes",
                        }
                    )
                    continue
                if not self._tape_tail_sha_matches(segment, resolved_tape):
                    # Corrupt conditioning tail (issue 123): the taped sha
                    # no longer matches the tail file — skip to the
                    # next-newest tape, metric-visible.
                    self._log_metric(
                        {
                            "event": "recovery_tape_skipped",
                            "segment_id": segment.name,
                            "reason": "conditioning tail sha mismatch",
                        }
                    )
                    continue
                return tape
        return None

    @staticmethod
    def _tape_tail_sha_matches(segment: Path, resolved_tape: Path) -> bool:
        """Best-effort taped-tail check (123; logic lives in supervisor_tape)."""
        return tape_tail_sha_matches(segment, resolved_tape)

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

    def _should_refresh_video_session(self, count: int | None, committed: int) -> bool:
        """Whether to refresh the video session before the next commit (issue 198).

        Only backends whose resident session cannot survive continuation
        blocks — and only when more segments remain in this batch, so a
        finished batch never pays a pointless reload.
        """
        if self._config.video.backend not in VIDEO_BACKENDS_NEEDING_FRESH_SESSION:
            return False
        return count is None or committed < count

    def _refresh_video_session(self, segment_id: str) -> None:
        """Restart the video worker and replay the recovery tape (issue 198).

        Proactive form of the manual-resume path: `restart()` replays init
        in a fresh process (shedding whatever the resident session
        accumulated), then `_resume_video_worker` replays causal context
        from the latest tape (no tape → fresh stream, first segment).
        The supervisor process — and its prefetch future — survives, so
        steady-state prefetch keeps hitting across the refresh.
        """
        self._video.restart()
        self._resume_video_worker(segment_id)
        self._log_metric(
            {
                "event": "video_session_refreshed",
                "segment_id": segment_id,
                "backend": self._config.video.backend,
            }
        )

    def _director_probe_blocked(self) -> bool:
        """Whether a director health probe would stall behind LLM work.

        The director worker serves one RPC at a time. An in-flight
        prefetch decide is the known case (skip-busy, leniency batch) —
        but the subtler one is a future that already timed out on our
        side (60 s budget) while the worker still chews the decide
        (120 s+ observed): its result is a non-dict, and a health probe
        issued now queues behind the unfinished decide and burns the
        full 5 s timeout (boba: 5 s every tail). Only a completed dict
        result proves the worker is free again.
        """
        future = self._prefetch_future
        if future is None:
            return False
        if self._prefetch_in_flight():
            return True
        try:
            return not isinstance(future.result(), dict)
        except Exception:
            return True

    def _sample_gauges(self, segment_id: str) -> None:
        """Best-effort resource snapshot after a commit (Phase 6 slice E).

        Never fails the commit — and never stalls it either (issue 017):
        every probe carries a short timeout, so a sick worker shows up as
        missing fields while adding at most GAUGE_TIMEOUT_SECONDS per
        worker. Sampled every `resource_gauge_interval_segments` segments
        (VoyageConfig, default 1).
        """
        try:
            interval = max(1, self._config.voyage.resource_gauge_interval_segments)
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
                if name == "director" and self._director_probe_blocked():
                    # The director's RPC queue is serial: a health probe
                    # issued while LLM work occupies it either blocks
                    # behind it or burns the full 5 s timeout (boba: 5 s
                    # every tail, 23.7 s once). Skip the probe — no fields
                    # this segment — instead of stalling the commit.
                    continue
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
            if (count is None or count > 0) and (
                self._config.video.backend in STREAMING_VIDEO_BACKENDS
            ):
                # Swansy continuity: a fresh process starts with an empty
                # video session tail, so without this the first segment of
                # every extension renders fresh (121f) and hard-cuts. Resume
                # from the latest committed tape before the first commit so
                # cross-invocation extensions continue like same-batch ones
                # (mid-batch refresh covers the rest). No tape (first
                # segment) is a no-op inside `_resume_video_worker`.
                try:
                    pending_start = read_state(self._run_dir)
                    self._resume_video_worker(
                        paths.format_segment_id(pending_start.next_segment_number)
                    )
                except VoyageError as exc:
                    failed = read_state(self._run_dir)
                    failed.status = "FAILED"
                    write_state(self._run_dir, failed)
                    self._log_metric({"event": "video_startup_resume_failed", "error": str(exc)})
                    raise
                except Exception as exc:
                    failed = read_state(self._run_dir)
                    failed.status = "FAILED"
                    write_state(self._run_dir, failed)
                    self._log_metric(
                        {
                            "event": "video_startup_resume_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise FatalWorkerError(
                        f"video startup resume failed with {type(exc).__name__}: {exc}"
                    ) from exc
            committed: list[str] = []
            stopped = False
            while count is None or len(committed) < count:
                control_started = time.monotonic()
                stop_requested = self._stop_requested()
                pause_requested = self._pause_requested()
                self._gap_ms["control_ms"] += (time.monotonic() - control_started) * 1000.0
                if stop_requested:
                    stopped = True
                    break
                if pause_requested:
                    break
                try:
                    segment_id = self.commit_one_segment()
                except VoyageError as exc:
                    failed = read_state(self._run_dir)
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
                if self._should_refresh_video_session(count, len(committed)):
                    # Issue 198: this backend's resident session OOMs on
                    # continuation blocks while a fresh process renders
                    # fine — refresh proactively so one invocation covers
                    # the whole batch (and the prefetch future survives).
                    pending = read_state(self._run_dir)
                    self._refresh_video_session(
                        paths.format_segment_id(pending.next_segment_number)
                    )
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
            self._close_model_pass_bar()
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
        drift_every = max(1, config.voyage.drift_every_n_segments)
        if target % drift_every != 0:
            # The next segment holds the current concept deterministically
            # (no LLM call) — a prefetch for it would be consumed as
            # `invalidated`, so skip the wasted 30-40 s of sidecar/worker
            # time. The waste is not free: on the llama backend it
            # contends with prompt enhancement at the next render start
            # on the single-slot sidecar server.
            return
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
        self._prefetch_submitted_at = time.monotonic()

    def _prefetch_in_flight(self) -> bool:
        """Whether a background prefetch decide is still running (pure check).

        The director worker serves one RPC at a time, so any other director
        call issued while this is true queues behind the 60 s prefetch
        budget — callers that cannot afford the wait (gauges) skip instead.
        """
        future = self._prefetch_future
        return future is not None and not future.done()

    def _take_prefetch(
        self,
        number: int,
        segment_id: str,
        *,
        invalidated: bool = False,
        invalidation_reason: str = "",
    ) -> dict[str, Any] | None:
        """Consume the prefetched raw proposal when it targets this segment.

        Hit = future done with a dict result (used as the accept loop's
        first candidate, still fully validated). Anything else is a miss:
        the next commit decides synchronously. Stale targets are dropped.
        A still-running future for this segment is waited out (bounded by
        its remaining RPC budget) instead of instantly missed: the miss
        path would queue a second decide behind the orphan on the serial
        worker lock, doubling the latency the prefetch was meant to hide.

        `invalidated` is the third outcome (issues 136 + 168): the caller
        knows the proposal cannot be used — the drift-cadence hold returns
        before the accept loop — so a ready future logs
        `director_prefetch_invalidated` (with `reason`) instead of `hit`
        and is dropped. Without it the soak hit-rate measures "prefetch
        was ready", not "prefetch was used", diverging exactly when the
        cadence is doing work.
        """
        future, target = self._prefetch_future, self._prefetch_target
        self._prefetch_future = None
        self._prefetch_target = None
        submitted_at, self._prefetch_submitted_at = self._prefetch_submitted_at, None

        def _age_ms() -> float:
            if submitted_at is None:
                return 0.0
            return round((time.monotonic() - submitted_at) * 1000.0, 3)

        if future is None or target != number:
            self._log_metric(
                {
                    "event": "director_prefetch_miss",
                    "segment_id": segment_id,
                    "prefetch_age_ms": _age_ms(),
                }
            )
            return None
        if not future.done() and not invalidated:
            # The prefetch overlapped this segment's whole video render, so
            # a still-running future is nearly done — or the worker is
            # wedged, whose recovery the sync path's restart budget owns.
            # Wait out its remaining RPC budget instead of instantly
            # missing and then queueing a second decide behind the orphan
            # on the serial worker lock: that queue is the observed
            # director-between-segments gap. A hold discards the proposal
            # unread, so it never waits.
            remaining = PREFETCH_TIMEOUT_SECONDS - _age_ms() / 1000.0
            if remaining > 0:
                try:
                    future.result(timeout=remaining)
                except Exception:  # noqa: BLE001 — falls through to done() re-check
                    pass
        if not future.done():
            self._log_metric(
                {
                    "event": "director_prefetch_miss",
                    "segment_id": segment_id,
                    "prefetch_age_ms": _age_ms(),
                }
            )
            return None
        try:
            raw = future.result()
        except Exception:  # noqa: BLE001 — prefetch must never break a commit
            self._log_metric(
                {
                    "event": "director_prefetch_miss",
                    "segment_id": segment_id,
                    "prefetch_age_ms": _age_ms(),
                }
            )
            return None
        if not isinstance(raw, dict):
            self._log_metric(
                {
                    "event": "director_prefetch_miss",
                    "segment_id": segment_id,
                    "prefetch_age_ms": _age_ms(),
                }
            )
            return None
        if invalidated:
            self._log_metric(
                {
                    "event": "director_prefetch_invalidated",
                    "segment_id": segment_id,
                    "reason": invalidation_reason or "discarded",
                    "prefetch_age_ms": _age_ms(),
                }
            )
            return None
        self._log_metric(
            {
                "event": "director_prefetch_hit",
                "segment_id": segment_id,
                "prefetch_age_ms": _age_ms(),
            }
        )
        return raw

    def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Embed via the director worker; None when unavailable (fallback).

        Bounded by EMBED_TIMEOUT_SECONDS (issue 017): a wedged director
        degrades to the token-set fallback instead of stalling the commit.
        Hostile/non-finite worker vectors (issue 103) degrade the same way.
        Transport hiccups outside the VoyageError taxonomy (issue 225:
        OSError from a respawn path) degrade the same way — embeddings are
        advisory, never commit-killing. Only the taxonomy-external,
        non-recoverable BaseExceptions (MemoryError, KeyboardInterrupt)
        propagate. Every fallback emits `director_embed_fallback` with the
        exception class so the degradation stays visible.
        """
        try:
            result = self._director.call(
                "embed",
                cast(dict[str, JsonValue], {"texts": texts}),
                timeout=EMBED_TIMEOUT_SECONDS,
            )
        except (VoyageError, OSError) as exc:
            self._log_metric(
                {
                    "event": "director_embed_fallback",
                    "error_class": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )
            return None
        vectors = result.get("vectors")
        if not isinstance(vectors, list):
            return None
        cleaned: list[list[float]] = []
        try:
            for row in vectors:
                if not isinstance(row, list):
                    return None
                values: list[float] = []
                for value in row:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        return None
                    values.append(float(value))
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
                else REVISITS_ALLOWED_SENTINEL
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
                "device": config.director.device,
                "temperature": config.director.temperature,
                "max_new_tokens": config.director.max_new_tokens,
                "enable_thinking": config.director.enable_thinking,
                "retry_feedback": retry_feedback,
            }
        )
        if config.director.backend == "llama":
            # Sidecar decide wiring (DESIGN §140 llama entry): the worker
            # routes only backend == "qwen" into `_qwen_decide`, where
            # endpoint presence selects the sidecar branch — so the wire
            # keeps "qwen" and `llama` stays a supervisor-side selector
            # (sidecar lifecycle + endpoint injection). Sending
            # backend="llama" would take the deterministic branch and
            # silently drop the LLM, so the mapping is explicit here.
            payload["backend"] = "qwen"
            payload["llama_endpoint"] = config.director.llama_endpoint
        return payload

    def _log_rejection(
        self,
        segment_id: str,
        attempt: int,
        reason: str,
        extra: dict[str, object],
        from_prefetch: bool,
    ) -> None:
        """Emit `director_rejection` (+ `director_prefetch_rejected`, Stage A).

        One shared seam for every `continue` path of the §74 accept loop,
        so the metric shape cannot drift between rejection reasons. A
        burned attempt-0 prefetch additionally emits
        `director_prefetch_rejected` — the retry-cost signal Stage B/C
        needs. Method (not a loop closure) so loop variables are always
        explicit parameters (B023).
        """
        if from_prefetch:
            self._log_metric(
                {"event": "director_prefetch_rejected", "segment_id": segment_id, "reason": reason}
            )
        self._log_metric(
            {
                "event": "director_rejection",
                "segment_id": segment_id,
                "attempt": attempt,
                "reason": reason,
                **extra,
            }
        )

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
    ) -> tuple[EvolutionDecision, dict[str, int]]:
        """§74 transaction: validate → style → score → accept.

        Bounded retries for schema/style failures, with rejection
        feedback; exhaustion falls back to the local deterministic
        director. Novelty never rejects (item 1): the prompt steers
        toward unvisited worlds, every generation is scored and recorded,
        and the first schema/style-valid generation renders — revisits
        carry novelty_accepted=False. Every rejection is recorded in
        the immutable concept history.

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
            if amendments:
                # Motion steering applies to holds too: the deterministic
                # "gentle motion, held wide shot" stage is the most
                # freeze-prone prompt on the live path, so a low-motion
                # reading still hardens it instead of being dropped.
                hold.video.stages = [
                    apply_feedback_amendments(stage_text, amendments)
                    for stage_text in hold.video.stages
                ]
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
            return hold, {"prompt_tokens": 0, "completion_tokens": 0}
        max_attempts = max(1, config.voyage.novelty_max_attempts)
        novelty_threshold = config.voyage.novelty_threshold
        feedback = ""
        spent_prompt_tokens = 0
        spent_completion_tokens = 0
        # A prefetched proposal stays usable with or without motion
        # steering: amendments apply post-hoc to both paths below, so a
        # prefetch generated without measured_context still enters the
        # accept loop as the first candidate instead of forcing a
        # redundant synchronous decide (the sequential director gap).
        # Validation, style and novelty still run on it.
        prefetch_pending = prefetched_raw is not None
        for attempt in range(max_attempts):
            served_prefetch = False
            if prefetch_pending:
                # First candidate comes from the parallel prefetch window
                # (still fully validated below — a stale proposal just
                # burns one attempt, then the loop calls the worker live).
                prefetch_pending = False
                served_prefetch = True
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
            served = _token_counts(raw)
            spent_prompt_tokens += served["prompt_tokens"]
            spent_completion_tokens += served["completion_tokens"]
            try:
                decision = EvolutionDecision.model_validate(raw)
            except Exception as exc:
                feedback = f"previous output failed schema validation: {exc}"
                self._log_rejection(
                    segment_id, attempt, "schema", {"detail": str(exc)[:300]}, served_prefetch
                )
                continue
            if not decision.video.stages:
                feedback = "previous output had no video stages; provide 3-5."
                self._log_rejection(segment_id, attempt, "empty_stages", {}, served_prefetch)
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
                self._log_rejection(
                    segment_id, attempt, "style", {"detail": str(exc)[:300]}, served_prefetch
                )
                continue
            vectors = self._embed_texts([decision.destination_concept])
            vector = vectors[0] if vectors else None
            accepted, last_score = store.check_novel(decision.destination_concept, vector)
            # Item 1: novelty never rejects — the prompt steers toward
            # unvisited worlds, the score is recorded, and the first
            # schema/style-valid generation always renders. A revisit
            # simply carries novelty_accepted=False. Single-serve accepts
            # are also the director speedup: no retry loop burns extra
            # ~120s+ LLM calls.
            record = store.append(
                decision.destination_concept,
                accepted=True,
                summary=decision.destination.summary,
                vector=vector,
                segment=state.next_segment_number,
            )
            decision.novelty_accepted = accepted or config.voyage.allow_concept_revisit
            kind = "novel" if decision.novelty_accepted else "revisit"
            suffix = (
                f"novelty {kind} (similarity {last_score:.3f}, "
                f"embeddings {'on' if vector is not None else 'fallback'}) "
                f"record {record.id}"
            )
            decision.notes = f"{decision.notes} | {suffix}" if decision.notes else suffix
            self._log_metric(
                {
                    "event": "novelty_scored",
                    "segment_id": segment_id,
                    "score": round(last_score, 3),
                    "threshold": novelty_threshold,
                    "embedded": vector is not None,
                    "accepted_novel": decision.novelty_accepted,
                }
            )
            return decision, {
                "prompt_tokens": spent_prompt_tokens,
                "completion_tokens": spent_completion_tokens,
            }
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
        self._log_metric(
            {
                "event": "director_fallback",
                "segment_id": segment_id,
                "attempts": max_attempts,
            }
        )
        return fallback, {
            "prompt_tokens": spent_prompt_tokens,
            "completion_tokens": spent_completion_tokens,
        }

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
        """Console plan dict (logic lives in supervisor_plan_info)."""
        return segment_plan_info(
            config,
            number,
            segment_id,
            decision,
            block_prompts,
            video_payload,
            num_blocks,
            prefetch_hit,
            drift_hold,
        )

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
        novelty → style → accept) and maps the accepted stages onto
        per-block prompts (§18.2). Records the `director` stage timing.
        Pure proposal: no media rendered, no state advanced.
        """
        concept_store_started = time.monotonic()
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
        self._gap_ms["concept_store_ms"] += (time.monotonic() - concept_store_started) * 1000.0
        # The drift-cadence hold is known before consumption (issues
        # 136 + 168): it returns before the accept loop — so a ready
        # proposal logs `invalidated` instead of `hit`, and the soak rate
        # measures use, not readiness.
        drift_every = max(1, config.voyage.drift_every_n_segments)
        drift_hold = number % drift_every != 0
        invalidation_reasons = ["drift_hold"] if drift_hold else []
        prefetched_raw = self._take_prefetch(
            number,
            segment_id,
            invalidated=bool(invalidation_reasons),
            invalidation_reason="+".join(invalidation_reasons),
        )
        prefetch_hit = prefetched_raw is not None
        if prefetch_hit and self._progress is not None:
            self._progress.note("director prefetch hit (used as first candidate)")
        director_started = time.monotonic()
        with self._stage("director", config.director.backend):
            measured_context = format_measured_context(style_spec, self._last_motion)
            amendments = feedback_amendments(self._last_motion, style_spec)
            decision, director_tokens = self._accept_director_decision(
                config,
                state,
                store,
                style_spec,
                segment_id,
                measured_context=measured_context,
                amendments=amendments or None,
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
        if (
            self._progress is not None
            and self._prefetch_target == number + 1
            and self._prefetch_in_flight()
        ):
            self._progress.note(
                f"director prefetch running for {paths.format_segment_id(number + 1)} (background)"
            )

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
        # Gap ledger emission (Stage A telemetry): the buckets accumulated
        # since the previous segment's propose — commit-tail gauges/rotate,
        # loop control reads, lock acquire, commit-head precheck, and the
        # ConceptStore init above — are attributed to this segment, then
        # reset for the next gap. stage_seconds keys are untouched.
        self._log_metric(
            {
                "event": "gap_breakdown",
                "segment_id": segment_id,
                "buckets": {key: round(value, 3) for key, value in self._gap_ms.items()},
            }
        )
        for key in self._gap_ms:
            self._gap_ms[key] = 0.0
        return ProposedSegment(
            decision=decision,
            prompt_plan=prompt_plan,
            block_prompts=block_prompts,
            num_blocks=num_blocks,
            prefetch_hit=prefetch_hit,
            drift_hold=drift_hold,
            director_tokens=director_tokens,
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
        # Track B enhancer (DESIGN §140, default ON): expand the staged
        # §18.2 prompts through the llama-server sidecar here — main commit
        # thread, BEFORE the `generate_blocks` worker RPC below — so
        # enhancement never overlaps the video forward (cuda:0 peak
        # 14.6 GiB) or any ACE residency (all-deferred audio commits
        # video-only; the sidecar holds ~5 GiB on cuda:1). When the knob
        # is off (`--no-prompt-enhance`) or the backend is not LTX, the
        # helper returns the inputs untouched and the payload below is
        # byte-identical.
        staged_prompt = proposed.prompt_plan.stages[0].prompt
        staged_blocks = list(proposed.block_prompts) if streaming else None
        if staged_blocks is not None:
            staged_blocks, enhance_summary = prompt_enhancer.enhance_many(
                staged_blocks,
                enabled=config.video.prompt_enhance,
                backend=config.video.backend,
                endpoint=config.director.llama_endpoint,
            )
            if staged_blocks:
                staged_prompt = staged_blocks[0]
        else:
            [staged_prompt], enhance_summary = prompt_enhancer.enhance_many(
                [staged_prompt],
                enabled=config.video.prompt_enhance,
                backend=config.video.backend,
                endpoint=config.director.llama_endpoint,
            )
        if config.video.prompt_enhance:
            # Track C observable: prove sidecar engagement per segment
            # (an ON run whose expansion never changes text — or never
            # reaches the server — is uninterpretable without this).
            self._log_metric(
                {
                    "event": "prompt_enhanced",
                    "segment": segment_id,
                    "backend": config.video.backend,
                    **enhance_summary,
                }
            )
        # LTX always-continue (ltx25-compare fix, 2026-10-01): drift never
        # forces a fresh segment — the worker continues from its tail
        # whenever one exists and goes fresh only when it has none (first
        # segment / missing tail). Sending scene_cut on destination change
        # rendered every drifted segment fresh (121f, hard cut). The only
        # intentional fresh is the periodic scene cut below (every Nth
        # segment per `voyage.scene_cut_every_n_segments`) for a strong
        # visual change.
        scene_cut = scene_cut_for_segment(number, config.voyage.scene_cut_every_n_segments)
        request = VideoBackendAdapter.request_from_config(
            config.video,
            segment_id=segment_id,
            prompt=staged_prompt,
            seed=video_seed(config.seed, number, 0),
            scene_cut=scene_cut,
            block_prompts=list(staged_blocks) if staged_blocks is not None else None,
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
            segment_result = self._generate_segment_with_live_prewarm(adapter, request, video_out)
        # Truthful frame accounting: the worker reports what it rendered
        # (a worker's decoded count can depend on internal chunking, not
        # the request), so the timeline always matches reality. Reports are
        # clamped (issue 006): an unbounded count would corrupt the timeline
        # directly, and a foreign tape would burn restart budget on doomed
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
        # LTXV sub-stage detail (Stage A telemetry): the worker reports
        # `stage_ms` under its `video` block; the adapter normalization
        # strips it, so it is read here from the raw worker result. Other
        # backends report none and merge as an empty mapping.
        video_stage_ms: dict[str, float] = {}
        if isinstance(video_block, dict):
            reported_stages = video_block.get("stage_ms")
            if isinstance(reported_stages, dict):
                video_stage_ms = {
                    str(key): float(value)
                    for key, value in reported_stages.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
        return RenderedVideo(
            frames=frames,
            duration=duration,
            video_time=video_time,
            recovery_tape=recovery_tape,
            video_stage_ms=video_stage_ms,
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
        """Video-only commit cover: every backend is deferred (DESIGN §140).

        No `audio.wav` is ever written at commit; ACE music renders at
        finalize from the stored director decisions. For streaming
        backends the conditioning tail (`video_tail.mp4`) is derived so
        the next segment chains instead of going fresh; `fake` renders
        statelessly and needs no tail. The `audio` stage timing is still
        recorded for metrics, but generation never opens an audio
        console stage: there is no audio generation during generation.
        This seam exists so the commit orchestration reads as four
        stages. `number`/`video_time`/`duration`/`decision`/
        `recovery_tape` stay in the signature for caller compatibility —
        only the backend name feeds the reason text.
        """
        audio_started = time.monotonic()
        if config.video.backend in STREAMING_VIDEO_BACKENDS:
            derive_conditioning_tail(segment, deferred_tail_frames(config.video.backend))
        stage_seconds["audio"] = round(time.monotonic() - audio_started, 3)
        return CoveredAudio(
            audio_plan=AudioPlan(segment_id=segment_id),
            audio_ahead=0.0,
            take_action="deferred",
            take_reason=(
                f"all backends deferred: video-only commit for {config.video.backend}, "
                "no audio.wav; ACE music renders at finalize"
            ),
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
        try:
            manifest = load_segment_manifest(segment)
            recorded_any = manifest.get("checksums")
            recorded = dict(recorded_any) if isinstance(recorded_any, dict) else {}
            metrics_raw_any = manifest.get("metrics")
            metrics_raw = dict(metrics_raw_any) if isinstance(metrics_raw_any, dict) else {}
            world_any = manifest.get("world_state")
            world_raw = dict(world_any) if isinstance(world_any, dict) else {}
            world_state = SegmentWorldState.model_validate(world_raw)
        except (OSError, ValueError) as exc:
            raise MediaError(
                f"segment {segment_id}: DONE exists but orphan metadata is unreadable "
                f"({exc}); refusing to re-render over it — inspect or remove "
                f"{segment} manually"
            ) from exc
        if not isinstance(recorded, dict) or not isinstance(recorded.get("video.mp4"), str):
            raise MediaError(
                f"segment {segment_id}: DONE exists but manifest checksums are malformed; "
                f"refusing to re-render over it — inspect or remove {segment} manually"
            )
        try:
            actual = sha256_file(video_out)
        except OSError as exc:
            raise MediaError(
                f"segment {segment_id}: DONE exists but video.mp4 is missing "
                f"({exc}); refusing to re-render over it — inspect or remove "
                f"{segment} manually"
            ) from exc
        if actual != recorded["video.mp4"]:
            raise MediaError(
                f"segment {segment_id}: DONE exists but video.mp4 fails checksum "
                "verification; refusing to re-render over it — inspect or "
                f"remove {segment} manually"
            )
        for name, extra in sorted(recorded.items()):
            if name == "video.mp4":
                continue  # already verified above
            if not isinstance(extra, str) or not extra:
                continue  # legacy manifest: media only means "not covered"
            try:
                actual_entry = sha256_file(segment / name)
            except OSError as exc:
                raise MediaError(
                    f"segment {segment_id}: DONE exists but {name} is missing "
                    f"({exc}); refusing to re-render over it — inspect or remove "
                    f"{segment} manually"
                ) from exc
            if actual_entry != extra:
                raise MediaError(
                    f"segment {segment_id}: DONE exists but {name} fails checksum "
                    "verification; refusing to re-render over it — inspect or "
                    f"remove {segment} manually"
                )
        frames = metrics_raw.get("frames") if isinstance(metrics_raw, dict) else None
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise MediaError(
                f"segment {segment_id}: DONE exists but manifest metrics carries no "
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
        # Video-only adoption (all backends deferred): the segment video
        # is validated against the configured geometry; audio renders at
        # finalize from the stored director decisions, so there is no
        # commit-time audio to verify and no A/V gate here.
        validate_video(
            video_out,
            self._config.video.width,
            self._config.video.height,
            self._config.video.fps,
        )
        fresh = read_state(self._run_dir)
        fresh.next_segment_number = number + 1
        fresh.committed_segments += 1
        fresh.timeline_frames += frames
        fresh.current_concept = world_state.current_concept
        fresh.destination_concept = world_state.destination_concept
        fresh.phase = world_state.phase
        fresh.decision_index = state.decision_index + 1
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
        decision = proposed.decision
        frames = rendered.frames
        duration = rendered.duration
        # 4. Validate before anything claims the segment is committed.
        # Video-only (all backends deferred): the segment video is checked
        # against the configured geometry; audio renders at finalize, so
        # there is no commit-time audio to validate and no A/V gate here.
        validate_started = time.monotonic()
        with self._stage("validate", "media checks"):
            video_info = validate_video(
                video_out, config.video.width, config.video.height, config.video.fps
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
            metrics_payload: dict[str, Any] = {
                "video": video_info,
                "frames": frames,
                # §23: RoPE mode is a first-class record — never change it
                # silently across resume; compare on recovery.
                # (A removed video backend was the only relative-RoPE
                # renderer; every live backend records False.)
                "use_relative_rope": False,
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
            }
            checksums: dict[str, str] = {
                "video.mp4": sha256_file(video_out),
            }
            tape_file = segment / "recovery.pt"
            if tape_file.is_file():
                checksums["recovery.pt"] = sha256_file(tape_file)
            manifest = build_segment_manifest(
                decision.model_dump(),
                proposed.prompt_plan.model_dump(),
                covered.audio_plan.model_dump(),
                world.model_dump(),
                metrics_payload,
                checksums,
            )
            write_segment_manifest(segment, manifest)
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
        # Control-plane compare-and-swap (issue 099) lives in the shared
        # helper so the orphan adoption path (issue 013) honors stop/pause
        # identically — see `_write_state_preserving_control_plane`.
        self._write_state_preserving_control_plane(fresh)
        stage_seconds["commit"] = round(time.monotonic() - commit_started, 3)
        # Post-commit motion sense (cheap tier, never fails the commit):
        # three 160px thumbnails + pixel-delta energy, typically 100-300ms
        # of ffmpeg decode, no GPU. Runs AFTER the state advance so sensing
        # can never strand a commit; the reading steers the NEXT proposal
        # via `_last_motion`. Unknown (None) reads steer nothing.
        motion = sense_motion(video_out)
        if motion.energy is not None:
            self._last_motion = {"motion_energy": motion.energy}
        else:
            self._last_motion = {}
        elapsed = round(time.monotonic() - started, 3)
        self._log_metric(
            {
                "event": "motion_sensed",
                "segment_id": segment_id,
                "motion_energy": motion.energy,
                "sense_seconds": motion.seconds,
            }
        )
        self._log_metric(
            {
                "event": "segment_committed",
                "segment_id": segment_id,
                "frames": frames,
                # Video-only commit (all backends deferred): no audio exists
                # at commit time, so drift is definitionally zero. The key
                # stays so metric/log readers never see a missing field.
                "av_drift_seconds": 0.0,
                "elapsed_seconds": elapsed,
                "stages": stage_seconds,
                "director_tokens": dict(proposed.director_tokens),
                "video_stage_ms": dict(rendered.video_stage_ms),
            }
        )
        gauges_started = time.monotonic()
        self._sample_gauges(segment_id)
        self._gap_ms["gauges_ms"] += (time.monotonic() - gauges_started) * 1000.0
        rotate_started = time.monotonic()
        self._rotate_worker_logs()
        self._gap_ms["rotate_ms"] += (time.monotonic() - rotate_started) * 1000.0
        if self._progress is not None:
            from voyage.audio.acestep import MAX_BPM
            from voyage.audio.beat import beats_for_segment

            beats, grid_bpm = beats_for_segment(
                duration, config.audio.beats_per_segment, max_bpm=MAX_BPM
            )
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
                    "motion_energy": motion.energy,
                    "motion_seconds": motion.seconds,
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
        `_cover_audio` (video-only deferred-audio cover) and
        `_commit_segment` (validate + metadata + DONE + state advance),
        each independently testable (issue 020).
        """
        started = time.monotonic()
        stage_seconds: dict[str, float] = {}
        config = self._config
        precheck_started = time.monotonic()
        state = read_state(self._run_dir)
        check_free_space(self._run_dir, config.min_free_space_gib)

        number = state.next_segment_number
        segment_id = paths.format_segment_id(number)
        segment = paths.segment_dir(self._run_dir, segment_id)
        segment.mkdir(parents=True, exist_ok=True)
        done_present = (segment / paths.DONE_MARKER).exists()
        done_children: list[str] = []
        if done_present:
            done_children = sorted(child.name for child in segment.iterdir())
        self._gap_ms["precheck_ms"] += (time.monotonic() - precheck_started) * 1000.0
        if done_present:
            # Crash-window orphan (issue 013): DONE went durable but the
            # state.json advance never landed, so this retry meets the same
            # number with DONE already present. Never re-render over a
            # previous render — adopt after checksum verification, else
            # refuse loudly. Exception: an artifact-free DONE dir (DONE is
            # written last, so the real crash window always leaves full
            # media + metadata alongside it) holds no render to protect —
            # fall through to a fresh render, metric-visible.
            if done_children == [paths.DONE_MARKER]:
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
        committed_id = self._commit_segment(
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
        # Wake the background pre-warm: the freshly committed segment is
        # now pollable (DONE + manifest), so upscale + interp can render
        # while the next segment generates. Never fails the commit. The
        # report describes the previous pass (work done during this
        # segment's render) — the newly woken pass reports next commit.
        self._notify_background_committed()
        self._report_background_prewarm()
        return committed_id

"""Generation-stack tests: config defaults, drift cadence, prefetch, blend.

Covers the rhythm-cut music-video stack on fake (CPU-only) backends:
new AudioConfig/VoyageConfig fields, toml roundtrip, override plumbing,
drift cadence holds, parallel director prefetch hits, and the
finalize-only overlap blend (duration-exact, previews untouched).
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.concepts import ConceptStore
from voyage.config import (
    AudioConfig,
    ProjectConfig,
    VoyageConfig,
    apply_draft_overrides,
    default_config_toml,
    load_config,
)
from voyage.media import (
    _probe_video_fps,
    build_final_audio,
    finalize_run,
    validate_video,
)
from voyage.media import probe as media_probe
from voyage.models import DirectorDestination, EvolutionDecision, StyleSpec
from voyage.persistence import read_state
from voyage.supervisor import PREFETCH_TIMEOUT_SECONDS, Supervisor, summarize_prefetch_outcome


def _write_toml(tmp_path: Path) -> Path:
    path = tmp_path / "voyage.toml"
    text = default_config_toml("stack", "pastel neon line-art, peaceful", 7)
    path.write_text(text, encoding="utf-8")
    return path


def test_audio_config_rhythm_defaults() -> None:
    audio = AudioConfig()
    assert audio.beats_per_segment == 4
    assert audio.final_overlap_fraction == pytest.approx(0.10)
    assert audio.final_overlap_cap_seconds == pytest.approx(0.5)


def test_audio_config_rhythm_validators() -> None:
    with pytest.raises(ValidationError):
        AudioConfig(beats_per_segment=0)
    with pytest.raises(ValidationError):
        AudioConfig(final_overlap_fraction=0.6)
    with pytest.raises(ValidationError):
        AudioConfig(final_overlap_fraction=-0.1)
    with pytest.raises(ValidationError):
        AudioConfig(final_overlap_cap_seconds=-1.0)


def test_voyage_config_drift_default_and_validator() -> None:
    assert VoyageConfig().drift_every_n_segments == 1
    with pytest.raises(ValidationError):
        VoyageConfig(drift_every_n_segments=0)


def test_toml_roundtrip_carries_new_fields(tmp_path: Path) -> None:
    config, _ = load_config(_write_toml(tmp_path))
    assert config.audio.beats_per_segment == 4
    assert config.audio.final_overlap_fraction == pytest.approx(0.10)
    assert config.audio.final_overlap_cap_seconds == pytest.approx(0.5)
    assert config.voyage.drift_every_n_segments == 1


def test_overrides_plumb_beats_and_drift(tmp_path: Path) -> None:
    config, _ = load_config(_write_toml(tmp_path))
    out = apply_draft_overrides(config, beats_per_segment=8, drift_every_n_segments=3)
    assert out.audio.beats_per_segment == 8
    assert out.voyage.drift_every_n_segments == 3
    assert config.audio.beats_per_segment == 4  # pure: source untouched
    with pytest.raises(ValidationError):
        apply_draft_overrides(config, beats_per_segment=0)


def _init_run(run_dir: Path, style: str = "pastel neon line-art, peaceful") -> None:
    initialize_run_directory(run_dir, run_id="stack", style=style, seed=7)


def _metric_events(run_dir: Path, event: str) -> list[dict[str, object]]:
    metrics = run_dir / paths.LOGS_DIRNAME / "metrics.jsonl"
    if not metrics.exists():
        return []
    events = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("event") == event:
                events.append(record)
    return events


def test_drift_cadence_holds_non_drift_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config = apply_draft_overrides(config, drift_every_n_segments=2)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        assert supervisor.commit_one_segment() == "000000"  # 0 % 2: drift (LLM path)
        assert supervisor.commit_one_segment() == "000001"  # 1 % 2: hold
    finally:
        supervisor.stop_workers()
    holds = _metric_events(run_dir, "drift_hold")
    assert len(holds) == 1
    assert holds[0]["segment_id"] == "000001"
    state = read_state(run_dir)
    assert state.committed_segments == 2


def test_director_prefetch_hits_next_commit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    # Segment 0 has no prefetch (nothing ran before it); segment 1 should
    # consume the proposal prefetched during segment 0's render window.
    misses = _metric_events(run_dir, "director_prefetch_miss")
    hits = _metric_events(run_dir, "director_prefetch_hit")
    assert [event["segment_id"] for event in misses] == ["000000"]
    assert [event["segment_id"] for event in hits] == ["000001"]


def _commit_two(run_dir: Path) -> None:
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


@pytest.mark.slow
def test_finalize_blend_is_timeline_exact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    out = tmp_path / "final.mp4"
    assert finalize_run(run_dir, out).exists()
    duration = float(media_probe(out).get("format", {}).get("duration", 0.0))
    # 2 fake segments x 48f @24fps = 4.0s; overlap blend must not shorten it.
    assert duration == pytest.approx(4.0, abs=0.15)


@pytest.mark.slow
def test_finalize_overlap_zero_keeps_legacy_splice(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    out = tmp_path / "final-legacy.mp4"
    assert finalize_run(run_dir, out, overlap_fraction=0.0).exists()
    duration = float(media_probe(out).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(4.0, abs=0.15)


def test_probe_video_fps_returns_zero_on_unparseable() -> None:
    """The presentation-fps probe never raises on odd ffprobe output."""
    video = {"codec_type": "video", "avg_frame_rate": "16/1"}
    assert _probe_video_fps({"streams": [video]}) == pytest.approx(16.0)
    assert _probe_video_fps({"streams": []}) == 0.0
    assert _probe_video_fps({}) == 0.0
    assert _probe_video_fps({"streams": [{"codec_type": "video"}]}) == 0.0
    assert _probe_video_fps({"streams": [{"codec_type": "video", "avg_frame_rate": "0/0"}]}) == 0.0
    assert (
        _probe_video_fps({"streams": [{"codec_type": "video", "avg_frame_rate": "bogus"}]}) == 0.0
    )


@pytest.mark.slow
def test_finalize_lifts_16fps_to_24fps_presentation(tmp_path: Path) -> None:
    """Sub-24fps sources (CausVid native 16fps) finalize at 24fps via
    motion-interpolated resampling, not frame duplication."""
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    config.video.fps = 16
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()
    out = tmp_path / "final-24.mp4"
    # Floors disabled: this pins the legacy 24fps presentation-floor path
    # at native geometry (default floors would lift to 1216x704@24).
    assert finalize_run(run_dir, out, fps=16, min_fps=0, min_width=0, min_height=0).exists()
    probed = validate_video(out, 768, 432, 24)
    assert probed["fps"] == pytest.approx(24.0, abs=0.5)
    # 2 fake segments x 48f @16fps = 6.0s of content; the 24fps presentation
    # carries ~144 frames over the same duration.
    assert probed["frames"] == pytest.approx(144, abs=4)
    duration = float(media_probe(out).get("format", {}).get("duration", 0.0))
    assert duration == pytest.approx(6.0, abs=0.3)


def test_build_final_audio_falls_back_without_takes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _init_run(run_dir)
    _commit_two(run_dir)
    # No takes ledger on this path only when audio/ is removed: blend must
    # degrade to the hard-splice concat, not raise.
    import shutil

    shutil.rmtree(run_dir / "audio", ignore_errors=True)
    usable = [paths.segment_dir(run_dir, "000000"), paths.segment_dir(run_dir, "000001")]
    dest = build_final_audio(run_dir, usable, tmp_path, 24, 48000, 2)
    assert dest.exists() and dest.stat().st_size > 0


@pytest.mark.slow
def test_generate_defaults_to_llama_director_with_offline_fallback(tmp_path: Path) -> None:
    """`generate` without --director resolves llama; the unstarted sidecar
    (this CPU-only image) degrades to the deterministic fallback via the
    worker §51 chain — the run still validates and finalizes."""
    from voyage.cli import main

    run_dir = tmp_path / "run"
    code = main(
        [
            "generate",
            "--backend",
            "fake",
            "--duration",
            "2s",
            "--style",
            "pastel neon line-art, peaceful",
            "--output",
            str(run_dir),
            "--run-id",
            "gen-qwen-default",
            "--seed",
            "11",
        ]
    )
    assert code == 0
    assert (run_dir / "final.mp4").exists()
    from voyage.config import load_config as _load

    config, _ = _load(run_dir / paths.CONFIG_FILENAME)
    assert config.director.backend == "llama"  # stored config matches the llama default
    state = read_state(run_dir)
    assert state.committed_segments == 1


# --- 088 fold: tests/test_prefetch_summary.py (5 tests, verbatim) ---
# Original module docstring (banner, issue ID stays greppable):
# """Issue 033 measurement: prefetch hit/miss aggregation (slim-testable).
#
# The commit path already emits `director_prefetch_hit/miss` per segment;
# `summarize_prefetch_outcome` turns those events into the hit rate the soak
# report needs before any prefetch restructuring. Pure reader — synthetic
# events only, no supervisor needed.
# """


def _events(*names: str) -> list[dict[str, Any]]:
    return [{"event": name, "segment_id": f"{index:06d}"} for index, name in enumerate(names)]


def test_empty_events_yield_no_rate() -> None:
    summary = summarize_prefetch_outcome([])
    assert summary == {"prefetch_hits": 0, "prefetch_misses": 0, "prefetch_hit_rate": None}


def test_mixed_events_count_and_rate() -> None:
    summary = summarize_prefetch_outcome(
        _events(
            "director_prefetch_miss",
            "director_prefetch_hit",
            "director_prefetch_hit",
            "segment_committed",
        )
    )
    assert summary["prefetch_hits"] == 2
    assert summary["prefetch_misses"] == 1
    assert summary["prefetch_hit_rate"] == 2 / 3


def test_unrelated_events_are_ignored() -> None:
    summary = summarize_prefetch_outcome(
        _events("segment_committed", "resource_gauges", "video_resumed")
    )
    assert summary == {"prefetch_hits": 0, "prefetch_misses": 0, "prefetch_hit_rate": None}


def test_all_hits_rate_is_one() -> None:
    summary = summarize_prefetch_outcome(_events("director_prefetch_hit", "director_prefetch_hit"))
    assert summary["prefetch_hit_rate"] == 1.0


def test_all_misses_rate_is_zero() -> None:
    summary = summarize_prefetch_outcome(
        _events("director_prefetch_miss", "director_prefetch_miss")
    )
    assert summary["prefetch_hit_rate"] == 0.0


# --- 088 fold: tests/test_prefetch_shutdown.py (2 tests, verbatim) ---
# Original module docstring (banner, issue ID stays greppable):
# """Issue 030: director prefetch is time-bounded and never gates shutdown.
#
# Why this file exists: the speculative `decide` prefetch ran with the
# 600 s worker default, and `stop_workers` used `shutdown(wait=False)`,
# which cannot stop a *running* future on non-daemon threads — a finished
# or SIGINTed run lingered at exit up to ten minutes behind a wedged
# director call. These tests pin the fix with stub directors (no GPU, no
# subprocess): the prefetch carries an explicit short budget, and
# `stop_workers` cancels/drains best-effort so it returns promptly even
# while a prefetch is wedged. The 137 resync contract is untouched — the
# prefetch still goes through `call()`, so a prefetch timeout still
# leaves the pipe resyncable (pinned by `test_rpc_timeout.py`).
# """


class _RecordingDirector:
    """Director stand-in: records `call` keyword arguments, returns a canned reply."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reply: dict[str, Any] = {"proposal": "stub"}

    def call(
        self,
        operation: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"operation": operation, "timeout": timeout})
        return dict(self.reply)

    def stop(self) -> None:
        """Stand-in for the worker shutdown (nothing to reap)."""


class _WedgedDirector:
    """Director stand-in: blocks inside `call` until released (wedged worker)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release = threading.Event()

    def call(
        self,
        operation: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"operation": operation, "timeout": timeout})
        self.release.wait(timeout=60.0)
        return {"proposal": "late"}

    def stop(self) -> None:
        """Stand-in for the worker shutdown (nothing to reap)."""


def _stub_decision() -> EvolutionDecision:
    return EvolutionDecision(
        decision_index=0,
        destination=DirectorDestination(canonical_name="lantern valley"),
    )


def _stubbed_supervisor(
    run_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    director: _RecordingDirector | _WedgedDirector,
) -> tuple[Supervisor, ProjectConfig]:
    """Supervisor with stubbed payload builder and director (no subprocess)."""
    initialize_run_directory(run_directory)
    config, _digest = load_config(run_directory / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_directory, config)
    monkeypatch.setattr(Supervisor, "_decide_payload", lambda self, *args, **kwargs: {})
    supervisor._director = director  # type: ignore[assignment]
    supervisor._workers_running = True
    supervisor._prefetch_executor = ThreadPoolExecutor(max_workers=1)
    return supervisor, config


def _submit_prefetch(supervisor: Supervisor, config: ProjectConfig, run_directory: Path) -> None:
    store = ConceptStore(run_directory)
    style = StyleSpec(prompt="pastel neon line-art, peaceful")
    supervisor._prefetch_decide_for_next(config, 0, _stub_decision(), store, style)


def _shutdown_executor(supervisor: Supervisor) -> None:
    executor, supervisor._prefetch_executor = supervisor._prefetch_executor, None
    supervisor._prefetch_future = None
    supervisor._prefetch_target = None
    if executor is not None:
        executor.shutdown(wait=True)


def test_prefetch_call_carries_bounded_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The speculative decide carries an explicit short budget, not the 600 s default."""
    assert math.isfinite(PREFETCH_TIMEOUT_SECONDS)
    assert 0.0 < PREFETCH_TIMEOUT_SECONDS < 600.0
    run_directory = tmp_path / "run"
    director = _RecordingDirector()
    supervisor, config = _stubbed_supervisor(run_directory, monkeypatch, director)
    try:
        _submit_prefetch(supervisor, config, run_directory)
        future = supervisor._prefetch_future
        assert future is not None
        assert future.result(timeout=15.0) == {"proposal": "stub"}
        assert len(director.calls) == 1
        assert director.calls[0]["operation"] == "decide"
        assert director.calls[0]["timeout"] == PREFETCH_TIMEOUT_SECONDS
    finally:
        _shutdown_executor(supervisor)


def test_stop_workers_returns_promptly_with_wedged_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stop_workers` never stalls behind a wedged prefetch thread."""
    run_directory = tmp_path / "run"
    director = _WedgedDirector()
    supervisor, config = _stubbed_supervisor(run_directory, monkeypatch, director)
    executor = supervisor._prefetch_executor
    _submit_prefetch(supervisor, config, run_directory)
    assert supervisor._prefetch_future is not None
    deadline = time.monotonic() + 10.0
    while not director.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert director.calls, "prefetch thread never entered the director call"
    started = time.monotonic()
    supervisor.stop_workers()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0
    assert supervisor._prefetch_future is None
    assert supervisor._prefetch_executor is None
    assert director.calls[0]["timeout"] == PREFETCH_TIMEOUT_SECONDS
    director.release.set()
    assert executor is not None
    executor.shutdown(wait=True)

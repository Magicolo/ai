"""Burst-parallel hardening: locked load-once, prepared-once, stream null paths, warp-grid memo.

Pins the torch best-practice fixes in `voyage/workers/augment_worker.py`:
modules are thread-safe to READ but not to WRITE, so the resident-net
cache check-then-set, `_prepare_model` (.half/.to/.eval), and the FILM/RIFE
warp-grid memo dicts are serialized; steady-state forwards run lock-free
with per-thread CUDA streams. CPU-only except the warp-grid test, which
needs real torch (`importorskip` — absent from the slim image).
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from voyage.workers import augment_worker


def _stub_torch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Minimal torch stub: device resolution + dtype sentinels (no tensors)."""

    class _FakeDevice:
        def __init__(self, name: str) -> None:
            self.type = name.split(":")[0]

        def __str__(self) -> str:
            return "cpu"

    torch_stub = ModuleType("torch")
    torch_stub.device = _FakeDevice  # type: ignore[attr-defined]
    torch_stub.float16 = "float16"  # type: ignore[attr-defined]
    torch_stub.float32 = "float32"  # type: ignore[attr-defined]
    cuda_stub = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    return torch_stub


@pytest.fixture()
def _clean_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the resident-net caches + prepared flags per test."""
    _stub_torch(monkeypatch)
    augment_worker._ESRGAN_CACHE.clear()
    augment_worker._FILM_CACHE.clear()
    augment_worker._RIFE_CACHE.clear()
    augment_worker._PREPARED_MODEL_KEYS.clear()
    augment_worker._DEVICE_FALLBACK_WARNED = False


def _run_threads(count: int, target: Any) -> None:
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_prepared_model_loads_once_under_threads(
    _clean_caches: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """8 racing first calls decode disk once and prepare once (no double load)."""
    weights = tmp_path / "rife.safetensors"
    weights.write_bytes(b"fake-rife")
    key = augment_worker._model_cache_key(weights, "cpu")
    loads = 0
    prepares = 0
    prepare_lock = threading.Lock()

    def _counting_loader() -> Any:
        nonlocal loads
        loads += 1
        return types.SimpleNamespace(eval=lambda: None)

    def _counting_prepare(model: Any, device: str) -> tuple[Any, Any]:
        nonlocal prepares
        with prepare_lock:
            prepares += 1
        return types.SimpleNamespace(type="cpu"), "float32"

    monkeypatch.setattr(augment_worker, "_prepare_model", _counting_prepare)
    seen: list[Any] = []
    seen_lock = threading.Lock()

    def _call() -> None:
        model, _, _ = augment_worker._get_prepared_model(
            augment_worker._RIFE_CACHE, key, "cpu", _counting_loader
        )
        with seen_lock:
            seen.append(model)

    _run_threads(8, _call)
    assert loads == 1
    assert prepares == 1
    assert len(seen) == 8
    assert all(model is seen[0] for model in seen)


def test_prepared_hit_skips_prepare(
    _clean_caches: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second wave on a prepared key reloads nothing and re-prepares nothing."""
    weights = tmp_path / "rife.safetensors"
    weights.write_bytes(b"fake-rife")
    key = augment_worker._model_cache_key(weights, "cpu")
    loads = 0
    prepares = 0

    def _counting_loader() -> Any:
        nonlocal loads
        loads += 1
        return types.SimpleNamespace(eval=lambda: None)

    def _counting_prepare(model: Any, device: str) -> tuple[Any, Any]:
        nonlocal prepares
        prepares += 1
        return types.SimpleNamespace(type="cpu"), "float32"

    monkeypatch.setattr(augment_worker, "_prepare_model", _counting_prepare)
    first, _, _ = augment_worker._get_prepared_model(
        augment_worker._RIFE_CACHE, key, "cpu", _counting_loader
    )
    _run_threads(
        8,
        lambda: augment_worker._get_prepared_model(
            augment_worker._RIFE_CACHE, key, "cpu", _counting_loader
        ),
    )
    assert loads == 1
    assert prepares == 1
    second, _, _ = augment_worker._get_prepared_model(
        augment_worker._RIFE_CACHE, key, "cpu", _counting_loader
    )
    assert second is first


def test_evict_clears_prepared_flags(
    _clean_caches: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Evict drops nets + prepared flags, so the next call reloads (no stale eval)."""
    weights = tmp_path / "rife.safetensors"
    weights.write_bytes(b"fake-rife")
    key = augment_worker._model_cache_key(weights, "cpu")
    loads = 0

    def _counting_loader() -> Any:
        nonlocal loads
        loads += 1
        return types.SimpleNamespace(eval=lambda: None)

    def _fake_prepare(model: Any, device: str) -> tuple[Any, Any]:
        return types.SimpleNamespace(type="cpu"), "float32"

    monkeypatch.setattr(augment_worker, "_prepare_model", _fake_prepare)
    augment_worker._get_prepared_model(augment_worker._RIFE_CACHE, key, "cpu", _counting_loader)
    assert augment_worker.evict_augment_models() == 1
    assert key not in augment_worker._PREPARED_MODEL_KEYS
    augment_worker._get_prepared_model(augment_worker._RIFE_CACHE, key, "cpu", _counting_loader)
    assert loads == 2


def test_stream_context_null_paths(_clean_caches: None) -> None:
    """CPU devices / None streams take the null context (current stream, no torch.cuda)."""
    cpu = types.SimpleNamespace(type="cpu")
    with augment_worker._inference_stream_context(cpu, object()):
        pass
    with augment_worker._inference_stream_context(cpu, None):
        pass


def test_warp_grids_concurrent_geometries(
    _clean_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed-geometry `_grids_for` races never KeyError and never lose a key.

    The old check → `clear()` → compute sequence let one thread's clear
    delete another's stored grid; the memo build now holds the lock and
    keys are never deleted, so concurrent geometries each resolve exactly
    once and same-geometry calls return the identical object.
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    torch = pytest.importorskip("torch", reason="warp grids need torch (absent from slim image)")
    net = augment_worker._build_rife_net(head_channels=4, block_channels=(8, 8))
    geometries = [(64, 128), (128, 64), (96, 96), (64, 64)]
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def _hammer(geometry: tuple[int, int]) -> None:
        height, width = geometry
        try:
            for _ in range(4):
                net._grids_for(height, width, "cpu")
        except BaseException as exc:  # noqa: BLE001 — collected across threads, re-raised below
            with errors_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=_hammer, args=(geometry,))
        for geometry in geometries
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    for height, width in geometries:
        assert (height, width, "cpu") in net.warp_grids
    repeat = net._grids_for(64, 128, "cpu")
    assert repeat is net.warp_grids[(64, 128, "cpu")]
    assert torch is not None


# --- Burst-parallel scoping, gate, and settle (S4) ---------------------------


def _parallel_segment(run_dir: Path, segment_id: str = "000000", *, frames: int = 8) -> Path:
    """Committed segment dir (mirrors the poller contract tests)."""
    import json

    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": frames},
        "checksums": {"video.mp4": "abc123"},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _parallel_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for position in range(count):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-frame")
        written.append(frame)
    return written


def _parallel_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
    assert dest_dir.is_dir()
    return list(frame_paths)


def _parallel_interp(frame_paths: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = (len(frame_paths) - 1) * multiplier + 1
    written = []
    for position in range(expected):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-interp")
        written.append(frame)
    return written


def _parallel_upscale_kwargs(**overrides):  # type: ignore[no-untyped-def]
    params = {
        "weights_path": Path("/models/realesrgan/realesr-animevideov3.pth"),
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "chunk_frames": 4,
        "decode_fn": _parallel_decode,
        "upscale_fn": _parallel_upscale,
    }
    params.update(overrides)
    return params


def _parallel_tasks_kwargs(**overrides):  # type: ignore[no-untyped-def]
    params = {
        "weights_key": "weights-abc",
        "out_width": 1216,
        "out_height": 704,
        "source_fps_key": 24,
        "upscale_factor": 2,
        "chunk_frames": 4,
        "multiplier": 4,
        "crf": 15,
        "preset": "veryfast",
    }
    params.update(overrides)
    return params


def test_parallel_gate_matrix() -> None:
    """Armed iff RIFE-only with both worker cards visible (pure gate)."""
    from voyage.augment_parallel import parallel_model_pass_armed

    assert parallel_model_pass_armed(interp_backend="rife", devices=("cuda:1", "cuda:0"))
    assert parallel_model_pass_armed(interp_backend="rife", devices=("cuda:0", "cuda:1"))
    assert not parallel_model_pass_armed(interp_backend="film", devices=("cuda:1", "cuda:0"))
    assert not parallel_model_pass_armed(interp_backend="rife", devices=("cuda:1",))
    assert not parallel_model_pass_armed(interp_backend="rife", devices=("cuda:0",))
    assert not parallel_model_pass_armed(interp_backend="rife", devices=())
    assert not parallel_model_pass_armed(interp_backend="rife", devices=("cuda:0", "cuda:2"))
    assert not parallel_model_pass_armed(interp_backend="", devices=("cuda:1", "cuda:0"))


def test_chunk_ids_scopes_upscale(tmp_path: Path) -> None:
    """chunk_ids=[0] renders one chunk and counts the other skipped."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _parallel_segment(tmp_path, frames=8)
    first = upscale_poll_once(tmp_path, **_parallel_upscale_kwargs(chunk_ids=[0]))
    assert first.chunks_done == 1
    assert first.chunks_skipped == 1
    second = upscale_poll_once(tmp_path, **_parallel_upscale_kwargs(chunk_ids=[1]))
    assert second.chunks_done == 1
    assert second.chunks_skipped == 1


def test_chunk_ids_out_of_range_counts_skipped(tmp_path: Path) -> None:
    """chunk_ids nobody owns renders nothing but counts every chunk skipped."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _parallel_segment(tmp_path, frames=8)
    result = upscale_poll_once(tmp_path, **_parallel_upscale_kwargs(chunk_ids=[7]))
    assert result.chunks_done == 0
    assert result.chunks_skipped == 2


def test_chunk_ids_scopes_interp(tmp_path: Path) -> None:
    """Interp chunk_ids=[1] renders only the second chunk."""
    from voyage.augment_interp_poller import interp_poll_once
    from voyage.augment_upscale_poller import upscale_poll_once

    _parallel_segment(tmp_path, frames=8)
    upscale_poll_once(tmp_path, **_parallel_upscale_kwargs())
    result = interp_poll_once(
        tmp_path,
        weights_path=Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        multiplier=4,
        interp_fn=_parallel_interp,
        chunk_ids=[1],
    )
    assert result.chunks_done == 1
    assert result.chunks_skipped == 1


def test_prune_partials_flag(tmp_path: Path) -> None:
    """prune_partials=False leaves a planted partial; default sweeps it."""
    from voyage.augment_upscale_poller import upscale_poll_once

    _parallel_segment(tmp_path, frames=8)
    upscale_poll_once(tmp_path, **_parallel_upscale_kwargs())
    plan_dir = next(iter((tmp_path / "augment").iterdir()))
    planted = plan_dir / "stale.partial"
    planted.write_bytes(b"stale")
    upscale_poll_once(tmp_path, **_parallel_upscale_kwargs(prune_partials=False))
    assert planted.is_file()
    upscale_poll_once(tmp_path, **_parallel_upscale_kwargs())
    assert not planted.exists()


def test_parallel_tasks_and_settle(tmp_path: Path) -> None:
    """Fresh segment queues 2 tasks; settle fails loud until both stages land."""
    import pytest

    from voyage.augment_interp_poller import interp_poll_once
    from voyage.augment_parallel import _verify_parallel_settle, build_parallel_tasks
    from voyage.augment_upscale_poller import upscale_poll_once
    from voyage.errors import MediaError

    _parallel_segment(tmp_path, frames=8)
    tasks = build_parallel_tasks(tmp_path, **_parallel_tasks_kwargs())
    assert [task.chunk_index for task in tasks] == [0, 1]
    settle_kwargs = _parallel_tasks_kwargs()
    del settle_kwargs["multiplier"]
    with pytest.raises(MediaError):
        _verify_parallel_settle(tmp_path, **settle_kwargs)
    upscale_poll_once(tmp_path, **_parallel_upscale_kwargs())
    with pytest.raises(MediaError):
        _verify_parallel_settle(tmp_path, **settle_kwargs)
    interp_poll_once(
        tmp_path,
        weights_path=Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        multiplier=4,
        interp_fn=_parallel_interp,
    )
    _verify_parallel_settle(tmp_path, **settle_kwargs)


def test_parallel_branch_runs_audio_first_then_bidirectional_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audio-first branch: music takes, then the bidirectional model pass.

    With RIFE + both GPUs visible finalize takes the audio-first branch
    (not the model∥music fork): music renders on cuda:0 while both cards
    are free, then the work-stealing deque renders every chunk (worker A
    front-to-back on cuda:1, worker B back-to-front on cuda:0). Pins: the
    parallel driver ran with the forwarded multiplier and backend, music
    completed first, and the output exists.
    """
    import shutil

    import voyage.audio_finalize as audio_finalize_module
    import voyage.augment as augment_module
    import voyage.augment_finalize as finalize_module
    import voyage.augment_parallel as parallel_module
    from tests.conftest import initialize_run_directory
    from voyage.augment import AugmentWeights
    from voyage.media import finalize_run
    from voyage.persistence import read_effective_config
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="parfirst", style="pastel neon line-art, peaceful")
    config = read_effective_config(run_dir)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()

    seen: dict[str, Any] = {}
    music_done = threading.Event()

    def _fake_parallel_pass(run_dir_arg: Path, **kwargs: Any) -> dict[str, float]:
        seen["parallel_ran"] = True
        seen["multiplier"] = kwargs["multiplier"]
        seen["interp_backend"] = kwargs["interp_backend"]
        seen["music_before_parallel"] = music_done.is_set()
        return {}

    def _fake_drain(run_dir_arg: Path, usable: list[Path], **kwargs: Any) -> tuple[Path, int]:
        intermediate = tmp_path / "model_intermediate.mp4"
        shutil.copy(usable[0] / "video.mp4", intermediate)
        return (intermediate, 24)

    real_ensure = audio_finalize_module.ensure_deferred_for_finalize

    def _recording_ensure(**kwargs: Any) -> Any:
        result = real_ensure(**kwargs)
        music_done.set()
        return result

    def _stub_weights(base: Path | str) -> AugmentWeights:
        # `weights_key_for` hashes real files — stand in three tiny legs.
        legs = Path(str(base))
        legs.mkdir(parents=True, exist_ok=True)
        (legs / "film").write_bytes(b"film")
        (legs / "esrgan").write_bytes(b"esrgan")
        (legs / "rife").write_bytes(b"rife")
        return AugmentWeights(
            film=legs / "film",
            realesrgan=legs / "esrgan",
            rife=legs / "rife",
        )

    monkeypatch.setattr(augment_module, "resolve_augment_weights", _stub_weights)
    monkeypatch.setattr(augment_module, "augment_devices", lambda: ("cuda:0", "cuda:1"))
    monkeypatch.setattr(augment_module, "model_pass_devices", lambda: ("cuda:1",))
    monkeypatch.setattr(parallel_module, "run_parallel_model_pass", _fake_parallel_pass)
    monkeypatch.setattr(finalize_module, "_drain_to_intermediate", _fake_drain)
    monkeypatch.setattr(audio_finalize_module, "ensure_deferred_for_finalize", _recording_ensure)
    out = tmp_path / "parfirst.mp4"
    finalize_run(run_dir, out, upscale=2, interpolate=1, models_dir=tmp_path / "models")
    assert out.exists() and out.stat().st_size > 0
    assert seen["parallel_ran"] is True
    assert seen["multiplier"] == 1
    assert seen["interp_backend"] == "rife"
    assert seen["music_before_parallel"] is True


class _StubProgressTracker:
    """Recording stand-in for the console bar tracker."""

    def __init__(self) -> None:
        self.done = 0
        self.total: int | None = None

    def update(self, advance: int) -> None:
        self.done += advance

    def set_total(self, total: int) -> None:
        self.total = total

    def set_extra(self, extra: str) -> None:
        del extra


class _StubBarContext:
    """Minimal context manager yielding the recording tracker."""

    def __init__(self, tracker: _StubProgressTracker) -> None:
        self.tracker = tracker

    def __enter__(self) -> _StubProgressTracker:
        return self.tracker

    def __exit__(self, *exc: Any) -> bool:
        return False


class _StubProgress:
    """Progress double with only `.bar()` (the driver never reads `.verbose`)."""

    def __init__(self) -> None:
        self.trackers: list[_StubProgressTracker] = []
        self.labels: list[str] = []

    def bar(self, label: str, total: int | None = None) -> _StubBarContext:
        tracker = _StubProgressTracker()
        tracker.total = total
        self.trackers.append(tracker)
        self.labels.append(label)
        return _StubBarContext(tracker)

    def info(self, message: str) -> None:
        del message


def test_parallel_driver_reports_chunk_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-leg chunk bars advance once per finished chunk half (kaolin was silent)."""
    import functools

    import voyage.augment_parallel as parallel_module
    from voyage.augment import AugmentWeights
    from voyage.augment_interp_poller import interp_poll_once
    from voyage.augment_upscale_poller import upscale_poll_once

    def _no_preload(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    monkeypatch.setattr(parallel_module, "_preload_worker_models", _no_preload)

    _parallel_segment(tmp_path, frames=8)
    legs = tmp_path / "legs"
    legs.mkdir(parents=True, exist_ok=True)
    (legs / "film").write_bytes(b"film")
    (legs / "esrgan").write_bytes(b"esrgan")
    (legs / "rife").write_bytes(b"rife")
    weights = AugmentWeights(
        film=legs / "film",
        realesrgan=legs / "esrgan",
        rife=legs / "rife",
    )
    progress = _StubProgress()
    parallel_module.run_parallel_model_pass(
        tmp_path,
        weights=weights,
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        source_fps=24,
        upscale_factor=2,
        multiplier=4,
        chunk_frames=4,
        crf=15,
        preset="veryfast",
        interp_backend="rife",
        upscale_poll_fn=functools.partial(
            upscale_poll_once,
            decode_fn=_parallel_decode,
            upscale_fn=_parallel_upscale,
        ),
        interp_poll_fn=functools.partial(
            interp_poll_once,
            interp_fn=_parallel_interp,
        ),
        progress=progress,
    )
    assert progress.labels == ["upscale chunks", "interpolate chunks"]
    assert len(progress.trackers) == 2
    for bar in progress.trackers:
        assert bar.total == 2
        assert bar.done == 2


def test_drain_to_intermediate_initializes_timings(tmp_path):
    """Regression (live kaolin `KeyError: 'drain_s'`, 2026-10-06).

    The bidirectional parallel branch calls `_drain_to_intermediate`
    directly with a timings dict the durable entry never initialized —
    the drain must zero-init its own keys instead of assuming them.
    """
    import types

    from voyage.augment_finalize import _drain_to_intermediate

    segment_dir = _parallel_segment(tmp_path, frames=8)
    weights = types.SimpleNamespace(
        realesrgan=tmp_path / "esrgan.pth",
        rife=tmp_path / "rife.safetensors",
    )

    def _stub_drain(plan_dir, *, out_fps):
        del out_fps
        return types.SimpleNamespace(
            chunks_drained=2,
            intermediate_mp4=plan_dir / "chunk_00.mp4",
        )

    def _stub_concat(parts, out):
        assert len(parts) == 1
        out.write_bytes(b"fake-intermediate")
        return out

    timings: dict[str, float] = {}
    final, fps = _drain_to_intermediate(
        tmp_path,
        [segment_dir],
        weights=weights,
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        source_fps=24.0,
        source_fps_key=24,
        upscale_factor=2,
        multiplier=1,
        crf=15,
        preset="veryfast",
        device="cpu",
        work_dir=tmp_path / "model_pass",
        interp_backend="rife",
        drain_fn=_stub_drain,
        concat_fn=_stub_concat,
        timings=timings,
        progress=None,
    )
    assert fps == 24
    assert final.name.endswith(".mp4")
    assert timings["drain_s"] >= 0.0
    assert timings["chunks_drained"] == 2.0
    assert timings["concat_s"] >= 0.0


def _gated_weights(tmp_path: Path) -> Any:
    from voyage.augment import AugmentWeights

    legs = tmp_path / "legs"
    legs.mkdir(parents=True, exist_ok=True)
    (legs / "film").write_bytes(b"film")
    (legs / "esrgan").write_bytes(b"esrgan")
    (legs / "rife").write_bytes(b"rife")
    return AugmentWeights(
        film=legs / "film",
        realesrgan=legs / "esrgan",
        rife=legs / "rife",
    )


def _run_gated_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate: threading.Event,
    preloads: list[str],
) -> Any:
    import functools

    import voyage.augment_parallel as parallel_module
    from voyage.augment_interp_poller import interp_poll_once
    from voyage.augment_upscale_poller import upscale_poll_once

    def _counting_preload(weights: Any, weights_key: str, interp_backend: str, device: str) -> None:
        preloads.append(device)

    monkeypatch.setattr(parallel_module, "_preload_worker_models", _counting_preload)
    _parallel_segment(tmp_path, frames=8)
    progress = _StubProgress()
    parallel_module.run_parallel_model_pass(
        tmp_path,
        weights=_gated_weights(tmp_path),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        source_fps=24,
        upscale_factor=2,
        multiplier=4,
        chunk_frames=4,
        crf=15,
        preset="veryfast",
        interp_backend="rife",
        upscale_poll_fn=functools.partial(
            upscale_poll_once,
            decode_fn=_parallel_decode,
            upscale_fn=_parallel_upscale,
        ),
        interp_poll_fn=functools.partial(
            interp_poll_once,
            interp_fn=_parallel_interp,
        ),
        progress=progress,
        second_worker_gate=gate,
    )
    return progress


def test_second_worker_waits_for_audio_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker B holds cuda:0 until the audio gate releases it (2-stream fork)."""
    preloads: list[str] = []
    progress = _run_gated_pass(tmp_path, monkeypatch, threading.Event(), preloads)
    # Worker A drained both tasks alone; worker B never preloaded on cuda:0.
    assert preloads == ["cuda:1"]
    assert progress.labels == ["upscale chunks", "interpolate chunks"]
    for bar in progress.trackers:
        assert bar.total == 2
        assert bar.done == 2


def test_second_worker_starts_when_gate_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A released gate starts both workers immediately (audio-first path)."""
    gate = threading.Event()
    gate.set()
    preloads: list[str] = []
    _run_gated_pass(tmp_path, monkeypatch, gate, preloads)
    assert sorted(preloads) == ["cuda:0", "cuda:1"]


def test_parallel_fails_fast_when_workers_lack_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Joint tasks need `sources=` workers — fail fast, never stall at settle."""
    import voyage.augment_joints as joints_module
    import voyage.augment_parallel as parallel_module
    from voyage.errors import MediaError

    _parallel_segment(tmp_path, "000000", frames=8)
    _parallel_segment(tmp_path, "000001", frames=8)
    joint_dir = tmp_path / "joint_000000_000001"
    monkeypatch.setattr(
        joints_module,
        "ensure_joint_units",
        lambda *args, **kwargs: [
            joints_module.JointUnit(
                joint_dir=joint_dir,
                joint_video=joint_dir / "joint.mp4",
                joint_key="jk",
                left_id="000000",
                right_id="000001",
                left_frames=8,
                right_frames=8,
            )
        ],
    )

    def _old_upscale(
        run_dir: Path,
        *,
        weights_path: Path,
        weights_key: str,
        out_width: int,
        out_height: int,
        out_fps: int,
        segment_ids: list[str] | None = None,
    ) -> Any:
        raise AssertionError("workers must never run past the fail-fast")

    def _old_interp(
        run_dir: Path,
        *,
        weights_path: Path,
        weights_key: str,
        out_width: int,
        out_height: int,
        out_fps: int,
        segment_ids: list[str] | None = None,
    ) -> Any:
        raise AssertionError("workers must never run past the fail-fast")

    with pytest.raises(MediaError, match="sources="):
        parallel_module.run_parallel_model_pass(
            tmp_path,
            weights=_gated_weights(tmp_path),
            weights_key="weights-abc",
            out_width=1216,
            out_height=704,
            source_fps=24,
            upscale_factor=2,
            multiplier=4,
            chunk_frames=4,
            crf=15,
            preset="veryfast",
            interp_backend="rife",
            upscale_poll_fn=_old_upscale,
            interp_poll_fn=_old_interp,
        )

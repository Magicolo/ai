"""Rank-2 worker/media perf fixes: 046/047/048/153/156/158 (TDD, CPU-only).

No torch/GPU required except where marked: torch-touching paths use
`pytest.importorskip` (skip cleanly in the slim image) or stubbed
`sys.modules["torch"]` fakes. Speed claims stay code-arithmetic +
synthetic-tensor tests, never live GPU renders.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import voyage.augment as augment_module
from voyage.errors import MediaError


def _completed(argv: list[str], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Completed-process stub for ffmpeg arg-list tests."""
    return subprocess.CompletedProcess(argv, returncode, "", "")


# ---------------------------------------------------------------------------
# 046 — augment chunk fast-seek + fresh dest_dir
# ---------------------------------------------------------------------------


def test_046_decode_rejects_stale_dest_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dest_dir already holding frame_*.png must fail before decoding."""

    def _fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _fake)
    dest = tmp_path / "chunk_00"
    dest.mkdir()
    (dest / "frame_000000.png").write_bytes(b"stale-payload")
    with pytest.raises(MediaError, match="not fresh"):
        augment_module.ffmpeg_decode_chunk(tmp_path / "source.mp4", dest, 0, 4)


def test_046_decode_fast_seek_uses_input_ss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With fps given, chunk K seeks to start/fps before -i (no linear rescan)."""
    calls: list[list[str]] = []

    def _fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        dest_dir = Path(argv[-1]).parent
        for index in range(32):
            (dest_dir / f"frame_{index:06d}.png").write_bytes(b"fake-png-payload")
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _fake)
    frames = augment_module.ffmpeg_decode_chunk(
        tmp_path / "source.mp4", tmp_path / "chunk_09", 288, 32, fps=32.0
    )
    assert len(frames) == 32
    (command,) = calls
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == f"{288 / 32.0:.6f}"
    vf_arg = command[command.index("-vf") + 1]
    assert "between(n\\,0\\,31)" in vf_arg


def test_046_decode_exact_fallback_without_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without fps the exact from-start select path is preserved."""
    calls: list[list[str]] = []

    def _fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        dest_dir = Path(argv[-1]).parent
        for index in range(4):
            (dest_dir / f"frame_{index:06d}.png").write_bytes(b"fake-png-payload")
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _fake)
    augment_module.ffmpeg_decode_chunk(tmp_path / "source.mp4", tmp_path / "chunk_00", 32, 4)
    (command,) = calls
    assert "-ss" not in command
    vf_arg = command[command.index("-vf") + 1]
    assert "between(n\\,32\\,35)" in vf_arg


def test_046_decode_rejects_bad_fps(tmp_path: Path) -> None:
    """Non-finite/non-positive fps fails before any ffmpeg spawn."""
    with pytest.raises(ValueError, match="fps"):
        augment_module.ffmpeg_decode_chunk(tmp_path / "s.mp4", tmp_path / "c", 0, 4, fps=0.0)
    with pytest.raises(ValueError, match="fps"):
        augment_module.ffmpeg_decode_chunk(
            tmp_path / "s.mp4", tmp_path / "c", 0, 4, fps=float("nan")
        )
    with pytest.raises(TypeError, match="fps"):
        augment_module.ffmpeg_decode_chunk(
            tmp_path / "s.mp4",
            tmp_path / "c",
            0,
            4,
            fps=True,
        )


# ---------------------------------------------------------------------------
# 047 — augment worker resident cache + list halving + gc + timing split
# ---------------------------------------------------------------------------


def test_047_model_cache_key_and_evict() -> None:
    """Cache keys separate (weights, device); evict drops everything."""
    from voyage.workers import augment_worker as worker

    worker.evict_augment_models()
    assert worker._model_cache_key(Path("/m/up.pth"), "cuda:0") == ("/m/up.pth", "cuda:0")
    assert worker._model_cache_key(Path("/m/up.pth"), "cuda:1") != worker._model_cache_key(
        Path("/m/up.pth"), "cuda:0"
    )
    assert worker.evict_augment_models() == 0


def test_047_prepare_model_halves_before_moving() -> None:
    """fp16 cast lands before the host-to-device move (no fp32 H2D transient)."""
    from voyage.workers import augment_worker as worker

    source = inspect.getsource(worker._prepare_model)
    assert source.index("half") < source.index(".to(")


def test_047_run_stacked_collects_before_empty_cache() -> None:
    """The OOM branch runs gc.collect() before empty_cache (the measured lesson)."""
    from voyage.workers import augment_worker as worker

    source = inspect.getsource(worker._run_stacked)
    assert "gc.collect()" in source
    assert source.index("gc.collect()") < source.index("empty_cache")


def test_047_upscale_halves_frame_list_before_stacking() -> None:
    """upscale_frames never stacks the full batch first (the chunking OOM)."""
    from voyage.workers import augment_worker as worker

    source = inspect.getsource(worker.upscale_frames)
    assert "torch.stack([torch.as_tensor(frame" not in source
    assert "_run_frame_batches" in source or "halv" in source.lower()


def test_047_upscale_reports_load_vs_infer_ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic-tensor run splits load_ms from infer_ms (benchmark attribution)."""
    torch = pytest.importorskip("torch", reason="timing split needs torch (absent in slim)")
    from voyage.workers import augment_worker as worker

    weights = tmp_path / "up.pth"
    weights.write_bytes(b"x" * 16)
    worker.evict_augment_models()

    def _stub(path: Path) -> Any:
        del path
        return worker._build_rrdb_net()

    monkeypatch.setattr(worker, "_load_esrgan_net", _stub)
    rows = torch.linspace(0.0, 1.0, 8).unsqueeze(1).expand(8, 8)
    frame = torch.stack([rows, rows, rows])
    timings: dict[str, float] = {}
    try:
        worker.upscale_frames([frame], weights, scale=1, device="cpu", timings=timings)
    finally:
        worker.evict_augment_models()
    assert set(timings) >= {"load_ms", "infer_ms"}
    assert timings["load_ms"] >= 0.0 and timings["infer_ms"] >= 0.0


# ---------------------------------------------------------------------------
# 153 — SFX stem cache fuzzy match + tmp+replace + prune/validate dedupe
# ---------------------------------------------------------------------------


def test_153_fuzzy_duration_hit() -> None:
    """A 1e-9 float perturbation still hits (same caption/seed/size)."""
    from voyage.sfx_finalize import SfxWindow, _stem_cache_hit

    window = SfxWindow("w0000", 0.0, 8.0, "rain", 7)
    record: dict[str, Any] = {
        "window_id": "w0000",
        "caption": "rain",
        "seed": 7,
        "model_size": "small_44k",
        "duration": 8.0 + 1e-9,
        "path": "audio/sfx/w0000.wav",
    }
    assert _stem_cache_hit(record, window, "small_44k") is True


def test_153_beyond_tolerance_misses() -> None:
    """Real plan changes (caption/size/duration drift) still miss."""
    from voyage.sfx_finalize import SfxWindow, _stem_cache_hit

    window = SfxWindow("w0000", 0.0, 8.0, "rain", 7)
    base: dict[str, Any] = {
        "window_id": "w0000",
        "caption": "rain",
        "seed": 7,
        "model_size": "small_44k",
        "duration": 8.0,
        "path": "audio/sfx/w0000.wav",
    }
    assert _stem_cache_hit({**base, "caption": "thunder"}, window, "small_44k") is False
    assert _stem_cache_hit({**base, "duration": 5.0}, window, "small_44k") is False
    assert _stem_cache_hit(base, window, "large_44k_v2") is False


class _FailingSfxWorker:
    """SubprocessWorker double whose render always fails (153 delete-before-success)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        del op, payload
        raise RuntimeError("simulated transient render failure")

    def stop(self) -> None:
        return None


def test_153_failed_render_keeps_old_stem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure leaves the old stem + ledger intact (no missing-stem)."""
    from voyage import sfx_finalize as finalize
    from voyage.sfx_finalize import append_sfx_window, render_sfx_bed, validate_sfx_ledger

    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _FailingSfxWorker)
    run_dir = tmp_path / "run"
    sfx_dir = run_dir / "audio" / "sfx"
    sfx_dir.mkdir(parents=True)
    (sfx_dir / "w0000.wav").write_bytes(b"OLD-STEM")
    append_sfx_window(
        sfx_dir / "sfx.jsonl",
        finalize.SfxWindow("w0000", 0.0, 8.0, "old-caption", 3),
        path="audio/sfx/w0000.wav",
        model_size="small_44k",
    )
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    with pytest.raises(RuntimeError, match="transient"):
        render_sfx_bed(
            run_dir,
            final_video,
            8.0,
            [(0.0, 8.0, "changed-caption")],
            tmp_path,
            "mmaudio",
            "/models",
            "cpu",
            "small_44k",
            3,
            48000,
            2,
            1,
        )
    assert (sfx_dir / "w0000.wav").read_bytes() == b"OLD-STEM"
    assert "missing" not in " ".join(validate_sfx_ledger(run_dir, 8.0))


class _WritingSfxWorker:
    """SubprocessWorker double that renders valid silent WAVs to the path given."""

    seen: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        import wave

        del op
        out = str(payload["output_path"])
        type(self).seen.append(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        duration = float(payload["duration_seconds"])
        frames = max(1, int(48000 * duration))
        with wave.open(out, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0" * frames * 4)
        return {"duration_seconds": duration}

    def stop(self) -> None:
        return None


def test_153_render_uses_tmp_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers render to a .partial.wav temp; success atomically replaces the stem."""
    from voyage.sfx_finalize import render_sfx_bed

    _WritingSfxWorker.seen = []
    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _WritingSfxWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    render_sfx_bed(
        run_dir,
        final_video,
        4.0,
        [(0.0, 4.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        0,
        48000,
        2,
        1,
    )
    assert _WritingSfxWorker.seen
    assert all(path.endswith(".partial.wav") for path in _WritingSfxWorker.seen)
    stem = run_dir / "audio" / "sfx" / "w0000.wav"
    assert stem.exists() and stem.stat().st_size > 44
    assert list((run_dir / "audio" / "sfx").glob("*.partial.wav")) == []


def test_153_validate_dedupes_duplicate_window_ids(tmp_path: Path) -> None:
    """Re-rendered (duplicate) + out-of-order ledger lines still validate clean."""
    import json

    from voyage.sfx_finalize import validate_sfx_ledger

    sfx_dir = tmp_path / "audio" / "sfx"
    sfx_dir.mkdir(parents=True)
    (sfx_dir / "w0000.wav").write_bytes(b"RIFF" + b"\0" * 32)
    (sfx_dir / "w0001.wav").write_bytes(b"RIFF" + b"\0" * 32)
    lines = [
        {"window_id": "w0001", "start": 7.0, "duration": 5.0, "path": "audio/sfx/w0001.wav"},
        {"window_id": "w0000", "start": 0.0, "duration": 8.0, "path": "audio/sfx/w0000.wav"},
        {"window_id": "w0000", "start": 0.0, "duration": 8.0, "path": "audio/sfx/w0000.wav"},
    ]
    (sfx_dir / "sfx.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    assert validate_sfx_ledger(tmp_path, 12.0) == []


def test_153_prunes_stale_partials_at_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Crashed-render .partial.wav files are pruned before planning (no orphans)."""
    from voyage.sfx_finalize import render_sfx_bed

    _WritingSfxWorker.seen = []
    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _WritingSfxWorker)
    run_dir = tmp_path / "run"
    sfx_dir = run_dir / "audio" / "sfx"
    sfx_dir.mkdir(parents=True)
    (sfx_dir / "w0000.partial.wav").write_bytes(b"CRASH-LEFTOVER")
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    render_sfx_bed(
        run_dir,
        final_video,
        4.0,
        [(0.0, 4.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        0,
        48000,
        2,
        1,
    )
    leftovers = list(sfx_dir.glob("*.partial.wav"))
    assert leftovers == []
    assert not any(leftover.read_bytes() == b"CRASH-LEFTOVER" for leftover in sfx_dir.iterdir())


# ---------------------------------------------------------------------------
# 156 — sfx/audio bench VRAM guards + session device index + report device
# ---------------------------------------------------------------------------


def _stub_torch(cuda: Any) -> None:
    """Install a stub torch module (cuda side only) for guard tests."""
    module = types.ModuleType("torch")
    module.cuda = cuda  # type: ignore[attr-defined]
    sys.modules["torch"] = module


def test_156_sfx_session_device_index() -> None:
    """cuda:N parses to N; anything else falls back to 0 (never crashes)."""
    from voyage.workers import sfx_mmaudio as worker

    worker.handle_init({})
    assert worker._session_device_index() == 0
    worker.handle_init({"device": "cuda:1"})
    assert worker._session_device_index() == 1
    worker.handle_init({"device": "cpu"})
    assert worker._session_device_index() == 0
    worker.handle_init({})


def test_156_sfx_peak_helpers_guard_off_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off-GPU: no reset call, peak reports None (never a torch RuntimeError)."""
    from voyage.workers import sfx_mmaudio as worker

    class _NoCuda:
        resets = 0

        def is_available(self) -> bool:
            return False

        def reset_peak_memory_stats(self, *args: Any) -> None:
            type(self).resets += 1

        def max_memory_allocated(self, *args: Any) -> int:
            raise AssertionError("must not read CUDA peak off-GPU")

    cuda = _NoCuda()
    _stub_torch(cuda)
    monkeypatch.setattr("torch.cuda", cuda, raising=False)
    worker.handle_init({"device": "cpu"})
    worker._reset_peak_stats()
    assert cuda.resets == 0
    assert worker._peak_gib() is None
    worker.handle_init({})
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_156_ace_peak_helpers_guard_off_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same guard contract on the ACE-Step benchmark path."""
    from voyage.workers import audio_acestep as worker

    class _NoCuda:
        resets = 0

        def is_available(self) -> bool:
            return False

        def reset_peak_memory_stats(self, *args: Any) -> None:
            type(self).resets += 1

        def max_memory_allocated(self, *args: Any) -> int:
            raise AssertionError("must not read CUDA peak off-GPU")

    cuda = _NoCuda()
    _stub_torch(cuda)
    monkeypatch.setattr("torch.cuda", cuda, raising=False)
    worker.handle_init({"device": "cpu"})
    worker._reset_peak_stats()
    assert cuda.resets == 0
    assert worker._peak_gib() is None
    worker.handle_init({})
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_156_sfx_benchmark_report_shape() -> None:
    """Report carries device + cuda_available; VRAM nulls off-GPU."""
    from voyage.workers import sfx_mmaudio as worker

    report = worker._benchmark_report(
        warmup=0,
        measured=1,
        duration=8.0,
        walls=[1.25],
        peaks=[],
        cuda_available=False,
        device="cpu",
    )
    assert report["device"] == "cpu"
    assert report["cuda_available"] is False
    assert report["vram_peak_gib"] is None
    assert report["vram_avg_gib"] is None
    assert report["window_wall_seconds"] == [1.25]


def test_156_benchmarks_use_guarded_helpers() -> None:
    """Both benchmarks bracket windows via the guarded helpers (no raw CUDA calls)."""
    from voyage.workers import audio_acestep as ace
    from voyage.workers import sfx_mmaudio as sfx

    for worker in (sfx, ace):
        source = inspect.getsource(worker.handle_benchmark)
        assert "_reset_peak_stats" in source
        assert "_peak_gib" in source


# ---------------------------------------------------------------------------
# 158 — sfx two-worker cuda:1 presence gate
# ---------------------------------------------------------------------------


def test_158_two_workers_need_two_visible_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-GPU visibility + --sfx-workers 2 fails fast with no ledger writes."""
    from voyage import sfx_finalize as finalize
    from voyage.sfx_finalize import render_sfx_bed

    monkeypatch.setattr(finalize, "augment_devices", lambda **kwargs: ("cuda:0",))
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    with pytest.raises(MediaError, match="2 visible GPUs"):
        render_sfx_bed(
            run_dir,
            final_video,
            8.0,
            [(0.0, 8.0, "rain")],
            tmp_path,
            "fake",
            "/models",
            "cpu",
            "small_44k",
            0,
            48000,
            2,
            2,
        )
    assert not (run_dir / "audio" / "sfx" / "sfx.jsonl").exists()


def test_158_two_workers_pass_with_two_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two visible GPUs + fake backend shards both windows (stubbed workers)."""
    from voyage import sfx_finalize as finalize
    from voyage.sfx_finalize import render_sfx_bed

    monkeypatch.setattr(finalize, "augment_devices", lambda **kwargs: ("cuda:0", "cuda:1"))
    _WritingSfxWorker.seen = []
    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _WritingSfxWorker)
    run_dir = tmp_path / "run"
    final_video = tmp_path / "final.mp4"
    final_video.write_bytes(b"fake-video")
    bed = render_sfx_bed(
        run_dir,
        final_video,
        12.0,
        [(0.0, 12.0, "rain")],
        tmp_path,
        "fake",
        "/models",
        "cpu",
        "small_44k",
        0,
        48000,
        2,
        2,
    )
    assert bed.exists()
    assert (run_dir / "audio" / "sfx" / "w0000.wav").exists()
    assert (run_dir / "audio" / "sfx" / "w0001.wav").exists()


def test_158_preflight_lists_mmaudio_sfx() -> None:
    """CPU-box + mmaudio-SFX is a preflight offender (021 membership holds)."""
    from voyage.cli import _cuda_offenders
    from voyage.config import ProjectConfig

    config = ProjectConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "probe",
            "style": "x",
            "seed": 0,
            "video": {"backend": "fake"},
            "audio": {"backend": "fake"},
            "sfx": {"backend": "mmaudio"},
        }
    )
    assert any(offender.startswith("sfx ") for offender in _cuda_offenders(config))


def test_158_sfx_workers_help_names_two_gpu_need(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--sfx-workers help states the 2-visible-GPU requirement."""
    from voyage.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["sfx", "--help"])
    assert exc.value.code == 0
    assert "2 visible GPUs" in capsys.readouterr().out

"""CPU-only tests for the Track D augment spike (`voyage.augment` + worker).

No GPU/torch required: worker weight checks run before any torch import,
so missing weights raise NotImplementedError torch-free; the two arch
smoke tests below are the only torch touchpoint and skip via
`pytest.importorskip` when torch is absent. ffmpeg helpers run against a
stubbed `run_capture` (arg-lists asserted, never executed).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import voyage.augment as augment_module
from voyage.augment import (
    AUGMENT_DEVICE_PRIMARY,
    AUGMENT_DEVICE_SECONDARY,
    DEFAULT_CHUNK_FRAMES,
    AugmentChunk,
    augment_devices,
    augment_plan,
    chunk_windows,
    ffmpeg_decode_chunk,
    ffmpeg_encode_chunk,
    interpolated_frame_count,
    run_augment_chunks,
)
from voyage.errors import MediaError
from voyage.workers import augment_worker


def test_default_chunk_size_is_32() -> None:
    """Chunk default mirrors VHS_BatchManager `frames_per_batch`."""
    assert DEFAULT_CHUNK_FRAMES == 32


def test_chunk_windows_tile_exactly() -> None:
    assert list(chunk_windows(100, 32)) == [(0, 32), (32, 32), (64, 32), (96, 4)]


def test_chunk_windows_defaults_to_32() -> None:
    assert list(chunk_windows(40)) == [(0, 32), (32, 8)]


def test_chunk_windows_empty_total_yields_nothing() -> None:
    assert list(chunk_windows(0, 32)) == []


def test_chunk_windows_small_total_is_one_window() -> None:
    assert list(chunk_windows(7, 32)) == [(0, 7)]


def test_chunk_windows_exact_multiple_has_no_remainder() -> None:
    assert list(chunk_windows(64, 32)) == [(0, 32), (32, 32)]


def test_chunk_windows_cover_without_gaps_or_overlaps() -> None:
    """Tiling invariant: windows partition [0, total) contiguously."""
    for total in (0, 1, 31, 32, 33, 100, 1000):
        covered = 0
        for start, count in chunk_windows(total, 32):
            assert start == covered
            covered += count
        assert covered == total


def test_chunk_windows_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="chunk"):
        list(chunk_windows(10, 0))
    with pytest.raises(ValueError, match="chunk"):
        list(chunk_windows(10, -4))
    with pytest.raises(ValueError, match="total_frames"):
        list(chunk_windows(-1, 32))
    with pytest.raises(TypeError, match="total_frames"):
        list(chunk_windows("100", 32))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="chunk"):
        list(chunk_windows(100, 32.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="total_frames"):
        list(chunk_windows(True, 32))


@pytest.mark.parametrize(
    ("sources", "multiplier", "expected"),
    [(1, 4, 1), (2, 4, 5), (5, 4, 17), (3, 2, 5), (4, 1, 4), (64, 4, 253)],
)
def test_interpolated_frame_count(sources: int, multiplier: int, expected: int) -> None:
    """(n-1)*m+1, with a single frame passing through."""
    assert interpolated_frame_count(sources, multiplier) == expected


def test_interpolated_frame_count_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="source_frames"):
        interpolated_frame_count(0, 4)
    with pytest.raises(ValueError, match="multiplier"):
        interpolated_frame_count(4, 0)
    with pytest.raises(TypeError, match="source_frames"):
        interpolated_frame_count(4.0, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="multiplier"):
        interpolated_frame_count(4, "4")  # type: ignore[arg-type]


def test_augment_plan_assigns_devices_round_robin() -> None:
    plan = augment_plan(
        64, chunk=32, multiplier=4, devices=(AUGMENT_DEVICE_PRIMARY, AUGMENT_DEVICE_SECONDARY)
    )
    assert [(chunk.index, chunk.device) for chunk in plan] == [
        (0, AUGMENT_DEVICE_PRIMARY),
        (1, AUGMENT_DEVICE_SECONDARY),
    ]
    assert all(chunk.expected_frames == 125 for chunk in plan)
    assert [chunk.start_frame for chunk in plan] == [0, 32]


def test_augment_plan_chunked_total_accounts_skipped_boundaries() -> None:
    """Independent chunks skip one boundary pair per joint: 250, not the unchunked 253."""
    plan = augment_plan(
        64, chunk=32, multiplier=4, devices=(AUGMENT_DEVICE_PRIMARY, AUGMENT_DEVICE_SECONDARY)
    )
    chunked = sum(chunk.expected_frames for chunk in plan)
    unchunked = interpolated_frame_count(64, 4)
    assert chunked == 250
    assert unchunked == 253
    assert unchunked - chunked == (len(plan) - 1) * (4 - 1)


def test_augment_plan_empty_total_is_empty_plan() -> None:
    assert augment_plan(0, chunk=32, multiplier=4, devices=(AUGMENT_DEVICE_PRIMARY,)) == []


def test_augment_plan_rejects_empty_devices() -> None:
    with pytest.raises(ValueError, match="at least one device"):
        augment_plan(64, devices=())


def test_augment_plan_validates_multiplier_even_when_empty() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        augment_plan(0, multiplier=0, devices=(AUGMENT_DEVICE_PRIMARY,))


def test_augment_devices_two_visible() -> None:
    assert augment_devices(env={"CUDA_VISIBLE_DEVICES": "0,1"}) == (
        AUGMENT_DEVICE_PRIMARY,
        AUGMENT_DEVICE_SECONDARY,
    )


def test_augment_devices_single_visible() -> None:
    assert augment_devices(env={"CUDA_VISIBLE_DEVICES": "0"}) == (AUGMENT_DEVICE_PRIMARY,)


def test_augment_devices_empty_env_means_no_devices() -> None:
    """An explicitly emptied CUDA_VISIBLE_DEVICES hides every GPU: skip augment."""
    assert augment_devices(env={"CUDA_VISIBLE_DEVICES": ""}) == ()


def test_augment_devices_falls_back_to_smi_count() -> None:
    assert augment_devices(env={}, smi_count=3) == (
        AUGMENT_DEVICE_PRIMARY,
        AUGMENT_DEVICE_SECONDARY,
    )
    assert augment_devices(env={}, smi_count=1) == (AUGMENT_DEVICE_PRIMARY,)
    assert augment_devices(env={}, smi_count=0) == (AUGMENT_DEVICE_PRIMARY,)


def test_run_augment_chunks_serial_preserves_order() -> None:
    plan = augment_plan(64, chunk=32, multiplier=2, devices=(AUGMENT_DEVICE_PRIMARY,))
    seen: list[tuple[int, str]] = []

    def _worker(chunk: AugmentChunk, device: str) -> int:
        seen.append((chunk.index, device))
        return chunk.index * 10

    assert run_augment_chunks(plan, _worker) == [0, 10]
    assert seen == [(0, AUGMENT_DEVICE_PRIMARY), (1, AUGMENT_DEVICE_PRIMARY)]


def test_run_augment_chunks_parallel_uses_both_devices() -> None:
    plan = augment_plan(
        64, chunk=32, multiplier=2, devices=(AUGMENT_DEVICE_PRIMARY, AUGMENT_DEVICE_SECONDARY)
    )

    seen: list[tuple[int, str]] = []

    def _recording(chunk: AugmentChunk, device: str) -> int:
        seen.append((chunk.index, device))
        return chunk.index * 10

    assert run_augment_chunks(plan, _recording) == [0, 10]
    assert sorted(device for _, device in seen) == [
        AUGMENT_DEVICE_PRIMARY,
        AUGMENT_DEVICE_SECONDARY,
    ]


def _completed(argv: list[str], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode, "", "")


def test_ffmpeg_decode_chunk_writes_and_returns_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def _fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        dest_dir = Path(argv[-1]).parent
        for index in range(4):
            (dest_dir / f"frame_{index:06d}.png").write_bytes(b"fake-png-payload")
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _fake)
    dest = tmp_path / "chunk_00"
    frames = ffmpeg_decode_chunk(tmp_path / "source.mp4", dest, 32, 4)
    assert [path.name for path in frames] == [f"frame_{index:06d}.png" for index in range(4)]
    assert all(frame.stat().st_size > 0 for frame in frames)
    (command,) = calls
    assert command[0] == "ffmpeg"
    assert "-y" in command and "-nostdin" in command
    assert all(isinstance(part, str) for part in command)
    assert "shell" not in [part.lower() for part in command]
    vf_arg = command[command.index("-vf") + 1]
    assert "between(n\\,32\\,35)" in vf_arg


def test_ffmpeg_decode_chunk_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _failing(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, returncode=1)

    monkeypatch.setattr(augment_module, "run_capture", _failing)
    with pytest.raises(MediaError, match="chunk decode failed"):
        ffmpeg_decode_chunk(tmp_path / "source.mp4", tmp_path / "chunk_00", 0, 4)


def test_ffmpeg_decode_chunk_empty_output_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _empty(argv: list[str]) -> subprocess.CompletedProcess[str]:
        Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _empty)
    with pytest.raises(MediaError, match="no frames"):
        ffmpeg_decode_chunk(tmp_path / "source.mp4", tmp_path / "chunk_00", 0, 4)


def test_ffmpeg_decode_chunk_zero_byte_frame_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _hollow(argv: list[str]) -> subprocess.CompletedProcess[str]:
        dest_dir = Path(argv[-1]).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "frame_000000.png").write_bytes(b"")
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _hollow)
    with pytest.raises(MediaError, match="empty frame"):
        ffmpeg_decode_chunk(tmp_path / "source.mp4", tmp_path / "chunk_00", 0, 4)


def test_ffmpeg_decode_chunk_names_start_at_zero(tmp_path: Path) -> None:
    """Real ffmpeg must emit 0-based names (image2 defaults to 1-based).

    The upscale poller decodes and upscales in one dir while
    `write_tensors_as_png_frames` writes 0-based names: 1-based decode
    output leaves a native-size `frame_<count>` orphan that breaks the
    interp count check. Needs real ffmpeg (skips when absent).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg unavailable")
    assert ffmpeg is not None
    source = tmp_path / "source.mp4"
    staged = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=24:duration=0.34",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert staged.returncode == 0, staged.stderr[-500:]
    dest = tmp_path / "chunk_00"
    frames = ffmpeg_decode_chunk(source, dest, 0, 4)
    expected = [f"frame_{index:06d}.png" for index in range(4)]
    assert [path.name for path in frames] == expected
    assert sorted(path.name for path in dest.glob("frame_*.png")) == expected


def test_ffmpeg_encode_chunk_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        Path(argv[-1]).write_bytes(b"fake-mp4-payload")
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _fake)
    dest = tmp_path / "chunk_00.mp4"
    assert ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", dest, 32) == dest
    (command,) = calls
    assert command[0] == "ffmpeg"
    assert "-c:v" in command and "libx264" in command
    assert command[command.index("-framerate") + 1] == "32"
    assert command[command.index("-crf") + 1] == "15"


def test_ffmpeg_encode_chunk_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _failing(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, returncode=1)

    monkeypatch.setattr(augment_module, "run_capture", _failing)
    with pytest.raises(MediaError, match="chunk encode failed"):
        ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", tmp_path / "chunk_00.mp4", 32)


def test_ffmpeg_encode_chunk_missing_output_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _silent(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv)

    monkeypatch.setattr(augment_module, "run_capture", _silent)
    with pytest.raises(MediaError, match="empty output"):
        ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", tmp_path / "chunk_00.mp4", 32)


def test_ffmpeg_encode_chunk_rejects_bad_crf(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="crf"):
        ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", tmp_path / "out.mp4", 32, crf=52)
    with pytest.raises(TypeError, match="fps"):
        ffmpeg_encode_chunk(tmp_path / "frame_%06d.png", tmp_path / "out.mp4", True)


def test_upscale_frames_missing_weights_raises_torch_free(tmp_path: Path) -> None:
    """Weights are checked before any torch import: registry owns download."""
    with pytest.raises(NotImplementedError, match="registry"):
        augment_worker.upscale_frames([], tmp_path / "realesr-animevideov3.pth")


def test_interpolate_pair_missing_weights_raises_torch_free(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="registry"):
        augment_worker.interpolate_pair(None, None, tmp_path / "film_net_fp16.safetensors")


def test_interpolate_triplet_missing_weights_raises_torch_free(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="registry"):
        augment_worker.interpolate_triplet(None, None, None, tmp_path / "film_net_fp16.safetensors")


def test_empty_weights_file_counts_as_missing(tmp_path: Path) -> None:
    hollow = tmp_path / "hollow.pth"
    hollow.write_bytes(b"")
    with pytest.raises(NotImplementedError, match="never fetches weights at runtime"):
        augment_worker.upscale_frames([], hollow)


def test_inference_precision_selects_fp16_on_cuda() -> None:
    assert augment_worker.inference_precision("cuda:0") == "fp16"
    assert augment_worker.inference_precision("cuda:1") == "fp16"
    assert augment_worker.inference_precision("cpu") == "fp32"


def test_validate_upscale_factor() -> None:
    assert augment_worker.validate_upscale_factor(2) == 2
    with pytest.raises(ValueError, match="one of"):
        augment_worker.validate_upscale_factor(3)
    with pytest.raises(TypeError, match="int"):
        augment_worker.validate_upscale_factor("2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        augment_worker.validate_upscale_factor(True)


def test_validate_blend_time() -> None:
    assert augment_worker.validate_blend_time(0.5) == 0.5
    assert augment_worker.validate_blend_time(0) == 0.0
    assert augment_worker.validate_blend_time(1) == 1.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        augment_worker.validate_blend_time(1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        augment_worker.validate_blend_time(float("nan"))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        augment_worker.validate_blend_time(float("inf"))
    with pytest.raises(TypeError, match="number"):
        augment_worker.validate_blend_time("0.5")  # type: ignore[arg-type]


def _tiny_frames(count: int, height: int = 16, width: int = 16) -> list[Any]:
    """Deterministic (3, H, W) float gradient frames (spike smoke-test seam)."""
    torch = pytest.importorskip("torch", reason="arch smoke needs torch (absent from slim image)")
    frames: list[Any] = []
    for index in range(count):
        rows = torch.linspace(0.0, 1.0, height).unsqueeze(1).expand(height, width)
        frames.append(torch.stack([rows, rows * (index + 1) / count, 1.0 - rows]))
    return frames


def test_rrdb_net_upscales_tiny_frame_on_cpu() -> None:
    """Spike seam: the vendored RRDBNet runs end to end (random init, no weights)."""
    torch = pytest.importorskip("torch", reason="arch smoke needs torch (absent from slim image)")
    model = augment_worker._build_rrdb_net()
    model.eval()
    [frame] = _tiny_frames(1)
    with torch.no_grad():
        out = model(frame.unsqueeze(0))
    assert tuple(out.shape) == (1, 3, 64, 64)


def test_film_stand_in_blends_tiny_pair_on_cpu() -> None:
    """Spike seam: the vendored FILM stand-in runs end to end (random init, no weights)."""
    torch = pytest.importorskip("torch", reason="arch smoke needs torch (absent from slim image)")
    model = augment_worker._build_film_net()
    model.eval()
    first, second = _tiny_frames(2)
    batched = torch.stack([torch.stack([first, second])])
    with torch.no_grad():
        out = model(batched, 0.5)
    assert tuple(out.shape) == (1, 3, 16, 16)


def test_run_stacked_halves_on_oom() -> None:
    """Spike seam: the halving loop splits a batch that OOMs once, preserving order."""
    torch = pytest.importorskip("torch", reason="halving smoke needs torch (absent from slim)")
    attempts: list[int] = []

    def _flaky(batch: Any) -> Any:
        attempts.append(int(batch.shape[0]))
        if int(batch.shape[0]) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB.")
        return batch * 2.0

    stacked = torch.stack([torch.full((1, 2, 2), float(index)) for index in range(2)])
    outputs = augment_worker._run_stacked(_flaky, stacked)
    assert attempts == [2, 1, 1]
    assert [float(item.flatten()[0]) for item in outputs] == [0.0, 2.0]

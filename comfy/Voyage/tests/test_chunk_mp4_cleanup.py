"""Durable chunk-mp4 truth: validator, heal, and backend-switch prune (CPU-only).

Pins the per-segment cleanup contract: once a chunk's durable mp4 is
complete, its pruned PNG dirs must read as healthy (validator clean,
heal keeps the records); an mp4 record without its file fails loud in
both; a backend switch deletes + strips stale-leg mp4s alongside the
stale interp pixels. Stub seams only (no torch/GPU/ffmpeg).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from voyage import augment_sidecar as sidecar
from voyage.augment_interp_poller import interp_poll_once
from voyage.augment_upscale_poller import upscale_poll_once
from voyage.cli_validate import _check_sidecar_plan_consistency


def _make_segment(
    run_dir: Path,
    segment_id: str = "000000",
    *,
    frames: int = 8,
    checksum: str = "abc123",
) -> Path:
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
        "checksums": {"video.mp4": checksum},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _stub_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for position in range(count):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-frame")
        written.append(frame)
    return written


def _stub_upscale(frame_paths: list[Path], dest_dir: Path) -> list[Path]:
    assert dest_dir.is_dir()
    return list(frame_paths)


def _stub_interp(frame_paths: list[Path], dest_dir: Path, multiplier: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = (len(frame_paths) - 1) * multiplier + 1
    written = []
    for position in range(expected):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-interp")
        written.append(frame)
    return written


def _stub_encode(png_dir: Path, dest: Path, fps: float) -> Path:
    dest.write_bytes(b"fake-chunk")
    return dest


def _full_poll(run_dir: Path) -> Path:
    _make_segment(run_dir, frames=8)
    upscale_poll_once(
        run_dir,
        weights_path=Path("/models/realesrgan/realesr-animevideov3.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        decode_fn=_stub_decode,
        upscale_fn=_stub_upscale,
    )
    result = interp_poll_once(
        run_dir,
        weights_path=Path("/models/frame_interpolation/film_net_fp16.safetensors"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        chunk_frames=4,
        multiplier=4,
        interp_fn=_stub_interp,
        chunk_encode_fn=_stub_encode,
    )
    assert result.chunks_done == 2
    return next(iter((run_dir / "augment").iterdir()))


def test_validator_clean_after_cleanup(tmp_path: Path) -> None:
    """Pruned PNGs + complete mp4s validate clean (no false findings)."""
    _full_poll(tmp_path)
    assert _check_sidecar_plan_consistency(tmp_path) == []


def test_validator_flags_mp4_record_without_file(tmp_path: Path) -> None:
    """An mp4 record whose file is gone fails loud (never silently skipped)."""
    plan_dir = _full_poll(tmp_path)
    (plan_dir / "chunk_01.mp4").unlink()
    findings = _check_sidecar_plan_consistency(tmp_path)
    assert any("stage chunk_mp4" in finding for finding in findings)


def test_heal_exempts_mp4_complete_png_groups(tmp_path: Path) -> None:
    """Heal keeps PNG-stage records once the durable mp4 supersedes them."""
    from voyage.augment_sidecar import heal_augment_ledgers

    plan_dir = _full_poll(tmp_path)
    assert heal_augment_ledgers(tmp_path) == 0
    records = sidecar.load_chunk_ledger(plan_dir / sidecar.CHUNKS_LEDGER_FILENAME)
    stages = {record["stage"] for record in records}
    assert {"upscaled", "interpolated", "chunk_mp4"} <= stages
    assert _check_sidecar_plan_consistency(tmp_path) == []


def test_heal_strips_mp4_record_without_file(tmp_path: Path) -> None:
    """An mp4 record without its file is dangling: stripped, then re-rendered."""
    from voyage.augment_sidecar import heal_augment_ledgers

    plan_dir = _full_poll(tmp_path)
    (plan_dir / "chunk_01.mp4").unlink()
    assert heal_augment_ledgers(tmp_path) >= 1
    records = sidecar.load_chunk_ledger(plan_dir / sidecar.CHUNKS_LEDGER_FILENAME)
    assert not [
        record
        for record in records
        if record["stage"] == "chunk_mp4" and record["chunk_index"] == 1
    ]


def test_prune_deletes_stale_chunk_mp4s(tmp_path: Path) -> None:
    """Backend switch deletes stale-leg mp4s + strips their records (PNGs too)."""
    from voyage.augment_sidecar import prune_stale_interp_plans

    plan_dir = _full_poll(tmp_path)
    other_leg = tmp_path / "film_net_fp16.safetensors"
    other_leg.write_bytes(b"film-weights")
    from voyage.hashing import sha256_file

    other_sha = sha256_file(other_leg)
    stale_key = f"film|{other_sha}|esrgan-sha"
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    records = sidecar.load_chunk_ledger(ledger)
    for record in records:
        if record["stage"] in ("interpolated", "chunk_mp4"):
            record["weights_key"] = stale_key
    ledger.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    weights = SimpleNamespace(film=other_leg, rife=tmp_path / "rife.pth")
    pruned = prune_stale_interp_plans(tmp_path, interp_backend="rife", weights=weights)
    assert pruned == 1
    assert not (plan_dir / "chunk_00.mp4").exists()
    assert not (plan_dir / "chunk_01.mp4").exists()
    assert not (plan_dir / "interpolated_00").exists()
    records = sidecar.load_chunk_ledger(ledger)
    assert not [record for record in records if record["stage"] in ("interpolated", "chunk_mp4")]
    assert [record for record in records if record["stage"] == "upscaled"]


def test_prune_keeps_current_backend_mp4s(tmp_path: Path) -> None:
    """Current-backend mp4s + records survive the stale-leg prune."""
    from voyage.augment_sidecar import prune_stale_interp_plans

    plan_dir = _full_poll(tmp_path)
    other_leg = tmp_path / "film_net_fp16.safetensors"
    other_leg.write_bytes(b"other-film-weights")
    weights = SimpleNamespace(film=other_leg, rife=tmp_path / "rife.pth")
    assert prune_stale_interp_plans(tmp_path, interp_backend="rife", weights=weights) == 0
    assert (plan_dir / "chunk_00.mp4").stat().st_size > 0
    records = sidecar.load_chunk_ledger(plan_dir / sidecar.CHUNKS_LEDGER_FILENAME)
    assert len([record for record in records if record["stage"] == "chunk_mp4"]) == 2

"""Resumable joints/model-pass skip: fingerprint + ledger truth (DESIGN §§56-57, §140).

Covers the stage-skip contract: a deterministic plan fingerprint over every
input that forks plan dirs, a cheap ledger-truth completeness check that never
re-does finished joints work, and a persisted coverage marker validator. No
GPU, no network, no torch — tmp_path scaffolds plus real ChunkKey shapes and
fake mp4 bytes (unprobable blobs read fail-open, mirroring
test_track_c_finalize_durability).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from voyage import augment_sidecar as sidecar
from voyage.augment_finalize import (
    augment_plan_fingerprint,
    augment_work_complete,
    stamp_augment_coverage,
)
from voyage.augment_joints import JOINT_DIR_TEMPLATE, joint_fix_key
from voyage.augment_morph import MORPH_BRIDGE


def _base_plan_kwargs() -> dict[str, Any]:
    return {
        "weights_key": "weights-test-key",
        "out_width": 64,
        "out_height": 64,
        "out_fps": 24,
        "upscale_factor": 2,
        "multiplier": 2,
        "crf": 15,
        "preset": "veryfast",
        "interp_backend": "rife",
        "segment_source_keys": ["segkey-000000", "segkey-000001"],
        "joint_fix_keys": [
            joint_fix_key(a_sha="a" * 64, b_sha="b" * 64, width=64, height=64, fps_key=24)
        ],
    }


def test_fingerprint_deterministic() -> None:
    first = augment_plan_fingerprint(**_base_plan_kwargs())
    second = augment_plan_fingerprint(**_base_plan_kwargs())
    assert isinstance(first, str) and len(first) == 64
    assert first == second


def test_fingerprint_flips_on_any_single_field() -> None:
    base = augment_plan_fingerprint(**_base_plan_kwargs())
    flips: list[dict[str, Any]] = []
    seed = _base_plan_kwargs()
    for field, replacement in [
        ("weights_key", "weights-other"),
        ("out_width", 128),
        ("out_height", 128),
        ("out_fps", 25),
        ("upscale_factor", 1),
        ("multiplier", 4),
        ("crf", 18),
        ("preset", "slow"),
        ("interp_backend", "film"),
    ]:
        mutated = dict(seed)
        mutated[field] = replacement
        flips.append(mutated)
    reordered_segments = dict(seed)
    reordered_segments["segment_source_keys"] = list(reversed(seed["segment_source_keys"]))
    flips.append(reordered_segments)
    forked_joint = dict(seed)
    forked_joint["joint_fix_keys"] = [
        joint_fix_key(a_sha="c" * 64, b_sha="b" * 64, width=64, height=64, fps_key=24)
    ]
    flips.append(forked_joint)
    for mutated in flips:
        assert augment_plan_fingerprint(**mutated) != base


def _write_joint_fix(run_dir: Path, position: int, fix_key: str, joint_sha: str) -> Path:
    from voyage.augment_joints import (
        JOINT_RECORD_FILENAME,
        JOINT_VIDEO_FILENAME,
        joint_sources_root,
    )

    joint_dir = joint_sources_root(run_dir) / JOINT_DIR_TEMPLATE.format(
        left=position, right=position + 1
    )
    joint_dir.mkdir(parents=True, exist_ok=True)
    (joint_dir / JOINT_VIDEO_FILENAME).write_bytes(b"\x00" * 64)
    record = {
        "source_key": fix_key,
        "joint_sha": joint_sha,
        "joint_frames": MORPH_BRIDGE,
        "left_id": f"{position:06d}",
        "right_id": f"{position + 1:06d}",
        "left_frames": 8,
        "right_frames": 8,
    }
    (joint_dir / JOINT_RECORD_FILENAME).write_text(json.dumps(record), encoding="utf-8")
    return joint_dir


def _complete_chunks_for(
    run_dir: Path,
    *,
    source_key: str,
    total_frames: int,
    weights_key: str,
    out_width: int,
    out_height: int,
    out_fps: int,
    upscale_factor: int,
    multiplier: int,
    chunk_frames: int,
    crf: int,
    preset: str,
) -> None:
    from voyage.augment import augment_plan
    from voyage.augment_sidecar import ChunkKey, append_chunk_record, plan_dir_for_segment

    plan_dir = plan_dir_for_segment(
        run_dir,
        source_key=source_key,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        crf=crf,
        preset=preset,
    )
    plan_dir.mkdir(parents=True, exist_ok=True)
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    for chunk in augment_plan(total_frames, chunk=chunk_frames, multiplier=multiplier):
        key = ChunkKey(
            chunk_index=chunk.index,
            start_frame=chunk.start_frame,
            source_frames=chunk.source_frames,
            expected_frames=chunk.expected_frames,
            upscale_factor=upscale_factor,
            multiplier=multiplier,
            crf=crf,
            preset=preset,
            source_key=source_key,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=out_fps,
            chunk_frames=chunk_frames,
        )
        append_chunk_record(ledger, key, stage="chunk_mp4", path=f"chunk_{chunk.index:02d}.mp4")
        (plan_dir / f"chunk_{chunk.index:02d}.mp4").write_bytes(b"\x00" * 64)


def _build_complete_tree(run_dir: Path) -> dict[str, Any]:
    seed = _base_plan_kwargs()
    weights_key = str(seed["weights_key"])
    out_width = int(seed["out_width"])
    out_height = int(seed["out_height"])
    out_fps = int(seed["out_fps"])
    upscale_factor = int(seed["upscale_factor"])
    multiplier = int(seed["multiplier"])
    crf = int(seed["crf"])
    preset = str(seed["preset"])
    segment_keys = list(seed["segment_source_keys"])
    fix_keys = list(seed["joint_fix_keys"])
    joint_sha = "jointsha-000000-000001"
    _write_joint_fix(run_dir, 0, fix_keys[0], joint_sha)
    segment_frames = [5, 5]
    chunk_frames = 4
    for source_key, total_frames in zip(segment_keys, segment_frames, strict=True):
        _complete_chunks_for(
            run_dir,
            source_key=source_key,
            total_frames=total_frames,
            weights_key=weights_key,
            out_width=out_width,
            out_height=out_height,
            out_fps=out_fps,
            upscale_factor=upscale_factor,
            multiplier=multiplier,
            chunk_frames=chunk_frames,
            crf=crf,
            preset=preset,
        )
    _complete_chunks_for(
        run_dir,
        source_key=joint_sha,
        total_frames=MORPH_BRIDGE,
        weights_key=weights_key,
        out_width=out_width,
        out_height=out_height,
        out_fps=out_fps,
        upscale_factor=upscale_factor,
        multiplier=multiplier,
        chunk_frames=chunk_frames,
        crf=crf,
        preset=preset,
    )
    fingerprint = augment_plan_fingerprint(**seed)
    return {
        "fingerprint": fingerprint,
        "segment_keys": segment_keys,
        "segment_frames": segment_frames,
        "fix_keys": fix_keys,
        "joint_sha": joint_sha,
        "chunk_frames": chunk_frames,
        "weights_key": weights_key,
        "out_width": out_width,
        "out_height": out_height,
        "out_fps": out_fps,
        "upscale_factor": upscale_factor,
        "multiplier": multiplier,
        "crf": crf,
        "preset": preset,
        "interp_backend": str(seed["interp_backend"]),
    }


def _complete_kwargs(built: dict[str, Any]) -> dict[str, Any]:
    return {
        "weights_key": built["weights_key"],
        "out_width": built["out_width"],
        "out_height": built["out_height"],
        "out_fps": built["out_fps"],
        "upscale_factor": built["upscale_factor"],
        "multiplier": built["multiplier"],
        "chunk_frames": built["chunk_frames"],
        "crf": built["crf"],
        "preset": built["preset"],
        "interp_backend": built["interp_backend"],
        "segment_source_keys": built["segment_keys"],
        "segment_frame_counts": built["segment_frames"],
        "joint_fix_keys": built["fix_keys"],
    }


def test_work_complete_true_on_synthetic_complete_tree(tmp_path: Path) -> None:
    built = _build_complete_tree(tmp_path)
    assert (
        augment_work_complete(
            tmp_path,
            str(built["fingerprint"]),
            2,
            1,
            **_complete_kwargs(built),
        )
        is True
    )


def test_work_complete_false_on_missing_fix_record(tmp_path: Path) -> None:
    built = _build_complete_tree(tmp_path)
    from voyage.augment_joints import JOINT_RECORD_FILENAME, joint_sources_root

    record = (
        joint_sources_root(tmp_path) / JOINT_DIR_TEMPLATE.format(left=0, right=1)
    ) / JOINT_RECORD_FILENAME
    record.unlink()
    assert (
        augment_work_complete(
            tmp_path,
            str(built["fingerprint"]),
            2,
            1,
            **_complete_kwargs(built),
        )
        is False
    )


def test_work_complete_false_on_forked_fix_key(tmp_path: Path) -> None:
    built = _build_complete_tree(tmp_path)
    from voyage.augment_joints import JOINT_RECORD_FILENAME, joint_sources_root

    record_path = (
        joint_sources_root(tmp_path) / JOINT_DIR_TEMPLATE.format(left=0, right=1)
    ) / JOINT_RECORD_FILENAME
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["source_key"] = "jointfix2x2|forked"
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        augment_work_complete(
            tmp_path,
            str(built["fingerprint"]),
            2,
            1,
            **_complete_kwargs(built),
        )
        is False
    )


def test_work_complete_false_on_missing_chunk_record(tmp_path: Path) -> None:
    built = _build_complete_tree(tmp_path)
    from voyage.augment_sidecar import plan_dir_for_segment

    plan_dir = plan_dir_for_segment(
        tmp_path,
        source_key=str(built["segment_keys"][0]),
        weights_key=str(built["weights_key"]),
        out_width=int(built["out_width"]),
        out_height=int(built["out_height"]),
        out_fps=int(built["out_fps"]),
        upscale_factor=int(built["upscale_factor"]),
        crf=int(built["crf"]),
        preset=str(built["preset"]),
    )
    ledger = plan_dir / sidecar.CHUNKS_LEDGER_FILENAME
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    ledger.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""), encoding="utf-8")
    assert (
        augment_work_complete(
            tmp_path,
            str(built["fingerprint"]),
            2,
            1,
            **_complete_kwargs(built),
        )
        is False
    )


def test_work_complete_false_on_missing_mp4_file(tmp_path: Path) -> None:
    built = _build_complete_tree(tmp_path)
    from voyage.augment_sidecar import plan_dir_for_segment

    plan_dir = plan_dir_for_segment(
        tmp_path,
        source_key=str(built["segment_keys"][1]),
        weights_key=str(built["weights_key"]),
        out_width=int(built["out_width"]),
        out_height=int(built["out_height"]),
        out_fps=int(built["out_fps"]),
        upscale_factor=int(built["upscale_factor"]),
        crf=int(built["crf"]),
        preset=str(built["preset"]),
    )
    candidates = sorted(plan_dir.glob("chunk_*.mp4"))
    assert candidates
    candidates[0].unlink()
    assert (
        augment_work_complete(
            tmp_path,
            str(built["fingerprint"]),
            2,
            1,
            **_complete_kwargs(built),
        )
        is False
    )


def test_stamp_accepts_good_marker() -> None:
    marker = {
        "fingerprint": "f" * 64,
        "segments": 64,
        "joints": 63,
        "jointed_timeline_sha": "a" * 64,
        "interp_backend": "rife",
        "weights_key": "weights-test-key",
    }
    stamped = stamp_augment_coverage(marker)
    assert stamped == marker
    assert stamped is not marker


def test_stamp_rejects_bad_shapes() -> None:
    good = {
        "fingerprint": "f" * 64,
        "segments": 2,
        "joints": 1,
        "jointed_timeline_sha": "a" * 64,
        "interp_backend": "rife",
        "weights_key": "weights-test-key",
    }
    bad_markers: list[dict[str, Any]] = []
    missing = dict(good)
    del missing["fingerprint"]
    bad_markers.append(missing)
    extra = dict(good)
    extra["surprise"] = 1
    bad_markers.append(extra)
    empty_fingerprint = dict(good)
    empty_fingerprint["fingerprint"] = ""
    bad_markers.append(empty_fingerprint)
    negative_segments = dict(good)
    negative_segments["segments"] = -1
    bad_markers.append(negative_segments)
    bad_backend = dict(good)
    bad_backend["interp_backend"] = "filmx"
    bad_markers.append(bad_backend)
    empty_weights = dict(good)
    empty_weights["weights_key"] = ""
    bad_markers.append(empty_weights)
    for bad in bad_markers:
        try:
            stamp_augment_coverage(bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"stamp accepted bad marker {bad!r}")


def test_poll_to_completion_should_stop_skips_segment_work(tmp_path: Path) -> None:
    from voyage.augment_finalize import _poll_to_completion

    segment_dir = tmp_path / "segments" / "000000"
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    (segment_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "transition": {},
                "prompt_plan": {},
                "audio_state": {},
                "world_state": {},
                "metrics": {"frames": 4},
                "checksums": {"video.mp4": "checksum-000000"},
            }
        ),
        encoding="utf-8",
    )
    work = tmp_path / "weights"
    work.mkdir(parents=True, exist_ok=True)
    film = work / "film.safetensors"
    film.write_bytes(b"film-weights")
    rife = work / "rife.safetensors"
    rife.write_bytes(b"rife-weights")
    esrgan = work / "esrgan.pth"
    esrgan.write_bytes(b"esrgan-weights")
    from voyage.augment import AugmentWeights

    weights = AugmentWeights(film=film, rife=rife, realesrgan=esrgan)
    from voyage.augment_finalize import weights_key_for

    weights_key = weights_key_for(weights, "rife")
    calls: list[str] = []

    def _fake_upscale(run_dir: Path, **kwargs: Any) -> Any:
        calls.append("upscale")
        return SimpleNamespace(chunks_done=0, frames_done=0, frames_skipped=0)

    def _fake_interp(run_dir: Path, **kwargs: Any) -> Any:
        calls.append("interp")
        return SimpleNamespace(chunks_done=0, frames_done=0, frames_skipped=0, chunks_waiting=0)

    _poll_to_completion(
        tmp_path,
        weights=weights,
        weights_key=weights_key,
        out_width=64,
        out_height=64,
        source_fps=24.0,
        upscale_factor=1,
        multiplier=1,
        chunk_frames=4,
        device="cpu",
        crf=15,
        preset="veryfast",
        upscale_poll_fn=_fake_upscale,
        interp_poll_fn=_fake_interp,
        interp_backend="rife",
        should_stop=lambda: True,
    )
    assert calls == []

"""Orphan plan-dir garbage collection (DESIGN §§56-57, §140).

Why this module exists: every settings/weights/segment re-render forks a
new `<plan-hash>/` (old hashes never reused), so `run/augment/` grows
without bound. `prune_orphan_plan_dirs` keeps the live set (current
committed segments' plan dirs plus their seam joints, recomputed through
the same derivation as the pollers) plus `morph_joints`, and deletes only
16-hex plan dirs older than the grace period — these tests pin that live
stays, old orphans go, young orphans wait, and unknown names survive.
Deliberately not wired into finalize (explicit later decision).
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from voyage.augment_drain import prune_orphan_plan_dirs
from voyage.augment_seam import seam_plan_dir
from voyage.augment_sidecar import plan_dir_for_segment


def _recipe() -> dict[str, object]:
    return {
        "weights_key": "wkey",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "upscale_factor": 2,
        "crf": 15,
        "preset": "veryfast",
    }


def _commit_segment(run_dir: Path, segment_id: str, checksum: str) -> Path:
    segment_dir = run_dir / "segments" / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "video.mp4").write_bytes(b"fake-video")
    manifest = {
        "format": 1,
        "transition": {},
        "prompt_plan": {},
        "audio_state": {},
        "world_state": {},
        "metrics": {"frames": 8},
        "checksums": {"video.mp4": checksum},
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (segment_dir / "DONE").write_text("done\n", encoding="utf-8")
    return segment_dir


def _touch_old(target: Path, reference_time: float, days_old: float) -> None:
    """Backdate a tree to `days_old` before `reference_time` (deterministic GC age)."""
    stamp = reference_time - days_old * 86400.0
    candidates = [target, *sorted(target.rglob("*"))] if target.is_dir() else [target]
    for child in candidates:
        with contextlib.suppress(OSError):
            os.utime(child, (stamp, stamp))
    with contextlib.suppress(OSError):
        os.utime(target, (stamp, stamp))


def test_gc_keeps_live_and_deletes_old_orphans(tmp_path: Path) -> None:
    """Live plan + seam dirs survive; an old forked hash is pruned."""
    run_dir = tmp_path / "run"
    (run_dir / "segments").mkdir(parents=True)
    _commit_segment(run_dir, "000000", "aaa")
    _commit_segment(run_dir, "000001", "bbb")
    recipe = _recipe()
    live_first = plan_dir_for_segment(run_dir, source_key="aaa", **recipe)  # type: ignore[arg-type]
    live_second = plan_dir_for_segment(run_dir, source_key="bbb", **recipe)  # type: ignore[arg-type]
    live_seam = seam_plan_dir(run_dir, key_a="aaa", key_b="bbb", **recipe)  # type: ignore[arg-type]
    for live in (live_first, live_second, live_seam):
        live.mkdir(parents=True, exist_ok=True)
        (live / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    orphan = run_dir / "augment" / ("0" * 16)
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    reference = 1_800_000_000.0
    _touch_old(orphan, reference, days_old=10.0)
    pruned = prune_orphan_plan_dirs(run_dir, grace_days=7.0, now_seconds=reference, **recipe)  # type: ignore[arg-type]
    assert pruned == 1
    assert not orphan.exists()
    assert live_first.is_dir()
    assert live_second.is_dir()
    assert live_seam.is_dir()


def test_gc_keeps_young_orphans_morph_joints_and_unknown_names(tmp_path: Path) -> None:
    """Young forks wait out the grace period; morph tree and odd names never go."""
    run_dir = tmp_path / "run"
    (run_dir / "segments").mkdir(parents=True)
    _commit_segment(run_dir, "000000", "aaa")
    recipe = _recipe()
    live = plan_dir_for_segment(run_dir, source_key="aaa", **recipe)  # type: ignore[arg-type]
    live.mkdir(parents=True, exist_ok=True)
    young = run_dir / "augment" / ("1" * 16)
    young.mkdir(parents=True, exist_ok=True)
    (young / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    morph_tree = run_dir / "augment" / "morph_joints"
    morph_tree.mkdir(parents=True, exist_ok=True)
    odd_name = run_dir / "augment" / "scratch-notes"
    odd_name.mkdir(parents=True, exist_ok=True)
    reference = 1_800_000_000.0
    _touch_old(young, reference, days_old=1.0)
    _touch_old(morph_tree, reference, days_old=30.0)
    _touch_old(odd_name, reference, days_old=30.0)
    pruned = prune_orphan_plan_dirs(run_dir, grace_days=7.0, now_seconds=reference, **recipe)  # type: ignore[arg-type]
    assert pruned == 0
    assert young.is_dir()
    assert morph_tree.is_dir()
    assert odd_name.is_dir()
    assert live.is_dir()


def test_gc_without_augment_dir_is_noop(tmp_path: Path) -> None:
    """Runs that never finalized have nothing to prune (clean zero, no raise)."""
    run_dir = tmp_path / "run"
    (run_dir / "segments").mkdir(parents=True)
    assert (
        prune_orphan_plan_dirs(
            run_dir,
            grace_days=7.0,
            now_seconds=1_800_000_000.0,
            **_recipe(),  # type: ignore[arg-type]
        )
        == 0
    )

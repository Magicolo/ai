"""Seam/morph auto-heal resume (DESIGN §§56-57, §140).

Why this module exists: ledger-hit-but-output-wrong used to fail loud
(`MediaError: clear the seam dir and retry` / morph key-clash `ValueError`),
stranding finalize on stale publishes, manual cleanups, and re-rendered
sides. Both renderers now drop the stale output plus the ledger and
re-render (chunk-poller parity: ledger-truth + output-truth, last-wins) —
these tests pin the heal (re-render, fresh record, no raise) plus the
unchanged happy-path resume (ledger hit + complete output = no re-render).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyage import augment_morph
from voyage.augment_seam import render_seam_once
from voyage.augment_sidecar import CHUNKS_LEDGER_FILENAME, STAGE_INTERPOLATED, load_chunk_ledger


def _seam_recipe() -> dict[str, Any]:
    return {
        "weights_key": "wkey",
        "out_width": 1216,
        "out_height": 704,
        "out_fps": 24,
        "upscale_factor": 2,
        "crf": 15,
        "preset": "veryfast",
    }


def _stub_seam_writer(
    call_log: list[str],
) -> Any:
    def _write_seam(
        first: Path, second: Path, dest: Path, multiplier: int, **kwargs: Any
    ) -> list[Path]:
        del first, second, kwargs
        call_log.append(f"render-{multiplier}")
        dest.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for position in range(multiplier - 1):
            frame = dest / f"frame_{position:06d}.png"
            frame.write_bytes(b"fresh-mid")
            written.append(frame)
        return written

    return _write_seam


def test_seam_heals_ledger_hit_with_short_output(tmp_path: Path) -> None:
    """Stale seam output (ledger hit, PNGs short) re-renders instead of raising."""
    seam_dir = tmp_path / "seam"
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"a")
    after.write_bytes(b"b")
    calls: list[str] = []
    first = render_seam_once(
        tmp_path,
        seam_dir,
        before_png=before,
        after_png=after,
        multiplier=4,
        source_key="aaa|bbb",
        interp_fn=_stub_seam_writer(calls),
        **_seam_recipe(),
    )
    assert first is True
    assert len(calls) == 1
    # Simulate stale publish: drop two of the three mids, ledger stays.
    for stale in sorted((seam_dir / "interpolated_00").glob("frame_*.png"))[1:]:
        stale.unlink()
    healed = render_seam_once(
        tmp_path,
        seam_dir,
        before_png=before,
        after_png=after,
        multiplier=4,
        source_key="aaa|bbb",
        interp_fn=_stub_seam_writer(calls),
        **_seam_recipe(),
    )
    assert healed is True
    assert len(calls) == 2
    finished = sorted((seam_dir / "interpolated_00").glob("frame_*.png"))
    assert len(finished) == 3
    assert all(frame.read_bytes() == b"fresh-mid" for frame in finished)
    records = load_chunk_ledger(seam_dir / CHUNKS_LEDGER_FILENAME)
    assert any(
        record.get("stage") == STAGE_INTERPOLATED and record.get("source_key") == "aaa|bbb"
        for record in records
    )


def test_seam_hit_with_complete_output_stays_noop(tmp_path: Path) -> None:
    """Unchanged happy path: ledger hit + complete output never re-renders."""
    seam_dir = tmp_path / "seam"
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"a")
    after.write_bytes(b"b")
    calls: list[str] = []
    assert (
        render_seam_once(
            tmp_path,
            seam_dir,
            before_png=before,
            after_png=after,
            multiplier=2,
            source_key="aaa|bbb",
            interp_fn=_stub_seam_writer(calls),
            **_seam_recipe(),
        )
        is True
    )
    assert (
        render_seam_once(
            tmp_path,
            seam_dir,
            before_png=before,
            after_png=after,
            multiplier=2,
            source_key="aaa|bbb",
            interp_fn=_stub_seam_writer(calls),
            **_seam_recipe(),
        )
        is False
    )
    assert len(calls) == 1


def _stub_bridge_writer(call_log: list[str]) -> Any:
    def _expand_bridge(
        anchor_a: Path, anchor_b: Path, weights: object, moment: float, device: str
    ) -> bytes:
        del anchor_a, anchor_b, weights, moment, device
        call_log.append("bridge-render")
        return b"fresh-bridge"

    return _expand_bridge


def _morph_key(first: str = "a", second: str = "b") -> str:
    return augment_morph.morph_joint_key(
        a_sha=first * 64, b_sha=second * 64, width=64, height=64, fps_key=6
    )


def test_morph_heals_hit_with_short_bridges(tmp_path: Path) -> None:
    """Stale morph bridges (ledger hit, PNGs short) re-render instead of raising."""
    joint_dir = tmp_path / "morph_000000_000001"
    anchor_a = tmp_path / "anchor_a.png"
    anchor_b = tmp_path / "anchor_b.png"
    anchor_a.write_bytes(b"anchor-a")
    anchor_b.write_bytes(b"anchor-b")
    key = _morph_key()
    calls: list[str] = []
    augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=key,
        interp_fn=_stub_bridge_writer(calls),
        weights=None,
        device="cpu",
    )
    assert len(calls) == 4
    for stale in sorted(joint_dir.glob("bridge_*.png"))[2:]:
        stale.unlink()
    healed = augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=key,
        interp_fn=_stub_bridge_writer(calls),
        weights=None,
        device="cpu",
    )
    assert len(healed.frames) == 4
    assert len(calls) == 8
    assert all(frame.is_file() and frame.stat().st_size > 0 for frame in healed.frames)


def test_morph_heals_key_clash_with_fresh_bridges(tmp_path: Path) -> None:
    """Re-rendered sides (new joint key, stale bridges) heal last-wins, never raise."""
    joint_dir = tmp_path / "morph_000000_000001"
    anchor_a = tmp_path / "anchor_a.png"
    anchor_b = tmp_path / "anchor_b.png"
    anchor_a.write_bytes(b"anchor-a")
    anchor_b.write_bytes(b"anchor-b")
    first_key = _morph_key("a", "b")
    second_key = _morph_key("c", "d")
    calls: list[str] = []
    augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=first_key,
        interp_fn=_stub_bridge_writer(calls),
        weights=None,
        device="cpu",
    )
    healed = augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=second_key,
        interp_fn=_stub_bridge_writer(calls),
        weights=None,
        device="cpu",
    )
    assert len(healed.frames) == 4
    record = json.loads((joint_dir / augment_morph.MORPH_RECORD_FILENAME).read_text())
    assert record["source_key"] == second_key


def test_morph_heals_torn_ledger(tmp_path: Path) -> None:
    """A torn joint ledger heals by re-rendering (never fail-loud)."""
    joint_dir = tmp_path / "morph_000000_000001"
    joint_dir.mkdir(parents=True, exist_ok=True)
    anchor_a = tmp_path / "anchor_a.png"
    anchor_b = tmp_path / "anchor_b.png"
    anchor_a.write_bytes(b"anchor-a")
    anchor_b.write_bytes(b"anchor-b")
    (joint_dir / augment_morph.MORPH_RECORD_FILENAME).write_text(
        '{"source_key": "torn', encoding="utf-8"
    )
    calls: list[str] = []
    healed = augment_morph.render_morph_once(
        joint_dir=joint_dir,
        a_anchor=anchor_a,
        b_anchor=anchor_b,
        source_key=_morph_key(),
        interp_fn=_stub_bridge_writer(calls),
        weights=None,
        device="cpu",
    )
    assert len(healed.frames) == 4
    assert len(calls) == 4

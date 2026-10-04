"""Native-resolution finalize: upscale factor rule + 1x passthrough (DESIGN §140).

By default ltx25's native output (1216x704) is considered sufficient — the
model pass interpolates at source resolution instead of upscaling 2x and
interpolating at 2432x1408 (which OOMs the 6 GB 2060 even solo). Covered:
the pure factor rule (`media.upscale_factor_for`), the torch-free 1x
passthrough in the upscale poller's default seam, and factor validation.
CPU-only, no torch/PIL/ffmpeg: the 1x path must never import the model
stack (the slim test image carries neither torch nor Pillow).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from voyage import media
from voyage.augment_upscale_poller import _default_upscale_pngs, upscale_poll_once


def test_factor_rule_skips_upscale_when_output_fits_source() -> None:
    """Equal or smaller output box means no genuine lift: factor 1."""
    assert media.upscale_factor_for(1216, 704, 1216, 704) == 1
    assert media.upscale_factor_for(1216, 704, 640, 352) == 1
    assert media.upscale_factor_for(1216, 704, 1216, 360) == 1


def test_factor_rule_upscales_on_genuine_lift() -> None:
    """Any output axis above source is a genuine lift: factor 2."""
    assert media.upscale_factor_for(768, 512, 1216, 704) == 2
    assert media.upscale_factor_for(1216, 704, 1216, 705) == 2
    assert media.upscale_factor_for(1216, 704, 1217, 704) == 2


def test_factor_rule_rejects_bad_geometry() -> None:
    """Zero/negative sizes fail loud, never silently pick a factor."""
    with pytest.raises(ValueError):
        media.upscale_factor_for(0, 704, 1216, 704)
    with pytest.raises(ValueError):
        media.upscale_factor_for(1216, 704, 1216, -1)


def _block_model_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any import of the model stack fails the test (1x must stay torch-free)."""

    class _Blocker:
        def find_module(self, name: str, _path: str | None = None) -> object:
            if name == "torch" or name.startswith(("torch.", "PIL")):
                return self
            if name.startswith("voyage.workers.augment_worker"):
                return self
            return None

        def load_module(self, name: str) -> object:
            raise ImportError(f"blocked model-stack import in 1x test: {name}")

    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])


def test_passthrough_copies_pngs_without_model_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Factor 1 is a CPU file copy: same names, identical bytes, no model."""
    _block_model_stack(monkeypatch)
    source_dir = tmp_path / "decoded"
    source_dir.mkdir()
    frames = []
    for index in range(3):
        frame = source_dir / f"frame_{index:06d}.png"
        frame.write_bytes(b"\x89PNG-fake-bytes-%d" % index)
        frames.append(frame)
    dest_dir = tmp_path / "upscaled_00.partial"
    written = _default_upscale_pngs(
        frames,
        dest_dir,
        weights_path=Path("/models/does-not-exist.pth"),
        device="cuda:1",
        upscale_factor=1,
    )
    assert [path.name for path in written] == [path.name for path in frames]
    for source, output in zip(frames, written, strict=True):
        assert output.read_bytes() == source.read_bytes()


def test_passthrough_rejects_bad_factor(tmp_path: Path) -> None:
    """Factors outside (1, 2, 4) fail at the seam, never reach the model."""
    with pytest.raises(ValueError):
        _default_upscale_pngs(
            [],
            tmp_path / "dest",
            weights_path=Path("/models/weights.pth"),
            device="cuda:1",
            upscale_factor=3,
        )
    with pytest.raises(ValueError):
        _default_upscale_pngs(
            [],
            tmp_path / "dest",
            weights_path=Path("/models/weights.pth"),
            device="cuda:1",
            upscale_factor=0,
        )


def _make_segment(run_dir: Path, segment_id: str, frames: int, checksum: str) -> None:
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


def _stub_decode(source_video: Path, dest_dir: Path, start: int, count: int) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for position in range(count):
        frame = dest_dir / f"frame_{position + 1:06d}.png"
        frame.write_bytes(b"fake-frame")
        written.append(frame)
    return written


def test_poller_factor_one_uses_default_seam_torch_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poller with factor 1 + default seam renders chunks with no model stack."""
    _block_model_stack(monkeypatch)
    _make_segment(tmp_path, "000000", frames=8, checksum="seg-key")
    result = upscale_poll_once(
        tmp_path,
        weights_path=Path("/models/realesrgan.pth"),
        weights_key="weights-abc",
        out_width=1216,
        out_height=704,
        out_fps=24,
        upscale_factor=1,
        chunk_frames=4,
        decode_fn=_stub_decode,
        device="cuda:1",
    )
    assert result.chunks_done == 2
    plan_dirs = list((tmp_path / "augment").iterdir())
    assert len(plan_dirs) == 1

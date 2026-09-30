"""CausVid video worker tests (Track 2.3, DESIGN §5.4).

CPU-only: every torch/causvid/omegaconf/imageio dependency is faked — no
GPU, no model downloads. The fake pipeline returns fixed numpy-backed
tensors (frame index encoded in pixel values, so commit slicing is
verified for real), and media I/O is stubbed via ``sys.modules`` following
the ``test_longlive_stages`` pattern.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voyage.backends import VideoSegmentResult
from voyage.workers import video_causvid
from voyage.workers.loop import serve
from voyage.workers.video_causvid import CausvidSession


class _FakeTensor:
    """Numpy-backed stand-in for a torch tensor (slicing is real)."""

    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)
        self.shape = self._array.shape

    def __getitem__(self, key: Any) -> _FakeTensor:
        return _FakeTensor(self._array[key])

    def permute(self, *order: int) -> _FakeTensor:
        return _FakeTensor(np.transpose(self._array, order))

    def transpose(self, dim0: int, dim1: int) -> _FakeTensor:
        return _FakeTensor(np.swapaxes(self._array, dim0, dim1))

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self._array, dim))

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> Any:
        return self._array

    @property
    def device(self) -> str:
        return "cpu"

    def to(self, *args: Any, **kwargs: Any) -> _FakeTensor:
        del args, kwargs
        return self

    def astype(self, dtype: Any) -> _FakeTensor:
        return _FakeTensor(self._array.astype(dtype))

    def __mul__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(self._array * other)

    def __sub__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(self._array - other)

    def __truediv__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(self._array / other)

    def __rtruediv__(self, other: Any) -> _FakeTensor:
        return _FakeTensor(other / self._array)


class _FakeGenerator:
    def __init__(self, torch: _FakeTorch) -> None:
        self._torch = torch
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> _FakeGenerator:
        self.seeds.append(seed)
        self._torch.generator_seeds.append(seed)
        return self


class _FakeCuda:
    def __init__(self) -> None:
        self.available = False
        self.empty_cache_calls = 0
        self.reset_calls = 0
        self.peak_bytes = 6 * 1024**3

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def max_memory_allocated(self) -> int:
        return self.peak_bytes

    def get_device_name(self, index: int) -> str:
        del index
        return "fake-gpu"

    def mem_get_info(self) -> tuple[int, int]:
        return (8 * 1024**3, 16 * 1024**3)


class _FakeModule:
    """Minimal ``torch.nn.Module``: attribute assignment + ``forward`` dispatch."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError

    def to(self, *args: Any, **kwargs: Any) -> _FakeModule:
        del args, kwargs
        return self


class _FakeNn:
    Module = _FakeModule


class _FakeTorch:
    """Stub for the session's lazy ``torch`` handle (no CUDA needed)."""

    nn = _FakeNn()

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.bfloat16 = "bfloat16"
        self.generator_seeds: list[int] = []
        self.global_seeds: list[int] = []
        self.randn_calls: list[dict[str, Any]] = []

    def manual_seed(self, seed: int) -> None:
        """Record global-RNG seeding (real torch seeds CPU + all CUDA)."""
        self.global_seeds.append(seed)

    def Generator(self, device: str = "cpu") -> _FakeGenerator:
        del device
        return _FakeGenerator(self)

    def randn(self, *args: Any, **kwargs: Any) -> _FakeTensor:
        shape = list(args[0]) if args else kwargs.get("shape", [1, 1])
        self.randn_calls.append({"shape": shape, "kwargs": kwargs})
        return _FakeTensor(np.zeros(shape, dtype=np.float32))

    def cat(self, tensors: list[Any], dim: int = 0) -> _FakeTensor:
        arrays = [t._array if isinstance(t, _FakeTensor) else np.asarray(t) for t in tensors]
        return _FakeTensor(np.concatenate(arrays, axis=dim))

    def from_numpy(self, array: Any) -> _FakeTensor:
        return _FakeTensor(np.asarray(array))

    def set_grad_enabled(self, enabled: bool) -> None:
        del enabled


class _FakeVaeModel:
    """Raw-VAE stub: mirrors the temporal chunking math (1, 4, 4, …).

    ``encode(x, scale)`` with ``x`` shaped ``(B, C, T, H, W)`` returns zeros
    shaped ``(B, 16, T', 4, 4)`` with ``T' = (T - 1) // 4 + 1`` — so the
    1-frame slice path yields 1 latent and the 9-frame resume window yields
    exactly ``overlap`` (3) latents.
    """

    def __init__(self, outer: _FakeVaeTape) -> None:
        self._outer = outer

    def encode(self, tensor: Any, scale: Any) -> _FakeTensor:
        del scale
        self._outer.encode_calls += 1
        array = tensor._array if isinstance(tensor, _FakeTensor) else np.asarray(tensor)
        latent_frames = (int(array.shape[2]) - 1) // 4 + 1
        return _FakeTensor(np.zeros((1, 16, latent_frames, 4, 4), dtype=np.float32))


class _FakeVaeTape:
    """Shared encode counter for the VAE stub."""

    def __init__(self) -> None:
        self.encode_calls = 0


class _FakeVae:
    """VAE-wrapper stub: ``mean``/``std`` + ``model.encode`` like upstream.

    Returns channel-time layout ``(B, C, T', H, W)`` like the raw model, so
    the worker's ``transpose(2, 1)`` back to ``(B, T', C, H, W)`` cats
    cleanly with the raw ``latents[:, -(overlap-1):]`` tail.
    """

    def __init__(self, latent_tail: tuple[int, ...] = (1, 16, 1, 4, 4)) -> None:
        del latent_tail
        self.mean = _FakeTensor(np.zeros(16, dtype=np.float32))
        self.std = _FakeTensor(np.ones(16, dtype=np.float32))
        self._tape = _FakeVaeTape()
        self.model = _FakeVaeModel(self._tape)

    @property
    def encode_calls(self) -> int:
        return self._tape.encode_calls


class _FakeTextEncoder:
    """T5 stub: `.to()` shuttles (no-op), call returns CUDA-ready embeds."""

    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> _FakeTextEncoder:
        self.devices.append(device)
        return self

    def __call__(self, prompts: Any) -> dict[str, Any]:
        del prompts
        return {"prompt_embeds": _FakeTensor(np.zeros((1, 16, 4, 4), dtype=np.float32))}


class _FakePipeline:
    """Fake upstream pipeline: 81 decoded frames, marker pixels, fixed latents."""

    DECODED = 81

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.vae = _FakeVae()
        self.text_encoder = _FakeTextEncoder()

    def inference(
        self,
        noise: Any,
        text_prompts: list[str],
        return_latents: bool = False,
        start_latents: Any = None,
    ) -> tuple[_FakeTensor, _FakeTensor]:
        del noise, return_latents
        self.calls.append({"prompts": list(text_prompts), "start": start_latents})
        video = np.zeros((1, self.DECODED, 3, 4, 6), dtype=np.float32)
        for index in range(self.DECODED):
            video[0, index] = index / self.DECODED
        latents = np.zeros((1, 21, 16, 4, 4), dtype=np.float32)
        return _FakeTensor(video), _FakeTensor(latents)


def _install_imageio_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub ``imageio.v2`` mimsave/mimread (slim has no imageio)."""
    recorded: dict[str, Any] = {"saves": [], "reads": []}

    def _mimsave(path: str, frames: Any, fps: int = 16, codec: str = "libx264") -> None:
        del codec
        recorded["saves"].append({"path": path, "frames": list(frames), "fps": fps})
        Path(path).write_bytes(b"fake-mp4")

    def _mimread(path: str) -> Any:
        recorded["reads"].append(path)
        raise OSError(f"no media stub for {path}")

    imageio_mod = types.ModuleType("imageio")
    v2_mod = types.ModuleType("imageio.v2")
    v2_mod.mimsave = _mimsave  # type: ignore[attr-defined]
    v2_mod.mimread = _mimread  # type: ignore[attr-defined]
    imageio_mod.v2 = v2_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2_mod)
    return recorded


def _test_session(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CausvidSession, _FakeTorch, _FakePipeline, dict[str, Any]]:
    """Resident session without __init__ (GPU-only): fakes injected directly."""
    recorded = _install_imageio_stub(monkeypatch)
    torch = _FakeTorch()
    pipeline = _FakePipeline()
    session = CausvidSession.__new__(CausvidSession)
    session._torch = torch  # type: ignore[attr-defined]
    session._device = "cuda:0"  # type: ignore[attr-defined]
    session._latent_shape = [1, 21, 16, 60, 104]  # type: ignore[attr-defined]
    session._overlap_frames = 3  # type: ignore[attr-defined]
    session._num_frame_per_block = 3  # type: ignore[attr-defined]
    session._config_sha256 = "0" * 64  # type: ignore[attr-defined]
    session._pipeline = pipeline  # type: ignore[attr-defined]
    session._start_latents = None  # type: ignore[attr-defined]
    session._pending_tail_path = None  # type: ignore[attr-defined]
    session._pending_overlap = 3  # type: ignore[attr-defined]
    session._last_prompt = None  # type: ignore[attr-defined]
    return session, torch, pipeline, recorded


def test_native_profile_constants() -> None:
    assert video_causvid.RECOVERY_PROFILE == "causvid"
    assert video_causvid.STATE_MODE == "reconstructable_prefix"
    assert video_causvid.NATIVE_FPS == 16
    assert (video_causvid.NATIVE_WIDTH, video_causvid.NATIVE_HEIGHT) == (832, 480)
    assert video_causvid.LATENT_SHAPE == [1, 21, 16, 60, 104]
    assert video_causvid.NUM_FRAME_PER_BLOCK == 3
    assert video_causvid.DEFAULT_OVERLAP_FRAMES == 3


def test_tail_drop_math_matches_upstream_script() -> None:
    """Overlap 3 → drop 9 → 72 novel; 21 latents decode to 81 frames."""
    assert video_causvid.dropped_tail_frames(3) == 9
    assert video_causvid.decoded_frames_for_latents(21) == 81
    assert video_causvid.novel_frames_per_rollout(81, 3) == 72
    assert video_causvid.split_tail_novel(81, 3) == (9, 72)


def test_rollout_zero_uses_uniform_accounting() -> None:
    """Rollout 0 drops its tail exactly like every later rollout (script parity)."""
    dropped, novel = video_causvid.split_tail_novel(81, 3)
    assert dropped + novel == 81
    assert novel == 72


def test_overlap_must_be_positive_block_multiple() -> None:
    video_causvid.validate_overlap_frames(3, 3)
    video_causvid.validate_overlap_frames(6, 3)
    for bad in (1, 2, 4, 5):
        with pytest.raises(ValueError, match="divisible by"):
            video_causvid.validate_overlap_frames(bad, 3)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="positive"):
            video_causvid.validate_overlap_frames(bad, 3)


def test_novel_requires_positive_remainder() -> None:
    with pytest.raises(ValueError, match="nothing would be committed"):
        video_causvid.novel_frames_per_rollout(9, 3)
    with pytest.raises(ValueError, match="nothing would be committed"):
        video_causvid.novel_frames_per_rollout(5, 3)


def test_latent_shape_validation() -> None:
    assert video_causvid.validate_latent_shape([1, 21, 16, 60, 104]) == [1, 21, 16, 60, 104]
    bad_shapes = ([1, 21, 16, 60], [1, 21, 16, 60, 104, 1], [1, 21, 0, 60, 104])
    for bad in [*bad_shapes, [1, 21, -16, 60, 104]]:
        with pytest.raises(ValueError, match="5 positive ints"):
            video_causvid.validate_latent_shape(bad)


def test_geometry_must_match_latent_spatial() -> None:
    video_causvid.validate_native_geometry(832, 480, [1, 21, 16, 60, 104])
    with pytest.raises(ValueError, match="does not match"):
        video_causvid.validate_native_geometry(768, 512, [1, 21, 16, 60, 104])
    with pytest.raises(ValueError, match="not divisible by"):
        video_causvid.validate_native_geometry(830, 480, [1, 21, 16, 60, 104])


def test_reencode_window_yields_overlap_latents() -> None:
    """The resume window re-encodes back to exactly `overlap` latent frames."""
    window = video_causvid.reencode_window_frames(3)
    assert window == video_causvid.dropped_tail_frames(3) == 9
    assert (window - 1) // 4 + 1 == 3


def test_select_tail_window_takes_newest_frames() -> None:
    frames = np.arange(20 * 2 * 2 * 3, dtype=np.uint8).reshape(20, 2, 2, 3)
    window = video_causvid.select_tail_window(frames, 3)
    assert window.shape[0] == 9
    assert window[0, 0, 0, 0] == frames[11, 0, 0, 0]
    with pytest.raises(ValueError, match="need 9"):
        video_causvid.select_tail_window(frames[:5], 3)


def test_commit_novel_slices_off_tail_markers() -> None:
    frames = np.zeros((81, 4, 6, 3), dtype=np.uint8)
    for index in range(81):
        frames[index] = index
    novel = video_causvid.commit_novel_frames(frames, 3)
    assert novel.shape[0] == 72
    assert int(novel[0, 0, 0, 0]) == 0
    assert int(novel[71, 0, 0, 0]) == 71


def _tape_kwargs(tail: Path) -> dict[str, Any]:
    return {
        "source_segment_id": "000007",
        "conditioning_tail_path": str(tail),
        "conditioning_tail_sha256": "1" * 64,
        "overlap_frames": 3,
        "num_frame_per_block": 3,
        "decoded_frames_per_rollout": 81,
        "novel_frames_per_rollout_count": 72,
        "rollouts": 2,
        "latent_shape": [1, 21, 16, 60, 104],
        "prompts": ["amber dunes", "teal spires"],
        "seeds": [7, 8],
        "width": 832,
        "height": 480,
        "fps": 16,
        "config_sha256": "2" * 64,
    }


def test_recovery_tape_round_trip(tmp_path: Path) -> None:
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"tail-bytes")
    tape = video_causvid.build_recovery_tape(**_tape_kwargs(tail))
    assert tape["profile"] == "causvid"
    assert tape["backend"] == "causvid"
    assert tape["state_mode"] == "reconstructable_prefix"
    assert tape["num_overlap_frames"] == 3
    assert tape["dropped_tail_frames"] == 9
    assert tape["novel_frames_per_rollout"] == 72
    assert tape["latent_shape"] == [1, 21, 16, 60, 104]
    assert tape["latent_tail_shape"] == [1, 2, 16, 60, 104]
    assert tape["latent_dtype"] == "bfloat16"
    assert tape["start_latents_from"] == video_causvid.START_FROM_UPSTREAM
    assert tape["code_commit"] == video_causvid.CAUSVID_COMMIT
    assert len(tape["profile_hash"]) == 64
    loaded = json.loads(json.dumps(tape))
    assert video_causvid.parse_recovery_tape(loaded) == loaded


def test_recovery_tape_rejects_foreign_json(tmp_path: Path) -> None:
    from voyage.workers import video_common

    with pytest.raises(ValueError, match="foreign or legacy"):
        video_causvid.parse_recovery_tape({"profile": "ltxv", "tail_png": "x"})
    with pytest.raises(ValueError, match="foreign or legacy"):
        video_causvid.parse_recovery_tape({})
    # Run-file pruning: a missing tail file parses (resume derives it);
    # only a missing tail *path* fails. Both-missing fails at ensure.
    tape = video_causvid.build_recovery_tape(**_tape_kwargs(tmp_path / "missing.mp4"))
    assert video_causvid.parse_recovery_tape(json.loads(json.dumps(tape))) == tape
    with pytest.raises(ValueError, match="conditioning tail missing"):
        video_common.ensure_conditioning_tail(tmp_path / "missing.mp4")


def test_load_tape_json_rejects_non_json_bytes(tmp_path: Path) -> None:
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_bytes(b"\x80\x04torch-pickle-not-json")
    with pytest.raises(ValueError, match="not JSON"):
        video_causvid._load_tape_json(str(tape_path))


def test_generate_blocks_end_to_end_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, torch, pipeline, _media = _test_session(monkeypatch)
    output = tmp_path / "seg" / "segment.mp4"
    result = session.generate_blocks(
        prompts=["amber dunes", "teal spires"],
        seeds=[7, 8],
        scene_cuts=[False, False],
        output_path=output,
        segment_id="000007",
    )
    # 2 rollouts x 81 decoded = 162 generated, 2 x 72 novel = 144 committed.
    assert result["frames"] == 144
    assert result["returned_frames"] == 144
    assert result["committed_frames"] == 144
    assert result["novel_frames"] == 144
    assert result["generated_frames"] == 162
    assert result["conditioning_frames"] == 18
    assert result["decoded_per_rollout"] == [81, 81]
    assert result["novel_per_rollout"] == [72, 72]
    assert result["native_fps"] == 16
    assert result["fps"] == 16
    assert result["overlap_frames"] == 3
    assert result["rollouts"] == 2
    assert result["fresh_rollouts"] == 1
    assert result["prompt_changed"] is False
    assert output.exists()
    # Continuation chains: rollout 1 rode rollout 0's tail latents.
    assert pipeline.calls[0]["start"] is None
    assert pipeline.calls[1]["start"] is not None
    assert session._start_latents is not None
    # Per-rollout determinism: one seeded generator per rollout.
    assert torch.generator_seeds == [7, 8]
    assert len(torch.randn_calls) == 2
    assert all("generator" in call["kwargs"] for call in torch.randn_calls)
    # Tape + (absent) tail beside the segment: run-file pruning records
    # the would-be path without persisting the file; the tape parses.
    tape = json.loads(Path(result["recovery_path"]).read_text(encoding="utf-8"))
    assert video_causvid.parse_recovery_tape(tape) == tape
    assert tape["novel_frames_per_rollout"] == 72
    assert tape["conditioning_tail_path"] == result["conditioning_tail_path"]
    assert "conditioning_tail_sha256" not in tape
    assert not Path(result["conditioning_tail_path"]).exists()


def test_generate_leaves_no_tail_file_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec (a): generate records the would-be tail path but writes no tail."""
    session, _torch, _pipeline, media = _test_session(monkeypatch)
    del _torch, _pipeline
    output = tmp_path / "seg" / "segment.mp4"
    result = session.generate_blocks(
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[True],
        output_path=output,
        segment_id="000007",
    )
    assert result["frames"] == 72
    assert output.exists()
    tail_path = Path(result["conditioning_tail_path"])
    assert tail_path.parent == output.parent
    assert tail_path.name == "video_tail.mp4"
    assert not tail_path.exists()
    assert [save["path"] for save in media["saves"]] == [str(output)]


def test_committed_markers_prove_tail_drop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame-index markers: a fresh rollout commits [0..71]; no tail is written."""
    session, _torch, _pipeline, media = _test_session(monkeypatch)
    del _torch, _pipeline
    session.generate_blocks(
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[True],
        output_path=tmp_path / "seg.mp4",
    )

    def _marker(frame: Any) -> int:
        return int(frame[0, 0, 0])

    assert len(media["saves"]) == 1
    segment_frames = media["saves"][0]["frames"]
    assert media["saves"][0]["fps"] == 16
    assert len(segment_frames) == 72
    for index, frame in enumerate(segment_frames):
        assert _marker(frame) == int(index / 81 * 255)


def test_rollout_seeds_global_rng_from_rollout_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream inference draws unseeded global-CUDA RNG per call (measured:
    advances cuda0 state; same-seed rollouts diverged ~0.5 latent mean-abs)
    — each rollout must seed it from its seed or nothing is reproducible."""
    session, torch, _pipeline, _media = _test_session(monkeypatch)
    session._run_rollout("amber dunes", 777, None, {"prompt_embeds": "x"})
    session._run_rollout("amber dunes", 778, None, {"prompt_embeds": "x"})
    assert torch.global_seeds == [777, 778]


def test_scene_cut_starts_fresh_then_chains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _torch, pipeline, _media = _test_session(monkeypatch)
    del _torch
    session.generate_blocks(
        prompts=["a", "b", "c"],
        seeds=[1, 2, 3],
        scene_cuts=[False, True, False],
        output_path=tmp_path / "seg.mp4",
    )
    assert pipeline.calls[0]["start"] is None  # cold session: fresh
    assert pipeline.calls[1]["start"] is None  # scene cut: fresh
    assert pipeline.calls[2]["start"] is not None  # chains onto rollout 1


def test_missing_resume_anchor_falls_back_to_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _torch, pipeline, _media = _test_session(monkeypatch)
    del _torch
    session._pending_tail_path = str(tmp_path / "gone.mp4")
    session.generate_blocks(
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[False],
        output_path=tmp_path / "seg.mp4",
    )
    assert pipeline.calls[0]["start"] is None
    assert session._pending_tail_path is None


def test_pipeline_errors_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, _torch, pipeline, _media = _test_session(monkeypatch)
    del _torch

    def _boom(**kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "inference", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        session.generate_blocks(
            prompts=["amber dunes"],
            seeds=[7],
            scene_cuts=[False],
            output_path=tmp_path / "seg.mp4",
        )


def test_generate_rejects_non_native_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """16 fps media is never relabeled as 24 (DESIGN §5.4 fps policy)."""
    session, _torch, _pipeline, _media = _test_session(monkeypatch)
    del _torch, _pipeline
    with pytest.raises(ValueError, match="native fps is 16"):
        session.generate_blocks(
            prompts=["amber dunes"],
            seeds=[7],
            scene_cuts=[False],
            output_path=tmp_path / "seg.mp4",
            fps=24,
        )


def test_result_dict_satisfies_segment_result_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _torch, _pipeline, _media = _test_session(monkeypatch)
    del _torch, _pipeline
    output = tmp_path / "seg.mp4"
    result = session.generate_blocks(
        prompts=["amber dunes"],
        seeds=[7],
        scene_cuts=[True],
        output_path=output,
    )
    segment = VideoSegmentResult(
        requested_frames=result["requested_frames"],
        returned_frames=result["returned_frames"],
        conditioning_frames=result["conditioning_frames"],
        novel_frames=result["novel_frames"],
        native_fps=result["native_fps"],
        output_path=str(output),
        backend="causvid",
        state_mode="reconstructable_prefix",
    )
    assert segment.requested_frames == 72
    assert segment.returned_frames == 72
    assert segment.novel_frames == 72
    assert segment.native_fps == 16


def test_handle_generate_blocks_requires_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video_causvid, "_SESSION", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        video_causvid.handle_generate_blocks(
            {
                "segment_id": "000000",
                "prompt": "amber dunes",
                "seed": 7,
                "output_path": str(tmp_path / "seg.mp4"),
                "fps": 16,
            }
        )


def test_handle_init_rejects_non_cuda_device(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        video_causvid.handle_init({"models_dir": str(tmp_path), "device": "cpu"})


def test_handle_init_missing_weights_says_download_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"voyage models download"):
        video_causvid.handle_init({"models_dir": str(tmp_path / "models")})


def test_handle_init_rejects_bad_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible by"):
        video_causvid.handle_init({"models_dir": str(tmp_path), "overlap_frames": 4})


def test_handle_resume_adopts_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, _torch, _pipeline, _media = _test_session(monkeypatch)
    del _torch, _pipeline
    tail = tmp_path / "video_tail.mp4"
    tail.write_bytes(b"tail-bytes")
    tape = video_causvid.build_recovery_tape(**_tape_kwargs(tail))
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_text(json.dumps(tape), encoding="utf-8")
    monkeypatch.setattr(video_causvid, "_SESSION", session)
    response = video_causvid.handle_resume({"recovery_path": str(tape_path)})
    assert response["resumed"] is True
    assert response["conditioning_tail_path"] == str(tail)
    assert response["start_latents_from"] == video_causvid.START_FROM_RESUME
    assert session._pending_tail_path == str(tail)


def test_handle_resume_rejects_legacy_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _torch, _pipeline, _media = _test_session(monkeypatch)
    del _torch, _pipeline
    tape_path = tmp_path / "recovery.pt"
    tape_path.write_bytes(b"\x80\x04torch-pickle-not-json")
    monkeypatch.setattr(video_causvid, "_SESSION", session)
    with pytest.raises(ValueError, match="not JSON"):
        video_causvid.handle_resume({"recovery_path": str(tape_path)})


def test_handle_benchmark_reports_generated_vs_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _torch, _pipeline, _media = _test_session(monkeypatch)
    del _torch, _pipeline
    sentinel = object()
    session._start_latents = sentinel  # type: ignore[attr-defined]
    fake_cuda = types.SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 6 * 1024**3,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(video_causvid, "_SESSION", session)
    report = video_causvid.handle_benchmark({"warmup": 1, "measured": 2})
    assert report["backend"] == "causvid"
    assert report["generated_frames_per_rollout"] == 81
    assert report["committed_frames_per_rollout"] == 72
    assert report["dropped_tail_frames_per_rollout"] == 9
    assert len(report["rollout_wall_seconds"]) == 2
    assert report["novel_fps_equivalent"] > 0
    # Benchmark never advances the stream: continuation state restored.
    assert session._start_latents is sentinel


def test_handle_evict_gpu_clears_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session, torch, _pipeline, _media = _test_session(monkeypatch)
    del _pipeline
    monkeypatch.setattr(video_causvid, "_SESSION", session)
    assert video_causvid.handle_evict_gpu({}) == {"evicted": True}
    assert video_causvid._SESSION is None
    assert torch.cuda.empty_cache_calls == 1


def test_worker_serve_map_covers_protocol() -> None:
    """The CausVid worker speaks every op the supervisor may send."""
    import inspect

    from voyage.workers import video_common

    # Since issue 019 the map is built by the shared factory: assert the
    # factory covers the protocol and main() delegates to it (with this
    # worker's handlers) instead of grepping main() for op literals.
    factory_source = inspect.getsource(video_common.standard_serve_map)
    for op in (
        "init",
        "health",
        "generate_blocks",
        "benchmark",
        "evict_gpu",
        "rebuild",
        "checkpoint",
        "resume",
        "shutdown",
    ):
        assert f'"{op}"' in factory_source
    main_source = inspect.getsource(video_causvid.main)
    assert "standard_serve_map" in main_source
    for handler in (
        "handle_init",
        "handle_health",
        "handle_generate_blocks",
        "handle_benchmark",
        "handle_evict_gpu",
        "handle_rebuild",
        "handle_resume",
    ):
        assert handler in main_source
    assert callable(serve)


def test_module_import_stays_lightweight() -> None:
    """Top-level import must not pull torch/causvid (CUDA-at-import rule)."""
    assert video_causvid._SESSION is None
    assert not any(name == "causvid" or name.startswith("causvid.") for name in sys.modules)
    assert video_causvid.CAUSVID_CHECKPOINT_FILE == ("autoregressive_checkpoint/model.pt")

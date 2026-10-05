"""Two-stream finalize: proxy-conditioned SFX bed in parallel with publish.

Covers the keyed ledger (proxy vs shipped stems never false-hit), the
atempo stretch builder, the stream-copy proxy reference, the dub helper,
and the parallel arm gate. Render legs use a silent-WAV worker double
(test_worker_perf_rank2 idiom); concat/stretch/dub legs use real ffmpeg
on tiny synth media (test_sfx_finalize idiom).
"""

from __future__ import annotations

import subprocess
import threading
import wave
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from voyage.console import ParallelFinalizeDisplay, VoyageConsole
from voyage.errors import MediaError
from voyage.media import SfxParallelRequest, sfx_parallel_armed
from voyage.sfx_finalize import (
    SFX_CONDITIONING_PROXY,
    SFX_CONDITIONING_SHIPPED,
    SfxWindow,
    append_sfx_window,
    atempo_chain_for_stretch,
    build_proxy_reference,
    load_sfx_ledger,
    stretch_and_dub_sfx_bed,
    validate_sfx_ledger,
)


def _chain_product(chain: str) -> float:
    """Multiply the atempo factors in a chain string back together."""
    product = 1.0
    for part in chain.split(","):
        product *= float(part.split("=")[1])
    return product


def test_atempo_identity_returns_none() -> None:
    assert atempo_chain_for_stretch(1.0) is None
    assert atempo_chain_for_stretch(1.0 + 1e-9) is None
    assert atempo_chain_for_stretch(1.0 - 1e-9) is None


def test_atempo_simple_speedup_chain() -> None:
    assert atempo_chain_for_stretch(1.5) == "atempo=1.5"
    assert atempo_chain_for_stretch(4.0) == "atempo=2.0,atempo=2.0"


def test_atempo_slowdown_chain_stays_in_range() -> None:
    chain = atempo_chain_for_stretch(0.1)
    assert chain is not None
    factors = [float(part.split("=")[1]) for part in chain.split(",")]
    assert all(0.5 <= factor <= 2.0 for factor in factors)
    assert _chain_product(chain) == pytest.approx(0.1)


def test_atempo_rejects_bogus_factors() -> None:
    for bogus in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(MediaError):
            atempo_chain_for_stretch(bogus)


def test_gate_arms_only_for_real_backends() -> None:
    def _request(backend: str) -> SfxParallelRequest:
        return SfxParallelRequest(
            backend=backend,
            models_dir="/models",
            device="cpu",
            model_size="small_44k",
            num_workers=1,
            fps=24,
        )

    assert sfx_parallel_armed(None) is False
    assert sfx_parallel_armed(_request("fake")) is False
    assert sfx_parallel_armed(_request("mmaudio")) is True


def _window() -> SfxWindow:
    return SfxWindow("w0000", 0.0, 8.0, "rain on glass", 7)


def test_cache_hit_requires_conditioning_source() -> None:
    from voyage.sfx_finalize import _stem_cache_hit

    shipped = {
        "caption": "rain on glass",
        "seed": 7,
        "model_size": "small_44k",
        "duration": 8.0,
        "conditioning_source": SFX_CONDITIONING_SHIPPED,
    }
    proxy = dict(shipped, conditioning_source=SFX_CONDITIONING_PROXY)
    legacy = {k: v for k, v in shipped.items() if k != "conditioning_source"}
    assert _stem_cache_hit(shipped, _window(), "small_44k", SFX_CONDITIONING_SHIPPED) is True
    assert _stem_cache_hit(shipped, _window(), "small_44k", SFX_CONDITIONING_PROXY) is False
    assert _stem_cache_hit(proxy, _window(), "small_44k", SFX_CONDITIONING_SHIPPED) is False
    assert _stem_cache_hit(proxy, _window(), "small_44k", SFX_CONDITIONING_PROXY) is True
    # Legacy lines predate the key and read as shipped pixels.
    assert _stem_cache_hit(legacy, _window(), "small_44k", SFX_CONDITIONING_SHIPPED) is True
    assert _stem_cache_hit(legacy, _window(), "small_44k", SFX_CONDITIONING_PROXY) is False


def test_proxy_group_validates_against_its_own_timeline(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = run_dir / "audio" / "sfx" / "sfx.jsonl"
    append_sfx_window(
        ledger,
        SfxWindow("w0000", 0.0, 8.0, "rain", 7),
        path="audio/sfx/w0000.wav",
        model_size="small_44k",
        conditioning_source=SFX_CONDITIONING_PROXY,
        conditioning_timeline=8.0,
    )
    (run_dir / "audio" / "sfx" / "w0000.wav").write_bytes(b"RIFF" + b"\0" * 100)
    # The proxy group tiles its source timeline: validating a longer
    # shipped timeline must not false-positive a shortfall on it.
    assert validate_sfx_ledger(run_dir, 12.0) == []


def test_proxy_and_shipped_records_coexist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = run_dir / "audio" / "sfx" / "sfx.jsonl"
    for source in (SFX_CONDITIONING_PROXY, SFX_CONDITIONING_SHIPPED):
        append_sfx_window(
            ledger,
            SfxWindow("w0000", 0.0, 8.0, "rain", 7),
            path="audio/sfx/w0000.wav",
            model_size="small_44k",
            conditioning_source=source,
            conditioning_timeline=8.0,
        )
    (run_dir / "audio" / "sfx" / "w0000.wav").write_bytes(b"RIFF" + b"\0" * 100)
    records = load_sfx_ledger(ledger)
    assert [r["conditioning_source"] for r in records] == [
        SFX_CONDITIONING_PROXY,
        SFX_CONDITIONING_SHIPPED,
    ]
    assert validate_sfx_ledger(run_dir, 8.0) == []


class _SilentSfxWorker:
    """SubprocessWorker double rendering silent WAVs (test_worker_perf_rank2 idiom)."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        del op
        type(self).calls.append(dict(payload))
        out = Path(str(payload["output_path"]))
        rate = int(payload["sample_rate"])
        channels = int(payload["channels"])
        frames = max(1, int(float(payload["duration_seconds"]) * rate))
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(b"\x00" * frames * channels * 2)
        return {"sfx": {"duration_seconds": float(payload["duration_seconds"])}}


def _render_bed(run_dir: Path, conditioning_source: str) -> Path:
    from voyage.sfx_finalize import render_sfx_bed

    _SilentSfxWorker.calls.clear()
    run_dir.mkdir(parents=True, exist_ok=True)
    reference = run_dir / "proxy_ref.mp4"
    reference.write_bytes(b"conditioning-pixels-stand-in")
    bed = render_sfx_bed(
        run_dir,
        reference,
        4.0,
        [(0.0, 4.0, "low rumble")],
        run_dir,
        "mmaudio",
        "/models",
        "cpu",
        "small_44k",
        3,
        48000,
        2,
        1,
        progress=None,
        conditioning_source=conditioning_source,
        conditioning_timeline=4.0,
    )
    assert bed.exists()
    return bed


def test_proxy_then_shipped_rerenders_without_false_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voyage.sfx_finalize import render_sfx_bed

    monkeypatch.setattr("voyage.rpc.SubprocessWorker", _SilentSfxWorker)
    run_dir = tmp_path / "run"
    _render_bed(run_dir, SFX_CONDITIONING_PROXY)
    assert len(_SilentSfxWorker.calls) == 1
    # Same plan on shipped pixels must re-render (different conditioning).
    _render_bed(run_dir, SFX_CONDITIONING_SHIPPED)
    assert len(_SilentSfxWorker.calls) == 1
    # ... and once more on shipped hits the shipped stem (no third render).
    render_sfx_bed(
        run_dir,
        run_dir / "proxy_ref.mp4",
        4.0,
        [(0.0, 4.0, "low rumble")],
        run_dir,
        "mmaudio",
        "/models",
        "cpu",
        "small_44k",
        3,
        48000,
        2,
        1,
        progress=None,
        conditioning_source=SFX_CONDITIONING_SHIPPED,
        conditioning_timeline=4.0,
    )
    assert len(_SilentSfxWorker.calls) == 1
    assert validate_sfx_ledger(run_dir, 4.0) == []


def _synth_clip(dest: Path, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x64:rate=8:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        check=True,
    )


def _synth_tone(dest: Path, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(dest),
        ],
        check=True,
    )


def _media_duration(path: Path) -> float:
    from voyage.media import probe

    return float(probe(path).get("format", {}).get("duration", 0.0) or 0.0)


def test_build_proxy_reference_concats_source_duration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    usable = []
    for index in range(2):
        segment = run_dir / "segments" / f"{index:06d}"
        segment.mkdir(parents=True)
        _synth_clip(segment / "video.mp4", 2.0)
        usable.append(segment)
    proxy, source_seconds = build_proxy_reference(run_dir, usable, tmp_path)
    assert proxy.exists()
    assert source_seconds == pytest.approx(4.0, abs=0.2)


def test_build_proxy_reference_rejects_empty_run(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="no usable segments"):
        build_proxy_reference(tmp_path / "run", [], tmp_path)


def test_stretch_and_dub_identity_and_slowmo(tmp_path: Path) -> None:
    staged = tmp_path / "staged.mp4"
    _synth_clip(staged, 4.0)
    music = tmp_path / "music.wav"
    _synth_tone(music, 4.0)
    bed = tmp_path / "bed.wav"
    _synth_tone(bed, 2.0)
    identity = stretch_and_dub_sfx_bed(staged, bed, music, tmp_path / "dub1.mp4", 48000, 2, 1.0)
    assert _media_duration(identity) == pytest.approx(4.0, abs=0.2)
    slowed = stretch_and_dub_sfx_bed(staged, bed, music, tmp_path / "dub2.mp4", 48000, 2, 2.0)
    assert _media_duration(slowed) == pytest.approx(4.0, abs=0.3)


def test_stream_view_emit_never_self_deadlocks() -> None:
    """Concurrent view emits (info/ok/error/line) always return.

    Regression: the view once locked `info` and re-acquired the same
    non-reentrant coordinator lock via base-`info` -> `self.styled`,
    wedging finalize forever at the first dub line. Only leaf emits
    (`line`/`styled`/`error`) may take the lock; `info`/`ok` inherit
    the base fan-out untouched.
    """
    stream = StringIO()
    display = ParallelFinalizeDisplay(VoyageConsole(stream=stream))
    view = display.stream_view("sfx bed")
    errors: list[BaseException] = []

    def _emit() -> None:
        try:
            for _ in range(25):
                view.info("hello")
                view.ok("done")
                view.error("boom")
                view.line("plain")
        except BaseException as exc:  # noqa: BLE001 — collected, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_emit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    display.close()
    assert "hello" in stream.getvalue()

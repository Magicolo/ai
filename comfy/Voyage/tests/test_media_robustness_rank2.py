"""Rank-2 media/CLI robustness issues 017/018/019/102/103 (CPU-only).

017: `validate_run` must map hostile inputs (dir-as-file, nested-metrics
    blowup) to error strings, never raise.
018: `probe` must wrap ffprobe JSON parse failures as `MediaError`.
019: `run_capture` must bound every ffmpeg/ffprobe spawn with `timeout=`
    and map `TimeoutExpired` to `MediaError`.
102: finalize staging must live on the preflight-measured filesystem
    (novel leg 3 only — concat quoting (053) and RAM staging (043) are
    owned elsewhere and asserted only via cross-refs here).
103: `_embed_texts` must degrade hostile/non-finite worker vectors to the
    token-set fallback (`None`), never raise or poison novelty.

Real ffmpeg where cheap (slim image ships it); synthetic fakes otherwise.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import initialize_run_directory
from voyage import paths
from voyage.config import load_config


def _commit_single_segment(run_dir: Path) -> None:
    """Commit one fake-backend segment so `validate`/`finalize` have a valid run."""
    from voyage.supervisor import Supervisor

    initialize_run_directory(run_dir, run_id="rank2", style="pastel neon line-art, peaceful")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)
    supervisor.start_workers()
    try:
        supervisor.commit_one_segment()
    finally:
        supervisor.stop_workers()


def _segment_dir(run_dir: Path) -> Path:
    segment = run_dir / paths.SEGMENTS_DIRNAME / "000000"
    assert segment.is_dir()
    return segment


# ---------------------------------------------------------------------------
# 017: validate_run is total over hostile inputs
# ---------------------------------------------------------------------------


def test_validate_dir_as_video_returns_error_not_raise(tmp_path: Path) -> None:
    """A directory where `video.mp4` belongs yields INVALID errors, no traceback."""
    from voyage.cli import validate_run

    run_dir = tmp_path / "run"
    _commit_single_segment(run_dir)
    segment = _segment_dir(run_dir)
    (segment / "video.mp4").unlink()
    (segment / "video.mp4").mkdir()
    errors = validate_run(run_dir)
    assert errors, "dir-as-video.mp4 must be reported, not silently valid"
    assert any("video.mp4" in error for error in errors)


def test_validate_dir_as_metrics_returns_error_not_raise(tmp_path: Path) -> None:
    """A directory where `manifest.json` belongs yields INVALID errors, no traceback."""
    from voyage.cli import validate_run

    run_dir = tmp_path / "run"
    _commit_single_segment(run_dir)
    segment = _segment_dir(run_dir)
    (segment / "manifest.json").unlink()
    (segment / "manifest.json").mkdir()
    errors = validate_run(run_dir)
    assert errors, "dir-as-manifest.json must be reported, not silently valid"
    assert any("manifest.json" in error for error in errors)


def test_validate_deeply_nested_metrics_returns_error_not_raise(tmp_path: Path) -> None:
    """Deeply nested `manifest.json` maps to an error string, never RecursionError."""
    from voyage.cli import validate_run

    run_dir = tmp_path / "run"
    _commit_single_segment(run_dir)
    segment = _segment_dir(run_dir)
    depth = 20000
    hostile = '{"format": 1, "metrics": {"frames": ' + "[" * depth + "1" + "]" * depth + "}}"
    (segment / "manifest.json").write_text(hostile, encoding="utf-8")
    errors = validate_run(run_dir)
    assert errors, "nested manifest.json must be reported, not silently valid"
    assert any("manifest.json" in error for error in errors)


def test_validate_dir_as_sha256_returns_error_not_raise(tmp_path: Path) -> None:
    """A directory where `manifest.json` belongs yields INVALID errors, no traceback."""
    from voyage.cli import validate_run

    run_dir = tmp_path / "run"
    _commit_single_segment(run_dir)
    segment = _segment_dir(run_dir)
    (segment / "manifest.json").unlink()
    (segment / "manifest.json").mkdir()
    errors = validate_run(run_dir)
    assert errors, "dir-as-manifest.json must be reported, not silently valid"
    assert any("manifest.json" in error for error in errors)


# ---------------------------------------------------------------------------
# 018: probe taxonomy — malformed JSON is a MediaError
# ---------------------------------------------------------------------------


def test_probe_invalid_json_raises_media_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncated ffprobe stdout maps to MediaError, never raw JSONDecodeError."""
    import voyage.media as media_module
    from voyage.errors import MediaError

    def _truncated(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "{truncated", "")

    monkeypatch.setattr(media_module, "run_capture", _truncated)
    with pytest.raises(MediaError, match="invalid JSON"):
        media_module.probe(Path("clip.mp4"))


def test_probe_garbage_stdout_raises_media_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-JSON stdout with exit 0 maps to MediaError."""
    import voyage.media as media_module
    from voyage.errors import MediaError

    def _garbage(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "not json at all", "")

    monkeypatch.setattr(media_module, "run_capture", _garbage)
    with pytest.raises(MediaError):
        media_module.probe(Path("clip.mp4"))


def test_probe_valid_json_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The taxonomy wrap must not break the happy path."""
    import voyage.media as media_module

    payload = json.dumps({"format": {"duration": "1.0"}, "streams": []})

    def _valid(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, payload, "")

    monkeypatch.setattr(media_module, "run_capture", _valid)
    assert media_module.probe(Path("clip.mp4"))["format"]["duration"] == "1.0"


# ---------------------------------------------------------------------------
# 019: run_capture bounds every spawn with timeout=
# ---------------------------------------------------------------------------


def test_run_capture_exposes_default_timeout() -> None:
    """The helper carries a mirroring default (600 s, like the RPC layer)."""
    import voyage.media as media_module

    assert media_module.FFMPEG_TIMEOUT_SECONDS == 600.0
    assert "timeout" in media_module.run_capture.__code__.co_varnames


def test_run_capture_timeout_expired_maps_to_media_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged child surfaces as MediaError, never a bare TimeoutExpired."""
    import subprocess as subprocess_module

    import voyage.media as media_module
    from voyage.errors import MediaError

    def _wedged(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess_module.TimeoutExpired(cmd="ffmpeg", timeout=0.01)

    monkeypatch.setattr(subprocess_module, "run", _wedged)
    with pytest.raises(MediaError, match="timed out"):
        media_module.run_capture(["ffmpeg", "-version"], timeout=0.01)


def test_run_capture_threads_timeout_to_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout value reaches `subprocess.run` (threaded, not dropped)."""
    import subprocess as subprocess_module

    import voyage.media as media_module

    seen: dict[str, Any] = {}
    real_run = subprocess_module.run

    def _recording(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return real_run(["true"], **kwargs)

    monkeypatch.setattr(subprocess_module, "run", _recording)
    media_module.run_capture(["true"], timeout=7.5)
    assert seen.get("timeout") == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# 102 leg 3 (novel): finalize staging lives on the run filesystem
# ---------------------------------------------------------------------------


def test_finalize_staging_uses_run_dir_with_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalize staging passes `dir=<run fs>` + greppable prefix (never bare /tmp)."""
    import voyage.media as media_module

    real_dir = tmp_path / "run"
    _commit_single_segment(real_dir)

    real_temporary_directory = tempfile.TemporaryDirectory
    calls: list[dict[str, Any]] = []

    def _recording(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _recording)
    out = real_dir.parent / "final-staging-probe.mp4"
    media_module.finalize_run(real_dir, out, min_fps=0, min_width=0, min_height=0)
    assert out.exists()
    finalize_calls = [call for call in calls if "voyage-final" in str(call.get("prefix", ""))]
    assert finalize_calls, f"no prefixed finalize staging among {calls!r}"
    staged = finalize_calls[0].get("dir")
    assert staged is not None, f"finalize staging missing dir= among {calls!r}"
    assert Path(str(staged)).resolve() == real_dir.resolve()


def test_finalize_publish_streams_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-ref leg 2 (043): the publish path never `read_bytes` a staged MP4."""
    import voyage.media as media_module

    run_dir = tmp_path / "run"
    _commit_single_segment(run_dir)
    real_read_bytes = Path.read_bytes

    def _guard(self: Path) -> bytes:
        if self.suffix == ".mp4":
            raise AssertionError(f"publish path read {self} into RAM (issues 043/102)")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guard)
    out = tmp_path / "final-noram.mp4"
    assert media_module.finalize_run(run_dir, out, min_fps=0, min_width=0, min_height=0).exists()


# ---------------------------------------------------------------------------
# 103: hostile/non-finite embed vectors degrade to the fallback
# ---------------------------------------------------------------------------


def _embed_with_vectors(tmp_path: Path, vectors: Any) -> list[list[float]] | None:
    """Run `_embed_texts` against a director double returning `vectors`."""
    from voyage.supervisor import Supervisor

    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, run_id="rank2embed", style="pastel neon line-art, peaceful")
    config, _ = load_config(run_dir / paths.CONFIG_FILENAME)
    supervisor = Supervisor(run_dir, config)

    def _hostile(op: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return {"vectors": vectors}

    supervisor._director.call = _hostile  # type: ignore[method-assign]
    return supervisor._embed_texts(["a misty harbor"])


def test_embed_garbage_string_degrades_to_fallback(tmp_path: Path) -> None:
    """`[['abc']]` returns None (fallback), never raises ValueError."""
    assert _embed_with_vectors(tmp_path, [["abc"]]) is None


def test_embed_none_component_degrades_to_fallback(tmp_path: Path) -> None:
    """`[[None]]` returns None (fallback), never raises TypeError."""
    assert _embed_with_vectors(tmp_path, [[None]]) is None


def test_embed_dict_component_degrades_to_fallback(tmp_path: Path) -> None:
    """`[[{}]]` returns None (fallback), never raises TypeError."""
    assert _embed_with_vectors(tmp_path, [[{}]]) is None


def test_embed_nan_degrades_to_fallback(tmp_path: Path) -> None:
    """`[['nan']]` coerces to NaN and must return None, never a poisoned vector."""
    assert _embed_with_vectors(tmp_path, [["nan"]]) is None


def test_embed_inf_degrades_to_fallback(tmp_path: Path) -> None:
    """`[[1e999]]` coerces to inf and must return None, never a poisoned vector."""
    assert _embed_with_vectors(tmp_path, [[1e999]]) is None


def test_embed_healthy_vectors_still_pass_through(tmp_path: Path) -> None:
    """The guard must not break the happy path."""
    assert _embed_with_vectors(tmp_path, [[0.5, -0.25]]) == [[0.5, -0.25]]

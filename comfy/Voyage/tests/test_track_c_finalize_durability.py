"""Track C finalize + state/durability helpers (DESIGN §56, AGENTS §11/§12).

New tests ONLY as `test_track_c_*.py` per the track brief. All tests are
pure/stdlib (tmp_path scaffolds, injected fakes) — no GPU, no network,
no ffmpeg, no torch. ffprobe-dependent gates are exercised via
monkeypatched seams or the unprobable → fail-open branches.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from voyage import paths
from voyage.augment_drain import (
    prune_stale_joint_artifacts,
    settle_uses_output_truth,
    sweep_chunk_mp4s,
)
from voyage.augment_finalize import (
    _JOINT_UNIT_CACHE,
    joint_units_signature,
    verify_settle_output_truth,
)
from voyage.augment_sidecar import (
    ChunkKey,
    append_chunk_record,
    chunk_mp4_complete,
    chunk_mp4_frames_match,
    load_chunk_ledger,
)
from voyage.cli_validate import (
    ORPHAN_PATTERNS,
    _check_morph_joints_ledger_vs_output,
    _warn_done_less_numeric_dirs,
)
from voyage.media import (
    check_expected_frames_gate,
    check_free_space_at_phase,
    expected_music_seconds,
    expected_presented_frames,
    free_gib,
    music_cache_usable,
    run_publish_encode,
    takes_ledger_normalized_hash,
)
from voyage.persistence import interp_backend_or_default, write_manifest_state_group
from voyage.segment_manifest import write_manifest_metrics_group


def _segment_with_frames(run_dir: Path, number: int, frames: int) -> Path:
    segment = run_dir / "segments" / f"{number:06d}"
    segment.mkdir(parents=True, exist_ok=True)
    (segment / "DONE").write_text("", encoding="utf-8")
    (segment / "video.mp4").write_bytes(b"\x00" * 64)
    (segment / "manifest.json").write_text(
        json.dumps(
            {"format": 1, "metrics": {"frames": frames}, "checksums": {}},
        ),
        encoding="utf-8",
    )
    return segment


# --- persistence helpers ----------------------------------------------------


def test_interp_backend_or_default_central() -> None:
    assert interp_backend_or_default("film") == "film"
    assert interp_backend_or_default("rife") == "rife"
    assert interp_backend_or_default(None) == "film"
    assert interp_backend_or_default("bogus") == "film"
    assert interp_backend_or_default(123) == "film"


def test_write_manifest_state_group_roundtrip(tmp_path: Path) -> None:
    from voyage.config import preset_config
    from voyage.persistence import read_manifest, read_state

    config = preset_config("trackc", "calm line art", 7)
    manifest = {
        **config.model_dump(mode="json"),
        "segments": 2,
        "final_video": None,
        "skip_bad": False,
        "no_sfx": False,
        "schema_version": paths.SCHEMA_VERSION,
    }
    from voyage.models import RunState

    state = RunState(
        name="trackc",
        status="CREATED",
        current_concept="calm line art",
        destination_concept="calm line art",
    )
    write_manifest_state_group(tmp_path, manifest, state)
    assert read_manifest(tmp_path)["segments"] == 2
    assert read_state(tmp_path).name == "trackc"
    assert not list(tmp_path.glob("*.partial"))


def test_write_manifest_metrics_group(tmp_path: Path) -> None:
    segment = tmp_path / "000000"
    segment.mkdir(parents=True)
    manifest: dict[str, object] = {"transition": {}, "prompt_plan": {}, "checksums": {}}
    metrics = {"frames": 5}
    dest = write_manifest_metrics_group(segment, manifest, metrics)
    assert dest == segment / "manifest.json"
    loaded = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["metrics"] == {"frames": 5}
    assert loaded["format"] == 1
    assert not list(segment.glob("*.partial"))


# --- media helpers ------------------------------------------------------------


def test_music_cache_usable_rejects_missing_and_zero() -> None:
    assert music_cache_usable(Path("/nonexistent.wav"), 10.0) is False
    assert music_cache_usable(Path("/nonexistent.wav"), 0.0) is False
    assert music_cache_usable(Path("/nonexistent.wav"), -1.0) is False


def test_music_cache_usable_probes_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"\x00" * 16)
    import voyage.media as media

    monkeypatch.setattr(media, "_audio_duration_seconds", lambda _p: 10.0)
    assert music_cache_usable(wav, 10.2) is True
    assert music_cache_usable(wav, 11.0) is False
    monkeypatch.setattr(
        media, "_audio_duration_seconds", lambda _p: (_ for _ in ()).throw(OSError("nope"))
    )
    assert music_cache_usable(wav, 10.0) is False


def test_expected_music_seconds_degenerate() -> None:
    assert expected_music_seconds([], 24.0, 1.0) == 0.0
    assert expected_music_seconds([], 0.0, 1.0) == 0.0
    assert expected_music_seconds([], 24.0, 0.0) == 0.0


def test_takes_ledger_hash_order_normalized(tmp_path: Path) -> None:
    ledger = tmp_path / "takes.jsonl"
    rows = [
        {"take_id": "take_0002", "covers_from": 20.0},
        {"take_id": "take_0001", "covers_from": 0.0},
    ]
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    forward = takes_ledger_normalized_hash(ledger, "take_id")
    ledger.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\n", encoding="utf-8")
    assert takes_ledger_normalized_hash(ledger, "take_id") == forward
    assert takes_ledger_normalized_hash(tmp_path / "missing.jsonl", "take_id") == "missing"


def test_run_publish_encode_logs_and_timeout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import voyage.media as media

    class _Proc:
        returncode = 0
        stderr = ""

    monkeypatch_calls: list[dict[str, object]] = []

    def _fake_capture(argv: list[str], timeout: float | None = None) -> _Proc:
        monkeypatch_calls.append({"timeout": timeout})
        assert argv[0] == "ffmpeg"
        return _Proc()

    import unittest.mock as mock

    with mock.patch.object(media, "run_capture", side_effect=_fake_capture):
        proc = run_publish_encode(["ffmpeg", "-y"], timeout=None, label="unit")
        assert proc.returncode == 0
    out = capsys.readouterr().out
    assert "publish [unit]" in out and "timeout=None" in out
    assert monkeypatch_calls[0]["timeout"] is None


def test_run_publish_encode_timeout_raises() -> None:
    import subprocess as _subprocess
    import unittest.mock as mock

    import voyage.media as media
    from voyage.errors import MediaError

    def _raise(argv: list[str], timeout: float | None = None) -> object:
        raise _subprocess.TimeoutExpired(cmd=argv, timeout=timeout or 0)

    with (
        mock.patch.object(media, "run_capture", side_effect=_raise),
        pytest.raises(MediaError, match="timed out"),
    ):
        run_publish_encode(["ffmpeg"], timeout=0.01, label="unit")


def test_free_gib_and_phase_preflight(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    free = free_gib(tmp_path)
    assert free is None or free > 0
    # Floor 0 still probes + logs (no raise).
    returned = check_free_space_at_phase(tmp_path, 0.0, "unit-phase")
    assert returned >= 0
    assert "unit-phase" in capsys.readouterr().out


def test_expected_presented_frames_and_gate(tmp_path: Path) -> None:
    _segment_with_frames(tmp_path, 0, 33)
    _segment_with_frames(tmp_path, 1, 33)
    usable = [tmp_path / "segments" / "000000", tmp_path / "segments" / "000001"]
    expected = expected_presented_frames(
        usable, source_fps=24.0, interpolate=2, out_fps=48, chunk_frames=32
    )
    assert expected > 0
    assert (
        check_expected_frames_gate(
            usable,
            source_fps=24.0,
            interpolate=2,
            out_fps=48,
            published_frames=expected,
        )
        == expected
    )
    from voyage.errors import MediaError

    with pytest.raises(MediaError, match="expected-frames gate"):
        check_expected_frames_gate(
            usable,
            source_fps=24.0,
            interpolate=2,
            out_fps=48,
            published_frames=expected + 50,
        )
    with pytest.raises(MediaError, match="unprobable"):
        check_expected_frames_gate(
            usable, source_fps=24.0, interpolate=2, out_fps=48, published_frames=None
        )


def test_finalize_options_publish_timeout_validation() -> None:
    from voyage.media import FinalizeOptions, resolve_finalize_settings

    opts = FinalizeOptions(publish_timeout_seconds=120.0)
    assert opts.publish_timeout_seconds == 120.0
    with pytest.raises(ValueError, match="publish_timeout_seconds"):
        FinalizeOptions(publish_timeout_seconds=-5.0)
    with pytest.raises(TypeError, match="publish_timeout_seconds"):
        FinalizeOptions(publish_timeout_seconds=True)
    resolved = resolve_finalize_settings(
        options=None,
        upscale=1,
        interpolate=1,
        crf=30,
        preset="slow",
        publish_timeout_seconds=45.0,
    )
    assert resolved.publish_timeout_seconds == 45.0
    resolved_default = resolve_finalize_settings(
        options=None, upscale=1, interpolate=1, crf=30, preset="slow"
    )
    assert resolved_default.publish_timeout_seconds is None


def test_model_pass_stage_rows_unified_keys() -> None:
    from voyage.media import _model_pass_stage_rows

    rows = _model_pass_stage_rows(
        {"upscale_poll_s": 1.0, "interp_poll_s": 2.0, "drain_s": 0.5, "morph_s": 0.25},
        0.0,
    )
    assert rows["upscale"] == 1.0
    assert rows["interpolate"] == 2.0
    assert rows["drain"] == 0.5
    assert rows["morph"] == 0.25


# --- paths / scratch ------------------------------------------------------------


def test_stale_scratch_prefixes_cover_track_c() -> None:
    for prefix in (
        "voyage-take-",
        "voyage-sfx-window-",
        "voyage-assemble-",
        "voyage-bench-",
        "ltx25-mux-",
        "ltx23-mux-",
    ):
        assert prefix in paths.STALE_SCRATCH_PREFIXES


def test_stale_scratch_prefixes_exclude_live_finalize_dirs(tmp_path: Path) -> None:
    """Live finalize tmpdirs must never be prunable (2026-10-07 B-vs-C).

    `finalize_sfx_pass` used to call the pruning `ensure_scratch_dir`
    mid-run as its tmp parent while the live `voyage-sfx-final-*` /
    `voyage-master-*` prefixes (plus the broad `voyage-sfx-` form that
    startswith-matches both live and window dirs) sat in STALE — the
    prune deleted the live tmpdir (ENOENT bed copy). The narrow
    `voyage-sfx-window-` per-window prefix stays listed (never live
    across a startup prune); everything live is excluded here and
    survives even the pruning ensure.
    """
    for live in ("voyage-sfx-final-", "voyage-master-", "voyage-sfx-"):
        assert live not in paths.STALE_SCRATCH_PREFIXES
    live_final = paths.scratch_dir(tmp_path) / "voyage-sfx-final-live123"
    live_final.mkdir(parents=True)
    (live_final / "bed.wav").write_bytes(b"\x00" * 16)
    assert paths.ensure_scratch_dir(tmp_path) == paths.scratch_dir(tmp_path)
    assert (live_final / "bed.wav").is_file()
    assert paths.ensure_scratch_dir_no_prune(tmp_path) == paths.scratch_dir(tmp_path)
    assert (live_final / "bed.wav").is_file()


def test_scratch_tmp_size_and_heal(tmp_path: Path) -> None:
    scratch = paths.scratch_dir(tmp_path)
    scratch.mkdir(parents=True)
    blob = scratch / "voyage-take-abc" / "take.wav"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x00" * 100)
    assert paths.scratch_tmp_size_bytes(tmp_path) == 100
    stale_final = tmp_path / "voyage-final-xyz"
    stale_final.mkdir()
    (stale_final / "concat.txt").write_text("x", encoding="utf-8")
    assert paths.heal_scratch_roots(tmp_path) >= 1
    assert not stale_final.exists()
    assert paths.scratch_tmp_size_bytes(tmp_path) is not None


# --- validate -----------------------------------------------------------------


def test_orphan_patterns_unified() -> None:
    assert ORPHAN_PATTERNS == ("*.partial", "*.partial.*", "*.tmp.npy", "*.tmp*")


def test_morph_joints_ledger_vs_output(tmp_path: Path) -> None:
    morph_root = tmp_path / "augment" / "morph_joints"
    morph_root.mkdir(parents=True)
    ledger = morph_root / "record.json"
    ledger.write_text(json.dumps([{"path": "joint_00_01.mp4"}]), encoding="utf-8")
    errors = _check_morph_joints_ledger_vs_output(tmp_path)
    assert any("morph_joints" in error for error in errors)
    (morph_root / "joint_00_01.mp4").write_bytes(b"\x00" * 32)
    assert _check_morph_joints_ledger_vs_output(tmp_path) == []


def test_done_less_numeric_dirs_warn(tmp_path: Path) -> None:
    segments = tmp_path / "segments"
    segments.mkdir(parents=True)
    stranded = segments / "000003"
    stranded.mkdir()
    (stranded / "video.mp4").write_bytes(b"\x00")
    warnings = _warn_done_less_numeric_dirs(segments)
    assert any("000003" in warning and "without DONE" in warning for warning in warnings)
    assert _warn_done_less_numeric_dirs(tmp_path / "nope") == []


# --- sidecar / drain ------------------------------------------------------------


def _chunk_key(*, index: int = 0, expected: int = 3) -> ChunkKey:
    return ChunkKey(
        chunk_index=index,
        start_frame=0,
        source_frames=3,
        expected_frames=expected,
        upscale_factor=1,
        multiplier=2,
        crf=15,
        preset="veryfast",
        source_key="src",
        weights_key="w",
        out_width=64,
        out_height=64,
        out_fps=24,
        chunk_frames=32,
    )


def test_chunk_mp4_frames_match_unprobable_and_missing(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    key = _chunk_key()
    assert chunk_mp4_frames_match(plan, key) is False
    mp4 = plan / "chunk_00.mp4"
    mp4.write_bytes(b"\x00" * 64)
    # ffprobe cannot parse a zero blob → unprobable → fail-open True.
    assert chunk_mp4_frames_match(plan, key) is True


def test_chunk_mp4_complete_requires_record(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "chunk_00.mp4").write_bytes(b"\x00" * 64)
    key = _chunk_key()
    assert chunk_mp4_complete(plan, [], key) is False
    ledger = plan / "chunks.jsonl"
    append_chunk_record(ledger, key, stage="chunk_mp4", path="chunk_00.mp4")
    assert chunk_mp4_complete(plan, load_chunk_ledger(ledger), key) is True


def test_settle_uses_output_truth(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    key = _chunk_key(expected=2)
    interp_dir = plan / "interpolated_00"
    interp_dir.mkdir()
    for index in range(2):
        (interp_dir / f"frame_{index:06d}.png").write_bytes(b"\x00" * 8)
    ledger = plan / "chunks.jsonl"
    append_chunk_record(ledger, key, stage="chunk_mp4", path="chunk_00.mp4")
    (plan / "chunk_00.mp4").write_bytes(b"\x00" * 64)
    from voyage.augment_drain import settle_uses_output_truth as _settle

    assert _settle(plan, load_chunk_ledger(ledger), key) is True
    (interp_dir / "frame_000001.png").unlink()
    # PNG dir now short but the durable mp4 still settles (mp4 is truth).
    assert settle_uses_output_truth(plan, load_chunk_ledger(ledger), key) is True


def test_sweep_live_dirs_only(tmp_path: Path) -> None:
    live = tmp_path / "augment" / "livehash12345678"
    orphan = tmp_path / "augment" / "deadbeefdeadbeef"
    for plan_dir in (live, orphan):
        plan_dir.mkdir(parents=True)
        interp = plan_dir / "interpolated_00"
        interp.mkdir()
        for index in range(2):
            (interp / f"frame_{index:06d}.png").write_bytes(b"\x00" * 8)
        key = _chunk_key(expected=2)
        append_chunk_record(plan_dir / "chunks.jsonl", key, stage="interpolated", path="x")

    def _fake_encode(png_dir: Path, dest: Path, fps: float) -> Path:
        dest.write_bytes(b"\x00" * 16)
        return dest

    ensured, _skipped = sweep_chunk_mp4s(tmp_path, encode_fn=_fake_encode, live_dirs={live})
    assert ensured >= 1
    assert (live / "chunk_00.mp4").exists()
    assert not (orphan / "chunk_00.mp4").exists()


def test_prune_stale_joint_artifacts_grace(tmp_path: Path) -> None:
    units = tmp_path / "augment" / "joint_sources"
    stale = units / "000000_000001"
    stale.mkdir(parents=True)
    (stale / "joint.mp4").write_bytes(b"\x00" * 8)
    old = time.time() - 10 * 86400.0
    import os

    os.utime(stale / "joint.mp4", (old, old))
    os.utime(stale, (old, old))
    pruned = prune_stale_joint_artifacts(tmp_path, grace_days=7.0)
    assert pruned >= 1
    assert not stale.exists()


def test_joint_unit_cache_shared() -> None:
    _JOINT_UNIT_CACHE.clear()

    class _Source:
        def __init__(self, segment_id: str, source_key: str) -> None:
            self.segment_id = segment_id
            self.source_key = source_key

    ordered = [_Source("000000", "a"), _Source("000001", "b")]
    signature = joint_units_signature(ordered)
    assert isinstance(signature, tuple) and len(signature) == 4
    from voyage.augment_finalize import _JOINT_UNIT_CACHE as _cache

    _cache[signature] = ("unit",)
    assert _cache[signature] == ("unit",)
    _cache.clear()


def test_verify_settle_output_truth_empty_run(tmp_path: Path) -> None:
    ok, missing = verify_settle_output_truth(
        tmp_path,
        weights_key="w",
        out_width=64,
        out_height=64,
        source_fps_key=24,
        upscale_factor=1,
        multiplier=2,
        chunk_frames=32,
        crf=15,
        preset="veryfast",
    )
    assert ok is True
    assert missing == []


# --- concepts -------------------------------------------------------------------


def test_concepts_extra_row_validation(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from voyage.concepts import ConceptStore, validate_concepts

    store = ConceptStore(tmp_path)
    store.append("red neon desert", True, vector=[1.0, 0.0])
    assert validate_concepts(tmp_path) == []
    import numpy as _np

    matrix = _np.load(str(tmp_path / "concept_vectors.npy"))
    stacked = _np.concatenate([matrix, matrix[:1]], axis=0)
    _np.save(str(tmp_path / "concept_vectors.npy"), stacked)
    errors = validate_concepts(tmp_path)
    assert any("extra untracked rows" in error for error in errors)


def test_truncate_concept_vectors_to_tracked(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from voyage.concepts import (
        ConceptStore,
        truncate_concept_vectors_to_tracked,
        validate_concepts,
    )

    store = ConceptStore(tmp_path)
    store.append("red neon desert", True, vector=[1.0, 0.0])
    import numpy as _np

    matrix = _np.load(str(tmp_path / "concept_vectors.npy"))
    _np.save(str(tmp_path / "concept_vectors.npy"), _np.concatenate([matrix, matrix[:1]], axis=0))
    assert any("extra untracked" in error for error in validate_concepts(tmp_path))
    removed = truncate_concept_vectors_to_tracked(tmp_path)
    assert removed == 1
    assert validate_concepts(tmp_path) == []

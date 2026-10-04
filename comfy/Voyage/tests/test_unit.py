"""Unit tests: atomic writes, seeds, config, concepts, prompts, RPC framing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voyage import hashing, media, paths
from voyage.atomic import atomic_write_bytes, atomic_write_json, read_json
from voyage.bench import summarize_gauges, timing_stats
from voyage.concepts import ConceptStore, token_set_similarity
from voyage.config import AudioConfig, preset_config
from voyage.errors import MediaError
from voyage.hashing import sha256_file, sha256_text
from voyage.models import PromptStage, WorkerRequest, WorkerResponse
from voyage.paths import (
    MAX_SEGMENT_NUMBER,
    MIN_SEGMENT_NUMBER,
    format_segment_id,
    resolve_stored_path,
    segment_dir,
)
from voyage.persistence import read_effective_config
from voyage.prompts import build_prompt_plan, compose_prompt
from voyage.rpc import decode_request, decode_response, encode_request, encode_response
from voyage.seeds import audio_seed, derive_seed, director_seed, video_seed
from voyage.workers import video_causvid, video_ltxv


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "state.json"
    atomic_write_json(dest, {"a": 1})
    assert read_json(dest) == {"a": 1}
    assert list(tmp_path.glob("*.partial")) == []


def test_atomic_write_replaces_without_partials(tmp_path: Path) -> None:
    dest = tmp_path / "run" / "state.json"
    atomic_write_bytes(dest, b"v1")
    atomic_write_bytes(dest, b"v2")
    assert dest.read_bytes() == b"v2"


def test_derive_seed_stable_and_separated() -> None:
    assert derive_seed(7, "video", 1, 2) == derive_seed(7, "video", 1, 2)
    assert derive_seed(7, "video", 1, 2) != derive_seed(7, "audio", 1, 2)
    assert derive_seed(7, "video", 1, 2) != derive_seed(8, "video", 1, 2)
    assert 0 <= video_seed(1, 2, 3) <= 2**31 - 1
    assert 0 <= audio_seed(1, 2, 3) <= 2**31 - 1
    assert 0 <= director_seed(1, 2) <= 2**31 - 1


def test_config_roundtrip(tmp_path: Path) -> None:
    config = preset_config("demo", "pastel neon line-art, peaceful", 42)
    assert config.name == "demo"
    assert config.seed == 42


def test_config_rejects_empty_style(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        preset_config("demo", "   ", 0)


def test_config_missing_file(tmp_path: Path) -> None:
    from voyage.errors import StateError

    with pytest.raises(StateError):
        read_effective_config(tmp_path / "nope")


def test_config_toml_escapes_style_injection(tmp_path: Path) -> None:
    hostile_style = 'x"\n[video]\nbackend="ltxv'
    hostile_run_id = 'run"x\n[evil]'
    config = preset_config(hostile_run_id, hostile_style, 0)
    assert config.style == hostile_style
    assert config.name == hostile_run_id


def test_config_toml_escapes_quotes_and_newlines(tmp_path: Path) -> None:
    tricky_style = 'neon "city"\nline2\ttab\\backslash\rcarriage'
    config = preset_config("demo", tricky_style, 1)
    assert config.style == tricky_style


def test_audio_config_rejects_non_finite_take_seconds() -> None:
    for hostile_value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            AudioConfig(take_seconds=hostile_value)
        with pytest.raises(ValueError):
            AudioConfig(ahead_seconds=hostile_value)
        with pytest.raises(ValueError):
            AudioConfig(crossfade_seconds=hostile_value)
        with pytest.raises(ValueError):
            AudioConfig(final_overlap_cap_seconds=hostile_value)


def test_audio_config_rejects_negative_take_seconds() -> None:
    with pytest.raises(ValueError):
        AudioConfig(take_seconds=-1.0)
    assert AudioConfig(take_seconds=45.0).take_seconds == 45.0


def test_derive_seed_separates_colon_labels() -> None:
    assert derive_seed(7, "a:b", "c") != derive_seed(7, "a", "b:c")
    assert derive_seed(7, "video", 1, 2) == derive_seed(7, "video", 1, 2)


def test_timing_stats_rejects_empty() -> None:
    with pytest.raises(ValueError):
        timing_stats([])
    single_stats = timing_stats([2.5])
    assert single_stats == {"count": 1, "mean": 2.5, "min": 2.5, "max": 2.5}


def test_summarize_gauges_empty_returns_none() -> None:
    summary = summarize_gauges([])
    assert summary["segments"] == 0
    assert summary["rss_first_mb"] is None
    assert summary["rss_last_mb"] is None
    assert summary["rss_delta_mb"] is None
    assert summary["disk_first_gib"] is None
    assert summary["disk_last_gib"] is None


def test_summarize_gauges_non_empty_stays_numeric() -> None:
    events = [
        {"rss_peak_mb": 100.0, "disk_free_gib": 9.0},
        {"rss_peak_mb": 120.0, "disk_free_gib": 8.5},
    ]
    summary = summarize_gauges(events)
    assert summary["segments"] == 2
    assert summary["rss_first_mb"] == 100.0
    assert summary["rss_last_mb"] == 120.0
    assert summary["rss_delta_mb"] == 20.0


def test_prompt_stage_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        PromptStage(stage=0, block_start=5, block_end=2, prompt="x")
    valid_stage = PromptStage(stage=0, block_start=2, block_end=5, prompt="x")
    assert valid_stage.block_end >= valid_stage.block_start
    equal_stage = PromptStage(stage=1, block_start=3, block_end=3, prompt="y")
    assert equal_stage.block_start == equal_stage.block_end


def test_format_segment_id_range() -> None:
    assert format_segment_id(0) == "000000"
    assert format_segment_id(42) == "000042"
    assert format_segment_id(999999) == "999999"
    with pytest.raises(ValueError):
        format_segment_id(-1)
    with pytest.raises(ValueError):
        format_segment_id(1000000)


def test_novelty_accepts_then_rejects_repeat(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty", similarity_threshold=0.55)
    first, _ = store.propose("luminous fungal metropolis at dusk")
    assert first.accepted
    second, score = store.propose("luminous fungal metropolis at dusk")
    assert not second.accepted
    assert score == 1.0


def test_novelty_history_is_append_only(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty")
    store.propose("crystalline ocean archive")
    lines = (tmp_path / "novelty" / "concepts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "concept-000000"


def test_novelty_records_rejections_immutably(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "novelty", similarity_threshold=0.55)
    store.propose("luminous fungal metropolis at dusk")
    rejected, _ = store.propose("luminous fungal metropolis at dusk")
    assert not rejected.accepted
    assert len(store.records()) == 2
    assert [record.accepted for record in store.records()] == [True, False]


def test_token_similarity_bounds() -> None:
    assert token_set_similarity("", "") == 1.0
    assert token_set_similarity("a b", "") == 0.0
    assert 0.0 <= token_set_similarity("neon city", "neon reef") <= 1.0


def test_prompt_plan_injects_style_everywhere() -> None:
    plan = build_prompt_plan("000001", "ST", "concept", "DRIFT", [0, 3], [2, 5])
    assert len(plan.stages) == 2
    assert all("ST" in stage.prompt for stage in plan.stages)
    assert compose_prompt("s", "c", "DRIFT").startswith("s")


def test_rpc_framing_roundtrip() -> None:
    request = WorkerRequest(id="req-000001", op="health", payload={})
    decoded = decode_request(encode_request(request))
    assert decoded == request
    response = WorkerResponse(id="req-000001", ok=True, result={"a": 1})
    assert decode_response(encode_response(response)) == response


# --- 088 fold: tests/test_hashing.py (7 tests) ---
# """Shared hashing helpers (issue 021).
#
# CPU-only, stdlib only: known vectors, chunked-vs-oneshot equivalence, and
# delegation of the four non-supervisor call sites. `supervisor.py` keeps its
# own copy — it belongs to another track (noted, not touched).
# """


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "weights.bin"
    target.write_bytes(b"voyage-weights-bytes" * 4096)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sha256_file_empty(tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


def test_sha256_file_streams_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-chunk reads hash identically (chunk size is an impl detail)."""
    monkeypatch.setattr(hashing, "CHUNK_SIZE_BYTES", 7)
    target = tmp_path / "chunked.bin"
    target.write_bytes(bytes(range(256)) * 64)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sha256_text_known_vector() -> None:
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_media_alias_delegates(tmp_path: Path) -> None:
    target = tmp_path / "audio.wav"
    target.write_bytes(b"media-bytes")
    assert media._sha256_file(target) == sha256_file(target)


def test_ltxv_sha256_file_delegates(tmp_path: Path) -> None:
    target = tmp_path / "video_tail.mp4"
    target.write_bytes(b"tail-bytes")
    assert video_ltxv.sha256_file(target) == sha256_file(target)


def test_causvid_sha256_helpers_delegate(tmp_path: Path) -> None:
    target = tmp_path / "video_tail.mp4"
    target.write_bytes(b"causvid-tail-bytes")
    assert video_causvid.sha256_file(target) == sha256_file(target)
    assert video_causvid.sha256_text("causvid") == sha256_text("causvid")


# --- 088 fold: tests/test_paths.py (8 tests) ---
# """Direct tests for the run-directory layout helpers (issue 038).
#
# `paths` was imported widely but never asserted: the six-digit id format,
# the segment-dir join, and the relocation-tolerant stored-path resolver
# had zero pins, so a layout regression would surface only as mysterious
# downstream failures.
# """


def test_format_segment_id_zero_pads() -> None:
    assert format_segment_id(0) == "000000"
    assert format_segment_id(1) == "000001"
    assert format_segment_id(42) == "000042"
    assert format_segment_id(MAX_SEGMENT_NUMBER) == "999999"


def test_format_segment_id_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="segment number"):
        format_segment_id(MIN_SEGMENT_NUMBER - 1)
    with pytest.raises(ValueError, match="segment number"):
        format_segment_id(MAX_SEGMENT_NUMBER + 1)


def test_format_segment_id_roundtrip_orders_lexicographically() -> None:
    rendered = [format_segment_id(number) for number in (0, 1, 9, 10, 999999)]
    assert rendered == sorted(rendered)
    assert all(len(segment_id) == 6 for segment_id in rendered)


def test_segment_dir_joins_layout(tmp_path: Path) -> None:
    segment_path = segment_dir(tmp_path, "000001")
    assert segment_path == tmp_path / paths.SEGMENTS_DIRNAME / "000001"


def test_resolve_stored_path_resolves_relative_against_run_dir(tmp_path: Path) -> None:
    resolved = resolve_stored_path(tmp_path, "audio/take_0000.wav")
    assert resolved == tmp_path / "audio" / "take_0000.wav"


def test_resolve_stored_path_keeps_existing_absolute(tmp_path: Path) -> None:
    """Outside-the-run absolutes raise (issue 015) — the old trust is gone."""
    existing = tmp_path / "video.mp4"
    existing.write_bytes(b"media")
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(tmp_path / "elsewhere", existing)


def test_resolve_stored_path_reanchors_moved_run(tmp_path: Path) -> None:
    """A stale absolute entry heals when the layout anchor exists at the new home."""
    run_dir = tmp_path / "run"
    relocated = run_dir / paths.SEGMENTS_DIRNAME / "000000" / "recovery.pt"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(b"tape")
    stale = Path("/old/home/audio") / "x" / "segments" / "000000" / "recovery.pt"
    assert resolve_stored_path(run_dir, stale) == relocated


def test_resolve_stored_path_returns_stale_when_unhealable(tmp_path: Path) -> None:
    """Unhealable outside-the-run absolutes raise (issue 015)."""
    stale = tmp_path / "nowhere" / "recovery.pt"
    with pytest.raises(MediaError, match="escapes the run dir"):
        resolve_stored_path(tmp_path / "run", stale)

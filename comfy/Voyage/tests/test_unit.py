"""Unit tests: atomic writes, seeds, config, concepts, prompts, RPC framing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voyage.atomic import atomic_write_bytes, atomic_write_json, read_json
from voyage.bench import summarize_gauges, timing_stats
from voyage.concepts import ConceptStore, token_set_similarity
from voyage.config import AudioConfig, default_config_toml, load_config
from voyage.errors import ConfigurationError
from voyage.models import PromptStage, WorkerRequest, WorkerResponse
from voyage.paths import format_segment_id
from voyage.prompts import build_prompt_plan, compose_prompt
from voyage.rpc import decode_request, decode_response, encode_request, encode_response
from voyage.seeds import audio_seed, derive_seed, director_seed, video_seed


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
    path = tmp_path / "voyage.toml"
    path.write_text(
        default_config_toml("demo", "pastel neon line-art, peaceful", 42),
        encoding="utf-8",
    )
    config, digest = load_config(path)
    assert config.run_id == "demo"
    assert config.seed == 42
    assert len(digest) == 64


def test_config_rejects_empty_style(tmp_path: Path) -> None:
    path = tmp_path / "voyage.toml"
    path.write_text(default_config_toml("demo", "   ", 0), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "nope.toml")


def test_config_toml_escapes_style_injection(tmp_path: Path) -> None:
    hostile_style = 'x"\n[video]\nbackend="ltxv'
    hostile_run_id = 'run"x\n[evil]'
    path = tmp_path / "voyage.toml"
    path.write_text(
        default_config_toml(hostile_run_id, hostile_style, 0),
        encoding="utf-8",
    )
    config, _digest = load_config(path)
    assert config.style == hostile_style
    assert config.run_id == hostile_run_id


def test_config_toml_escapes_quotes_and_newlines(tmp_path: Path) -> None:
    tricky_style = 'neon "city"\nline2\ttab\\backslash\rcarriage'
    path = tmp_path / "voyage.toml"
    path.write_text(default_config_toml("demo", tricky_style, 1), encoding="utf-8")
    config, _digest = load_config(path)
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

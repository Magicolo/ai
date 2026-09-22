"""Unit tests: atomic writes, seeds, config, concepts, prompts, RPC framing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voyage.atomic import atomic_write_bytes, atomic_write_json, read_json
from voyage.concepts import ConceptStore, token_set_similarity
from voyage.config import default_config_toml, load_config
from voyage.errors import ConfigurationError
from voyage.models import WorkerRequest, WorkerResponse
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


def test_novelty_accepts_then_rejects_repeat(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "concepts.jsonl", similarity_threshold=0.55)
    first, _ = store.propose("luminous fungal metropolis at dusk")
    assert first.accepted
    second, score = store.propose("luminous fungal metropolis at dusk")
    assert not second.accepted
    assert score == 1.0


def test_novelty_history_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "concepts.jsonl"
    store = ConceptStore(path)
    store.propose("crystalline ocean archive")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["index"] == 0


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

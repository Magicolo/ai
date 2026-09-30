"""Director loader staleness + generation param validation (issue 075).

CPU-only: cache comparisons and param checks run before any model
loads, so no Qwen/MiniLM weights are needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from voyage.workers import director as director_worker


def test_validate_max_new_tokens_bounds() -> None:
    for bad_budget in (0, -5, 4097, 1000000):
        with pytest.raises(ValueError, match="max_new_tokens"):
            director_worker.validate_max_new_tokens(bad_budget)
    for good_budget in (1, 256, 1024, 4096):
        director_worker.validate_max_new_tokens(good_budget)


def test_validate_temperature_rejects_non_finite_and_negative() -> None:
    for bad_temperature in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            director_worker.validate_temperature(bad_temperature)
    for good_temperature in (0.0, 0.7, 1.5):
        director_worker.validate_temperature(good_temperature)


def test_qwen_loader_returns_cached_model_for_same_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_model = object()
    sentinel_tokenizer = object()
    monkeypatch.setattr(
        director_worker,
        "_QWEN",
        {
            "model": sentinel_model,
            "tokenizer": sentinel_tokenizer,
            "model_id": "model-a",
            "device": "cuda:1",
        },
    )
    model, tokenizer = director_worker._load_qwen("model-a", device="cuda:1")
    assert model is sentinel_model
    assert tokenizer is sentinel_tokenizer


def test_qwen_loader_reloads_when_model_id_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_model = object()
    monkeypatch.setattr(
        director_worker,
        "_QWEN",
        {
            "model": sentinel_model,
            "tokenizer": object(),
            "model_id": "model-a",
            "device": "cuda:1",
        },
    )
    # Slim has no torch/transformers: the reload attempt must raise (not
    # silently keep serving model-a's weights), and the failed load must
    # not clobber the resident entry.
    with pytest.raises(ImportError):
        director_worker._load_qwen("model-b", device="cuda:1")
    model, _ = director_worker._load_qwen("model-a", device="cuda:1")
    assert model is sentinel_model


def test_embedder_loader_reloads_when_model_id_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_model = object()
    monkeypatch.setattr(
        director_worker, "_EMBEDDER", {"model": sentinel_model, "model_id": "model-a"}
    )
    assert director_worker._load_embedder("model-a") is sentinel_model
    with pytest.raises(ImportError):
        director_worker._load_embedder("model-b")


def test_inspector_loader_reloads_when_model_id_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_model = object()
    monkeypatch.setattr(
        director_worker,
        "_INSPECTOR",
        {"model": sentinel_model, "processor": object(), "model_id": "model-a"},
    )
    model, _ = director_worker._load_inspector("model-a")
    assert model is sentinel_model
    with pytest.raises(ImportError):
        director_worker._load_inspector("model-b")


def test_handle_embed_rejects_empty_and_non_string_texts() -> None:
    with pytest.raises(ValueError, match="non-empty texts"):
        director_worker.handle_embed({"texts": []})
    # SCOPE NOTE 2026-09-25 (issue 007 supervisor track): `checked_request`
    # now isinstance-enforces before any handler code runs, so a non-list
    # `texts` fails fast with TypeError instead of reaching the handler's
    # ValueError. Intent unchanged (reject before embedding); only the
    # contract moved one layer earlier. Worker-track: adjust if desired.
    with pytest.raises(TypeError, match="must be list"):
        director_worker.handle_embed({"texts": "not-a-list"})
    with pytest.raises(ValueError, match="must all be strings"):
        director_worker.handle_embed({"texts": ["amber dunes", 123]})


def test_handle_inspect_rejects_bad_token_budgets() -> None:
    for bad_budget in (0, -3, 5000):
        with pytest.raises(ValueError, match="max_new_tokens"):
            director_worker.handle_inspect(
                {"frame_path": "/tmp/frame.png", "max_new_tokens": bad_budget}
            )


def test_qwen_decide_rejects_bad_generation_params() -> None:
    base: dict[str, Any] = {"decision_index": 0}
    with pytest.raises(ValueError, match="temperature"):
        director_worker._qwen_decide({**base, "temperature": float("nan")})
    with pytest.raises(ValueError, match="temperature"):
        director_worker._qwen_decide({**base, "temperature": -1.0})
    with pytest.raises(ValueError, match="max_new_tokens"):
        director_worker._qwen_decide({**base, "max_new_tokens": 0})
    with pytest.raises(ValueError, match="max_new_tokens"):
        director_worker._qwen_decide({**base, "max_new_tokens": 1000000})

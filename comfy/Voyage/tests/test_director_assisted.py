"""Director assisted-generation upgrades: PLD, constrained JSON, llama sidecar.

Covers the three client-side `_qwen_generate` upgrades (DESIGN §§8-9 Qwen
decider, §51 retry chain): prompt-lookup decoding (option 2), the
Outlines-or-coercion constrained-JSON path (option 4a, interim coercion
until `outlines` lands in the director image — probed absent in the slim
container 2026-10-01), and the llama-server HTTP branch (option 3 client)
against the exact sidecar contract. All CPU-only: model loading and the
HTTP transport are stubbed, no GPU, no network, no model weights.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from typing import Any

import pytest

from voyage.director import DIRECTOR_SYSTEM_PROMPT
from voyage.models import EvolutionDecision
from voyage.workers import director as director_worker

LLAMA_TEST_ENDPOINT = "http://127.0.0.1:8080"
"""Sidecar base URL used across the llama-branch tests (never contacted)."""

EXPECTED_COMPLETIONS_PATH = "/v1/chat/completions"
"""OpenAI-compatible chat endpoint path the sidecar serves (contract)."""

EXPECTED_SCHEMA_NAME = "evolution_decision"
"""`response_format` schema name the sidecar track pinned (contract)."""

PROMPT_TOKEN_COUNT = 5
"""Fake prompt width: input_ids shape (1, 5), so completions are countable."""

GENERATED_TOKEN_TOTAL = 8
"""Fake output width: output[0] holds 8 tokens, 3 survive the prompt slice."""


def _canned_decision_json() -> str:
    """Minimal schema-plausible decision body (index added by `_qwen_decide`)."""
    return json.dumps(
        {
            "destination": {"canonical_name": "assisted valley"},
            "video": {"stages": ["a calm neon valley holds"]},
        }
    )


class _FakeInputIds:
    """Minimal `input_ids` stand-in: shape for counting, `.to` for CUDA moves."""

    def __init__(self) -> None:
        self.shape = (1, PROMPT_TOKEN_COUNT)

    def to(self, target: str) -> _FakeInputIds:
        del target
        return self


class _FakeGeneratedRow:
    """Output row supporting `output[0][prompt_tokens:]` slicing with `.shape`."""

    def __init__(self, total_tokens: int) -> None:
        self._total_tokens = total_tokens
        self.shape = (total_tokens,)

    def __getitem__(self, item: Any) -> _FakeGeneratedRow:
        if isinstance(item, slice):
            start = item.start or 0
            return _FakeGeneratedRow(max(0, self._total_tokens - start))
        raise TypeError(f"fake row supports slices only (got {item!r})")


class _FakeModel:
    """Resident-model stand-in recording every `generate` call's kwargs."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls
        self.device = "cpu"

    def generate(self, **kwargs: Any) -> list[_FakeGeneratedRow]:
        self._calls.append(dict(kwargs))
        return [_FakeGeneratedRow(GENERATED_TOKEN_TOTAL)]


class _PldRejectingModel(_FakeModel):
    """First `generate` with the PLD kwarg raises, the retry without succeeds."""

    def generate(self, **kwargs: Any) -> list[_FakeGeneratedRow]:
        self._calls.append(dict(kwargs))
        if "prompt_lookup_num_tokens" in kwargs:
            raise TypeError(
                "generate() got an unexpected keyword argument 'prompt_lookup_num_tokens'"
            )
        return [_FakeGeneratedRow(GENERATED_TOKEN_TOTAL)]


class _FakeTokenizer:
    """Tokenizer stand-in: fixed prompt width, canned decode text."""

    def __init__(self, decoded_text: str) -> None:
        self.eos_token_id = 0
        self._decoded_text = decoded_text

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        del messages, kwargs
        return "fake-chat-prompt"

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        del prompt, kwargs
        return {"input_ids": _FakeInputIds()}

    def decode(self, generated: Any, **kwargs: Any) -> str:
        del generated, kwargs
        return self._decoded_text


def _install_qwen_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decoded_text: str,
    model: _FakeModel | None = None,
) -> list[dict[str, Any]]:
    """Stub the torch import, loader, and resident pair; return generate calls."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(director_worker, "_require_module", lambda _name: None)
    torch_stub = types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        inference_mode=lambda: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    resident_model: _FakeModel = model if model is not None else _FakeModel(calls)
    if model is not None:
        resident_model._calls = calls  # test seam shares the call log
    tokenizer = _FakeTokenizer(decoded_text)

    def _fake_load(model_id: str, *, device: str) -> tuple[Any, Any]:
        del model_id, device
        return resident_model, tokenizer

    monkeypatch.setattr(director_worker, "_load_qwen", _fake_load)
    return calls


def _decide_payload(**overrides: Any) -> dict[str, Any]:
    """Minimal `_qwen_decide` payload with a working deterministic fallback."""
    payload: dict[str, Any] = {
        "decision_index": 0,
        "current_world": "assisted valley",
        "destination_concept": "assisted valley",
        "style_charter": "pastel neon line-art",
    }
    payload.update(overrides)
    return payload


def test_pld_kwarg_present_in_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Option 2: the AWQ `generate` call carries `prompt_lookup_num_tokens=10`."""
    calls = _install_qwen_fakes(monkeypatch, decoded_text=_canned_decision_json())
    text, prompt_tokens, completion_tokens = director_worker._qwen_generate(
        "model-under-test",
        "describe a valley",
        0.7,
        64,
        False,
        device="cpu",
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"
    assert prompt_tokens == PROMPT_TOKEN_COUNT
    assert completion_tokens == GENERATED_TOKEN_TOTAL - PROMPT_TOKEN_COUNT
    assert calls[0]["prompt_lookup_num_tokens"] == 10
    assert calls[0]["max_new_tokens"] == 64
    assert calls[0]["temperature"] == 0.7


def test_pld_rejection_falls_back_without_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stack without PLD support degrades to a plain generate (same contract)."""
    calls = _install_qwen_fakes(
        monkeypatch,
        decoded_text=_canned_decision_json(),
        model=_PldRejectingModel([]),
    )
    text, prompt_tokens, completion_tokens = director_worker._qwen_generate(
        "model-under-test",
        "describe a valley",
        0.7,
        64,
        False,
        device="cpu",
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"
    assert (prompt_tokens, completion_tokens) == (
        PROMPT_TOKEN_COUNT,
        GENERATED_TOKEN_TOTAL - PROMPT_TOKEN_COUNT,
    )
    assert len(calls) == 2
    assert calls[0]["prompt_lookup_num_tokens"] == 10
    assert "prompt_lookup_num_tokens" not in calls[1]


def test_coerce_decision_json_returns_canonical_first_try() -> None:
    """Option 4a interim: fenced/chatty output coerces to parseable JSON."""
    fenced = "Here is your decision:\n```json\n" + _canned_decision_json() + "\n```"
    coerced = director_worker._coerce_decision_json(fenced)
    assert json.loads(coerced)["destination"]["canonical_name"] == "assisted valley"


def test_coerce_decision_json_rejects_garbage() -> None:
    """Unparseable output raises ValueError into the §51 retry chain."""
    with pytest.raises(ValueError, match="no JSON object"):
        director_worker._coerce_decision_json("no object here, just prose")


def test_qwen_generate_coerces_chatty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The AWQ path returns clean JSON first-try on a canned constrained output."""
    fenced = "```json\n" + _canned_decision_json() + "\n```"
    _install_qwen_fakes(monkeypatch, decoded_text=fenced)
    text, _prompt_tokens, _completion_tokens = director_worker._qwen_generate(
        "model-under-test",
        "describe a valley",
        0.7,
        64,
        False,
        device="cpu",
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"


def _install_llama_fake(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any] | BaseException,
) -> dict[str, Any]:
    """Stub the module-level HTTP seam; return the captured url + body."""
    posted: dict[str, Any] = {}

    def _fake_post(request_url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        posted["url"] = request_url
        posted["body"] = request_body
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(director_worker, "_post_llama_chat", _fake_post)
    return posted


def _llama_success_response() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": _canned_decision_json()}}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 23},
    }


def test_llama_branch_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Option 3 client: endpoint URL, schema name, thinking-off, passthrough."""
    posted = _install_llama_fake(monkeypatch, _llama_success_response())
    text, prompt_tokens, completion_tokens = director_worker._qwen_generate(
        "Qwen/Qwen3-8B",
        "describe a valley",
        0.5,
        128,
        False,
        device="cuda:1",
        llama_endpoint=LLAMA_TEST_ENDPOINT,
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"
    assert (prompt_tokens, completion_tokens) == (17, 23)
    assert posted["url"] == LLAMA_TEST_ENDPOINT + EXPECTED_COMPLETIONS_PATH
    body = posted["body"]
    assert body["messages"][0] == {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}
    assert body["messages"][1] == {"role": "user", "content": "describe a valley"}
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 128
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == EXPECTED_SCHEMA_NAME
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == EvolutionDecision.model_json_schema()


def test_llama_branch_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing-slash endpoint still builds one clean completions URL."""
    posted = _install_llama_fake(monkeypatch, _llama_success_response())
    director_worker._qwen_generate(
        "Qwen/Qwen3-8B",
        "describe a valley",
        0.5,
        128,
        False,
        device="cuda:1",
        llama_endpoint=LLAMA_TEST_ENDPOINT + "/",
    )
    assert posted["url"] == LLAMA_TEST_ENDPOINT + EXPECTED_COMPLETIONS_PATH


def test_llama_branch_missing_usage_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent `usage` keeps the (text, tokens) contract with 0 counts."""
    posted = _install_llama_fake(
        monkeypatch,
        {"choices": [{"message": {"content": _canned_decision_json()}}]},
    )
    text, prompt_tokens, completion_tokens = director_worker._qwen_generate(
        "Qwen/Qwen3-8B",
        "describe a valley",
        0.5,
        128,
        False,
        device="cuda:1",
        llama_endpoint=LLAMA_TEST_ENDPOINT,
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"
    assert (prompt_tokens, completion_tokens) == (0, 0)
    assert posted["url"] == LLAMA_TEST_ENDPOINT + EXPECTED_COMPLETIONS_PATH


def test_llama_branch_extracts_json_belt_and_braces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chatty sidecar content still parses via the existing `_extract_json`."""
    chatty = "Sure! " + _canned_decision_json() + " Hope this helps."
    _install_llama_fake(
        monkeypatch,
        {
            "choices": [{"message": {"content": chatty}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    )
    text, _prompt_tokens, _completion_tokens = director_worker._qwen_generate(
        "Qwen/Qwen3-8B",
        "describe a valley",
        0.5,
        128,
        False,
        device="cuda:1",
        llama_endpoint=LLAMA_TEST_ENDPOINT,
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"


def test_llama_connection_failure_uses_existing_retry_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused connection degrades to the deterministic fallback (§51)."""
    _install_llama_fake(monkeypatch, ConnectionError("connection refused"))
    dumped = director_worker._qwen_decide(_decide_payload(llama_endpoint=LLAMA_TEST_ENDPOINT))
    assert dumped["fallback"] is True
    assert "assisted valley" in dumped["destination"]["canonical_name"]


def test_llama_missing_content_uses_existing_retry_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty sidecar content retries then falls back — never raises outward."""
    _install_llama_fake(
        monkeypatch,
        {"choices": [{"message": {"content": ""}}], "usage": {}},
    )
    dumped = director_worker._qwen_decide(_decide_payload(llama_endpoint=LLAMA_TEST_ENDPOINT))
    assert dumped["fallback"] is True


def test_llama_connection_error_is_catchable_before_decide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The branch raises stdlib exceptions §51 already catches (no new types)."""
    _install_llama_fake(monkeypatch, ConnectionError("connection refused"))
    with pytest.raises(ConnectionError, match="connection refused"):
        director_worker._qwen_generate(
            "Qwen/Qwen3-8B",
            "describe a valley",
            0.5,
            128,
            False,
            device="cuda:1",
            llama_endpoint=LLAMA_TEST_ENDPOINT,
        )


def test_awq_path_untouched_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """No endpoint configured means the current AWQ path runs unchanged."""
    calls = _install_qwen_fakes(monkeypatch, decoded_text=_canned_decision_json())

    def _boom(request_url: str, request_body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("HTTP seam must stay unused without an endpoint")

    monkeypatch.setattr(director_worker, "_post_llama_chat", _boom)
    text, _prompt_tokens, _completion_tokens = director_worker._qwen_generate(
        "model-under-test",
        "describe a valley",
        0.7,
        64,
        False,
        device="cpu",
    )
    assert json.loads(text)["destination"]["canonical_name"] == "assisted valley"
    assert len(calls) == 1


def test_qwen_decide_accepts_llama_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid sidecar reply accepts with usage counts, like the AWQ path."""
    _install_llama_fake(monkeypatch, _llama_success_response())
    dumped = director_worker._qwen_decide(_decide_payload(llama_endpoint=LLAMA_TEST_ENDPOINT))
    assert dumped["fallback"] is False
    assert dumped["prompt_tokens"] == 17
    assert dumped["completion_tokens"] == 23
    assert dumped["destination"]["canonical_name"] == "assisted valley"


def test_handle_init_records_llama_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """`init` stores the sidecar URL; None clears it back to the AWQ default."""
    monkeypatch.setattr(director_worker, "_CONFIG", {"backend": "qwen"})
    director_worker.handle_init({"llama_endpoint": LLAMA_TEST_ENDPOINT})
    assert director_worker._CONFIG["llama_endpoint"] == LLAMA_TEST_ENDPOINT
    director_worker.handle_init({"llama_endpoint": None})
    assert "llama_endpoint" not in director_worker._CONFIG
    with pytest.raises(TypeError, match="llama_endpoint"):
        director_worker.handle_init({"llama_endpoint": 123})


def test_qwen_decide_forwards_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored `init` endpoint reaches `_qwen_generate` without payload help."""
    seen: dict[str, Any] = {}

    def _recording_generate(*args: Any, **kwargs: Any) -> tuple[str, int, int]:
        seen["kwargs"] = dict(kwargs)
        return _canned_decision_json(), 1, 2

    monkeypatch.setattr(director_worker, "_qwen_generate", _recording_generate)
    monkeypatch.setattr(
        director_worker,
        "_CONFIG",
        {"backend": "qwen", "llama_endpoint": LLAMA_TEST_ENDPOINT},
    )
    dumped = director_worker._qwen_decide(_decide_payload())
    assert dumped["fallback"] is False
    assert seen["kwargs"]["llama_endpoint"] == LLAMA_TEST_ENDPOINT

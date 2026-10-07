"""Issue 263: narrow fail-soft with distinct parse-vs-transport accounting.

`enhance` catches only sidecar-shaped failures
(RuntimeError/OSError for transport, ValueError/TypeError for parse);
programming errors propagate loud. `enhance_many` reports
transport_failures vs parse_failures separately.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from voyage import prompt_enhancer


class _FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


def _install_httpx_fake(monkeypatch: pytest.MonkeyPatch, behavior: Any) -> None:
    def _fake_post(url: str, json: Any = None, timeout: Any = None) -> Any:
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=_fake_post))


def _completion(content: str | None) -> _FakeResponse:
    message: dict[str, Any] = {} if content is None else {"content": content}
    return _FakeResponse(
        200,
        {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def test_transport_vs_parse_kinds_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-200/ConnectionError read as transport; choiceless as parse (263)."""
    _install_httpx_fake(monkeypatch, _FakeResponse(500, {"error": "busy"}))
    _, _, kind = prompt_enhancer.enhance_with_failure_kind("hello")
    assert kind == "transport"
    _install_httpx_fake(monkeypatch, ConnectionError("refused"))
    _, _, kind = prompt_enhancer.enhance_with_failure_kind("hello")
    assert kind == "transport"
    _install_httpx_fake(monkeypatch, _FakeResponse(200, {"choices": []}))
    _, _, kind = prompt_enhancer.enhance_with_failure_kind("hello")
    assert kind == "parse"
    _install_httpx_fake(monkeypatch, _completion(None))
    _, _, kind = prompt_enhancer.enhance_with_failure_kind("hello")
    assert kind == "parse"


def test_programming_errors_propagate_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """AttributeError/KeyError from the contract are not swallowed (263)."""
    monkeypatch.setattr(
        prompt_enhancer,
        "_parse_expansion",
        lambda _parsed, _url: (_ for _ in ()).throw(AttributeError("bug")),
    )
    _install_httpx_fake(monkeypatch, _completion("expanded"))
    with pytest.raises(AttributeError):
        prompt_enhancer.enhance("hello")
    with pytest.raises(AttributeError):
        prompt_enhancer.enhance_with_failure_kind("hello")


def test_enhance_many_counts_transport_vs_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary carries separate transport/parse failure counters (263)."""
    calls: list[str] = []
    original = prompt_enhancer.enhance_with_failure_kind

    def _sequenced(
        text: str, system_prompt: str = "", endpoint: str = "", timeout: float = 0.0
    ) -> tuple[str, dict[str, int], str | None]:
        calls.append(text)
        if len(calls) == 1:
            return text, {"prompt_tokens": 0, "completion_tokens": 0}, "transport"
        return text, {"prompt_tokens": 0, "completion_tokens": 0}, "parse"

    monkeypatch.setattr(prompt_enhancer, "enhance_with_failure_kind", _sequenced)
    _, summary = prompt_enhancer.enhance_many(["a", "b"], enabled=True, backend="ltx25")
    assert summary["transport_failures"] == 1
    assert summary["parse_failures"] == 1
    assert summary["prompts_changed"] == 0
    monkeypatch.setattr(prompt_enhancer, "enhance_with_failure_kind", original)


def test_enhance_still_fails_soft_on_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expected transport/parse failures still degrade to input text (263)."""
    for behavior in (
        _FakeResponse(500, {"error": "busy"}),
        ConnectionError("down"),
        _FakeResponse(200, {"choices": []}),
        _completion("   "),
    ):
        _install_httpx_fake(monkeypatch, behavior)
        expanded, counts = prompt_enhancer.enhance("hello")
        assert expanded == "hello"
        assert counts == {"prompt_tokens": 0, "completion_tokens": 0}

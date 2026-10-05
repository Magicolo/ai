"""Track B Qwen enhancer prototype (DESIGN §140): opt-in, default OFF.

CPU-only: the sidecar HTTP transport is stubbed (fake `httpx` in
`sys.modules` — no live server, no network), the config/CLI layers are
pure, and `configure` runs with a stubbed model ensure. Pins the four
contract points: off-by-default (existing manifests/payloads untouched),
the free-form body (no `response_format`), fail-soft on transport
failure, and real expansion when the knob is on.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from voyage import prompt_enhancer
from voyage.config import VideoConfig, preset_config, resolve_config

ORIGINAL_PROMPT = "a quiet neon harbor at dusk"
EXPANDED_PROMPT = (
    "a quiet neon harbor at dusk, slow push-in over glassy water, "
    "pastel signs glowing through mist, continuous single shot"
)


class _FakeResponse:
    """Minimal httpx response stand-in: status + JSON body."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


def _install_httpx_fake(
    monkeypatch: pytest.MonkeyPatch,
    behavior: Any,
) -> dict[str, Any]:
    """Serve `behavior` from `httpx.post`; return the captured call."""
    posted: dict[str, Any] = {}

    def _fake_post(url: str, json: Any = None, timeout: Any = None) -> Any:
        posted["url"] = url
        posted["body"] = json
        posted["timeout"] = timeout
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=_fake_post))
    return posted


def _completion_response(
    content: str | None,
    prompt_tokens: int = 7,
    completion_tokens: int = 11,
    status_code: int = 200,
) -> _FakeResponse:
    """Canned sidecar reply carrying `content` plus a usage block."""
    message: dict[str, Any] = {} if content is None else {"content": content}
    return _FakeResponse(
        status_code,
        {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def test_prompt_enhance_defaults_on() -> None:
    """The knob defaults True: fresh runs expand prompts (2026-10-05).

    Seams stay invisible with expansion and adherence is no worse, so
    the default flipped from opt-in to on; `--no-prompt-enhance` opts
    out per run. Old manifests keep their stored value on updates.
    """
    assert VideoConfig().prompt_enhance is True
    assert preset_config("probe", "pastel neon line-art, peaceful", 7).video.prompt_enhance is True
    base = preset_config("probe", "pastel neon line-art, peaceful", 7)
    before = base.model_dump()
    assert resolve_config(base).video.prompt_enhance is True
    assert resolve_config(base).model_dump() == before


def test_prompt_enhance_resolves_through_config() -> None:
    """Explicit opt-out survives resolution; backend presets never flip it."""
    base = preset_config("probe", "pastel neon line-art, peaceful", 7)
    assert resolve_config(base, prompt_enhance=True).video.prompt_enhance is True
    assert resolve_config(base, prompt_enhance=False).video.prompt_enhance is False
    assert resolve_config(base, backend="ltx25").video.prompt_enhance is True


def test_enhancer_body_is_free_form() -> None:
    """The expansion body carries NO response_format (director JSON path untouched)."""
    body = prompt_enhancer.build_enhancer_body(ORIGINAL_PROMPT)
    assert "response_format" not in body
    assert body["messages"] == [
        {"role": "system", "content": prompt_enhancer.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": ORIGINAL_PROMPT},
    ]
    assert body["max_tokens"] == prompt_enhancer.DEFAULT_MAX_TOKENS


def test_enhance_expands_prompt_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy sidecar returns the expanded text plus its token counts."""
    posted = _install_httpx_fake(monkeypatch, _completion_response(EXPANDED_PROMPT))
    expanded, tokens = prompt_enhancer.enhance(ORIGINAL_PROMPT)
    assert expanded == EXPANDED_PROMPT
    assert tokens == {"prompt_tokens": 7, "completion_tokens": 11}
    assert posted["url"].endswith(prompt_enhancer.COMPLETIONS_PATH)
    assert "response_format" not in posted["body"]


def test_enhance_fails_soft_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server absence degrades to the input text (the commit never fails)."""
    _install_httpx_fake(monkeypatch, ConnectionError("refused"))
    expanded, tokens = prompt_enhancer.enhance(ORIGINAL_PROMPT)
    assert expanded == ORIGINAL_PROMPT
    assert tokens == {"prompt_tokens": 0, "completion_tokens": 0}


def test_enhance_fails_soft_on_bad_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-200, choiceless, contentless, and empty replies all degrade."""
    bad_replies = [
        _FakeResponse(500, {"error": "busy"}),
        _FakeResponse(200, {"choices": []}),
        _completion_response(None),
        _completion_response("   "),
    ]
    for reply in bad_replies:
        _install_httpx_fake(monkeypatch, reply)
        expanded, tokens = prompt_enhancer.enhance(ORIGINAL_PROMPT)
        assert expanded == ORIGINAL_PROMPT
        assert tokens == {"prompt_tokens": 0, "completion_tokens": 0}


def test_enhance_skips_http_for_blank_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank input returns as-is without touching the transport."""
    posted = _install_httpx_fake(monkeypatch, _completion_response(EXPANDED_PROMPT))
    expanded, tokens = prompt_enhancer.enhance("   ")
    assert expanded == "   "
    assert tokens == {"prompt_tokens": 0, "completion_tokens": 0}
    assert posted == {}


def test_enhance_for_backend_gates_on_knob_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off — or on a non-LTX backend — returns the inputs untouched."""
    prompts = [ORIGINAL_PROMPT, "a second stage"]
    assert prompt_enhancer.enhance_for_backend(prompts, enabled=False, backend="ltx25") == prompts
    assert prompt_enhancer.enhance_for_backend(prompts, enabled=True, backend="fake") == prompts
    assert prompt_enhancer.enhance_for_backend(prompts, enabled=True, backend="ltxv") == prompts


def test_enhance_for_backend_expands_ltx_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """On + ltx25/ltx23 maps the expansion over every staged prompt."""
    _install_httpx_fake(monkeypatch, _completion_response(EXPANDED_PROMPT))
    for backend in ("ltx25", "ltx23"):
        assert prompt_enhancer.enhance_for_backend(
            [ORIGINAL_PROMPT, ORIGINAL_PROMPT], enabled=True, backend=backend
        ) == [EXPANDED_PROMPT, EXPANDED_PROMPT]


def test_prompt_enhance_overrides_mapping() -> None:
    """CLI flags map to resolver kwargs; absent maps to nothing."""
    from voyage.cli_core import _prompt_enhance_overrides

    assert _prompt_enhance_overrides(argparse.Namespace()) == {}
    assert _prompt_enhance_overrides(argparse.Namespace(prompt_enhance=None)) == {}
    assert _prompt_enhance_overrides(argparse.Namespace(prompt_enhance=True)) == {
        "prompt_enhance": True
    }
    assert _prompt_enhance_overrides(argparse.Namespace(no_prompt_enhance=True)) == {
        "prompt_enhance": False
    }
    with pytest.raises(ValueError, match="only one of"):
        _prompt_enhance_overrides(argparse.Namespace(prompt_enhance=True, no_prompt_enhance=True))


def _configure_namespace(name: str, **overrides: object) -> argparse.Namespace:
    """Minimal `configure` namespace (mirrors tests/test_configure.py)."""
    base: dict[str, object] = {
        "name": name,
        "backend": None,
        "from_run": None,
        "duration": None,
        "segments": None,
        "style": None,
        "seed": None,
        "final_video": None,
        "skip_bad": False,
        "no_download": True,
        "force": False,
        "director": None,
        "director_device": None,
        "blocks": None,
        "take_seconds": None,
        "quantization": None,
        "beats_per_segment": None,
        "drift_every_n": None,
        "music_caption": None,
        "video_caption": None,
        "upscale": None,
        "interpolate": None,
        "presentation_fps": None,
        "no_sfx": False,
        "sfx_backend": None,
        "sfx_caption": None,
        "sfx_device": None,
        "sfx_model_size": None,
        "sfx_workers": 1,
        "verbose": False,
        "no_color": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _never_touch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voyage.models_ensure.ensure_models", lambda *a, **k: 0)


def test_configure_stores_prompt_enhance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--prompt-enhance` persists into the manifest (generate honors it)."""
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        "enhanced", style="dark harbors", segments=2, seed=7, prompt_enhance=True
    )
    assert cli_configure.cmd_configure(args) == 0
    manifest = json.loads((tmp_path / "output" / "enhanced" / "manifest.json").read_text())
    assert manifest["video"]["prompt_enhance"] is True


def test_configure_defaults_prompt_enhance_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No flag means the manifest carries True (fresh runs expand)."""
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace("plain", style="dark harbors", segments=2, seed=7)
    assert cli_configure.cmd_configure(args) == 0
    manifest = json.loads((tmp_path / "output" / "plain" / "manifest.json").read_text())
    assert manifest["video"]["prompt_enhance"] is True


def test_configure_update_keeps_and_clears_prompt_enhance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Updates touch the knob only when flagged (inherit, then explicit off)."""
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    assert (
        cli_configure.cmd_configure(
            _configure_namespace("toggle", style="s", segments=2, seed=1, prompt_enhance=True)
        )
        == 0
    )
    assert cli_configure.cmd_configure(_configure_namespace("toggle", style="t")) == 0
    manifest = json.loads((tmp_path / "output" / "toggle" / "manifest.json").read_text())
    assert manifest["video"]["prompt_enhance"] is True
    assert manifest["style"] == "t"
    assert cli_configure.cmd_configure(_configure_namespace("toggle", no_prompt_enhance=True)) == 0
    manifest = json.loads((tmp_path / "output" / "toggle" / "manifest.json").read_text())
    assert manifest["video"]["prompt_enhance"] is False


def test_configure_refuses_both_enhance_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--prompt-enhance` + `--no-prompt-enhance` exits 2 without writing."""
    from voyage import cli_configure

    monkeypatch.chdir(tmp_path)
    args = _configure_namespace(
        "clash",
        style="s",
        segments=2,
        seed=1,
        prompt_enhance=True,
        no_prompt_enhance=True,
    )
    assert cli_configure.cmd_configure(args) == 2
    assert not (tmp_path / "output" / "clash").exists()


def test_enhance_many_reports_engagement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Track C observable: counts + changed tally prove sidecar engagement."""
    posted = _install_httpx_fake(monkeypatch, _completion_response(EXPANDED_PROMPT, 7, 11))
    expanded, summary = prompt_enhancer.enhance_many(
        [ORIGINAL_PROMPT, ORIGINAL_PROMPT],
        enabled=True,
        backend="ltx25",
    )
    assert expanded == [EXPANDED_PROMPT, EXPANDED_PROMPT]
    assert summary["prompts_total"] == 2
    assert summary["prompts_changed"] == 2
    assert summary["expanded_chars"] == 2 * len(EXPANDED_PROMPT)
    assert summary["prompt_tokens"] == 14
    assert summary["completion_tokens"] == 22
    assert posted["url"].endswith("/v1/chat/completions")
    assert "response_format" not in posted["body"]


def test_enhance_many_off_returns_zeroed_summary() -> None:
    """Knob off: texts untouched and the summary proves no HTTP happened."""
    expanded, summary = prompt_enhancer.enhance_many(
        [ORIGINAL_PROMPT], enabled=False, backend="ltx25"
    )
    assert expanded == [ORIGINAL_PROMPT]
    assert summary == {
        "prompts_total": 1,
        "prompts_changed": 0,
        "expanded_chars": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def test_enhance_many_fail_soft_counts_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dead server: input text kept, zero tokens, changed tally stays 0."""
    _install_httpx_fake(monkeypatch, ConnectionError("sidecar down"))
    expanded, summary = prompt_enhancer.enhance_many(
        [ORIGINAL_PROMPT], enabled=True, backend="ltx25"
    )
    assert expanded == [ORIGINAL_PROMPT]
    assert summary["prompts_changed"] == 0
    assert summary["prompt_tokens"] == 0
    assert summary["completion_tokens"] == 0

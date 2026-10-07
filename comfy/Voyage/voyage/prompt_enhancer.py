"""Qwen prompt-enhancer prototype (DESIGN §140 LTX track, §§8-9 decider).

Track B prototype: an opt-in, additive prompt-expansion stage in front of
the LTX video path. The LTX-2.5 reference renders through a separate
prompt-enhancer LLM (gemma-4-E2B-it plus its T2V system prompt); Voyage
has no enhancer stage today — `video_ltx25.py` sends the director's stage
text straight into `CLIPTextEncode` — so this module reuses the
already-running llama-server sidecar (Qwen3.5-4B-Q4_K_M on 127.0.0.1:8080,
the same OpenAI-compatible `/v1/chat/completions` the director's
EvolutionDecision JSON path uses) for free-form expansion.

Default ON (`VideoConfig.prompt_enhance = True`): on ltx25/ltx23 the
supervisor expands the staged prompts through the sidecar (see below);
on any backend outside `ENHANCER_BACKENDS`, `enhance_many` returns the
input prompts untouched with a zeroed summary, so non-LTX payloads stay
byte-identical. When the sidecar is unreachable (e.g. deterministic
director runs, where it never starts), expansion degrades to the input
text with zero token counts and the per-segment `prompt_enhanced`
metric event records the non-engagement instead of failing the commit.
The supervisor calls it at the top of `_render_video` — in the main
commit thread, BEFORE the `generate_blocks` worker RPC — so enhancement
never overlaps the video forward (cuda:0 peak 14.6 GiB) or any ACE
residency (all-deferred audio commits video-only; the sidecar itself
holds ~5 GiB on cuda:1). The director JSON path is untouched: this body
carries NO `response_format` (free-form text out).

No GPU here by construction (§12 hard ban): stdlib plus a lazy `httpx`
import only — `torch`/`transformers` must never appear in this module.
"""

from __future__ import annotations

import importlib
from typing import Any

COMPLETIONS_PATH = "/v1/chat/completions"
"""OpenAI-compatible chat path the sidecar serves (same contract as the director)."""

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
"""Sidecar base URL (mirrors `config.DEFAULT_LLAMA_ENDPOINT` as a literal:
this module stays stdlib-only, so it cannot import `voyage.config`;
the supervisor passes `config.director.llama_endpoint` at the call site)."""

DEFAULT_TIMEOUT_SECONDS = 60.0
"""One enhancement call budget (same class as the director's sidecar timeout)."""

DEFAULT_MAX_TOKENS = 512
"""Expansion cap: an enriched stage paragraph, not a second screenplay."""

DEFAULT_TEMPERATURE = 0.7
"""Sampling temperature for the expansion (mirrors the director default)."""

ENHANCER_BACKENDS = frozenset({"ltx25", "ltx23"})
"""Backends the prototype applies to (ltx23 shares the ltx25 prompt path,
so parity is one set member; every other backend returns inputs untouched)."""

DEFAULT_SYSTEM_PROMPT = """\
You are a text-to-video prompt expander. Expand the user's short scene
description into one rich, concrete, continuous single-shot video prompt:
physical cinematography (one unbroken camera move, no cuts, no scene
change), tangible subjects, materials, light, palette, atmosphere, and
gentle continuous motion. Reply with the expanded prompt text only —
no JSON, no quotes, no preamble, no explanation.\
"""
"""Free-form T2V expansion doctrine (prototype analogue of the reference
enhancer system prompt; the director's JSON contract never applies here)."""


def build_enhancer_body(
    text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """Build the free-form sidecar chat body (pure; no `response_format`).

    Same message shape the director sends (system doctrine + user text),
    minus the `response_format` json_schema the EvolutionDecision path
    pins — the enhancer returns prose, so constraining it to a schema
    would reject every valid expansion.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _as_count(value: object) -> int:
    """Coerce a sidecar `usage` count to a safe int (pure; garbage is 0)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value == value and value >= 0:
        return int(value)
    return 0


def _post_chat(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST one chat completion to the sidecar (module-level test seam).

    `httpx` imports lazily (the slim gate image has no httpx; a
    module-scope import would break every slim import of this module).
    Non-200 and misshapen replies raise; transport errors propagate as
    `RuntimeError`/`OSError` so `enhance` can stay narrow (issue 263)
    without importing httpx at module scope. A missing httpx install
    reads as transport-down (`RuntimeError`), never as an import crash.
    """
    try:
        httpx_client: Any = importlib.import_module("httpx")
    except ImportError as exc:
        raise RuntimeError(f"enhancer httpx unavailable for {url}: {exc}") from exc
    candidate = getattr(httpx_client, "HTTPError", None)
    http_error_type: type[BaseException] | None = None
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        http_error_type = candidate
    if http_error_type is not None:
        try:
            response: Any = httpx_client.post(url, json=body, timeout=timeout)
        except BaseException as exc:
            if isinstance(exc, http_error_type):
                raise RuntimeError(  # noqa: TRY004 - transport failure, not a type error
                    f"enhancer transport failed for {url}: {exc}"
                ) from exc
            raise
    else:
        response = httpx_client.post(url, json=body, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"enhancer {url} answered status {response.status_code}")
    parsed: Any = response.json()
    if not isinstance(parsed, dict):
        raise TypeError(f"enhancer {url} answered a non-object body")
    return parsed


def _parse_expansion(parsed: dict[str, Any], url: str) -> tuple[str, dict[str, int]]:
    """Pull `(expanded_text, tokens)` out of a sidecar reply (pure).

    Raises ValueError on choiceless/contentless/empty replies so the
    fail-soft `except` in `enhance` degrades to the input text. Lives in
    its own function (not inline in the `try`) so the raises are never
    mistaken for transport errors by the linter or the reader.
    """
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"enhancer {url} answered without choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"enhancer {url} answered without message content")
    expanded = content.strip()
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return expanded, {"prompt_tokens": 0, "completion_tokens": 0}
    return expanded, {
        "prompt_tokens": _as_count(usage.get("prompt_tokens")),
        "completion_tokens": _as_count(usage.get("completion_tokens")),
    }


def enhance_with_failure_kind(
    text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, dict[str, int], str | None]:
    """Expand one prompt, reporting the fail-soft kind (issue 263).

    Returns `(expanded_text, counts, failure_kind)` where `failure_kind`
    is None on success (including blank-input skip), `"transport"` when
    the sidecar was unreachable or answered non-200 (`RuntimeError` from
    `_post_chat`, including normalized `httpx.HTTPError` and missing
    httpx, plus `OSError` such as `ConnectionError`), and `"parse"` when
    the reply was misshapen (`ValueError`/`TypeError` from
    `_parse_expansion` or the non-object guard in `_post_chat`).
    Programming errors (`AttributeError`, `KeyError`, ...) propagate
    loud — they are bugs, not sidecar-down. The split lets engagement
    accounting separate "down" from "malformed" instead of merging both
    into zero-count non-engagement.
    """
    zero_counts = {"prompt_tokens": 0, "completion_tokens": 0}
    if not text.strip():
        return text, dict(zero_counts), None
    body = build_enhancer_body(text, system_prompt)
    try:
        url = endpoint.rstrip("/") + COMPLETIONS_PATH
        parsed = _post_chat(url, body, timeout)
        expanded, counts = _parse_expansion(parsed, url)
    except (RuntimeError, OSError):
        return text, dict(zero_counts), "transport"
    except (ValueError, TypeError):
        return text, dict(zero_counts), "parse"
    else:
        return expanded, counts, None


def enhance(
    text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, dict[str, int]]:
    """Expand one prompt through the sidecar; fail soft to the input text.

    Returns `(expanded_text, {"prompt_tokens": n, "completion_tokens": m})`.
    Blank input skips the HTTP call (nothing to expand); transport
    (`RuntimeError`/`OSError`, including normalized `httpx.HTTPError`)
    and parse (`ValueError`/`TypeError`) failures return the input
    unchanged with zero counts — the commit must never fail because an
    advisory prototype stage is unavailable. Programming errors propagate
    loud (issue 263). See `enhance_with_failure_kind` for the kind split.
    """
    expanded, counts, _kind = enhance_with_failure_kind(text, system_prompt, endpoint, timeout)
    return expanded, counts


def enhance_for_backend(
    prompts: list[str],
    *,
    enabled: bool,
    backend: str,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[str]:
    """Expand staged prompts when the prototype applies; else return them untouched.

    Thin wrapper over `enhance_many` for callers that only need the texts
    (the supervisor uses `enhance_many` directly for its engagement
    summary). `enabled` is `config.video.prompt_enhance` (default True);
    the backend gate keeps every non-LTX path byte-identical even if the
    knob is ever mis-set. Per-prompt failures already degrade inside
    `enhance`, so this never raises for transport reasons — it maps the
    expansion over the staged list.
    """
    return enhance_many(
        prompts,
        enabled=enabled,
        backend=backend,
        endpoint=endpoint,
        timeout=timeout,
        system_prompt=system_prompt,
    )[0]


def enhance_many(
    prompts: list[str],
    *,
    enabled: bool,
    backend: str,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[list[str], dict[str, int]]:
    """Expand staged prompts and report engagement (Track C observable).

    Same gating as `enhance_for_backend`, but returns `(expanded,
    summary)` where summary carries `prompts_total`, `prompts_changed`
    (expansion differs from input), `expanded_chars`, `prompt_tokens`,
    `completion_tokens`, plus `transport_failures` and `parse_failures`
    (issue 263: distinct counters so engagement accounting separates
    "sidecar down" from "malformed reply" instead of merging both into
    zero-count non-engagement) — so a run's metrics prove whether the
    sidecar engaged, instead of leaving ON-vs-OFF pairs to wall-time
    guesswork. Never raises for transport/parse reasons (see `enhance`).
    """
    total = len(prompts)
    summary = {
        "prompts_total": total,
        "prompts_changed": 0,
        "expanded_chars": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "transport_failures": 0,
        "parse_failures": 0,
    }
    if not enabled or backend not in ENHANCER_BACKENDS:
        return list(prompts), summary
    expanded: list[str] = []
    for prompt in prompts:
        text, counts, failure_kind = enhance_with_failure_kind(
            prompt, system_prompt, endpoint, timeout
        )
        expanded.append(text)
        summary["expanded_chars"] += len(text)
        summary["prompt_tokens"] += counts["prompt_tokens"]
        summary["completion_tokens"] += counts["completion_tokens"]
        if failure_kind == "transport":
            summary["transport_failures"] += 1
        elif failure_kind == "parse":
            summary["parse_failures"] += 1
        if text != prompt:
            summary["prompts_changed"] += 1
    return expanded, summary

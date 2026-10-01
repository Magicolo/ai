"""Director worker: `python -m voyage.workers.director`.

Backends: `deterministic` (no model weights) and `qwen` (Qwen decider:
cuda:1 via 4-bit AWQ by default, bf16 on CPU with device="cpu", DESIGN
§§8-9). Ops: `decide`, `embed`, `inspect`.
Invalid model JSON follows the §51 chain inside the worker — stricter
retry, lower temperature, then deterministic fallback — so a bad LLM
response never corrupts persistent state. `inspect` (Qwen3.5-9B VLM,
Phase 5) follows a retry→skip chain instead: a failed inspection
returns `inspected: False` and the voyage continues on deterministic
metrics alone — a slow or missing inspector must never stop a healthy
voyage (§44).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from voyage.director import (
    DIRECTOR_SYSTEM_PROMPT as SYSTEM_PROMPT,
)
from voyage.director import (
    DeterministicDirector,
    build_director_user_message,
    deterministic_decision,
)
from voyage.models import EvolutionDecision, TransitionPhase
from voyage.workers.loop import checked_request, serve, validate_benchmark_counts

_CONFIG: dict[str, Any] = {"backend": "qwen"}
_QWEN: dict[str, Any] = {}
_EMBEDDER: dict[str, Any] = {}
_INSPECTOR: dict[str, Any] = {}

INSPECTOR_MODEL_ID = "Qwen/Qwen3.5-9B"
"""Default VLM weights (pinned in the model registry, Step 5)."""

INIT_STR_KEYS = (
    "backend",
    "model_id",
    "device",
    "embedding_model_id",
    "inspector_model_id",
    "models_dir",
)
"""String-valued `init` fields the worker records (models_dir added for the
volume-deleted gap: the supervisor passes video.models_dir so snapshot
resolution never guesses the mount; device added for the GPU decider: the
supervisor passes director.device so the Qwen load branches to cuda)."""

INSPECT_PROMPT = (
    "Describe what is visible in this frame of an abstract infinite "
    "voyage animation in one or two sentences: dominant shapes, motion "
    "direction, palette. Reply with a JSON object only, no prose: "
    '{"scene_summary": "..."}'
)
"""Tight single-frame prompt: one frame per segment keeps the ~2min CPU
budget bounded (Step 0 probe: 92s for 128 tokens)."""

MAX_NEW_TOKENS_LIMIT = 4096
"""Upper bound for `max_new_tokens`: unbounded budgets run away on CPU."""

DEFAULT_CPU_MODEL_ID = "Qwen/Qwen3-8B"
"""bf16 decider served from system RAM on the legacy CPU path."""

DEFAULT_CUDA_MODEL_ID = "Qwen/Qwen3-4B-AWQ"
"""4-bit AWQ decider for CUDA devices: the 8B bf16 cannot fit a 6GB second
GPU (OOM at materialization, 5.49GiB > 5.6GB), while the 4B-AWQ serves at
2.8GiB peak with valid first-attempt JSON (probe 2026-09-30)."""

PROMPT_LOOKUP_NUM_TOKENS = 10
"""Draft tokens for prompt-lookup decoding (option 2, DESIGN §§8-9).

Lookahead-style speculative path on the resident AWQ stack (~23s per
directive today): the model verifies cheap n-gram draft tokens instead of
autoregressing every step. 10 matches the upstream prompt-lookup default
order; a stack that rejects the kwarg falls back to plain generate (§51).
"""

LLAMA_REQUEST_TIMEOUT_SECONDS = 60.0
"""HTTP timeout for one sidecar chat completion (option 3 client, §51).

Why 60: the in-process AWQ decider answers in ~23s, so the sidecar serves
the same budget class — 60s leaves headroom for queueing without hanging
a segment commit behind a dead server (the §51 chain retries, then the
deterministic fallback renders).
"""

LLAMA_COMPLETIONS_PATH = "/v1/chat/completions"
"""OpenAI-compatible chat path the sidecar serves (option 3 contract)."""

LLAMA_DECISION_SCHEMA_NAME = "evolution_decision"
"""`response_format` schema name pinned with the sidecar track (contract)."""

LLAMA_ENDPOINT_KEY = "llama_endpoint"
"""Decide/init payload field carrying the sidecar base URL (contract).

`None`/absent/empty means the current in-process AWQ path, untouched —
the sidecar is opt-in per payload, never a config flip with blast radius.
"""


def _normalize_device(device: object) -> str:
    """Validate a decider placement string (fail fast, never guess)."""
    text = str(device or "cuda:1")
    if text != "cpu" and not text.startswith("cuda"):
        raise ValueError(f"director device must be 'cpu' or 'cuda[:N]' (got {text!r})")
    return text


def _require_module(module_name: str) -> None:
    """Fail fast with ImportError when an optional model stack is absent.

    Why a `find_spec` guard instead of a bare import: the worker loop maps
    `ImportError` (like any unexpected exception) to retryable WORKER_ERROR,
    but the message from a bare `import torch` deep inside a loader names
    the module without saying which model stack needed it. The guard raises
    the same `ImportError` class the loaders historically raised (pinned by
    `test_director_request_validation`), with the needing stack named, and
    — because it runs before any cache mutation — a failed load never
    clobbers the resident entry. GPU stays behind these lazy imports (§12
    hard ban: no `torch`/`transformers`/`sentence_transformers` at module
    scope, verified by grep).
    """
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"director worker needs optional dependency {module_name!r} "
            "for this backend (slim image carries the deterministic backend only)"
        )


def validate_max_new_tokens(max_new_tokens: int) -> None:
    """Reject token budgets that generate nothing or run away (issue 075)."""
    if max_new_tokens < 1 or max_new_tokens > MAX_NEW_TOKENS_LIMIT:
        raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS_LIMIT} (got {max_new_tokens})")


def validate_temperature(temperature: float) -> None:
    """Reject non-finite/negative sampling temperatures (issue 075)."""
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"temperature must be finite and >= 0 (got {temperature})")


def _models_dir() -> str:
    """Where this worker's /models snapshots live (init payload wins)."""
    configured = _CONFIG.get("models_dir")
    if isinstance(configured, str) and configured:
        return configured
    return os.environ.get("VOYAGE_MODELS_DIR", "/models")


def _resolve_model_source(model_id: str) -> str:
    """Map a hub id to its single /models snapshot, fetching when absent.

    Local directories pass through untouched (E2E/inspector local paths);
    ids outside the registry keep today's hub/cache behavior. A known repo
    resolves to `<models_dir>/<relative_dir>`; when that snapshot's own
    checklist fails, the owning spec is fetched into /models (the
    HF_HUB_OFFLINE guard is lifted for that fetch — it protects the
    ephemeral cache, not the persistent volume) and re-checked. Download
    failure raises so the caller's fallback chain (deterministic /
    inspect-skip) engages with the error recorded — the worker never
    serves half-fetched weights.
    """
    from voyage import model_registry  # lazy: registry is torch-free (§12)

    if os.path.isdir(model_id):
        return model_id
    ref = model_registry.resolve_snapshot(model_id)
    if ref is None:
        return model_id
    models_dir = Path(_models_dir())
    if model_registry.snapshot_present(models_dir, ref):
        return str(models_dir / ref.relative_dir)
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        model_registry.download_model(models_dir, ref.spec_name)
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    if not model_registry.snapshot_present(models_dir, ref):
        raise RuntimeError(f"snapshot {ref.relative_dir} still incomplete after download")
    return str(models_dir / ref.relative_dir)


def _effective_qwen_id(model_id: str, device: str) -> str:
    """Apply the 8B→4B-AWQ substitution for CUDA devices (pure helper).

    Kept separate from `_load_qwen` so the cache key and the stored id
    use one normalization — keying on the raw request reloaded the full
    model on every call (live 2026-09-30).
    """
    if device != "cpu" and model_id == DEFAULT_CPU_MODEL_ID:
        return DEFAULT_CUDA_MODEL_ID
    return model_id


def _cuda_available() -> bool:
    """True when the resident torch stack reports a usable CUDA device.

    Goes through `getattr` (not `torch.cuda.is_available()`) so unit
    stubs of the torch module without a `cuda` attribute keep working.
    """
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available() if callable(is_available) else False)


def _load_qwen(model_id: str, *, device: str) -> tuple[Any, Any]:
    # `device` is keyword-only and required: the pre-GPU signature took a
    # bare model_id (CPU always), and a silent "cuda:1" default would flip
    # any unmigrated caller onto the second GPU behind its back.
    # Issue 075: reload (not reuse) when the id changed — a long-lived
    # worker re-`init` with a new model must not keep deciding with the
    # old weights. Single resident entry (no per-id growth, cf. issue 030);
    # the cache key is (effective_id, device) so a placement change reloads.
    effective_id = _effective_qwen_id(model_id, device)
    if (
        "model" not in _QWEN
        or _QWEN.get("model_id") != effective_id
        or _QWEN.get("device") != device
    ):
        _require_module("torch")
        _require_module("transformers")
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        # Offline-first: use cached weights when present, fail fast when
        # absent (a missing Qwen stack must degrade to the deterministic
        # fallback in seconds — never hang a commit on a model download).
        # Explicit HF_HUB_OFFLINE=0 re-enables downloads.
        offline = os.environ.get("HF_HUB_OFFLINE", "1") != "0"
        if device == "cpu":
            source = _resolve_model_source(model_id)
            tokenizer = AutoTokenizer.from_pretrained(
                source, trust_remote_code=False, local_files_only=offline
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=False,
                    local_files_only=offline,
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=False,
                    local_files_only=offline,
                )
        else:
            # CUDA path: AWQ-quantized decider via the transformers
            # gptqmodel backend (no `import awq` — autoawq is deprecated and
            # must NOT be installed). The 8B bf16 default cannot fit a
            # small second GPU, so it substitutes the 4B-AWQ pin with a
            # loud warning; explicit model ids pass through untouched.
            # Absent CUDA (single-GPU/CI boxes) falls back to CPU loudly —
            # an OOM never falls back (it propagates as retryable).
            if not _cuda_available():
                print(
                    "WARNING: director device "
                    f"{device!r} has no CUDA — falling back to the CPU path",
                    file=sys.stderr,
                )
                return _load_qwen(model_id, device="cpu")
            if model_id != effective_id:
                print(
                    "WARNING: director model "
                    f"{model_id!r} cannot fit a small CUDA device — substituting "
                    f"{effective_id!r} (pass an explicit --director model id to override)",
                    file=sys.stderr,
                )
            source = _resolve_model_source(effective_id)
            tokenizer = AutoTokenizer.from_pretrained(
                source, trust_remote_code=False, local_files_only=offline
            )
            # attn_implementation="eager": transformers 5.17 defaults to
            # flash-attention, whose kernels require Ampere (sm_80+) — the
            # RTX 2060 second GPU is Turing (sm_75) and every decide fell
            # back to deterministic with "FlashAttention only supports Ampere
            # GPUs or newer" (verified live 2026-09-30; eager rescue probe
            # /tmp/ab_eager.py generated clean JSON on cuda:1). Eager costs
            # some throughput but the director is latency-tolerant.
            model = AutoModelForCausalLM.from_pretrained(
                source,
                device_map=device,
                attn_implementation="eager",
                trust_remote_code=False,
                local_files_only=offline,
            )
            model_id = effective_id
        model.eval()
        _QWEN["model"] = model
        _QWEN["tokenizer"] = tokenizer
        _QWEN["model_id"] = model_id
        _QWEN["device"] = device
    return _QWEN["model"], _QWEN["tokenizer"]


def _load_inspector(model_id: str) -> tuple[Any, Any]:
    """Lazily load the Qwen3.5-9B VLM + processor on CPU (bf16, mmap-fast).

    trust_remote_code stays False (issue 056): transformers 5.17.0 ships
    NATIVE qwen3_5 modeling (Qwen3_5ForConditionalGeneration) plus the
    multimodal auto class, verified live 2026-09-29 — the True flag was a
    4.57.6-era requirement (that stack lacks both) and is now dead. The
    weights (~19GB BF16) live in system RAM — never on the 16GB GPU.
    """
    # Issue 075: same reload-on-id-change contract as `_load_qwen`.
    if "model" not in _INSPECTOR or _INSPECTOR.get("model_id") != model_id:
        _require_module("torch")
        _require_module("transformers")
        source = _resolve_model_source(model_id)
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(source, trust_remote_code=False)
        model = AutoModelForMultimodalLM.from_pretrained(
            source,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.eval()
        _INSPECTOR["model"] = model
        _INSPECTOR["processor"] = processor
        _INSPECTOR["model_id"] = model_id
    return _INSPECTOR["model"], _INSPECTOR["processor"]


def _load_embedder(model_id: str) -> Any:
    # Issue 075: same reload-on-id-change contract as `_load_qwen`.
    if "model" not in _EMBEDDER or _EMBEDDER.get("model_id") != model_id:
        _require_module("sentence_transformers")
        source = _resolve_model_source(model_id)
        from sentence_transformers import SentenceTransformer

        _EMBEDDER["model"] = SentenceTransformer(source, device="cpu")
        _EMBEDDER["model_id"] = model_id
    return _EMBEDDER["model"]


def _inspector_generate(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
    """Run one non-thinking VLM pass over a single frame PNG (greedy)."""
    _require_module("torch")
    import torch

    model, processor = _load_inspector(model_id)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": frame_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return str(processor.decode(generated)).strip()


def _normalize_inspect(text: str) -> dict[str, Any]:
    """Parse one VLM reply into a scene summary (pure; raises on garbage)."""
    data = _extract_json(text)
    summary = data.get("scene_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("VLM reply has no scene_summary string")
    return {"scene_summary": summary.strip()}


def handle_inspect(payload: dict[str, Any]) -> dict[str, Any]:
    """Retry→skip chain: two attempts, then `inspected: False` (never raises).

    The inspector is advisory (§44): deterministic metrics carry the
    feedback loop, so a VLM failure degrades to a missing scene summary
    instead of aborting the segment.
    """
    checked_request(payload, frame_path=str)
    model_id = str(payload.get("model_id") or _CONFIG.get("inspector_model_id", INSPECTOR_MODEL_ID))
    max_new_tokens = int(payload.get("max_new_tokens", 256))
    validate_max_new_tokens(max_new_tokens)
    attempts = [INSPECT_PROMPT, INSPECT_PROMPT + " JSON object only. No prose."]
    last_error = "no attempts"
    for attempt_prompt in attempts:
        try:
            text = _inspector_generate(
                model_id, str(payload["frame_path"]), attempt_prompt, max_new_tokens
            )
            result = _normalize_inspect(text)
            result["inspected"] = True
            result["model_id"] = model_id
            return result
        except Exception as exc:  # noqa: BLE001 — inspector must never raise
            last_error = str(exc)
    return {"inspected": False, "error": last_error, "model_id": model_id}


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first `{...}` object out of chatty model output (pure).

    Strips markdown fences first (the strict retry prompt still returns
    fenced JSON often); raises ValueError when no object survives so the
    §51 chain can retry or fall back deterministically.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rsplit("```", 1)[0].strip():
            cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model output is not a JSON object")
    return parsed


def _coerce_decision_json(raw_text: str) -> str:
    """Coerce chatty model output to canonical parseable JSON (option 4a).

    Interim for the Outlines constrained-decode path (DESIGN §§8-9): the
    `outlines` package is absent from every voyage image (probed via
    `find_spec` in the slim container 2026-10-01 — do NOT pip install it;
    container hygiene §10/§11), so true logit-level constrained decoding
    cannot run yet. Until the director image gains `outlines`, this strict
    wrapper is the guarantee: extract the first JSON object (fences and
    prose tolerated) and re-dump it canonically, so downstream always gets
    parseable JSON first-try or a `ValueError` the §51 chain retries (then
    deterministic fallback — never corrupt state). When `outlines` lands,
    replace this body with an `OutlinesLogitsProcessor` built from
    `EvolutionDecision.model_json_schema()` (the schema helper the llama
    branch already wires below) and keep this signature and contract.
    Token counts are NOT estimated here: both live paths report exact
    counts (AWQ tensor widths, sidecar `usage` block), so no estimation
    layer exists to drift.
    """
    parsed = _extract_json(raw_text)
    return json.dumps(parsed, sort_keys=True)


def _non_negative_int(value: object) -> int:
    """Coerce a sidecar `usage` count to a safe int (pure; garbage is 0)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return int(value)
    return 0


def _build_llama_request_body(
    user_message: str, temperature: float, max_new_tokens: int
) -> dict[str, Any]:
    """Build the exact sidecar chat body (option 3 client, pure).

    Contract pinned with the sidecar track: system prompt + user message,
    temperature/max_tokens passthrough, a strict `evolution_decision`
    JSON-schema response format sourced from
    `EvolutionDecision.model_json_schema()`, and thinking disabled.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": LLAMA_DECISION_SCHEMA_NAME,
                "strict": True,
                "schema": EvolutionDecision.model_json_schema(),
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _post_llama_chat(request_url: str, request_body: dict[str, Any]) -> dict[str, Any]:
    """POST one chat completion to the sidecar (module-level test seam).

    Why a seam: tests stub this function with a fake transport — no live
    server, no network in the suite. `httpx` imports lazily (the slim
    gate image has no httpx; the director venv serves it — a module-scope
    import would break every slim import of this worker). Explicit timeout,
    no streaming. Non-200 and misshapen replies raise `RuntimeError` /
    `ValueError`; transport errors (e.g. `ConnectionError`) propagate
    unchanged — all are `Exception` subclasses the existing §51 chain
    already catches, so this branch degrades exactly like the AWQ path
    (retry, then deterministic fallback) and never raises new types
    outward past `_qwen_decide`.
    """
    httpx_client: Any = importlib.import_module("httpx")
    response: Any = httpx_client.post(
        request_url, json=request_body, timeout=LLAMA_REQUEST_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        raise RuntimeError(f"llama-server {request_url} answered status {response.status_code}")
    parsed: Any = response.json()
    if not isinstance(parsed, dict):
        raise ValueError(f"llama-server {request_url} answered a non-object body")
    return cast("dict[str, Any]", parsed)


def _qwen_generate_llama(
    llama_endpoint: str, user_message: str, temperature: float, max_new_tokens: int
) -> tuple[str, int, int]:
    """Serve one completion from the sidecar, keeping the (text, tokens) contract.

    The sidecar guarantees schema-shaped JSON; the reply still runs
    through `_extract_json` belt-and-braces (a proxy could wrap content)
    and the option-4a coercion, so both paths return canonical JSON.
    Counts come from the response `usage` block (0s when absent — the
    supervisor tolerates zero counts, and exact-when-present beats
    tokenizer estimation that could drift from the server's counting).
    """
    request_url = llama_endpoint.rstrip("/") + LLAMA_COMPLETIONS_PATH
    request_body = _build_llama_request_body(user_message, temperature, max_new_tokens)
    parsed = _post_llama_chat(request_url, request_body)
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"llama-server {request_url} answered without choices")
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"llama-server {request_url} answered without message content")
    text = _coerce_decision_json(content)
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return text, 0, 0
    return (
        text,
        _non_negative_int(usage.get("prompt_tokens")),
        _non_negative_int(usage.get("completion_tokens")),
    )


def _qwen_generate(
    model_id: str,
    user_message: str,
    temperature: float,
    max_new_tokens: int,
    enable_thinking: bool,
    *,
    device: str,
    llama_endpoint: str | None = None,
) -> tuple[str, int, int]:
    """Generate one completion, reporting token usage (Stage A telemetry).

    Returns the decoded text plus (prompt_tokens, completion_tokens) counted
    from the live input/output tensor widths — the supervisor aggregates
    them into `segment_committed.director_tokens` so LLM cost is visible
    per segment instead of vanishing into the `director` stage seconds.

    Three client-side upgrades ride this function (DESIGN §§8-9, §51):
    option 2 adds prompt-lookup decoding to the AWQ `generate` call (with
    graceful fallback when the stack rejects the kwarg); option 4a coerces
    the decoded text through `_coerce_decision_json` (interim until the
    director image gains `outlines`); option 3 serves the whole completion
    from the llama-server sidecar when `llama_endpoint` is set (exact
    contract in `_build_llama_request_body`, transport seam in
    `_post_llama_chat`). `None`/empty endpoint keeps the current AWQ path
    byte-for-byte, so existing callers and tests are unaffected.
    """
    if isinstance(llama_endpoint, str) and llama_endpoint:
        return _qwen_generate_llama(llama_endpoint, user_message, temperature, max_new_tokens)
    _require_module("torch")
    import torch

    model, tokenizer = _load_qwen(model_id, device=device)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        # The AWQ model lives on CUDA; input tensors must follow it there
        # (the CPU path keeps host tensors — no .to() no-op churn).
        target = str(model.device)
        inputs = {key: value.to(target) for key, value in inputs.items()}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
        "prompt_lookup_num_tokens": PROMPT_LOOKUP_NUM_TOKENS,
    }
    if temperature > 0.0:
        generate_kwargs.update(
            {"temperature": temperature, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
        )
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)
    except TypeError as generate_error:
        # Stacks without prompt-lookup support (older transformers/GPTQModel)
        # reject the kwarg: retry once without it instead of failing the
        # decide. Anything else re-raises into the §51 chain untouched.
        if "prompt_lookup_num_tokens" not in str(generate_error):
            raise
        del generate_kwargs["prompt_lookup_num_tokens"]
        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    generated = output[0][prompt_tokens:]
    completion_tokens = int(generated.shape[0])
    raw_text = str(tokenizer.decode(generated, skip_special_tokens=True)).strip()
    return _coerce_decision_json(raw_text), prompt_tokens, completion_tokens


def _qwen_decide(payload: dict[str, Any]) -> dict[str, Any]:
    """§51 chain: generate → validate → stricter retry → fallback."""
    model_id = str(payload.get("model_id") or _CONFIG.get("model_id", "Qwen/Qwen3-8B"))
    device = _normalize_device(payload.get("device") or _CONFIG.get("device", "cuda:1"))
    temperature = float(payload.get("temperature", 0.7))
    max_new_tokens = int(payload.get("max_new_tokens", 1024))
    validate_temperature(temperature)
    validate_max_new_tokens(max_new_tokens)
    enable_thinking = bool(payload.get("enable_thinking", False))
    configured_endpoint = payload.get(LLAMA_ENDPOINT_KEY, _CONFIG.get(LLAMA_ENDPOINT_KEY))
    if configured_endpoint is not None and not isinstance(configured_endpoint, str):
        raise TypeError(
            f"decide field {LLAMA_ENDPOINT_KEY!r} must be str or None, "
            f"got {type(configured_endpoint).__name__}"
        )
    llama_endpoint = configured_endpoint if configured_endpoint else None
    decision_index = int(payload["decision_index"])
    phase = cast(TransitionPhase, str(payload.get("phase", "ESTABLISH")))
    user_message = build_director_user_message(
        style_charter=str(payload.get("style_charter", "")),
        current_world=str(payload.get("current_world", "")),
        current_transition=str(payload.get("current_transition", "")),
        recent_summary=str(payload.get("recent_summary", "")),
        forbidden_summary=str(payload.get("forbidden_summary", "")),
        audio_state=str(payload.get("audio_state", "")),
        controller_metrics=str(payload.get("controller_metrics", "")),
        retry_feedback=str(payload.get("retry_feedback", "")),
        measured_context=str(payload.get("measured_context", "")),
        previous_captions=str(payload.get("previous_captions", "")),
    )
    attempts = [
        (user_message, temperature),
        (user_message + "\n\nSTRICT: JSON object only. No prose.", max(0.1, temperature - 0.2)),
    ]
    last_error = "no attempts"
    prompt_tokens = 0
    completion_tokens = 0
    for attempt_number, (attempt_message, attempt_temp) in enumerate(attempts):
        try:
            text, prompt_tokens, completion_tokens = _qwen_generate(
                model_id,
                attempt_message,
                attempt_temp,
                max_new_tokens,
                enable_thinking,
                device=device,
                llama_endpoint=llama_endpoint,
            )
            data = _extract_json(text)
            data["decision_index"] = decision_index
            data["phase"] = data.get("phase", phase)
            decision = EvolutionDecision.model_validate(data)
            dumped = decision.model_dump()
            dumped["fallback"] = False
            dumped["prompt_tokens"] = prompt_tokens
            dumped["completion_tokens"] = completion_tokens
            return dumped
        except Exception as exc:  # noqa: BLE001 — chain must survive any bad output
            last_error = str(exc)
            if attempt_number == 0:
                attempts[1] = (
                    user_message
                    + "\n\nPREVIOUS OUTPUT REJECTED — fix these errors and "
                    + f"respond with a corrected JSON object only:\n{last_error}",
                    max(0.1, temperature - 0.2),
                )
    fallback = deterministic_decision(
        decision_index,
        str(payload.get("current_world", "")),
        str(payload.get("destination_concept", "")),
        phase,
        str(payload.get("style_charter", "")),
    )
    dumped = fallback.model_dump()
    dumped["fallback"] = True
    dumped["notes"] = f"qwen-unparseable ({last_error}); {fallback.notes}"
    dumped["prompt_tokens"] = prompt_tokens
    dumped["completion_tokens"] = completion_tokens
    return dumped


def handle_decide(payload: dict[str, Any]) -> dict[str, Any]:
    checked_request(payload, decision_index=int, phase=str)
    backend = str(payload.get("backend") or _CONFIG.get("backend", "qwen"))
    if backend == "qwen":
        return _qwen_decide(payload)
    checked_request(
        payload,
        current_concept=str,
        destination_concept=str,
        style=str,
    )
    director = DeterministicDirector(style=str(payload["style"]))
    phase = cast(TransitionPhase, str(payload["phase"]))
    decision = director.propose(
        decision_index=int(payload["decision_index"]),
        current_concept=str(payload["current_concept"]),
        destination_concept=str(payload["destination_concept"]),
        phase=phase,
    )
    dumped = decision.model_dump()
    dumped["fallback"] = True
    return dumped


def handle_embed(payload: dict[str, Any]) -> dict[str, Any]:
    checked_request(payload, texts=list)
    raw_texts = payload["texts"]
    if not isinstance(raw_texts, list) or not raw_texts:
        raise ValueError("embed needs a non-empty texts list")
    if not all(isinstance(item, str) for item in raw_texts):
        raise ValueError("embed texts must all be strings (no silent coercion)")
    texts = list(raw_texts)
    model_id = str(
        payload.get("embedding_model_id")
        or _CONFIG.get("embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2")
    )
    model = _load_embedder(model_id)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {"vectors": [[float(value) for value in row] for row in vectors.tolist()]}


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    """Record the director backend + model ids (no weights load here).

    Optional fields are type-checked when present so a mistyped `init`
    fails as INVALID_PAYLOAD (fatal) instead of misdirecting every later
    `decide`. Weights stay lazy: a long-lived worker re-`init` with a new
    id reloads on next use (issue 075), and the process can start while
    video still owns the GPU. The sidecar URL (`llama_endpoint`, option 3)
    is stored alongside: a non-empty string arms the HTTP branch of
    `_qwen_generate`, `None`/empty clears back to the AWQ default.
    """
    for key in INIT_STR_KEYS:
        if key in payload and not isinstance(payload[key], str):
            raise TypeError(f"init field {key!r} must be str, got {type(payload[key]).__name__}")
    if LLAMA_ENDPOINT_KEY in payload:
        endpoint_value = payload[LLAMA_ENDPOINT_KEY]
        if endpoint_value is None:
            _CONFIG.pop(LLAMA_ENDPOINT_KEY, None)
        elif isinstance(endpoint_value, str) and endpoint_value:
            _CONFIG[LLAMA_ENDPOINT_KEY] = endpoint_value
        elif isinstance(endpoint_value, str):
            _CONFIG.pop(LLAMA_ENDPOINT_KEY, None)
        else:
            raise TypeError(
                f"init field {LLAMA_ENDPOINT_KEY!r} must be str or None, "
                f"got {type(endpoint_value).__name__}"
            )
    unknown = sorted(set(payload) - set(INIT_STR_KEYS) - {LLAMA_ENDPOINT_KEY})
    if unknown:
        raise TypeError(f"init got unknown field(s) {unknown} (known: {sorted(INIT_STR_KEYS)})")
    _CONFIG.update({key: payload[key] for key in INIT_STR_KEYS if key in payload})
    return {"status": "READY", "backend": _CONFIG["backend"]}


def handle_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Liveness probe reporting which optional stacks are resident."""
    del payload
    return {
        "status": "READY",
        "backend": _CONFIG["backend"],
        "device": _CONFIG.get("device", "cuda:1"),
        "qwen_loaded": "model" in _QWEN,
        "qwen_device": _QWEN.get("device"),
        "embedder_loaded": "model" in _EMBEDDER,
        "inspector_loaded": "model" in _INSPECTOR,
    }


def handle_shutdown(payload: dict[str, Any]) -> dict[str, Any]:
    """Stop the worker loop; resident CPU stacks die with the process."""
    del payload
    return {"stopped": True}


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured deterministic decisions (startup excluded)."""
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    validate_benchmark_counts(warmup, measured)
    probe = {
        "decision_index": 0,
        "phase": "DRIFT",
        "current_concept": "a misty harbor",
        "destination_concept": "a glass desert",
        "style": "pastel neon line-art, peaceful",
        "backend": "deterministic",
    }
    walls: list[float] = []
    for decision_index in range(warmup + measured):
        started = time.monotonic()
        handle_decide({**probe, "decision_index": decision_index})
        elapsed = time.monotonic() - started
        if decision_index >= warmup:
            walls.append(elapsed)
    mean = sum(walls) / len(walls)
    return {
        "backend": "deterministic",
        "warmup_decisions": warmup,
        "measured_decisions": measured,
        "decision_wall_seconds": [round(wall, 3) for wall in walls],
        "decisions_per_second": round(1.0 / mean, 3),
    }


def main() -> None:
    """Serve the director op map over the shared JSONL loop."""
    serve(
        {
            "init": handle_init,
            "health": handle_health,
            "decide": handle_decide,
            "embed": handle_embed,
            "inspect": handle_inspect,
            "benchmark": handle_benchmark,
            "shutdown": handle_shutdown,
        }
    )


if __name__ == "__main__":
    main()

"""Director worker: `python -m voyage.workers.director`.

Backends: `deterministic` (no model weights) and `qwen` (Qwen3-8B on
CPU + MiniLM embeddings, DESIGN §§8-9). Ops: `decide`, `embed`, `inspect`.
Invalid model JSON follows the §51 chain inside the worker — stricter
retry, lower temperature, then deterministic fallback — so a bad LLM
response never corrupts persistent state. `inspect` (Qwen3.5-9B VLM,
Phase 5) follows a retry→skip chain instead: a failed inspection
returns `inspected: False` and the voyage continues on deterministic
metrics alone — a slow or missing inspector must never stop a healthy
voyage (§44).
"""

from __future__ import annotations

import json
import time
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
from voyage.workers.loop import checked_request, serve

_CONFIG: dict[str, Any] = {"backend": "deterministic"}
_QWEN: dict[str, Any] = {}
_EMBEDDER: dict[str, Any] = {}
_INSPECTOR: dict[str, Any] = {}

INSPECTOR_MODEL_ID = "Qwen/Qwen3.5-9B"
"""Default VLM weights (pinned in the model registry, Step 5)."""

INSPECT_PROMPT = (
    "Describe what is visible in this frame of an abstract infinite "
    "voyage animation in one or two sentences: dominant shapes, motion "
    "direction, palette. Reply with a JSON object only, no prose: "
    '{"scene_summary": "..."}'
)
"""Tight single-frame prompt: one frame per segment keeps the ~2min CPU
budget bounded (Step 0 probe: 92s for 128 tokens)."""


def _load_qwen(model_id: str) -> tuple[Any, Any]:
    if "model" not in _QWEN:
        import os

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
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=False, local_files_only=offline
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=False,
                local_files_only=offline,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=False,
                local_files_only=offline,
            )
        model.eval()
        _QWEN["model"] = model
        _QWEN["tokenizer"] = tokenizer
    return _QWEN["model"], _QWEN["tokenizer"]


def _load_inspector(model_id: str) -> tuple[Any, Any]:
    """Lazily load the Qwen3.5-9B VLM + processor on CPU (bf16, mmap-fast).

    trust_remote_code is required: the model ships custom modeling and
    processor code (Step 0 probe). The weights (~19GB BF16) live in
    system RAM — never on the 16GB GPU.
    """
    if "model" not in _INSPECTOR:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        _INSPECTOR["model"] = model
        _INSPECTOR["processor"] = processor
    return _INSPECTOR["model"], _INSPECTOR["processor"]


def _load_embedder(model_id: str) -> Any:
    if "model" not in _EMBEDDER:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER["model"] = SentenceTransformer(model_id, device="cpu")
    return _EMBEDDER["model"]


def _inspector_generate(model_id: str, frame_path: str, prompt: str, max_new_tokens: int) -> str:
    """Run one non-thinking VLM pass over a single frame PNG (greedy)."""
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


def _qwen_generate(
    model_id: str,
    user_message: str,
    temperature: float,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    import torch

    model, tokenizer = _load_qwen(model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generate_kwargs.update(
            {"temperature": temperature, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
        )
    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    generated = output[0][inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def _qwen_decide(payload: dict[str, Any]) -> dict[str, Any]:
    """§51 chain: generate → validate → stricter retry → fallback."""
    model_id = str(payload.get("model_id") or _CONFIG.get("model_id", "Qwen/Qwen3-8B"))
    temperature = float(payload.get("temperature", 0.7))
    max_new_tokens = int(payload.get("max_new_tokens", 1024))
    enable_thinking = bool(payload.get("enable_thinking", False))
    decision_index = int(payload["decision_index"])
    phase = cast("TransitionPhase", str(payload.get("phase", "ESTABLISH")))
    user_message = build_director_user_message(
        style_charter=str(payload.get("style_charter", "")),
        current_world=str(payload.get("current_world", "")),
        current_transition=str(payload.get("current_transition", "")),
        recent_summary=str(payload.get("recent_summary", "")),
        forbidden_summary=str(payload.get("forbidden_summary", "")),
        audio_state=str(payload.get("audio_state", "")),
        controller_metrics=str(payload.get("controller_metrics", "")),
        retry_feedback=str(payload.get("retry_feedback", "")),
    )
    attempts = [
        (user_message, temperature),
        (user_message + "\n\nSTRICT: JSON object only. No prose.", max(0.1, temperature - 0.2)),
    ]
    last_error = "no attempts"
    for attempt_number, (attempt_message, attempt_temp) in enumerate(attempts):
        try:
            text = _qwen_generate(
                model_id, attempt_message, attempt_temp, max_new_tokens, enable_thinking
            )
            data = _extract_json(text)
            data["decision_index"] = decision_index
            data["phase"] = data.get("phase", phase)
            decision = EvolutionDecision.model_validate(data)
            dumped = decision.model_dump()
            dumped["fallback"] = False
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
    return dumped


def handle_decide(payload: dict[str, Any]) -> dict[str, Any]:
    checked_request(payload, decision_index=int, phase=str)
    backend = str(payload.get("backend") or _CONFIG.get("backend", "deterministic"))
    if backend == "qwen":
        return _qwen_decide(payload)
    checked_request(
        payload,
        current_concept=str,
        destination_concept=str,
        style=str,
    )
    director = DeterministicDirector(style=str(payload["style"]))
    phase = cast("TransitionPhase", str(payload["phase"]))
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
    texts = [str(item) for item in payload["texts"]]
    model_id = str(
        payload.get("embedding_model_id")
        or _CONFIG.get("embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2")
    )
    model = _load_embedder(model_id)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {"vectors": [[float(value) for value in row] for row in vectors.tolist()]}


def handle_init(payload: dict[str, Any]) -> dict[str, Any]:
    _CONFIG.update(
        {
            key: payload[key]
            for key in ("backend", "model_id", "embedding_model_id", "inspector_model_id")
            if key in payload
        }
    )
    return {"status": "READY", "backend": _CONFIG["backend"]}


def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Time warmup + measured deterministic decisions (startup excluded)."""
    warmup = int(payload.get("warmup", 1))
    measured = int(payload.get("measured", 3))
    probe = {
        "decision_index": 0,
        "phase": "DRIFT",
        "current_concept": "a misty harbor",
        "destination_concept": "a glass desert",
        "style": "pastel neon line-art, peaceful",
        "backend": "deterministic",
    }
    walls: list[float] = []
    for index in range(warmup + measured):
        started = time.monotonic()
        handle_decide({**probe, "decision_index": index})
        elapsed = time.monotonic() - started
        if index >= warmup:
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
    serve(
        {
            "init": handle_init,
            "health": lambda _payload: {
                "status": "READY",
                "backend": _CONFIG["backend"],
                "qwen_loaded": "model" in _QWEN,
                "embedder_loaded": "model" in _EMBEDDER,
                "inspector_loaded": "model" in _INSPECTOR,
            },
            "decide": handle_decide,
            "embed": handle_embed,
            "inspect": handle_inspect,
            "benchmark": handle_benchmark,
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()

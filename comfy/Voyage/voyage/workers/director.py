"""Director worker: `python -m voyage.workers.director`.

Backends: `deterministic` (no model weights) and `qwen` (Qwen3-8B on
CPU + MiniLM embeddings, DESIGN §§8-9). Ops: `decide`, `embed`.
Invalid model JSON follows the §51 chain inside the worker — stricter
retry, lower temperature, then deterministic fallback — so a bad LLM
response never corrupts persistent state.
"""

from __future__ import annotations

import json
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


def _load_qwen(model_id: str) -> tuple[Any, Any]:
    if "model" not in _QWEN:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=False
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32, device_map="cpu", trust_remote_code=False
            )
        model.eval()
        _QWEN["model"] = model
        _QWEN["tokenizer"] = tokenizer
    return _QWEN["model"], _QWEN["tokenizer"]


def _load_embedder(model_id: str) -> Any:
    if "model" not in _EMBEDDER:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER["model"] = SentenceTransformer(model_id, device="cpu")
    return _EMBEDDER["model"]


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
            for key in ("backend", "model_id", "embedding_model_id")
            if key in payload
        }
    )
    return {"status": "READY", "backend": _CONFIG["backend"]}


def main() -> None:
    serve(
        {
            "init": handle_init,
            "health": lambda _payload: {
                "status": "READY",
                "backend": _CONFIG["backend"],
                "qwen_loaded": "model" in _QWEN,
                "embedder_loaded": "model" in _EMBEDDER,
            },
            "decide": handle_decide,
            "embed": handle_embed,
            "shutdown": lambda _payload: {"stopped": True},
        }
    )


if __name__ == "__main__":
    main()

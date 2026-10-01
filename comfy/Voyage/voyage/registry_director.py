"""Director triple family pins + builders (DESIGN §8 + §140 GPU-director entry).

Split from `voyage.registry_records` (issue 082): the Phase 3 director
LLM + GPU decider + shared embedding row as importable pins with no
cross-family coupling. `voyage.registry_records` re-exports every name
below so existing importers keep working; new code imports from here
directly.

- `QWEN_*`: Qwen3-8B dense Hub pin + layout + floors + license (Phase 3
  director stack, CPU-served bf16).
- `QWEN4B_AWQ_*`: Qwen3-4B-AWQ Hub pin + layout + floors + license (GPU
  decider, cuda:1 from the director venv).
- `MINILM_*`: shared sentence-transformers embedding Hub pin + layout +
  floors + license (consumed by BOTH `_record_director` builders below).
- `_record_director` / `_describe_director` + `_record_director_awq` /
  `_describe_director_awq`: manifest values + exact OK strings.
- No EXPECTED ingest hash: this row carries none (same as the
  inspector/audio/sfx/CausVid rows — the manifest record carries no sha).

Single-source note (batch-15 decision, explicit): the shared `MINILM_*`
pins move HERE alongside the QWEN rows — not duplicated, not left
behind in `registry_records`. Both record builders resolve the embedding
pins from this one module, so a pin bump touches exactly one file.
Silent pin duplication is forbidden.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue

# Phase 3 director LLM (DESIGN §8). Qwen3-8B dense, Apache 2.0, ungated.
# BF16 weights (~16.4 GiB); served on CPU from system RAM in the director
# worker process — never on the video GPU.
QWEN_HF_REPO = "Qwen/Qwen3-8B"

QWEN_HF_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

QWEN_SUBDIR = "Qwen3-8B"

QWEN_ALLOW = [
    "model-0000[1-5]-of-00005.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

QWEN_MIN_BYTES = 15_000_000_000

QWEN_LICENSE = "Apache 2.0"

QWEN_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE"

# GPU decider (unified worker image). Qwen3-4B AWQ-quantized, Apache 2.0,
# ungated. Single-shard int4 weights (~2.6 GiB); served on cuda:1 from the
# director venv — the 8B bf16 OOMs at materialization on a 6GB second GPU
# (5.49GiB > 5.6GB; probe 2026-09-30: 4B-AWQ loads in 2.5s, 2.8GiB peak,
# valid first-attempt JSON at temp 0.7).
QWEN4B_AWQ_HF_REPO = "Qwen/Qwen3-4B-AWQ"

QWEN4B_AWQ_HF_REVISION = "74d4bd2bd4bff9cafc9345221320bffb08b406a3"

QWEN4B_AWQ_SUBDIR = "Qwen3-4B-AWQ"

QWEN4B_AWQ_ALLOW = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

QWEN4B_AWQ_MIN_BYTES = 2_000_000_000

QWEN4B_AWQ_LICENSE = "Apache 2.0"

QWEN4B_AWQ_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3-4B-AWQ/blob/main/LICENSE"

# Shared concept-store embedding (DESIGN §21). Single source for both
# director builders above (batch-15 decision — see module docstring).
MINILM_HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"

MINILM_HF_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

MINILM_SUBDIR = "all-MiniLM-L6-v2"

MINILM_ALLOW = [
    "model.safetensors",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

MINILM_MIN_BYTES = 50_000_000

MINILM_LICENSE = "Apache 2.0"


def _record_director(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the Phase 3 director stack."""
    qwen_dir = models_dir / QWEN_SUBDIR
    qwen_shards = sorted(qwen_dir.glob("model-*-of-*.safetensors"))
    qwen_bytes = sum(part.stat().st_size for part in qwen_shards)
    minilm_dir = models_dir / MINILM_SUBDIR
    minilm_weights = minilm_dir / "model.safetensors"
    return {
        "repo": QWEN_HF_REPO,
        "revision": QWEN_HF_REVISION,
        "model_dir": str(qwen_dir),
        "checkpoint_bytes": qwen_bytes,
        "shards": [part.name for part in qwen_shards],
        "license": QWEN_LICENSE,
        "license_url": QWEN_LICENSE_URL,
        "embedding_repo": MINILM_HF_REPO,
        "embedding_revision": MINILM_HF_REVISION,
        "embedding_dir": str(minilm_dir),
        "embedding_bytes": minilm_weights.stat().st_size if minilm_weights.exists() else 0,
        "embedding_license": MINILM_LICENSE,
    }


def _describe_director(models_dir: Path) -> str:
    """Exact OK string for the director stack (byte-stable)."""
    qwen_dir = models_dir / QWEN_SUBDIR
    qwen_shards = sorted(qwen_dir.glob("model-*-of-*.safetensors"))
    qwen_bytes = sum(part.stat().st_size for part in qwen_shards)
    minilm_weights = models_dir / MINILM_SUBDIR / "model.safetensors"
    return (
        f"director-qwen8b OK (Qwen3-8B {qwen_bytes / 1024**3:.1f} GiB + MiniLM "
        f"{minilm_weights.stat().st_size / 1024**2:.0f} MiB)"
    )


def _record_director_awq(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the GPU decider stack (4B-AWQ + MiniLM)."""
    qwen_dir = models_dir / QWEN4B_AWQ_SUBDIR
    weights = qwen_dir / "model.safetensors"
    minilm_dir = models_dir / MINILM_SUBDIR
    minilm_weights = minilm_dir / "model.safetensors"
    return {
        "repo": QWEN4B_AWQ_HF_REPO,
        "revision": QWEN4B_AWQ_HF_REVISION,
        "model_dir": str(qwen_dir),
        "checkpoint_bytes": weights.stat().st_size if weights.exists() else 0,
        "license": QWEN4B_AWQ_LICENSE,
        "license_url": QWEN4B_AWQ_LICENSE_URL,
        "embedding_repo": MINILM_HF_REPO,
        "embedding_revision": MINILM_HF_REVISION,
        "embedding_dir": str(minilm_dir),
        "embedding_bytes": minilm_weights.stat().st_size if minilm_weights.exists() else 0,
        "embedding_license": MINILM_LICENSE,
    }


def _describe_director_awq(models_dir: Path) -> str:
    """Exact OK string for the GPU decider stack (byte-stable)."""
    weights = models_dir / QWEN4B_AWQ_SUBDIR / "model.safetensors"
    minilm_weights = models_dir / MINILM_SUBDIR / "model.safetensors"
    return (
        f"director-qwen4b-awq OK (Qwen3-4B-AWQ {weights.stat().st_size / 1024**3:.1f} GiB "
        f"+ MiniLM {minilm_weights.stat().st_size / 1024**2:.0f} MiB)"
    )

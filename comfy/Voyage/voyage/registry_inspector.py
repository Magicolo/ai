"""Phase 5 visual-inspector family pins + builders (DESIGN §§43-44, 100, 132).

Split from `voyage.registry_records` (issue 082): the Qwen3.5-9B
multimodal VLM row as importable pins with no cross-family coupling.
`voyage.registry_records` re-exports every name below so existing
importers keep working; new code imports from here directly.

- `QWEN35_*`: Hub pin + allow-list + floor + license.
- `_record_inspector` / `_describe_inspector`: manifest value + exact
  OK string.
- No EXPECTED ingest hash: the inspector manifest record carries no sha
  (same open residual as the CausVid DMD checkpoint — see
  `registry_records.py`).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue

# Phase 5 visual inspector (DESIGN §§43-44, 100, 132). Qwen3.5-9B
# multimodal VLM, Apache 2.0, ungated. BF16 weights (~19 GiB); served on
# CPU from system RAM in the director worker — never on the video GPU.
# The allow-list MUST include chat_template.jinja: the Qwen3.5 processor
# needs it and snapshot_download without it fails the inspector load
# (Step 0 probe lesson). Shard names carry a `model.safetensors-` prefix
# (unlike Qwen3-8B's `model-` prefix), hence the distinct shard glob.
QWEN35_HF_REPO = "Qwen/Qwen3.5-9B"

QWEN35_HF_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

QWEN35_SUBDIR = "Qwen3.5-9B"

QWEN35_ALLOW = [
    "model.safetensors-*-of-*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]

QWEN35_MIN_BYTES = 18_000_000_000

QWEN35_LICENSE = "Apache 2.0"

QWEN35_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE"


def _record_inspector(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the Phase 5 VLM inspector."""
    target_dir = models_dir / QWEN35_SUBDIR
    shards = sorted(target_dir.glob("model.safetensors-*-of-*.safetensors"))
    weights_bytes = sum(part.stat().st_size for part in shards)
    return {
        "repo": QWEN35_HF_REPO,
        "revision": QWEN35_HF_REVISION,
        "dir": str(target_dir),
        "bytes": weights_bytes,
        "license": QWEN35_LICENSE,
        "license_url": QWEN35_LICENSE_URL,
    }


def _describe_inspector(models_dir: Path) -> str:
    """Exact OK string for the VLM inspector (byte-stable)."""
    target_dir = models_dir / QWEN35_SUBDIR
    shards = sorted(target_dir.glob("model.safetensors-*-of-*.safetensors"))
    weights_bytes = sum(part.stat().st_size for part in shards)
    return f"inspector-qwen35 OK (Qwen3.5-9B {weights_bytes / 1024**3:.1f} GiB)"

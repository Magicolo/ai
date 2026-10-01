"""Phase 4 audio family pins + builders (DESIGN §§6, 37).

Split from `voyage.registry_records` (issue 082): the ACE-Step 1.5
music-stack row as importable pins with no cross-family coupling.
`voyage.registry_records` re-exports every name below so existing
importers keep working; new code imports from here directly.

- `ACE_*`: Hub pins + layout + floors + license (Phase 4 music stack).
- `_ACE_CHECKPOINTS_RELATIVE` / `_ACE_LM_RELATIVE`: derived layout
  relatives (kept with the pins they derive from).
- `_record_audio` / `_describe_audio`: manifest value + exact OK string.
- No EXPECTED ingest hash: this row carries none (same as the inspector
  row — the manifest record carries no sha).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue

# Phase 4 music stack (DESIGN §§6, 37). ACE-Step 1.5 turbo DiT + 0.6B
# planner LM (spec V1: 2B turbo + 0.6B LM, 8GB floor; XL rejected).
# Both repos ungated. Layout lesson from the Step 6 E2E: the handler's
# initialize_service gates on MAIN_MODEL_COMPONENTS (turbo + vae + text
# encoder + the 1.7B default LM) directly under <project>/checkpoints/,
# and auto-downloads the full 9.4GB bundle when anything is missing — so
# the registry pre-downloads ALL FOUR main components there (the 1.7B LM
# is load-bearing for the gate even though generation uses the 0.6B via
# an explicit lm_model_path), plus the 0.6B planner LM alongside.
ACE_MAIN_REPO = "ACE-Step/Ace-Step1.5"

ACE_MAIN_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"

ACE_MAIN_SUBDIR = "acestep"

ACE_CHECKPOINTS_SUBDIR = "checkpoints"

ACE_MAIN_ALLOW = [
    "acestep-v15-turbo/*",
    "vae/*",
    "Qwen3-Embedding-0.6B/*",
    "acestep-5Hz-lm-1.7B/*",
    "config.json",
]

ACE_TURBO_MIN_BYTES = 4_000_000_000

ACE_LM17_MIN_BYTES = 3_000_000_000

ACE_MAIN_LICENSE = "Apache 2.0 (upstream code repo; no LICENSE file in weight repo)"

ACE_LM_REPO = "ACE-Step/acestep-5Hz-lm-0.6B"

ACE_LM_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80"

ACE_LM_SUBDIR = "acestep-5Hz-lm-0.6B"

ACE_LM_ALLOW = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]

ACE_LM_MIN_BYTES = 1_000_000_000

_ACE_CHECKPOINTS_RELATIVE = f"{ACE_MAIN_SUBDIR}/{ACE_CHECKPOINTS_SUBDIR}"

_ACE_LM_RELATIVE = f"{ACE_MAIN_SUBDIR}/{ACE_CHECKPOINTS_SUBDIR}/{ACE_LM_SUBDIR}"


def _record_audio(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the Phase 4 music stack."""
    checkpoints_dir = models_dir / ACE_MAIN_SUBDIR / ACE_CHECKPOINTS_SUBDIR
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    ace_dir = models_dir / ACE_MAIN_SUBDIR
    planner_weights = checkpoints_dir / ACE_LM_SUBDIR / "model.safetensors"
    return {
        "repo": ACE_MAIN_REPO,
        "revision": ACE_MAIN_REVISION,
        "model_dir": str(ace_dir),
        "turbo_bytes": turbo_weights.stat().st_size if turbo_weights.exists() else 0,
        "license": ACE_MAIN_LICENSE,
        "planner_repo": ACE_LM_REPO,
        "planner_revision": ACE_LM_REVISION,
        "planner_dir": str(checkpoints_dir / ACE_LM_SUBDIR),
        "planner_bytes": planner_weights.stat().st_size if planner_weights.exists() else 0,
    }


def _describe_audio(models_dir: Path) -> str:
    """Exact OK string for the music stack (byte-stable)."""
    checkpoints_dir = models_dir / ACE_MAIN_SUBDIR / ACE_CHECKPOINTS_SUBDIR
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    planner_weights = checkpoints_dir / ACE_LM_SUBDIR / "model.safetensors"
    return (
        f"audio-acestep OK (turbo {turbo_weights.stat().st_size / 1024**3:.1f} GiB "
        f"+ planner LM {planner_weights.stat().st_size / 1024**3:.1f} GiB)"
    )

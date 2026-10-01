"""Phase 7 LTXV family pins + builders (DESIGN Phase 7, §§84-85).

Split from `voyage.registry_records` (issue 082): the LTXV 2B
distilled-DiT row as importable pins with no cross-family coupling.
`voyage.registry_records` re-exports every name below so existing
importers keep working; new code imports from here directly.

- `LTXV_*`: Hub pins + layout + floors + code commit (Phase 7).
- `EXPECTED_LTXV_*`: ingest-time hash pins (issue 071; provenance in
  `registry_records.py` — measured live 2026-09-30 from the provisioned
  volume at the pinned revision).
- `_record_ltxv` / `_describe_ltxv`: manifest value + exact OK string.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# Phase 7 LTXV alternative backend: 2B distilled DiT + spatial upscaler
# (0.9.8) plus the PixArt T5 tokenizer/encoder the pipeline needs for
# CPU-precomputed bf16 embeds. Both HF repos ungated; weights verified
# present in ~/.cache/voyage-models/ltxv-2b (Slice 1 probe). License: check
# the weight repo for the exact Lightricks community-license text.
LTXV_HF_REPO = "Lightricks/LTX-Video"

LTXV_HF_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"

LTXV_SUBDIR = "ltxv-2b"

LTXV_DIT_FILE = "ltxv-2b-0.9.8-distilled.safetensors"

LTXV_UPSC_FILE = "ltxv-spatial-upscaler-0.9.8.safetensors"

LTXV_DIT_MIN_BYTES = 6_000_000_000

LTXV_UPSC_MIN_BYTES = 400_000_000

LTXV_TE_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"

LTXV_TE_REVISION = "b89adadeccd9ead2adcb9fa2825d3fabec48d404"

LTXV_TE_SUBDIR = "PixArt-XL-2-1024-MS"

LTXV_TE_ALLOW = ["tokenizer/*", "text_encoder/*"]

LTXV_COMMIT = "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"

LTXV_COMMIT_SHORT = "4b2d053"

EXPECTED_LTXV_DIT_SHA256 = "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f"

EXPECTED_LTXV_UPSC_SHA256 = "5b076031c6f860db9037a54f3bb819f10bfb5532ea26a6d30062292428a0c208"


def _record_ltxv(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the Phase 7 LTXV stack."""
    ltxv_dir = models_dir / LTXV_SUBDIR
    dit_path = ltxv_dir / LTXV_DIT_FILE
    upsc_path = ltxv_dir / LTXV_UPSC_FILE
    return {
        "repo": LTXV_HF_REPO,
        "revision": LTXV_HF_REVISION,
        "model_dir": str(ltxv_dir),
        "checkpoint_bytes": dit_path.stat().st_size,
        "files": [LTXV_DIT_FILE, LTXV_UPSC_FILE],
        "code_commit": LTXV_COMMIT,
        "text_encoder_repo": LTXV_TE_REPO,
        "text_encoder_revision": LTXV_TE_REVISION,
        # Per-file shas (071): verify_model checks each against these so a
        # mutated weight fails ensure even though presence + floors pass.
        "checkpoint_shas": {
            f"{LTXV_SUBDIR}/{LTXV_DIT_FILE}": sha256_file(dit_path),
            f"{LTXV_SUBDIR}/{LTXV_UPSC_FILE}": sha256_file(upsc_path),
        },
    }


def _describe_ltxv(models_dir: Path) -> str:
    """Exact OK string for the LTXV stack (byte-stable)."""
    dit_path = models_dir / LTXV_SUBDIR / LTXV_DIT_FILE
    return f"ltxv-2b OK (DiT {dit_path.stat().st_size / 1024**3:.1f} GiB + upscaler)"

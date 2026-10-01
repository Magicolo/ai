"""LTX-2.5 family pins + builders (DESIGN §140; backend `ltxv` successor `ltx25`).

Split from `voyage.registry_records` (same pattern as issue 082): the
LTX-2.5 Q3 working-quant row as importable pins with no cross-family
coupling. `voyage.registry_records` re-exports every name below so
existing importers keep working; new code imports from here directly.

- `LTX25_*`: Hub pins + layout + floors + code commit (C1 primary,
  LTX2-REPORT.md recommendation; experiments E1-E4/S21-S30).
- `EXPECTED_LTX25_*`: ingest-time hash pins (issue 071 pattern;
  measured live 2026-10-01 from the consolidated volume at the pinned
  revisions, CPU-only sha256).
- `_record_ltx25` / `_describe_ltx25`: manifest value + exact OK string.

Stack (all implicit — no user-facing quant/TE/VAE knobs): Q3_K_M DiT
+ Gemma4 Q2_K text encoder + conv video VAE + audio VAE + spatial
upscaler (Mode A quality path default). E5 int2/qint2 files and the
diffusion VAE also live in the volume but are out of scope for this
spec (E5 deferred; conv is the default per S22).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# LTX-2.5 Q3 working quant (C1 primary). Worker:
# `voyage/workers/video_ltx25.py` (backend `ltx25`); ComfyUI pinned
# @2f35f4a + ComfyUI-GGUF @6ea2651 + gemma4 patch drives the GGUF path
# in-process. Provenance: LTX2.md E1-E4, LTX2-REPORT.md.
LTX25_DIT_REPO = "Abiray/LTX-2.5-Distilled-GGUF"

LTX25_DIT_REVISION = "7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0"

LTX25_SUBDIR = "ltx25"

LTX25_DIT_FILE = "LTX-2.5-Distilled-Q3_K_M.gguf"

LTX25_DIT_MIN_BYTES = 12_000_000_000

# Gemma4 12B text encoder with projection (GATED repo — downloads need
# a token with access; never put tokens in code). Provenance confirmed:
# local sha matches elix3r's published SHA256SUMS value.
LTX25_TE_REPO = "elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF"

LTX25_TE_REVISION = "2a18e836d286eb0570ec9f013c3591eb8c614d57"

LTX25_TE_FILE = "gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf"

LTX25_TE_MIN_BYTES = 5_500_000_000

# VAEs + spatial upscaler (GATED repo). Local layout mirrors the Hub:
# `hf_hub_download(..., local_dir=<models>/ltx25)` replicates the repo
# structure, so checks use the replicated subdir paths below
# (`vae/...`, `latent_upscale_models/...`).
LTX25_VAE_REPO = "Lightricks/LTX-2.5"

LTX25_VAE_REVISION = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"

LTX25_VIDEO_VAE_SUBFOLDER = "vae"

LTX25_VIDEO_VAE_FILE = "ltx-2.5-video-vae-conv-bf16.safetensors"

LTX25_VIDEO_VAE_MIN_BYTES = 1_300_000_000

LTX25_AUDIO_VAE_SUBFOLDER = "vae"

LTX25_AUDIO_VAE_FILE = "ltx-2.5-audio-vae-bf16.safetensors"

LTX25_AUDIO_VAE_MIN_BYTES = 300_000_000

LTX25_UPSC_SUBFOLDER = "latent_upscale_models"

LTX25_UPSC_FILE = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

LTX25_UPSC_MIN_BYTES = 900_000_000

LTX25_COMMIT = "2f35f4a08176d993cded35dac3332be4f7287f41"

LTX25_COMMIT_SHORT = "2f35f4a"

LTX25_GGUF_COMMIT = "6ea2651"

EXPECTED_LTX25_DIT_SHA256 = "f593274e67dc0c4714240539c57b76fc6ecac2256fbab9fb32b2a0f72b8d3dea"

EXPECTED_LTX25_TE_SHA256 = "a70012eeea2fbee7c9c058fca5710842ec8cb9d9e94e630f2d1a0647a8d300f6"

EXPECTED_LTX25_VIDEO_VAE_SHA256 = "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"

EXPECTED_LTX25_AUDIO_VAE_SHA256 = "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5"

EXPECTED_LTX25_UPSC_SHA256 = "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8"


def _record_ltx25(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the LTX-2.5 Q3 stack."""
    ltx25_dir = models_dir / LTX25_SUBDIR
    dit_path = ltx25_dir / LTX25_DIT_FILE
    te_path = ltx25_dir / LTX25_TE_FILE
    return {
        "repo": LTX25_DIT_REPO,
        "revision": LTX25_DIT_REVISION,
        "model_dir": str(ltx25_dir),
        "checkpoint_bytes": dit_path.stat().st_size,
        "files": [
            LTX25_DIT_FILE,
            LTX25_TE_FILE,
            f"{LTX25_VIDEO_VAE_SUBFOLDER}/{LTX25_VIDEO_VAE_FILE}",
            f"{LTX25_AUDIO_VAE_SUBFOLDER}/{LTX25_AUDIO_VAE_FILE}",
            f"{LTX25_UPSC_SUBFOLDER}/{LTX25_UPSC_FILE}",
        ],
        "code_commit": LTX25_COMMIT,
        "text_encoder_repo": LTX25_TE_REPO,
        "text_encoder_revision": LTX25_TE_REVISION,
        "vae_repo": LTX25_VAE_REPO,
        "vae_revision": LTX25_VAE_REVISION,
        # Per-file shas (071): verify_model checks each against these so a
        # mutated weight fails ensure even though presence + floors pass.
        "checkpoint_shas": {
            f"{LTX25_SUBDIR}/{LTX25_DIT_FILE}": sha256_file(dit_path),
            f"{LTX25_SUBDIR}/{LTX25_TE_FILE}": sha256_file(te_path),
        },
    }


def _describe_ltx25(models_dir: Path) -> str:
    """Exact OK string for the LTX-2.5 stack (byte-stable)."""
    dit_path = models_dir / LTX25_SUBDIR / LTX25_DIT_FILE
    return f"ltx25 OK (DiT {dit_path.stat().st_size / 1024**3:.1f} GiB + Gemma4 TE + VAEs)"

"""LTX-2.3 family pins + builders (DESIGN §140; backend `ltx23`).

Split from `voyage.registry_records` (same pattern as issue 082): the
LTX-2.3 Q3 working-quant row as importable pins with no cross-family
coupling. `voyage.registry_records` re-exports every name below so
existing importers keep working; new code imports from here directly.

- `LTX23_*`: Hub pins + layout + floors + code commit (C2 fallback,
  LTX2-REPORT.md recommendation; experiments E6-E9, TE-2, S22).
- `EXPECTED_LTX23_*`: ingest-time hash pins (issue 071 pattern;
  measured live 2026-10-01 from the consolidated volume at the pinned
  revisions, CPU-only sha256).
- `_record_ltx23` / `_describe_ltx23`: manifest value + exact OK string.

Stack (all implicit — no user-facing quant/TE/VAE knobs): Q3_K_M DiT
+ DualCLIP Gemma3-Q2_K backbone + connectors + distilled video/audio
VAEs. Mode A uses the SHARED spatial upscaler from the ltx25 volume
(no model duplication); the ltx23 spec checks its presence but does
not re-pin its hash (pinned once in `registry_ltx25.py`).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file
from voyage.registry_ltx25 import LTX25_SUBDIR, LTX25_UPSC_FILE

# LTX-2.3 Q3 working quant (C2 fallback — only family with a viable
# 2-GPU split per TE-2). Worker: `voyage/workers/video_ltx23.py`
# (backend `ltx23`); same pinned ComfyUI stack as ltx25. Provenance:
# LTX2.md E6-E9, LTX2-REPORT.md.
LTX23_DIT_REPO = "unsloth/LTX-2.3-GGUF"

LTX23_DIT_REVISION = "96e8ed4925ead3db9ff4d0084f165ef6a74f28d0"

LTX23_SUBDIR = "ltx23"

LTX23_DIT_SUBFOLDER = "distilled"

LTX23_DIT_FILE = "ltx-2.3-22b-distilled-Q3_K_M.gguf"

LTX23_DIT_MIN_BYTES = 10_000_000_000

# Gemma3 12B QAT backbone (ungated). DualCLIP pairs it with the
# connectors file below (`DualCLIPLoaderGGUF`, no patch needed —
# pinned ComfyUI supports the Gemma3 LTX path natively).
LTX23_TE_REPO = "unsloth/gemma-3-12b-it-qat-GGUF"

LTX23_TE_REVISION = "858acec7ec0541a46c39985c95d3b52d8f3ab183"

LTX23_TE_FILE = "gemma-3-12b-it-qat-Q2_K.gguf"

LTX23_TE_MIN_BYTES = 4_400_000_000

# Connectors + distilled VAEs (same repo as the DiT). Local layout
# mirrors the Hub: `hf_hub_download(..., local_dir=<models>/ltx23)`
# replicates the repo structure, so checks use the replicated subdir
# paths below (`distilled/`, `text_encoders/`, `vae/`).
LTX23_CONN_SUBFOLDER = "text_encoders"

LTX23_CONN_FILE = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"

LTX23_CONN_MIN_BYTES = 2_100_000_000

LTX23_VIDEO_VAE_SUBFOLDER = "vae"

LTX23_VIDEO_VAE_FILE = "ltx-2.3-22b-distilled_video_vae.safetensors"

LTX23_VIDEO_VAE_MIN_BYTES = 1_300_000_000

LTX23_AUDIO_VAE_SUBFOLDER = "vae"

LTX23_AUDIO_VAE_FILE = "ltx-2.3-22b-distilled_audio_vae.safetensors"

LTX23_AUDIO_VAE_MIN_BYTES = 300_000_000

LTX23_VAE_REVISION = LTX23_DIT_REVISION
"""Hub revision of both distilled VAEs (issue 285).

Video + audio VAEs ship in the same `unsloth/LTX-2.3-GGUF` repo as the
DiT, so one repo-level pin versions all three files — unlike ltx25,
whose VAEs live in the separate gated `Lightricks/LTX-2.5` repo with
its own `LTX25_VAE_REVISION`. A future split (VAEs moving repos) must
promote this to an independent literal.
"""

# Shared Mode-A upscaler (lives in the ltx25 volume — not duplicated).
LTX23_UPSC_RELATIVE_PATH = f"{LTX25_SUBDIR}/{LTX25_UPSC_FILE}"

LTX23_COMMIT = "2f35f4a08176d993cded35dac3332be4f7287f41"

LTX23_COMMIT_SHORT = "2f35f4a"

LTX23_GGUF_COMMIT = "6ea2651"

EXPECTED_LTX23_DIT_SHA256 = "388614a12f3d38c8bb08e42e92e5c73cb8cc1a1e5368b4cf02687ffa42c75269"

EXPECTED_LTX23_TE_SHA256 = "bc8e8b4de07b06795f0105b3dc5901352b3eb62ecc33ef67cac01b4d86a6d9c0"

EXPECTED_LTX23_CONN_SHA256 = "c61cbb396e2a8175d8b2da51f0fdac885a4ccd22c9f64dafa5aa2c455dc8a507"

EXPECTED_LTX23_VIDEO_VAE_SHA256 = "e68d6d8f8a42942ac9b862cc315beb3bc30805a8876c7ad63ba5bf7a2b8e168a"

EXPECTED_LTX23_AUDIO_VAE_SHA256 = "3cd6a6eb8cb28f5ecc12f1f3126952b2a3d2b0b42ad3270e63cefafafe0d9b57"


def _record_ltx23(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the LTX-2.3 Q3 stack."""
    ltx23_dir = models_dir / LTX23_SUBDIR
    dit_path = ltx23_dir / LTX23_DIT_SUBFOLDER / LTX23_DIT_FILE
    te_path = ltx23_dir / LTX23_TE_FILE
    conn_path = ltx23_dir / LTX23_CONN_SUBFOLDER / LTX23_CONN_FILE
    video_vae_path = ltx23_dir / LTX23_VIDEO_VAE_SUBFOLDER / LTX23_VIDEO_VAE_FILE
    audio_vae_path = ltx23_dir / LTX23_AUDIO_VAE_SUBFOLDER / LTX23_AUDIO_VAE_FILE
    return {
        "repo": LTX23_DIT_REPO,
        "revision": LTX23_DIT_REVISION,
        "model_dir": str(ltx23_dir),
        "checkpoint_bytes": dit_path.stat().st_size,
        "files": [
            f"{LTX23_DIT_SUBFOLDER}/{LTX23_DIT_FILE}",
            LTX23_TE_FILE,
            f"{LTX23_CONN_SUBFOLDER}/{LTX23_CONN_FILE}",
            f"{LTX23_VIDEO_VAE_SUBFOLDER}/{LTX23_VIDEO_VAE_FILE}",
            f"{LTX23_AUDIO_VAE_SUBFOLDER}/{LTX23_AUDIO_VAE_FILE}",
        ],
        "code_commit": LTX23_COMMIT,
        "text_encoder_repo": LTX23_TE_REPO,
        "text_encoder_revision": LTX23_TE_REVISION,
        "shared_upscaler": LTX23_UPSC_RELATIVE_PATH,
        # Per-file shas (071, issues 208/209): one entry per expected_hashes
        # path, so verify_model checks the whole stack. The DiT key carries
        # the distilled/ subfolder (208: the bare-subdir key never resolves);
        # the shared Mode-A upscaler is pinned once in registry_ltx25 and is
        # not re-recorded here.
        "checkpoint_shas": {
            f"{LTX23_SUBDIR}/{LTX23_DIT_SUBFOLDER}/{LTX23_DIT_FILE}": sha256_file(dit_path),
            f"{LTX23_SUBDIR}/{LTX23_TE_FILE}": sha256_file(te_path),
            f"{LTX23_SUBDIR}/{LTX23_CONN_SUBFOLDER}/{LTX23_CONN_FILE}": (sha256_file(conn_path)),
            f"{LTX23_SUBDIR}/{LTX23_VIDEO_VAE_SUBFOLDER}/{LTX23_VIDEO_VAE_FILE}": (
                sha256_file(video_vae_path)
            ),
            f"{LTX23_SUBDIR}/{LTX23_AUDIO_VAE_SUBFOLDER}/{LTX23_AUDIO_VAE_FILE}": (
                sha256_file(audio_vae_path)
            ),
        },
    }


def _describe_ltx23(models_dir: Path) -> str:
    """Exact OK string for the LTX-2.3 stack (byte-stable)."""
    dit_path = models_dir / LTX23_SUBDIR / LTX23_DIT_SUBFOLDER / LTX23_DIT_FILE
    return f"ltx23 OK (DiT {dit_path.stat().st_size / 1024**3:.1f} GiB + Gemma3 TE + VAEs)"

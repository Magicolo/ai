"""SonicMaster mastering family pins + builders (DESIGN §§84-85, §140 Track A).

Split pattern mirrors `voyage.registry_film` / `voyage.registry_rife`
(issue 082): the three-file mastering weights as importable pins with no
cross-family coupling. `voyage.registry_records` re-exports every name
below so existing importers keep working; new code imports from here
directly.

- `SONICMASTER_*`: public mastering model Hub pin + layout + floors +
  license (Track A).
- `SONICMASTER_VAE_*`: gated Stable Audio Open VAE Hub pin + layout +
  floors + license (Track A) — weights + config, both under `vae/`.
- `EXPECTED_SONICMASTER_*`: ingest-time hash pins (issue 071 pattern;
  measured digests — model + VAE weights are the Hub LFS oids at the
  pinned revisions, cross-checked against the ear-test probe files; the
  VAE config was downloaded at the pinned revision and hashed).
- `_record_mastering` / `_describe_mastering`: manifest value + exact
  OK string.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# SonicMaster mastering model (Track A): public single-file safetensors.
# Verified public 2026-10-07: the nateraw/sonicmaster mirror id is dead
# (401 unauthenticated / 404 + "Repository not found" via the API with a
# token), while amaai-lab/SonicMaster is public (private:false, gated:false,
# Apache-2.0) — and it is the exact file the ear-test probe ran (local
# sha256 matches the LFS oid below). Floor holds ~13% headroom below the
# expected ~3.45 GB file.
SONICMASTER_HF_REPO = "amaai-lab/SonicMaster"

SONICMASTER_HF_REVISION = "93c7fa73c4dd446c9e656af34524b5f6a416afda"

SONICMASTER_SUBDIR = "sonicmaster"

SONICMASTER_MODEL_FILE = "model.safetensors"

SONICMASTER_MODEL_REPO_PATH = f"{SONICMASTER_SUBDIR}/{SONICMASTER_MODEL_FILE}"

SONICMASTER_MODEL_MIN_BYTES = 3_000_000_000

SONICMASTER_LICENSE = "Apache-2.0"

SONICMASTER_LICENSE_URL = "https://huggingface.co/amaai-lab/SonicMaster"

# Stable Audio Open VAE (Track A): gated repo — downloads need a token
# with access; never put tokens in code. Verified 2026-10-07: the old
# revision pin is an invalid rev id and vae.safetensors-at-root does not
# exist — the real files live under vae/ at the revision below (repo
# gated:auto). Two files so the runner can load the VAE standalone:
# weights + config. Floors hold ~10% headroom below the expected ~624 MB
# weights and the 350-byte config.
SONICMASTER_VAE_REPO = "stabilityai/stable-audio-open-1.0"

SONICMASTER_VAE_REVISION = "f21265c1e2710b3bd2386596943f0007f55f802e"

SONICMASTER_VAE_SUBFOLDER = "vae"

SONICMASTER_VAE_FILE = "diffusion_pytorch_model.safetensors"

SONICMASTER_VAE_CONFIG_FILE = "config.json"

SONICMASTER_VAE_REPO_PATH = (
    f"{SONICMASTER_SUBDIR}/{SONICMASTER_VAE_SUBFOLDER}/{SONICMASTER_VAE_FILE}"
)

SONICMASTER_VAE_CONFIG_REPO_PATH = (
    f"{SONICMASTER_SUBDIR}/{SONICMASTER_VAE_SUBFOLDER}/{SONICMASTER_VAE_CONFIG_FILE}"
)

SONICMASTER_VAE_MIN_BYTES = 560_000_000

SONICMASTER_VAE_CONFIG_MIN_BYTES = 300

SONICMASTER_VAE_LICENSE = "Stability AI Community License (gated)"

SONICMASTER_VAE_LICENSE_URL = "https://huggingface.co/stabilityai/stable-audio-open-1.0"

# Measured digests (issue 071 shape, 64 lower hex): the model + VAE
# weights shas are the Hub LFS oids at the pinned revisions, each
# cross-checked against the local ear-test probe file (identical sha256 —
# the probe ran these exact bytes). The VAE config sha was downloaded at
# the pinned revision and hashed (350 bytes). Re-measure live from the
# provisioned volume if either revision ever moves — the gate in
# tests/test_registry_pins.py fails closed on any truncated pin, so never
# re-pin by hand, always re-download + re-hash.
EXPECTED_SONICMASTER_MODEL_SHA256 = (
    "d6af20753f79824321b62a797780ff04ff277590804793f56bd83a2f3b7a5ed4"
)

EXPECTED_SONICMASTER_VAE_SHA256 = "2131cdb52020b2473707465449d8bdb4f6cca61c93150a947baca02bc58ffd7b"

EXPECTED_SONICMASTER_VAE_CONFIG_SHA256 = (
    "858b6cce27aa1b2d1e8d331734d2d9690d9ea1dad2fb1e6bd23ee67a61058ad9"
)


def _record_mastering(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the SonicMaster mastering weights (Track A)."""
    mastering_dir = models_dir / SONICMASTER_SUBDIR
    model_path = mastering_dir / SONICMASTER_MODEL_FILE
    vae_dir = mastering_dir / SONICMASTER_VAE_SUBFOLDER
    vae_path = vae_dir / SONICMASTER_VAE_FILE
    vae_config_path = vae_dir / SONICMASTER_VAE_CONFIG_FILE
    return {
        "repo": SONICMASTER_HF_REPO,
        "revision": SONICMASTER_HF_REVISION,
        "model_dir": str(mastering_dir),
        "checkpoint_bytes": model_path.stat().st_size,
        "files": [
            SONICMASTER_MODEL_FILE,
            f"{SONICMASTER_VAE_SUBFOLDER}/{SONICMASTER_VAE_FILE}",
            f"{SONICMASTER_VAE_SUBFOLDER}/{SONICMASTER_VAE_CONFIG_FILE}",
        ],
        "license": SONICMASTER_LICENSE,
        "license_url": SONICMASTER_LICENSE_URL,
        "vae_repo": SONICMASTER_VAE_REPO,
        "vae_revision": SONICMASTER_VAE_REVISION,
        "vae_license": SONICMASTER_VAE_LICENSE,
        "vae_license_url": SONICMASTER_VAE_LICENSE_URL,
        # Per-file shas (071, issues 208/209): one entry per
        # expected_hashes path, so verify_model checks the whole stack.
        "checkpoint_shas": {
            SONICMASTER_MODEL_REPO_PATH: sha256_file(model_path),
            SONICMASTER_VAE_REPO_PATH: sha256_file(vae_path),
            SONICMASTER_VAE_CONFIG_REPO_PATH: sha256_file(vae_config_path),
        },
    }


def _describe_mastering(models_dir: Path) -> str:
    """Exact OK string for the SonicMaster weights (byte-stable)."""
    model_path = models_dir / SONICMASTER_SUBDIR / SONICMASTER_MODEL_FILE
    vae_path = models_dir / SONICMASTER_SUBDIR / SONICMASTER_VAE_SUBFOLDER / SONICMASTER_VAE_FILE
    size_gib = model_path.stat().st_size / 1024**3
    vae_mib = vae_path.stat().st_size / 1024**2
    return f"sonicmaster OK (model {size_gib:.1f} GiB + VAE {vae_mib:.0f} MiB)"

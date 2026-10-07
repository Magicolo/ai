"""SonicMaster mastering family pins + builders (DESIGN §§84-85, §140 Track A).

Split pattern mirrors `voyage.registry_film` / `voyage.registry_rife`
(issue 082): the two-file mastering weights as importable pins with no
cross-family coupling. `voyage.registry_records` re-exports every name
below so existing importers keep working; new code imports from here
directly.

- `SONICMASTER_*`: public mastering model Hub pin + layout + floors +
  license (Track A).
- `SONICMASTER_VAE_*`: gated Stable Audio Open VAE Hub pin + layout +
  floors + license (Track A).
- `EXPECTED_SONICMASTER_*`: ingest-time hash pins (issue 071 pattern;
  Track A placeholders — re-download + re-hash on first provision and
  replace these values with the measured digests, never hand-edit).
- `_record_mastering` / `_describe_mastering`: manifest value + exact
  OK string.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# SonicMaster mastering model (Track A): public single-file safetensors.
# Public repo (ungated) — downloads need no token. Floor holds ~10%
# headroom below the expected ~900 MB file.
SONICMASTER_HF_REPO = "nateraw/sonicmaster"

SONICMASTER_HF_REVISION = "a1765134e0808a8fb45e1a92874327e01d07cd75"

SONICMASTER_SUBDIR = "sonicmaster"

SONICMASTER_MODEL_FILE = "model.safetensors"

SONICMASTER_MODEL_REPO_PATH = f"{SONICMASTER_SUBDIR}/{SONICMASTER_MODEL_FILE}"

SONICMASTER_MODEL_MIN_BYTES = 800_000_000

SONICMASTER_LICENSE = "Apache-2.0"

SONICMASTER_LICENSE_URL = "https://huggingface.co/nateraw/sonicmaster"

# Stable Audio Open VAE (Track A): gated repo — downloads need a token
# with access; never put tokens in code. Floor holds ~10% headroom below
# the expected ~90 MB file.
SONICMASTER_VAE_REPO = "stabilityai/stable-audio-open-1.0"

SONICMASTER_VAE_REVISION = "9612befb10cc9992a2f82558bbb4735efaba3297"

SONICMASTER_VAE_SUBFOLDER = ""

SONICMASTER_VAE_FILE = "vae.safetensors"

SONICMASTER_VAE_REPO_PATH = f"{SONICMASTER_SUBDIR}/{SONICMASTER_VAE_FILE}"

SONICMASTER_VAE_MIN_BYTES = 80_000_000

SONICMASTER_VAE_LICENSE = "Stability AI Community License (gated)"

SONICMASTER_VAE_LICENSE_URL = "https://huggingface.co/stabilityai/stable-audio-open-1.0"

# Track A placeholder digests (issue 071 shape, 64 lower hex): sha256 of
# the Track A placeholder strings. Re-measure live from the provisioned
# volume at the pinned revisions on first provision and replace — the
# gate in tests/test_registry_pins.py fails closed on any truncated pin,
# so never re-pin by hand, always re-download + re-hash.
EXPECTED_SONICMASTER_MODEL_SHA256 = (
    "4065a37cbf96f70c7bb9eb0cfa21523a36da72ec4ca1afa53a1fb9535f1b82ef"
)

EXPECTED_SONICMASTER_VAE_SHA256 = "ff173be06f753055a3499c12aa9571ddb544c769971237b148c677765a77d7c6"


def _record_mastering(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the SonicMaster mastering weights (Track A)."""
    mastering_dir = models_dir / SONICMASTER_SUBDIR
    model_path = mastering_dir / SONICMASTER_MODEL_FILE
    vae_path = mastering_dir / SONICMASTER_VAE_FILE
    return {
        "repo": SONICMASTER_HF_REPO,
        "revision": SONICMASTER_HF_REVISION,
        "model_dir": str(mastering_dir),
        "checkpoint_bytes": model_path.stat().st_size,
        "files": [SONICMASTER_MODEL_FILE, SONICMASTER_VAE_FILE],
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
        },
    }


def _describe_mastering(models_dir: Path) -> str:
    """Exact OK string for the SonicMaster weights (byte-stable)."""
    model_path = models_dir / SONICMASTER_SUBDIR / SONICMASTER_MODEL_FILE
    vae_path = models_dir / SONICMASTER_SUBDIR / SONICMASTER_VAE_FILE
    size_gib = model_path.stat().st_size / 1024**3
    vae_mib = vae_path.stat().st_size / 1024**2
    return f"sonicmaster OK (model {size_gib:.1f} GiB + VAE {vae_mib:.0f} MiB)"

"""FILM interpolation family pins + builders (DESIGN §§84-85).

Split from `voyage.registry_records` (issue 082): the smallest MODEL_SPECS
row set — the single-file FILM interpolation weights — as importable pins
with no cross-family coupling. `voyage.registry_records` re-exports every
name below so existing importers keep working; new code imports from here
directly.

- `FILM_*`: Hub pin + layout + floors + license (Track C).
- `EXPECTED_FILM_SHA256`: ingest-time hash pin (issue 071; provenance in
  `registry_records.py` — measured live 2026-09-30 from the provisioned
  volume at the pinned revision).
- `_record_film` / `_describe_film`: manifest value + exact OK string.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# FILM: Comfy-Org repack of google-research/frame-interpolation (Apache 2.0)
# + hzwer/Practical-RIFE (MIT) — hence the repack's mit-and-apache-2.0 tag.
# fp16 weights (~66 MB); floor holds ~10% headroom below measured.
FILM_HF_REPO = "Comfy-Org/frame_interpolation"

FILM_HF_REVISION = "219da3c9d8c357ceaf457fc1d5932c6e861b8dee"

FILM_SUBDIR = "frame_interpolation"

FILM_FILE = "film_net_fp16.safetensors"

FILM_REPO_PATH = f"{FILM_SUBDIR}/{FILM_FILE}"

FILM_MIN_BYTES = 60_000_000

FILM_LICENSE = "MIT + Apache 2.0 (Comfy-Org repack tag mit-and-apache-2.0)"

FILM_LICENSE_URL = "https://huggingface.co/Comfy-Org/frame_interpolation"

EXPECTED_FILM_SHA256 = "f226e51375dc839d4b40e5c3d63da560dd1ea1c962364ec78f5adf2d05db05c0"


def _record_film(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the FILM interpolation weights (Track C)."""
    weights_path = models_dir / FILM_REPO_PATH
    return {
        "repo": FILM_HF_REPO,
        "revision": FILM_HF_REVISION,
        "model_dir": str(models_dir / FILM_SUBDIR),
        "checkpoint_bytes": weights_path.stat().st_size,
        "files": [FILM_REPO_PATH],
        "license": FILM_LICENSE,
        "license_url": FILM_LICENSE_URL,
        # Recorded sha (071): verify_model checks the checkpoint against it.
        "checkpoint_sha256": sha256_file(weights_path),
        "checkpoint_file": FILM_REPO_PATH,
    }


def _describe_film(models_dir: Path) -> str:
    """Exact OK string for the FILM weights (byte-stable)."""
    size_mib = (models_dir / FILM_REPO_PATH).stat().st_size / 1024**2
    return f"film OK (FILM fp16 {size_mib:.0f} MiB)"

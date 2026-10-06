"""RIFE interpolation family pins + builders (DESIGN §140 RIFE port).

Split from `voyage.registry_records` (same pattern as issue 082 / the FILM
split): the single-file RIFE interpolation weights as importable pins with
no cross-family coupling. `voyage.registry_records` re-exports every name
below so existing importers keep working; new code imports from here
directly.

- `RIFE_*`: Hub pin + layout + floors + license (Phase 1 RIFE port).
- `EXPECTED_RIFE_SHA256`: ingest-time hash pin (issue 071 pattern;
  measured live 2026-10-05 from the `/tmp` download at the pinned
  revision, before provisioning into the volume).
- `_record_rife` / `_describe_rife`: manifest value + exact OK string.

Why v4.25-heavy (not standard / 4.26 / lite): the 2026-10-05 best-quality
probe (top-motion kaolin pairs at 2048x1152, fp16) measured heavy as
sharpest (0.09257 vs FILM 0.09759) and closest to FILM (meandiff
0.02881), with identical peak VRAM (0.65 GiB, activation-dominated) and
noise-scale speed deltas vs the ~9-14x FILM margin; contact-sheet eyeball
on the highest-motion pair showed no ghosting, warping, or thin-line
shimmer. 4.26 has user-reported large-block artifacts; lite is softest.
The Practical-RIFE maintainer recommends the 4.25 family for most scenes
(more flow blocks, anime improved).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# RIFE: hzwer/Practical-RIFE (MIT), via the Comfy-Org repack at the same
# pinned revision as the FILM weights (siblings in one repo revision).
# fp16 heavy weights (~87 MB); floor holds ~10% headroom below measured.
RIFE_HF_REPO = "Comfy-Org/frame_interpolation"

RIFE_HF_REVISION = "219da3c9d8c357ceaf457fc1d5932c6e861b8dee"

RIFE_SUBDIR = "frame_interpolation"

RIFE_FILE = "rife_v4.25_heavy.safetensors"

RIFE_REPO_PATH = f"{RIFE_SUBDIR}/{RIFE_FILE}"

RIFE_MIN_BYTES = 78_000_000

RIFE_LICENSE = "MIT (hzwer/Practical-RIFE)"

RIFE_LICENSE_URL = "https://github.com/hzwer/Practical-RIFE"

EXPECTED_RIFE_SHA256 = "8d0f6be4655a7c1814f27c9e299677454cb2d348dcd16d0e2107fd9899ac7a7"


def _record_rife(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the RIFE interpolation weights (Phase 1 RIFE port)."""
    weights_path = models_dir / RIFE_REPO_PATH
    return {
        "repo": RIFE_HF_REPO,
        "revision": RIFE_HF_REVISION,
        "model_dir": str(models_dir / RIFE_SUBDIR),
        "checkpoint_bytes": weights_path.stat().st_size,
        "files": [RIFE_REPO_PATH],
        "license": RIFE_LICENSE,
        "license_url": RIFE_LICENSE_URL,
        # Recorded sha (071 pattern): verify_model checks the checkpoint.
        "checkpoint_sha256": sha256_file(weights_path),
        "checkpoint_file": RIFE_REPO_PATH,
    }


def _describe_rife(models_dir: Path) -> str:
    """Exact OK string for the RIFE weights (byte-stable)."""
    size_mib = (models_dir / RIFE_REPO_PATH).stat().st_size / 1024**2
    return f"rife OK (RIFE v4.25-heavy fp16 {size_mib:.0f} MiB)"

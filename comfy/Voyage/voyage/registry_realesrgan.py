"""Real-ESRGAN anime upscaler family pins + builders (DESIGN §§84-85).

Split from `voyage.registry_records` (issue 082): the next-smallest
single-file MODEL_SPECS row set — the Real-ESRGAN anime upscaler weights —
as importable pins with no cross-family coupling.
`voyage.registry_records` re-exports every name below so existing
importers keep working; new code imports from here directly.

- `REALESRGAN_*`: Hub pin + layout + floors + license (Track C).
- `EXPECTED_REALESRGAN_SHA256`: ingest-time hash pin (issue 071;
   provenance: measured live 2026-10-02 from the downloaded file at the
   pinned revision — matches the Hub LFS oid exactly).
- `_record_realesrgan` / `_describe_realesrgan`: manifest value + exact
  OK string.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# Real-ESRGAN anime-video-XS: xinntao/Real-ESRGAN v0.2.5.0 release asset
# (SRVGGNetCompact PReLU, 16 conv / 64 feat, native 4x, 2,504,012 bytes,
# BSD-3-Clause), re-hosted 1:1 on the Hub — xinntao ships no HF repo, so
# the registry pins the nateraw mirror (its LFS oid matches the measured
# sha256 b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d
# exactly at the pinned revision). Adopted 2026-10-02: ~11.5x faster
# than the RRDB anime-6B at 768x512 on the 2060 (0.055 vs 0.63 s/frame),
# same 2x recipe (native x4 + Lanczos 0.5). Floor holds ~15% headroom
# below measured.
REALESRGAN_HF_REPO = "nateraw/real-esrgan"

REALESRGAN_HF_REVISION = "44ad8adf6069185b86df22349b12f255821c86ab"

REALESRGAN_SUBDIR = "realesrgan"

REALESRGAN_ANIME_FILE = "realesr-animevideov3.pth"

REALESRGAN_ANIME_MIN_BYTES = 2_100_000

REALESRGAN_UPSTREAM_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
)

REALESRGAN_LICENSE = "BSD 3-Clause"

REALESRGAN_LICENSE_URL = "https://huggingface.co/nateraw/real-esrgan/blob/main/LICENSE"

EXPECTED_REALESRGAN_SHA256 = "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d"


def _record_realesrgan(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the Real-ESRGAN anime upscaler weights (Track C)."""
    weights_path = models_dir / REALESRGAN_SUBDIR / REALESRGAN_ANIME_FILE
    relative_path = f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}"
    return {
        "repo": REALESRGAN_HF_REPO,
        "revision": REALESRGAN_HF_REVISION,
        "model_dir": str(models_dir / REALESRGAN_SUBDIR),
        "checkpoint_bytes": weights_path.stat().st_size,
        "files": [REALESRGAN_ANIME_FILE],
        "upstream_url": REALESRGAN_UPSTREAM_URL,
        "license": REALESRGAN_LICENSE,
        "license_url": REALESRGAN_LICENSE_URL,
        # Recorded sha (071): verify_model checks the checkpoint against it.
        "checkpoint_sha256": sha256_file(weights_path),
        "checkpoint_file": relative_path,
    }


def _describe_realesrgan(models_dir: Path) -> str:
    """Exact OK string for the Real-ESRGAN anime weights (byte-stable)."""
    weights_path = models_dir / REALESRGAN_SUBDIR / REALESRGAN_ANIME_FILE
    size_mib = weights_path.stat().st_size / 1024**2
    return f"realesrgan-anime OK (anime-video-XS {size_mib:.0f} MiB)"

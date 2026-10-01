"""Real-ESRGAN anime upscaler family pins + builders (DESIGN §§84-85).

Split from `voyage.registry_records` (issue 082): the next-smallest
single-file MODEL_SPECS row set — the Real-ESRGAN anime upscaler weights —
as importable pins with no cross-family coupling.
`voyage.registry_records` re-exports every name below so existing
importers keep working; new code imports from here directly.

- `REALESRGAN_*`: Hub pin + layout + floors + license (Track C).
- `EXPECTED_REALESRGAN_SHA256`: ingest-time hash pin (issue 071;
  provenance in `registry_records.py` — measured live 2026-09-30 from
  the provisioned volume at the pinned revision).
- `_record_realesrgan` / `_describe_realesrgan`: manifest value + exact
  OK string.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# Real-ESRGAN anime 6B: xinntao/Real-ESRGAN v0.2.2.4 release asset (RRDBNet
# 6-block, 4x, 17,938,799 bytes, BSD-3-Clause (c) 2021 Xintao Wang),
# re-hosted 1:1 on the Hub — xinntao ships no HF repo, so the registry pins
# the amd mirror (its card records the upstream release URL + sha256
# f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da).
# Floor holds ~15% headroom below measured.
REALESRGAN_HF_REPO = "amd/realesrgan-x4plus-anime-6b"

REALESRGAN_HF_REVISION = "b14ff5f8ecb5a4b56ce4049a58d0bca1f8814690"

REALESRGAN_SUBDIR = "realesrgan"

REALESRGAN_ANIME_FILE = "RealESRGAN_x4plus_anime_6B.pth"

REALESRGAN_ANIME_MIN_BYTES = 15_000_000

REALESRGAN_UPSTREAM_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
    "RealESRGAN_x4plus_anime_6B.pth"
)

REALESRGAN_LICENSE = "BSD 3-Clause (c) 2021 Xintao Wang"

REALESRGAN_LICENSE_URL = "https://huggingface.co/amd/realesrgan-x4plus-anime-6b/blob/main/LICENSE"

EXPECTED_REALESRGAN_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"


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
    return f"realesrgan-anime OK (anime 6B {size_mib:.0f} MiB)"

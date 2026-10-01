"""Stream D CausVid family pins + builders (DESIGN §5.4).

Split from `voyage.registry_records` (issue 082): the CausVid DMD
checkpoint + Wan2.1-T2V-1.3B base row as importable pins with no
cross-family coupling. `voyage.registry_records` re-exports every name
below so existing importers keep working; new code imports from here
directly.

- `CAUSVID_*`: upstream code commit + DMD checkpoint Hub pin + layout +
  floors + license (Stream D backend).
- `WAN21_*`: Wan2.1-T2V-1.3B base Hub pin + subset allow-list + floors +
  license (DiT arch / T5 encoder / VAE underneath the DMD checkpoint).
- `_record_causvid` / `_describe_causvid`: manifest value + exact OK
  string.
- No EXPECTED ingest hash: the CausVid DMD checkpoint carries none (the
  weight file was pruned 2026-09-24 — same open residual as the
  inspector row, see `registry_records.py`).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file

# Stream D CausVid backend (DESIGN §5.4, TASK §§19/23.4/25.3, §30.1).
# Worker: `voyage/workers/video_causvid.py` (backend `causvid`, registered
# in supervisor VIDEO_WORKER_MODULES + STREAMING_VIDEO_BACKENDS, CLI via
# `generate --backend causvid` / `models download/verify causvid`); this
# block pins the upstream sources and exposes the download/verify entry
# points mirroring the ltxv pattern.
# Pins probed 2026-09-24 (see docs/UPSTREAM_CAUSVID_NOTES.md for URLs,
# geometry/fps/overlap notes, license implications, open worker questions).
# Upstream code pin (git commit, not a floating branch; master HEAD at probe
# time — tip commit 2025-08-07 "Update README.md").
CAUSVID_COMMIT = "adb6a5ecd07666b4d0290042915c8406e6d5ce22"

CAUSVID_COMMIT_SHORT = "adb6a5e"

# DMD causal generator checkpoint (CC BY-NC-SA 4.0, ungated). The worker will
# strict-load `torch.load(<checkpoint>)['generator']` per the upstream
# long-video script; bidirectional/warp/ODE/LMDB siblings are skipped.
CAUSVID_HF_REPO = "tianweiy/CausVid"

CAUSVID_HF_REVISION = "b545eb2728fc9d1515023a270b847f7b24b3aa89"

CAUSVID_SUBDIR = "causvid"

CAUSVID_CHECKPOINT_SUBDIR = "autoregressive_checkpoint"

CAUSVID_CHECKPOINT_NAME = "model.pt"

CAUSVID_CHECKPOINT_FILE = f"{CAUSVID_CHECKPOINT_SUBDIR}/{CAUSVID_CHECKPOINT_NAME}"

# Measured 2026-09-24: autoregressive_checkpoint/model.pt is 11,352,649,716
# bytes (~10.6 GiB) — a full training snapshot keyed on ['generator'], not a
# params-only file, hence far above a 1.3B bf16 param count. Floor holds ~12%
# headroom below measured (same convention as the Wan2.1 subset floors).
CAUSVID_CKPT_MIN_BYTES = 10_000_000_000

CAUSVID_LICENSE = "CC BY-NC-SA 4.0 (non-commercial; share-alike on adaptations)"

CAUSVID_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en"

# Wan2.1-T2V-1.3B base providing the DiT arch, T5 encoder, tokenizer and VAE
# underneath the CausVid DMD checkpoint. Ungated, Apache 2.0. Downloaded as
# a subset (diffusion shard + VAE + T5 + tokenizer). File sizes measured from
# the HF API file listing at pin time (no download): DiT 5.68GB, VAE 508MB,
# T5 11.36GB.
WAN21_HF_REPO = "Wan-AI/Wan2.1-T2V-1.3B"

WAN21_HF_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"

WAN21_SUBDIR = "Wan2.1-T2V-1.3B"

WAN21_ALLOW = [
    "diffusion_pytorch_model.safetensors",
    "config.json",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
]

WAN21_DIT_MIN_BYTES = 5_000_000_000

WAN21_VAE_MIN_BYTES = 400_000_000

WAN21_T5_MIN_BYTES = 10_000_000_000

WAN21_LICENSE = "Apache 2.0"

WAN21_LICENSE_URL = "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/LICENSE.txt"


def _record_causvid(models_dir: Path) -> dict[str, JsonValue]:
    """Manifest value for the Stream D CausVid stack."""
    causvid_dir = models_dir / CAUSVID_SUBDIR
    checkpoint_path = causvid_dir / CAUSVID_CHECKPOINT_FILE
    wan21_dir = models_dir / WAN21_SUBDIR
    return {
        "repo": CAUSVID_HF_REPO,
        "revision": CAUSVID_HF_REVISION,
        "model_dir": str(causvid_dir),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "files": [CAUSVID_CHECKPOINT_FILE],
        "code_commit": CAUSVID_COMMIT,
        "license": CAUSVID_LICENSE,
        "license_url": CAUSVID_LICENSE_URL,
        "base_repo": WAN21_HF_REPO,
        "base_revision": WAN21_HF_REVISION,
        "base_dir": str(wan21_dir),
        "base_license": WAN21_LICENSE,
    }


def _describe_causvid(models_dir: Path) -> str:
    """Exact OK string for the CausVid stack (byte-stable)."""
    gib = (models_dir / CAUSVID_SUBDIR / CAUSVID_CHECKPOINT_FILE).stat().st_size / 1024**3
    return f"causvid OK (DMD {gib:.1f} GiB + Wan2.1-1.3B base)"

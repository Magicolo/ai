"""Pinned model registry (DESIGN §§84-85).

Every integration records provider/repo/revision/license/local-path/checksum
in the run manifest. Downloads are explicit (`voyage models download`) —
never from `voyage run`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Upstream code pin (git commit, not a floating branch).
LONGLIVE_COMMIT = "6b36d20ec6f7958d29d11a704dfa64611a9f2572"
LONGLIVE_COMMIT_SHORT = "6b36d20"

# Merged BF16 generator checkpoint (base AR + DMD LoRA merged). Ungated.
LONGLIVE_HF_REPO = "Efficient-Large-Model/LongLive-2.0-5B"
LONGLIVE_HF_REVISION = "8521079b863720a57c1a8d9b19c8d9e6ccb04c0f"
LONGLIVE_HF_FILE = "model_bf16.pt"
LONGLIVE_LICENSE = "NVIDIA Open Model License Agreement"
LONGLIVE_LICENSE_URL = (
    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
)

# Base Wan model providing T5 encoder, tokenizer, VAE and arch config.
# Ungated. Downloaded as a subset (diffusion shards + VAE + T5 + tokenizer).
WAN_HF_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_SUBDIR = "Wan2.2-TI2V-5B"
WAN_ALLOW = [
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "config.json",
    "configuration.json",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
]
WAN_LICENSE = "Apache 2.0 (see repo LICENSE; record exact text at download)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_longlive2_bf16(models_dir: Path) -> dict[str, Any]:
    """Explicit download (DESIGN §85). Returns a manifest-ready record dict."""
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    wan_dir = models_dir / "wan_models" / WAN_SUBDIR
    snapshot_download(
        repo_id=WAN_HF_REPO,
        local_dir=str(wan_dir),
        allow_patterns=WAN_ALLOW,
    )
    blob = hf_hub_download(
        repo_id=LONGLIVE_HF_REPO,
        filename=LONGLIVE_HF_FILE,
        revision=LONGLIVE_HF_REVISION,
        local_dir=str(models_dir / "longlive2"),
    )
    generator_path = Path(blob)
    record = {
        "video": {
            "repo": LONGLIVE_HF_REPO,
            "revision": LONGLIVE_HF_REVISION,
            "model_id": LONGLIVE_HF_FILE,
            "checkpoint_sha256": _sha256(generator_path),
            "checkpoint_bytes": generator_path.stat().st_size,
            "license": LONGLIVE_LICENSE,
            "license_url": LONGLIVE_LICENSE_URL,
            "code_commit": LONGLIVE_COMMIT,
            "wan_repo": WAN_HF_REPO,
            "wan_dir": str(wan_dir),
            "wan_license": WAN_LICENSE,
        }
    }
    manifest_path = models_dir / "manifest.json"
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def verify_longlive2_bf16(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of every required weight file."""
    missing: list[str] = []
    generator = models_dir / "longlive2" / LONGLIVE_HF_FILE
    if not generator.exists() or generator.stat().st_size < 1_000_000_000:
        missing.append(str(generator))
    wan_dir = models_dir / "wan_models" / WAN_SUBDIR
    for pattern in WAN_ALLOW:
        if pattern.endswith("*"):
            matches = list(wan_dir.glob(pattern))
            if not matches:
                missing.append(f"{wan_dir}/{pattern}")
        elif not (wan_dir / pattern).exists():
            missing.append(str(wan_dir / pattern))
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    size_gib = generator.stat().st_size / 1024**3
    return True, f"longlive2-bf16 OK (generator {size_gib:.1f} GiB + Wan subset)"


def models_dir_layout(models_dir: Path) -> dict[str, str]:
    return {
        "wan_dir": str(models_dir / "wan_models" / WAN_SUBDIR),
        "generator_ckpt": str(models_dir / "longlive2" / LONGLIVE_HF_FILE),
        "manifest": str(models_dir / "manifest.json"),
    }

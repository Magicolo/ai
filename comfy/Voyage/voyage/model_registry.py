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

# Phase 3 director LLM (DESIGN §8). Qwen3-8B dense, Apache 2.0, ungated.
# BF16 weights (~16.4 GiB); served on CPU from system RAM in the director
# worker process — never on the video GPU.
QWEN_HF_REPO = "Qwen/Qwen3-8B"
QWEN_HF_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN_SUBDIR = "Qwen3-8B"
QWEN_ALLOW = [
    "model-0000[1-5]-of-00005.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]
QWEN_MIN_BYTES = 15_000_000_000
QWEN_LICENSE = "Apache 2.0"
QWEN_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE"

# Phase 5 visual inspector (DESIGN §§43-44, 100, 132). Qwen3.5-9B
# multimodal VLM, Apache 2.0, ungated. BF16 weights (~19 GiB); served on
# CPU from system RAM in the director worker — never on the video GPU.
# The allow-list MUST include chat_template.jinja: the Qwen3.5 processor
# needs it and snapshot_download without it fails the inspector load
# (Step 0 probe lesson). Shard names carry a `model.safetensors-` prefix
# (unlike Qwen3-8B's `model-` prefix), hence the distinct shard glob.
QWEN35_HF_REPO = "Qwen/Qwen3.5-9B"
QWEN35_HF_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN35_SUBDIR = "Qwen3.5-9B"
QWEN35_ALLOW = [
    "model.safetensors-*-of-*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]
QWEN35_MIN_BYTES = 18_000_000_000
QWEN35_LICENSE = "Apache 2.0"
QWEN35_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE"

# Phase 4 music stack (DESIGN §§6, 37). ACE-Step 1.5 turbo DiT + 0.6B
# planner LM (spec V1: 2B turbo + 0.6B LM, 8GB floor; XL rejected).
# Both repos ungated. Layout lesson from the Step 6 E2E: the handler's
# initialize_service gates on MAIN_MODEL_COMPONENTS (turbo + vae + text
# encoder + the 1.7B default LM) directly under <project>/checkpoints/,
# and auto-downloads the full 9.4GB bundle when anything is missing — so
# the registry pre-downloads ALL FOUR main components there (the 1.7B LM
# is load-bearing for the gate even though generation uses the 0.6B via
# an explicit lm_model_path), plus the 0.6B planner LM alongside.
ACE_MAIN_REPO = "ACE-Step/Ace-Step1.5"
ACE_MAIN_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
ACE_MAIN_SUBDIR = "acestep"
ACE_CHECKPOINTS_SUBDIR = "checkpoints"
ACE_MAIN_ALLOW = [
    "acestep-v15-turbo/*",
    "vae/*",
    "Qwen3-Embedding-0.6B/*",
    "acestep-5Hz-lm-1.7B/*",
    "config.json",
]
ACE_TURBO_MIN_BYTES = 4_000_000_000
ACE_LM17_MIN_BYTES = 3_000_000_000
ACE_MAIN_LICENSE = "Apache 2.0 (upstream code repo; no LICENSE file in weight repo)"
ACE_LM_REPO = "ACE-Step/acestep-5Hz-lm-0.6B"
ACE_LM_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80"
ACE_LM_SUBDIR = "acestep-5Hz-lm-0.6B"
ACE_LM_ALLOW = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]
ACE_LM_MIN_BYTES = 1_000_000_000
MINILM_HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_HF_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_SUBDIR = "all-MiniLM-L6-v2"
MINILM_ALLOW = [
    "model.safetensors",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]
MINILM_MIN_BYTES = 50_000_000
MINILM_LICENSE = "Apache 2.0"


# Phase 7 LTXV alternative backend: 2B distilled DiT + spatial upscaler
# (0.9.8) plus the PixArt T5 tokenizer/encoder the pipeline needs for
# CPU-precomputed bf16 embeds. Both HF repos ungated; weights verified
# present in ~/.cache/voyage-models/ltxv-2b (Slice 1 probe). License: check
# the weight repo for the exact Lightricks community-license text.
LTXV_HF_REPO = "Lightricks/LTX-Video"
LTXV_HF_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"
LTXV_SUBDIR = "ltxv-2b"
LTXV_DIT_FILE = "ltxv-2b-0.9.8-distilled.safetensors"
LTXV_UPSC_FILE = "ltxv-spatial-upscaler-0.9.8.safetensors"
LTXV_DIT_MIN_BYTES = 6_000_000_000
LTXV_UPSC_MIN_BYTES = 400_000_000
LTXV_TE_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"
LTXV_TE_REVISION = "b89adadeccd9ead2adcb9fa2825d3fabec48d404"
LTXV_TE_SUBDIR = "PixArt-XL-2-1024-MS"
LTXV_TE_ALLOW = ["tokenizer/*", "text_encoder/*"]
LTXV_COMMIT = "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"
LTXV_COMMIT_SHORT = "4b2d053"


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


def _merge_manifest_record(models_dir: Path, key: str, value: dict[str, Any]) -> dict[str, Any]:
    """Merge one record into models_dir/manifest.json (DESIGN §85)."""
    manifest_path = models_dir / "manifest.json"
    record: dict[str, Any] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            record = loaded
    record[key] = value
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def download_director_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 3 director stack (DESIGN §§8-9, 85).

    Qwen3-8B snapshot (bf16 shards + tokenizer) plus MiniLM-L6-v2
    (safetensors + tokenizer/configs, skips the onnx/openvino/tf extras).
    Merges into the shared manifest; returns the merged record.
    """
    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    qwen_dir = models_dir / QWEN_SUBDIR
    snapshot_download(
        repo_id=QWEN_HF_REPO,
        revision=QWEN_HF_REVISION,
        local_dir=str(qwen_dir),
        allow_patterns=QWEN_ALLOW,
    )
    minilm_dir = models_dir / MINILM_SUBDIR
    snapshot_download(
        repo_id=MINILM_HF_REPO,
        revision=MINILM_HF_REVISION,
        local_dir=str(minilm_dir),
        allow_patterns=MINILM_ALLOW,
    )
    qwen_shards = sorted(qwen_dir.glob("model-*-of-*.safetensors"))
    qwen_bytes = sum(p.stat().st_size for p in qwen_shards)
    minilm_weights = minilm_dir / "model.safetensors"
    record = _merge_manifest_record(
        models_dir,
        "director",
        {
            "repo": QWEN_HF_REPO,
            "revision": QWEN_HF_REVISION,
            "model_dir": str(qwen_dir),
            "checkpoint_bytes": qwen_bytes,
            "shards": [p.name for p in qwen_shards],
            "license": QWEN_LICENSE,
            "license_url": QWEN_LICENSE_URL,
            "embedding_repo": MINILM_HF_REPO,
            "embedding_revision": MINILM_HF_REVISION,
            "embedding_dir": str(minilm_dir),
            "embedding_bytes": minilm_weights.stat().st_size if minilm_weights.exists() else 0,
            "embedding_license": MINILM_LICENSE,
        },
    )
    return record


def verify_director_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the director stack."""
    missing: list[str] = []
    qwen_dir = models_dir / QWEN_SUBDIR
    qwen_shards = sorted(qwen_dir.glob("model-*-of-*.safetensors"))
    qwen_bytes = sum(p.stat().st_size for p in qwen_shards) if qwen_shards else 0
    for pattern in QWEN_ALLOW:
        if "[" in pattern:
            continue  # covered by the shard glob above
        if not (qwen_dir / pattern).exists():
            missing.append(str(qwen_dir / pattern))
    if qwen_bytes < QWEN_MIN_BYTES:
        missing.append(f"{qwen_dir}/model-*-of-*.safetensors ({qwen_bytes} bytes)")
    minilm_dir = models_dir / MINILM_SUBDIR
    minilm_weights = minilm_dir / "model.safetensors"
    if not minilm_weights.exists() or minilm_weights.stat().st_size < MINILM_MIN_BYTES:
        missing.append(str(minilm_weights))
    for pattern in MINILM_ALLOW[1:]:
        if "/" in pattern:
            if not (minilm_dir / pattern).exists():
                missing.append(str(minilm_dir / pattern))
        elif not (minilm_dir / pattern).exists():
            missing.append(str(minilm_dir / pattern))
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    return True, (
        f"director-qwen8b OK (Qwen3-8B {qwen_bytes / 1024**3:.1f} GiB + MiniLM "
        f"{minilm_weights.stat().st_size / 1024**2:.0f} MiB)"
    )


def download_inspector_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 5 VLM inspector (DESIGN §§43-44).

    Qwen3.5-9B multimodal snapshots into <models>/Qwen3.5-9B. The
    allow-list must include chat_template.jinja: without it the processor
    fails to build (Step 0 probe lesson). Merges into the shared manifest;
    returns the merged record.
    """
    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    target_dir = models_dir / QWEN35_SUBDIR
    snapshot_download(
        repo_id=QWEN35_HF_REPO,
        revision=QWEN35_HF_REVISION,
        local_dir=str(target_dir),
        allow_patterns=QWEN35_ALLOW,
    )
    shards = sorted(target_dir.glob("model.safetensors-*-of-*.safetensors"))
    weights_bytes = sum(p.stat().st_size for p in shards)
    record = _merge_manifest_record(
        models_dir,
        "inspector",
        {
            "repo": QWEN35_HF_REPO,
            "revision": QWEN35_HF_REVISION,
            "dir": str(target_dir),
            "bytes": weights_bytes,
            "license": QWEN35_LICENSE,
            "license_url": QWEN35_LICENSE_URL,
        },
    )
    return record


def verify_inspector_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the VLM inspector."""
    missing: list[str] = []
    target_dir = models_dir / QWEN35_SUBDIR
    shards = sorted(target_dir.glob("model.safetensors-*-of-*.safetensors"))
    weights_bytes = sum(p.stat().st_size for p in shards) if shards else 0
    for pattern in QWEN35_ALLOW:
        if "*" in pattern:
            continue  # covered by the shard glob above
        if not (target_dir / pattern).exists():
            missing.append(str(target_dir / pattern))
    if weights_bytes < QWEN35_MIN_BYTES:
        missing.append(f"{target_dir}/model shards ({weights_bytes} bytes)")
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    return True, f"inspector-qwen35 OK (Qwen3.5-9B {weights_bytes / 1024**3:.1f} GiB)"


def download_audio_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 4 music stack (DESIGN §§6, 37, 85).

    All four MAIN_MODEL_COMPONENTS (turbo DiT + VAE + text encoder + the
    1.7B default LM, which only satisfies the handler's gate) under
    <models>/acestep/checkpoints/ plus the 0.6B planner LM submodel as
    <models>/acestep/checkpoints/acestep-5Hz-lm-0.6B — the exact layout
    the ACE handler's initialize_service expects (Step 6 E2E lesson: any
    other layout triggers a full 9.4GB auto-download at first run).
    Merges into the shared manifest; returns the merged record.
    """
    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    ace_dir = models_dir / ACE_MAIN_SUBDIR
    checkpoints_dir = ace_dir / ACE_CHECKPOINTS_SUBDIR
    snapshot_download(
        repo_id=ACE_MAIN_REPO,
        revision=ACE_MAIN_REVISION,
        local_dir=str(checkpoints_dir),
        allow_patterns=ACE_MAIN_ALLOW,
    )
    snapshot_download(
        repo_id=ACE_LM_REPO,
        revision=ACE_LM_REVISION,
        local_dir=str(checkpoints_dir / ACE_LM_SUBDIR),
        allow_patterns=ACE_LM_ALLOW,
    )
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    lm_weights = checkpoints_dir / ACE_LM_SUBDIR / "model.safetensors"
    record = _merge_manifest_record(
        models_dir,
        "audio",
        {
            "repo": ACE_MAIN_REPO,
            "revision": ACE_MAIN_REVISION,
            "model_dir": str(ace_dir),
            "turbo_bytes": turbo_weights.stat().st_size if turbo_weights.exists() else 0,
            "license": ACE_MAIN_LICENSE,
            "planner_repo": ACE_LM_REPO,
            "planner_revision": ACE_LM_REVISION,
            "planner_dir": str(checkpoints_dir / ACE_LM_SUBDIR),
            "planner_bytes": lm_weights.stat().st_size if lm_weights.exists() else 0,
        },
    )
    return record


def verify_audio_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the music stack."""
    missing: list[str] = []
    checkpoints_dir = models_dir / ACE_MAIN_SUBDIR / ACE_CHECKPOINTS_SUBDIR
    for pattern in ACE_MAIN_ALLOW:
        if pattern.endswith("/*"):
            matches = list((checkpoints_dir / pattern[:-2]).glob("*"))
            if not matches:
                missing.append(f"{checkpoints_dir}/{pattern}")
        elif not (checkpoints_dir / pattern).exists():
            missing.append(str(checkpoints_dir / pattern))
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    if not turbo_weights.exists() or turbo_weights.stat().st_size < ACE_TURBO_MIN_BYTES:
        missing.append(str(turbo_weights))
    lm17_weights = checkpoints_dir / "acestep-5Hz-lm-1.7B" / "model.safetensors"
    if not lm17_weights.exists() or lm17_weights.stat().st_size < ACE_LM17_MIN_BYTES:
        missing.append(str(lm17_weights))
    lm_weights = checkpoints_dir / ACE_LM_SUBDIR / "model.safetensors"
    if not lm_weights.exists() or lm_weights.stat().st_size < ACE_LM_MIN_BYTES:
        missing.append(str(lm_weights))
    for pattern in ACE_LM_ALLOW[1:]:
        if not (checkpoints_dir / ACE_LM_SUBDIR / pattern).exists():
            missing.append(str(checkpoints_dir / ACE_LM_SUBDIR / pattern))
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    return True, (
        f"audio-acestep OK (turbo {turbo_weights.stat().st_size / 1024**3:.1f} GiB "
        f"+ planner LM {lm_weights.stat().st_size / 1024**3:.1f} GiB)"
    )


def download_ltxv_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 7 LTXV stack (DESIGN Phase 7).

    2B-distilled DiT + spatial upscaler from Lightricks/LTX-Video plus the
    PixArt T5 tokenizer/encoder subfolders the worker needs for
    CPU-precomputed bf16 embeds. Merges into the shared manifest; returns
    the merged record.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    ltxv_dir = models_dir / LTXV_SUBDIR
    ltxv_dir.mkdir(parents=True, exist_ok=True)
    for filename in (LTXV_DIT_FILE, LTXV_UPSC_FILE):
        hf_hub_download(
            repo_id=LTXV_HF_REPO,
            revision=LTXV_HF_REVISION,
            filename=filename,
            local_dir=str(ltxv_dir),
        )
    te_dir = models_dir / LTXV_TE_SUBDIR
    snapshot_download(
        repo_id=LTXV_TE_REPO,
        revision=LTXV_TE_REVISION,
        local_dir=str(te_dir),
        allow_patterns=LTXV_TE_ALLOW,
    )
    dit_path = ltxv_dir / LTXV_DIT_FILE
    record = _merge_manifest_record(
        models_dir,
        "ltxv",
        {
            "repo": LTXV_HF_REPO,
            "revision": LTXV_HF_REVISION,
            "model_dir": str(ltxv_dir),
            "checkpoint_bytes": dit_path.stat().st_size,
            "files": [LTXV_DIT_FILE, LTXV_UPSC_FILE],
            "code_commit": LTXV_COMMIT,
            "text_encoder_repo": LTXV_TE_REPO,
            "text_encoder_revision": LTXV_TE_REVISION,
        },
    )
    return record


def verify_ltxv_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the LTXV stack."""
    missing: list[str] = []
    ltxv_dir = models_dir / LTXV_SUBDIR
    dit_path = ltxv_dir / LTXV_DIT_FILE
    if not dit_path.exists() or dit_path.stat().st_size < LTXV_DIT_MIN_BYTES:
        missing.append(str(dit_path))
    upsc_path = ltxv_dir / LTXV_UPSC_FILE
    if not upsc_path.exists() or upsc_path.stat().st_size < LTXV_UPSC_MIN_BYTES:
        missing.append(str(upsc_path))
    te_dir = models_dir / LTXV_TE_SUBDIR
    for pattern in LTXV_TE_ALLOW:
        matches = list((te_dir / pattern[:-2]).glob("*"))
        if not matches:
            missing.append(f"{te_dir}/{pattern}")
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    return True, f"ltxv-2b OK (DiT {dit_path.stat().st_size / 1024**3:.1f} GiB + upscaler)"


def models_dir_layout(models_dir: Path) -> dict[str, str]:
    return {
        "wan_dir": str(models_dir / "wan_models" / WAN_SUBDIR),
        "generator_ckpt": str(models_dir / "longlive2" / LONGLIVE_HF_FILE),
        "ltxv_dir": str(models_dir / LTXV_SUBDIR),
        "qwen_dir": str(models_dir / QWEN_SUBDIR),
        "minilm_dir": str(models_dir / MINILM_SUBDIR),
        "acestep_dir": str(models_dir / ACE_MAIN_SUBDIR),
        "manifest": str(models_dir / "manifest.json"),
    }

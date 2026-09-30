"""Pinned model registry (DESIGN §§84-85).

Every integration records provider/repo/revision/license/local-path/checksum
in the run manifest. Downloads are explicit (`voyage models download`) —
never from `voyage run`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voyage.hashing import sha256_file

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


def _sha256(path: Path) -> str:
    """Legacy alias of :func:`voyage.hashing.sha256_file` (issue 021).

    Kept so existing callers keep working; new code imports hashing directly.
    """
    return sha256_file(path)


def verify_checkpoint_sha256(checkpoint: Path, expected_sha256: str) -> None:
    """Fail closed when a checkpoint disagrees with its recorded hash (005).

    Pure helper: hashes ``checkpoint`` and raises ``ValueError`` on any
    mismatch (possible tampering or truncated download). Exact hex compare
    (case-insensitive); empty expectations are rejected, never skipped.
    """
    if not expected_sha256:
        raise ValueError(f"no recorded sha256 for checkpoint {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"checkpoint {checkpoint} sha256 mismatch: expected {expected_sha256}, "
            f"got {actual} — refusing to torch.load an untrusted file"
        )


def verify_checkpoint_against_manifest(models_dir: Path, key: str, checkpoint: Path) -> None:
    """sha256-verify a checkpoint against the download manifest (005).

    Reads ``models_dir/manifest.json`` and, when it carries a
    ``checkpoint_sha256`` for ``key``, verifies ``checkpoint`` against it
    before any ``torch.load``. No manifest (or no sha for this key — e.g.
    volumes provisioned outside `voyage models download`) passes through
    so fresh provisioned stacks still load; a present sha that disagrees
    raises ``ValueError`` (fail closed).
    """
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        return
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return
    entry = loaded.get(key)
    if not isinstance(entry, dict):
        return
    recorded = entry.get("checkpoint_sha256")
    if not isinstance(recorded, str) or not recorded:
        return
    verify_checkpoint_sha256(checkpoint, recorded)


def download_longlive2_bf16(models_dir: Path) -> dict[str, Any]:
    """Explicit download (DESIGN §85). Returns a manifest-ready record dict."""
    return download_model(models_dir, "longlive2-bf16")


def verify_longlive2_bf16(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of every required weight file."""
    return verify_model(models_dir, "longlive2-bf16")


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


# Table-driven registry (issue 026): the six download/verify pairs differ
# only in hub parameters, file layouts and message strings, so the
# mechanics (hub calls, manifest merge, missing-list format, glob/size
# checks) live once in download_model/verify_model and each backend is one
# MODEL_SPECS row plus small record/message builders.


@dataclass(frozen=True)
class SnapshotSpec:
    """One snapshot_download call: repo pinned at revision into relative_dir."""

    repo_id: str
    revision: str | None  # None = floating (Wan2.2 has no pinned revision yet — see 011)
    relative_dir: str  # relative to models_dir
    allow_patterns: tuple[str, ...]


@dataclass(frozen=True)
class FileSpec:
    """One hf_hub_download call (single file, optional subfolder)."""

    repo_id: str
    revision: str
    filename: str
    subfolder: str  # "" when the file sits at the repo root
    relative_dir: str  # relative to models_dir


@dataclass(frozen=True)
class RequiredFile:
    """One presence (+ optional size-floor) check, in checklist order."""

    relative_path: str  # relative to models_dir
    min_bytes: int  # 0 = presence only


@dataclass(frozen=True)
class RequiredGlob:
    """One non-emptiness check: the relative glob must match >= 1 file."""

    relative_pattern: str  # relative to models_dir


@dataclass(frozen=True)
class ShardFloor:
    """One shard-glob byte-sum floor (director/inspector weight shards)."""

    relative_glob: str  # relative to models_dir
    min_bytes: int
    missing_label: str  # e.g. "model-*-of-*.safetensors" or "model shards"


@dataclass(frozen=True)
class ModelSpec:
    """One registry row: how to fetch it, check it and describe it."""

    name: str  # CLI target, e.g. "longlive2-bf16"
    manifest_key: str
    snapshots: tuple[SnapshotSpec, ...]
    files: tuple[FileSpec, ...]
    record_builder: Callable[[Path], dict[str, Any]]  # manifest value for the key
    checks: tuple[RequiredFile | RequiredGlob | ShardFloor, ...]  # in order
    success_message: Callable[[Path], str]  # exact OK string (byte-stable)


_WAN22_RELATIVE = f"wan_models/{WAN_SUBDIR}"
_ACE_CHECKPOINTS_RELATIVE = f"{ACE_MAIN_SUBDIR}/{ACE_CHECKPOINTS_SUBDIR}"
_ACE_LM_RELATIVE = f"{ACE_MAIN_SUBDIR}/{ACE_CHECKPOINTS_SUBDIR}/{ACE_LM_SUBDIR}"


def _record_longlive2(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the LongLive 2.0 stack (fetch part lives in the table)."""
    wan_dir = models_dir / "wan_models" / WAN_SUBDIR
    generator_path = models_dir / "longlive2" / LONGLIVE_HF_FILE
    return {
        "repo": LONGLIVE_HF_REPO,
        "revision": LONGLIVE_HF_REVISION,
        "model_id": LONGLIVE_HF_FILE,
        "checkpoint_sha256": sha256_file(generator_path),
        "checkpoint_bytes": generator_path.stat().st_size,
        "license": LONGLIVE_LICENSE,
        "license_url": LONGLIVE_LICENSE_URL,
        "code_commit": LONGLIVE_COMMIT,
        "wan_repo": WAN_HF_REPO,
        "wan_dir": str(wan_dir),
        "wan_license": WAN_LICENSE,
    }


def _record_director(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the Phase 3 director stack."""
    qwen_dir = models_dir / QWEN_SUBDIR
    qwen_shards = sorted(qwen_dir.glob("model-*-of-*.safetensors"))
    qwen_bytes = sum(part.stat().st_size for part in qwen_shards)
    minilm_dir = models_dir / MINILM_SUBDIR
    minilm_weights = minilm_dir / "model.safetensors"
    return {
        "repo": QWEN_HF_REPO,
        "revision": QWEN_HF_REVISION,
        "model_dir": str(qwen_dir),
        "checkpoint_bytes": qwen_bytes,
        "shards": [part.name for part in qwen_shards],
        "license": QWEN_LICENSE,
        "license_url": QWEN_LICENSE_URL,
        "embedding_repo": MINILM_HF_REPO,
        "embedding_revision": MINILM_HF_REVISION,
        "embedding_dir": str(minilm_dir),
        "embedding_bytes": minilm_weights.stat().st_size if minilm_weights.exists() else 0,
        "embedding_license": MINILM_LICENSE,
    }


def _record_inspector(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the Phase 5 VLM inspector."""
    target_dir = models_dir / QWEN35_SUBDIR
    shards = sorted(target_dir.glob("model.safetensors-*-of-*.safetensors"))
    weights_bytes = sum(part.stat().st_size for part in shards)
    return {
        "repo": QWEN35_HF_REPO,
        "revision": QWEN35_HF_REVISION,
        "dir": str(target_dir),
        "bytes": weights_bytes,
        "license": QWEN35_LICENSE,
        "license_url": QWEN35_LICENSE_URL,
    }


def _record_audio(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the Phase 4 music stack."""
    checkpoints_dir = models_dir / ACE_MAIN_SUBDIR / ACE_CHECKPOINTS_SUBDIR
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    ace_dir = models_dir / ACE_MAIN_SUBDIR
    planner_weights = checkpoints_dir / ACE_LM_SUBDIR / "model.safetensors"
    return {
        "repo": ACE_MAIN_REPO,
        "revision": ACE_MAIN_REVISION,
        "model_dir": str(ace_dir),
        "turbo_bytes": turbo_weights.stat().st_size if turbo_weights.exists() else 0,
        "license": ACE_MAIN_LICENSE,
        "planner_repo": ACE_LM_REPO,
        "planner_revision": ACE_LM_REVISION,
        "planner_dir": str(checkpoints_dir / ACE_LM_SUBDIR),
        "planner_bytes": planner_weights.stat().st_size if planner_weights.exists() else 0,
    }


def _record_ltxv(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the Phase 7 LTXV stack."""
    ltxv_dir = models_dir / LTXV_SUBDIR
    dit_path = ltxv_dir / LTXV_DIT_FILE
    return {
        "repo": LTXV_HF_REPO,
        "revision": LTXV_HF_REVISION,
        "model_dir": str(ltxv_dir),
        "checkpoint_bytes": dit_path.stat().st_size,
        "files": [LTXV_DIT_FILE, LTXV_UPSC_FILE],
        "code_commit": LTXV_COMMIT,
        "text_encoder_repo": LTXV_TE_REPO,
        "text_encoder_revision": LTXV_TE_REVISION,
    }


def _record_causvid(models_dir: Path) -> dict[str, Any]:
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


def _describe_longlive2(models_dir: Path) -> str:
    """Exact OK string for the LongLive 2.0 stack (byte-stable)."""
    size_gib = (models_dir / "longlive2" / LONGLIVE_HF_FILE).stat().st_size / 1024**3
    return f"longlive2-bf16 OK (generator {size_gib:.1f} GiB + Wan subset)"


def _describe_director(models_dir: Path) -> str:
    """Exact OK string for the director stack (byte-stable)."""
    qwen_dir = models_dir / QWEN_SUBDIR
    qwen_shards = sorted(qwen_dir.glob("model-*-of-*.safetensors"))
    qwen_bytes = sum(part.stat().st_size for part in qwen_shards)
    minilm_weights = models_dir / MINILM_SUBDIR / "model.safetensors"
    return (
        f"director-qwen8b OK (Qwen3-8B {qwen_bytes / 1024**3:.1f} GiB + MiniLM "
        f"{minilm_weights.stat().st_size / 1024**2:.0f} MiB)"
    )


def _describe_inspector(models_dir: Path) -> str:
    """Exact OK string for the VLM inspector (byte-stable)."""
    target_dir = models_dir / QWEN35_SUBDIR
    shards = sorted(target_dir.glob("model.safetensors-*-of-*.safetensors"))
    weights_bytes = sum(part.stat().st_size for part in shards)
    return f"inspector-qwen35 OK (Qwen3.5-9B {weights_bytes / 1024**3:.1f} GiB)"


def _describe_audio(models_dir: Path) -> str:
    """Exact OK string for the music stack (byte-stable)."""
    checkpoints_dir = models_dir / ACE_MAIN_SUBDIR / ACE_CHECKPOINTS_SUBDIR
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    planner_weights = checkpoints_dir / ACE_LM_SUBDIR / "model.safetensors"
    return (
        f"audio-acestep OK (turbo {turbo_weights.stat().st_size / 1024**3:.1f} GiB "
        f"+ planner LM {planner_weights.stat().st_size / 1024**3:.1f} GiB)"
    )


def _describe_ltxv(models_dir: Path) -> str:
    """Exact OK string for the LTXV stack (byte-stable)."""
    dit_path = models_dir / LTXV_SUBDIR / LTXV_DIT_FILE
    return f"ltxv-2b OK (DiT {dit_path.stat().st_size / 1024**3:.1f} GiB + upscaler)"


def _describe_causvid(models_dir: Path) -> str:
    """Exact OK string for the CausVid stack (byte-stable)."""
    gib = (models_dir / CAUSVID_SUBDIR / CAUSVID_CHECKPOINT_FILE).stat().st_size / 1024**3
    return f"causvid OK (DMD {gib:.1f} GiB + Wan2.1-1.3B base)"


MODEL_SPECS: dict[str, ModelSpec] = {
    "longlive2-bf16": ModelSpec(
        name="longlive2-bf16",
        manifest_key="video",
        snapshots=(SnapshotSpec(WAN_HF_REPO, None, _WAN22_RELATIVE, tuple(WAN_ALLOW)),),
        files=(
            FileSpec(LONGLIVE_HF_REPO, LONGLIVE_HF_REVISION, LONGLIVE_HF_FILE, "", "longlive2"),
        ),
        record_builder=_record_longlive2,
        checks=(
            RequiredFile(f"longlive2/{LONGLIVE_HF_FILE}", 1_000_000_000),
            RequiredFile(
                f"{_WAN22_RELATIVE}/diffusion_pytorch_model-00001-of-00003.safetensors", 0
            ),
            RequiredFile(
                f"{_WAN22_RELATIVE}/diffusion_pytorch_model-00002-of-00003.safetensors", 0
            ),
            RequiredFile(
                f"{_WAN22_RELATIVE}/diffusion_pytorch_model-00003-of-00003.safetensors", 0
            ),
            RequiredFile(f"{_WAN22_RELATIVE}/diffusion_pytorch_model.safetensors.index.json", 0),
            RequiredFile(f"{_WAN22_RELATIVE}/config.json", 0),
            RequiredFile(f"{_WAN22_RELATIVE}/configuration.json", 0),
            RequiredFile(f"{_WAN22_RELATIVE}/Wan2.2_VAE.pth", 0),
            RequiredFile(f"{_WAN22_RELATIVE}/models_t5_umt5-xxl-enc-bf16.pth", 0),
            RequiredGlob(f"{_WAN22_RELATIVE}/google/umt5-xxl/*"),
        ),
        success_message=_describe_longlive2,
    ),
    "director-qwen8b": ModelSpec(
        name="director-qwen8b",
        manifest_key="director",
        snapshots=(
            SnapshotSpec(QWEN_HF_REPO, QWEN_HF_REVISION, QWEN_SUBDIR, tuple(QWEN_ALLOW)),
            SnapshotSpec(MINILM_HF_REPO, MINILM_HF_REVISION, MINILM_SUBDIR, tuple(MINILM_ALLOW)),
        ),
        files=(),
        record_builder=_record_director,
        checks=(
            RequiredFile(f"{QWEN_SUBDIR}/config.json", 0),
            RequiredFile(f"{QWEN_SUBDIR}/generation_config.json", 0),
            RequiredFile(f"{QWEN_SUBDIR}/tokenizer.json", 0),
            RequiredFile(f"{QWEN_SUBDIR}/tokenizer_config.json", 0),
            RequiredFile(f"{QWEN_SUBDIR}/vocab.json", 0),
            RequiredFile(f"{QWEN_SUBDIR}/merges.txt", 0),
            ShardFloor(
                f"{QWEN_SUBDIR}/model-*-of-*.safetensors",
                QWEN_MIN_BYTES,
                "model-*-of-*.safetensors",
            ),
            RequiredFile(f"{MINILM_SUBDIR}/model.safetensors", MINILM_MIN_BYTES),
            RequiredFile(f"{MINILM_SUBDIR}/config.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/config_sentence_transformers.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/sentence_bert_config.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/modules.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/1_Pooling/config.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/tokenizer.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/tokenizer_config.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/special_tokens_map.json", 0),
            RequiredFile(f"{MINILM_SUBDIR}/vocab.txt", 0),
        ),
        success_message=_describe_director,
    ),
    "inspector-qwen35": ModelSpec(
        name="inspector-qwen35",
        manifest_key="inspector",
        snapshots=(
            SnapshotSpec(QWEN35_HF_REPO, QWEN35_HF_REVISION, QWEN35_SUBDIR, tuple(QWEN35_ALLOW)),
        ),
        files=(),
        record_builder=_record_inspector,
        checks=(
            RequiredFile(f"{QWEN35_SUBDIR}/model.safetensors.index.json", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/config.json", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/tokenizer.json", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/tokenizer_config.json", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/vocab.json", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/merges.txt", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/chat_template.jinja", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/preprocessor_config.json", 0),
            RequiredFile(f"{QWEN35_SUBDIR}/video_preprocessor_config.json", 0),
            ShardFloor(
                f"{QWEN35_SUBDIR}/model.safetensors-*-of-*.safetensors",
                QWEN35_MIN_BYTES,
                "model shards",
            ),
        ),
        success_message=_describe_inspector,
    ),
    "audio-acestep": ModelSpec(
        name="audio-acestep",
        manifest_key="audio",
        snapshots=(
            SnapshotSpec(
                ACE_MAIN_REPO, ACE_MAIN_REVISION, _ACE_CHECKPOINTS_RELATIVE, tuple(ACE_MAIN_ALLOW)
            ),
            SnapshotSpec(ACE_LM_REPO, ACE_LM_REVISION, _ACE_LM_RELATIVE, tuple(ACE_LM_ALLOW)),
        ),
        files=(),
        record_builder=_record_audio,
        checks=(
            RequiredGlob(f"{_ACE_CHECKPOINTS_RELATIVE}/acestep-v15-turbo/*"),
            RequiredGlob(f"{_ACE_CHECKPOINTS_RELATIVE}/vae/*"),
            RequiredGlob(f"{_ACE_CHECKPOINTS_RELATIVE}/Qwen3-Embedding-0.6B/*"),
            RequiredGlob(f"{_ACE_CHECKPOINTS_RELATIVE}/acestep-5Hz-lm-1.7B/*"),
            RequiredFile(f"{_ACE_CHECKPOINTS_RELATIVE}/config.json", 0),
            RequiredFile(
                f"{_ACE_CHECKPOINTS_RELATIVE}/acestep-v15-turbo/model.safetensors",
                ACE_TURBO_MIN_BYTES,
            ),
            RequiredFile(
                f"{_ACE_CHECKPOINTS_RELATIVE}/acestep-5Hz-lm-1.7B/model.safetensors",
                ACE_LM17_MIN_BYTES,
            ),
            RequiredFile(f"{_ACE_LM_RELATIVE}/model.safetensors", ACE_LM_MIN_BYTES),
            RequiredFile(f"{_ACE_LM_RELATIVE}/config.json", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/tokenizer.json", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/tokenizer_config.json", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/vocab.json", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/merges.txt", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/special_tokens_map.json", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/added_tokens.json", 0),
            RequiredFile(f"{_ACE_LM_RELATIVE}/chat_template.jinja", 0),
        ),
        success_message=_describe_audio,
    ),
    "ltxv-2b": ModelSpec(
        name="ltxv-2b",
        manifest_key="ltxv",
        snapshots=(
            SnapshotSpec(LTXV_TE_REPO, LTXV_TE_REVISION, LTXV_TE_SUBDIR, tuple(LTXV_TE_ALLOW)),
        ),
        files=(
            FileSpec(LTXV_HF_REPO, LTXV_HF_REVISION, LTXV_DIT_FILE, "", LTXV_SUBDIR),
            FileSpec(LTXV_HF_REPO, LTXV_HF_REVISION, LTXV_UPSC_FILE, "", LTXV_SUBDIR),
        ),
        record_builder=_record_ltxv,
        checks=(
            RequiredFile(f"{LTXV_SUBDIR}/{LTXV_DIT_FILE}", LTXV_DIT_MIN_BYTES),
            RequiredFile(f"{LTXV_SUBDIR}/{LTXV_UPSC_FILE}", LTXV_UPSC_MIN_BYTES),
            RequiredGlob(f"{LTXV_TE_SUBDIR}/tokenizer/*"),
            RequiredGlob(f"{LTXV_TE_SUBDIR}/text_encoder/*"),
        ),
        success_message=_describe_ltxv,
    ),
    "causvid": ModelSpec(
        name="causvid",
        manifest_key="causvid",
        snapshots=(
            SnapshotSpec(WAN21_HF_REPO, WAN21_HF_REVISION, WAN21_SUBDIR, tuple(WAN21_ALLOW)),
        ),
        files=(
            FileSpec(
                CAUSVID_HF_REPO,
                CAUSVID_HF_REVISION,
                CAUSVID_CHECKPOINT_NAME,
                CAUSVID_CHECKPOINT_SUBDIR,
                CAUSVID_SUBDIR,
            ),
        ),
        record_builder=_record_causvid,
        checks=(
            RequiredFile(f"{CAUSVID_SUBDIR}/{CAUSVID_CHECKPOINT_FILE}", CAUSVID_CKPT_MIN_BYTES),
            RequiredFile(f"{WAN21_SUBDIR}/diffusion_pytorch_model.safetensors", 0),
            RequiredFile(f"{WAN21_SUBDIR}/config.json", 0),
            RequiredFile(f"{WAN21_SUBDIR}/Wan2.1_VAE.pth", 0),
            RequiredFile(f"{WAN21_SUBDIR}/models_t5_umt5-xxl-enc-bf16.pth", 0),
            RequiredGlob(f"{WAN21_SUBDIR}/google/umt5-xxl/*"),
            RequiredFile(
                f"{WAN21_SUBDIR}/diffusion_pytorch_model.safetensors", WAN21_DIT_MIN_BYTES
            ),
            RequiredFile(f"{WAN21_SUBDIR}/Wan2.1_VAE.pth", WAN21_VAE_MIN_BYTES),
            RequiredFile(f"{WAN21_SUBDIR}/models_t5_umt5-xxl-enc-bf16.pth", WAN21_T5_MIN_BYTES),
        ),
        success_message=_describe_causvid,
    ),
}


def _require_spec(spec_name: str) -> ModelSpec:
    """Look up a spec by CLI target name (fail loud on unknown)."""
    try:
        return MODEL_SPECS[spec_name]
    except KeyError:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"unknown model spec {spec_name!r} (known: {known})") from None


def _run_spec_downloads(models_dir: Path, spec: ModelSpec) -> None:
    """Perform a spec's hub fetches (record building lives in download_model)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in spec.snapshots:
        snapshot_kwargs: dict[str, Any] = {
            "repo_id": snapshot.repo_id,
            "local_dir": str(models_dir / snapshot.relative_dir),
            "allow_patterns": list(snapshot.allow_patterns),
        }
        if snapshot.revision is not None:
            snapshot_kwargs["revision"] = snapshot.revision
        snapshot_download(**snapshot_kwargs)
    for filereq in spec.files:
        (models_dir / filereq.relative_dir).mkdir(parents=True, exist_ok=True)
        file_kwargs: dict[str, Any] = {
            "repo_id": filereq.repo_id,
            "revision": filereq.revision,
            "filename": filereq.filename,
            "local_dir": str(models_dir / filereq.relative_dir),
        }
        if filereq.subfolder:
            file_kwargs["subfolder"] = filereq.subfolder
        hf_hub_download(**file_kwargs)


def download_model(models_dir: Path, spec_name: str) -> dict[str, Any]:
    """Table-driven download: fetch a spec's files, merge its manifest record."""
    spec = _require_spec(spec_name)
    _run_spec_downloads(models_dir, spec)
    return _merge_manifest_record(models_dir, spec.manifest_key, spec.record_builder(models_dir))


def _collect_missing(
    models_dir: Path,
    spec: ModelSpec,
    checks: list[RequiredFile | RequiredGlob | ShardFloor] | None = None,
) -> list[str]:
    """Run a spec's checklist in order; missing entries as display strings.

    `checks` optionally narrows the run (per-snapshot presence); None runs
    the full spec checklist (verify path).
    """
    missing: list[str] = []
    for check in spec.checks if checks is None else checks:
        if isinstance(check, RequiredFile):
            candidate = models_dir / check.relative_path
            if check.min_bytes > 0:
                if not candidate.exists() or candidate.stat().st_size < check.min_bytes:
                    missing.append(str(candidate))
            elif not candidate.exists():
                missing.append(str(candidate))
        elif isinstance(check, RequiredGlob):
            if not list(models_dir.glob(check.relative_pattern)):
                missing.append(f"{models_dir}/{check.relative_pattern}")
        else:
            assert isinstance(check, ShardFloor)
            shards = sorted(models_dir.glob(check.relative_glob))
            shard_bytes = sum(part.stat().st_size for part in shards)
            if shard_bytes < check.min_bytes:
                shard_parent = models_dir / check.relative_glob.split("/")[0]
                missing.append(f"{shard_parent}/{check.missing_label} ({shard_bytes} bytes)")
    return missing


def verify_model(models_dir: Path, spec_name: str) -> tuple[bool, str]:
    """Table-driven verify: checklist first, then the spec's exact OK string."""
    spec = _require_spec(spec_name)
    missing = _collect_missing(models_dir, spec)
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    return True, spec.success_message(models_dir)


def download_director_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 3 director stack (DESIGN §§8-9, 85).

    Qwen3-8B snapshot (bf16 shards + tokenizer) plus MiniLM-L6-v2
    (safetensors + tokenizer/configs, skips the onnx/openvino/tf extras).
    Merges into the shared manifest; returns the merged record.
    """
    return download_model(models_dir, "director-qwen8b")


def verify_director_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the director stack."""
    return verify_model(models_dir, "director-qwen8b")


def download_inspector_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 5 VLM inspector (DESIGN §§43-44).

    Qwen3.5-9B multimodal snapshots into <models>/Qwen3.5-9B. The
    allow-list must include chat_template.jinja: without it the processor
    fails to build (Step 0 probe lesson). Merges into the shared manifest;
    returns the merged record.
    """
    return download_model(models_dir, "inspector-qwen35")


def verify_inspector_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the VLM inspector."""
    return verify_model(models_dir, "inspector-qwen35")


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
    return download_model(models_dir, "audio-acestep")


def verify_audio_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the music stack."""
    return verify_model(models_dir, "audio-acestep")


def download_ltxv_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Phase 7 LTXV stack (DESIGN Phase 7).

    2B-distilled DiT + spatial upscaler from Lightricks/LTX-Video plus the
    PixArt T5 tokenizer/encoder subfolders the worker needs for
    CPU-precomputed bf16 embeds. Merges into the shared manifest; returns
    the merged record.
    """
    return download_model(models_dir, "ltxv-2b")


def verify_ltxv_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the LTXV stack."""
    return verify_model(models_dir, "ltxv-2b")


def download_causvid_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Stream D CausVid stack (DESIGN §5.4).

    The autoregressive DMD checkpoint from tianweiy/CausVid plus the
    Wan2.1-T2V-1.3B base subset (DiT shard + VAE + T5 + tokenizer) the
    worker needs underneath it. Merges into the shared manifest; returns
    the merged record. Backs the `models download causvid` CLI target for
    the `causvid` worker (`voyage/workers/video_causvid.py`).
    """
    return download_model(models_dir, "causvid")


def verify_causvid_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the CausVid stack."""
    return verify_model(models_dir, "causvid")


def models_dir_layout(models_dir: Path) -> dict[str, str]:
    """Every models-tree root the registry downloads (086).

    Covers all shipped stacks: Wan2.2 + LongLive generator, Wan2.1 +
    CausVid DMD, LTXV DiT/upscaler + its PixArt text encoder, Qwen3-8B
    director, Qwen3.5-9B inspector, MiniLM embeddings, ACE-Step music.
    """
    return {
        "wan_dir": str(models_dir / "wan_models" / WAN_SUBDIR),
        "generator_ckpt": str(models_dir / "longlive2" / LONGLIVE_HF_FILE),
        "wan21_dir": str(models_dir / WAN21_SUBDIR),
        "causvid_dir": str(models_dir / CAUSVID_SUBDIR),
        "ltxv_dir": str(models_dir / LTXV_SUBDIR),
        "ltxv_text_encoder_dir": str(models_dir / LTXV_TE_SUBDIR),
        "qwen_dir": str(models_dir / QWEN_SUBDIR),
        "inspector_dir": str(models_dir / QWEN35_SUBDIR),
        "minilm_dir": str(models_dir / MINILM_SUBDIR),
        "acestep_dir": str(models_dir / ACE_MAIN_SUBDIR),
        "manifest": str(models_dir / "manifest.json"),
    }
class SnapshotRef:
    """A known hub snapshot's single /models home plus its owning spec.

    The owning spec drives fetch (`download_model`) and full-stack verify
    (`verify_model`); `relative_dir` is the snapshot's own directory for
    load-from-path and per-snapshot presence checks.
    """

    spec_name: str
    repo_id: str
    revision: str | None
    relative_dir: str


def resolve_snapshot(repo_id: str) -> SnapshotRef | None:
    """Map any known hub repo id to its /models snapshot (None when unknown).

    First spec wins; repo ids are unique across snapshot rows today. Only
    snapshot repos map — FileSpec single-file rows have no loadable
    directory, so they keep hub behavior.
    """
    for spec_name, spec in MODEL_SPECS.items():
        for snapshot in spec.snapshots:
            if snapshot.repo_id == repo_id:
                return SnapshotRef(
                    spec_name=spec_name,
                    repo_id=snapshot.repo_id,
                    revision=snapshot.revision,
                    relative_dir=snapshot.relative_dir,
                )
    return None


def snapshot_present(models_dir: Path, ref: SnapshotRef) -> bool:
    """True when the snapshot's own checklist passes (no cross-snapshot coupling).

    A qwen load must not fail just because the MiniLM side of its spec is
    missing (and vice versa) — each snapshot gates only its own files, so
    a partial volume still serves whatever is complete.
    """
    spec = _require_spec(ref.spec_name)
    prefix = ref.relative_dir.rstrip("/") + "/"

    def _belongs(check: RequiredFile | RequiredGlob | ShardFloor) -> bool:
        if isinstance(check, RequiredFile):
            return check.relative_path.startswith(prefix)
        if isinstance(check, RequiredGlob):
            return check.relative_pattern.startswith(prefix)
        return check.relative_glob.startswith(prefix)

    return not _collect_missing(models_dir, spec, [c for c in spec.checks if _belongs(c)])

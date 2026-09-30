"""Pinned model registry (DESIGN §§84-85).

Every integration records provider/repo/revision/license/local-path/checksum
in the run manifest. Downloads are explicit (`voyage models download`) —
never from `voyage run`.
"""

from __future__ import annotations

import json
import os
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
# Still floating (issue 070): every sibling snapshot pins a full 40-hex
# revision, but the Wan2.2 base downloads `main` at whatever it points to
# on provision day, so two provisions can yield different base weights with
# identical manifests. Pin procedure (needs network + provisioned bytes —
# the volume was pruned 2026-09-24, so nothing below is resolvable CPU-only):
#   1. resolve the verified main commit:
#      python -c "from huggingface_hub import HfApi;
#                 print(HfApi().model_info('Wan-AI/Wan2.2-TI2V-5B').sha)"
#   2. re-provision the Wan subset at that revision, sha256 the shards to
#      confirm they match the running volume, then set this constant to the
#      40-hex revision (the SnapshotSpec + manifest record already read it).
#   3. shrink tests/test_registry_pins.py's floating set to the empty set.
# Do NOT invent a hash — a wrong pin fails every provision loudly.
WAN_HF_REVISION: str | None = None
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

# GPU decider (unified worker image). Qwen3-4B AWQ-quantized, Apache 2.0,
# ungated. Single-shard int4 weights (~2.6 GiB); served on cuda:1 from the
# director venv — the 8B bf16 OOMs at materialization on a 6GB second GPU
# (5.49GiB > 5.6GB; probe 2026-09-30: 4B-AWQ loads in 2.5s, 2.8GiB peak,
# valid first-attempt JSON at temp 0.7).
QWEN4B_AWQ_HF_REPO = "Qwen/Qwen3-4B-AWQ"
QWEN4B_AWQ_HF_REVISION = "74d4bd2bd4bff9cafc9345221320bffb08b406a3"
QWEN4B_AWQ_SUBDIR = "Qwen3-4B-AWQ"
QWEN4B_AWQ_ALLOW = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]
QWEN4B_AWQ_MIN_BYTES = 2_000_000_000
QWEN4B_AWQ_LICENSE = "Apache 2.0"
QWEN4B_AWQ_LICENSE_URL = "https://huggingface.co/Qwen/Qwen3-4B-AWQ/blob/main/LICENSE"

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

# SFX effects stack (SFX slice 2, three-caption doctrine). Upstream
# hkchengrex/MMAudio (CVPR 2025 video-to-audio, CC-BY-NC-4.0
# non-commercial — same class as the CausVid DMD checkpoint): native
# .pth weights (no comfy-loader machinery), 44 kHz variants only (the
# pipeline is 44.1/48 kHz end to end). small_44k (157 M params, 601 MB)
# is the 2060 ladder candidate; large_44k_v2 (1.03 B, 3.9 GB,
# upstream-recommended) is the 4060 default.
# Code pin: main HEAD 2026-02-23 (docs-only tip commit — inference code
# untouched since the vendored ComfyUI copy). Weights pin: HF main
# 2026-02-19. The 44 kHz BigVGAN vocoder auto-downloads upstream from
# nvidia (pinned here instead — explicit, never at run time); CLIP text
# tower loads from the registry-pinned DFN5B .bin via open_clip's
# builtin ViT-H-14-378-quickgelu arch entry (no hub round-trip).
MMAUDIO_CODE_COMMIT = "974010a026c731054592d8f777218bd9d85a6c24"
MMAUDIO_CODE_COMMIT_SHORT = "974010a"
MMAUDIO_HF_REPO = "hkchengrex/MMAudio"
MMAUDIO_HF_REVISION = "eb13a1a98fdbec91753775c57b074ccdfc60587c"
MMAUDIO_SUBDIR = "mmaudio"
MMAUDIO_WEIGHT_FILES = (
    "weights/mmaudio_small_44k.pth",
    "weights/mmaudio_medium_44k.pth",
    "weights/mmaudio_large_44k_v2.pth",
)
MMAUDIO_EXT_FILES = (
    "ext_weights/v1-44.pth",
    "ext_weights/synchformer_state_dict.pth",
)
MMAUDIO_SMALL_MIN_BYTES = 500_000_000
MMAUDIO_MEDIUM_MIN_BYTES = 2_000_000_000
MMAUDIO_LARGE_MIN_BYTES = 3_400_000_000
MMAUDIO_VAE_MIN_BYTES = 1_000_000_000
MMAUDIO_SYNCHFORMER_MIN_BYTES = 800_000_000
MMAUDIO_LICENSE = "CC BY-NC 4.0 (non-commercial)"
MMAUDIO_LICENSE_URL = "https://huggingface.co/hkchengrex/MMAudio/blob/main/README.md"
MMAUDIO_VOCODER_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"
MMAUDIO_VOCODER_REVISION = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"
MMAUDIO_VOCODER_SUBDIR = f"{MMAUDIO_SUBDIR}/vocoder/bigvgan_v2_44khz_128band_512x"
MMAUDIO_VOCODER_ALLOW = (
    "*.py",
    "config.json",
    "bigvgan_generator.pt",
    "alias_free_activation/*",
)
MMAUDIO_VOCODER_MIN_BYTES = 400_000_000
MMAUDIO_VOCODER_LICENSE = "MIT"
MMAUDIO_CLIP_REPO = "apple/DFN5B-CLIP-ViT-H-14-384"
MMAUDIO_CLIP_REVISION = "01b771ed0d1395ca5ffdd279897d665ebe00dfd2"
MMAUDIO_CLIP_SUBDIR = f"{MMAUDIO_SUBDIR}/clip"
MMAUDIO_CLIP_ALLOW = ("open_clip_pytorch_model.bin", "config.json")
MMAUDIO_CLIP_MIN_BYTES = 3_000_000_000
MMAUDIO_CLIP_LICENSE = "Apple AMLR (research, see repo LICENSE)"


# Finalize-stage augmentation weights (Track C): FILM frame interpolation +
# Real-ESRGAN anime upscaler. Inference-only weights fetched at runtime via
# `voyage models download film/realesrgan-anime` (or pulled automatically by
# `generate`'s ensure step on CUDA backends) — the video image carries only
# the torch-native loaders (safetensors/Pillow leaf deps in
# worker/Dockerfile.video), never retraining code.
#
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
#
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


# Expected ingest hashes (issue 071): `download_model` verifies these BEFORE
# merging the manifest record, so a poisoned first fetch can never become the
# attested baseline. Provenance per row: the LongLive generator hash is
# manifest-attested (the pruned provisioned volume's manifest.json "video"
# record, itself fetched at the pinned LONGLIVE_HF_REVISION via FileSpec, so
# hub-side integrity held at fetch time); the LTXV/FILM/Real-ESRGAN hashes
# were measured live 2026-09-30 from the provisioned volume (all FileSpec
# pinned-revision fetches). No constant exists for the CausVid DMD checkpoint:
# its manifest record carries no sha and the weight file was pruned
# 2026-09-24 — re-provision, measure, and add it here (residual).
EXPECTED_LONGLIVE_SHA256 = "ec9063a44ea3c91e8ff55edcdd58dba3f1bcf6ac9091249629cb57fcebe35fd8"
EXPECTED_LTXV_DIT_SHA256 = "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f"
EXPECTED_LTXV_UPSC_SHA256 = "5b076031c6f860db9037a54f3bb819f10bfb5532ea26a6d30062292428a0c208"
EXPECTED_FILM_SHA256 = "f226e51375dc839d4b40e5c3d63da560dd1ea1c962364ec78f5adf2d05db05c0"
EXPECTED_REALESRGAN_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"


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


def verify_checkpoint_against_manifest(
    models_dir: Path,
    key: str,
    checkpoint: Path,
    *,
    allow_missing_manifest: bool = False,
) -> None:
    """sha256-verify a checkpoint against the download manifest (005, 071).

    Reads ``models_dir/manifest.json`` and, when it carries a
    ``checkpoint_sha256`` for ``key``, verifies ``checkpoint`` against it
    before any ``torch.load``. A present sha that disagrees raises
    ``ValueError`` (fail closed).

    Unknown-manifest + known-key (no manifest, no entry, or no sha for this
    key — e.g. volumes provisioned outside `voyage models download`) fails
    closed by default (071): pass ``allow_missing_manifest=True`` or set
    ``VOYAGE_ALLOW_MISSING_MANIFEST=1`` to keep the external-volume
    pass-through explicitly. Provisioned workers never opt in.
    """
    manifest_path = models_dir / "manifest.json"
    recorded: str | None = None
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            entry = loaded.get(key)
            if isinstance(entry, dict):
                maybe_sha = entry.get("checkpoint_sha256")
                if isinstance(maybe_sha, str) and maybe_sha:
                    recorded = maybe_sha
    if recorded is None:
        env_opt_in = os.environ.get("VOYAGE_ALLOW_MISSING_MANIFEST", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if allow_missing_manifest or env_opt_in:
            return
        raise ValueError(
            f"no recorded sha256 for {key} in {manifest_path} — refusing to load "
            f"{checkpoint} (provision with `voyage models download`, or opt in explicitly "
            "with allow_missing_manifest=True / VOYAGE_ALLOW_MISSING_MANIFEST=1 "
            "for external volumes)"
        )
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
    revision: str | None  # None = floating (Wan2.2 only — see 070)
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
class ExpectedHash:
    """One ingest-time hash pin: the file must match before first use (071)."""

    relative_path: str  # relative to models_dir
    expected_sha256: str  # 64-hex baseline (see EXPECTED_* provenance notes)


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
    expected_hashes: tuple[ExpectedHash, ...] = ()  # ingest pins (071, checked pre-merge)
    manifest_checkpoint: str | None = None  # checkpoint relpath carrying checkpoint_sha256


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
        "wan_revision": WAN_HF_REVISION,  # None = still floating (issue 070)
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
    upsc_path = ltxv_dir / LTXV_UPSC_FILE
    return {
        "repo": LTXV_HF_REPO,
        "revision": LTXV_HF_REVISION,
        "model_dir": str(ltxv_dir),
        "checkpoint_bytes": dit_path.stat().st_size,
        "files": [LTXV_DIT_FILE, LTXV_UPSC_FILE],
        "code_commit": LTXV_COMMIT,
        "text_encoder_repo": LTXV_TE_REPO,
        "text_encoder_revision": LTXV_TE_REVISION,
        # Per-file shas (071): verify_model checks each against these so a
        # mutated weight fails ensure even though presence + floors pass.
        "checkpoint_shas": {
            f"{LTXV_SUBDIR}/{LTXV_DIT_FILE}": sha256_file(dit_path),
            f"{LTXV_SUBDIR}/{LTXV_UPSC_FILE}": sha256_file(upsc_path),
        },
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


def _record_sfx(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the SFX effects stack."""
    sfx_dir = models_dir / MMAUDIO_SUBDIR
    large_weights = sfx_dir / "weights" / "mmaudio_large_44k_v2.pth"
    return {
        "repo": MMAUDIO_HF_REPO,
        "revision": MMAUDIO_HF_REVISION,
        "model_dir": str(sfx_dir),
        "variants": list(MMAUDIO_WEIGHT_FILES),
        "large_bytes": large_weights.stat().st_size if large_weights.exists() else 0,
        "code_commit": MMAUDIO_CODE_COMMIT,
        "license": MMAUDIO_LICENSE,
        "license_url": MMAUDIO_LICENSE_URL,
        "vocoder_repo": MMAUDIO_VOCODER_REPO,
        "vocoder_revision": MMAUDIO_VOCODER_REVISION,
        "vocoder_license": MMAUDIO_VOCODER_LICENSE,
        "clip_repo": MMAUDIO_CLIP_REPO,
        "clip_revision": MMAUDIO_CLIP_REVISION,
        "clip_license": MMAUDIO_CLIP_LICENSE,
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


def _record_director_awq(models_dir: Path) -> dict[str, Any]:
    """Manifest value for the GPU decider stack (4B-AWQ + MiniLM)."""
    qwen_dir = models_dir / QWEN4B_AWQ_SUBDIR
    weights = qwen_dir / "model.safetensors"
    minilm_dir = models_dir / MINILM_SUBDIR
    minilm_weights = minilm_dir / "model.safetensors"
    return {
        "repo": QWEN4B_AWQ_HF_REPO,
        "revision": QWEN4B_AWQ_HF_REVISION,
        "model_dir": str(qwen_dir),
        "checkpoint_bytes": weights.stat().st_size if weights.exists() else 0,
        "license": QWEN4B_AWQ_LICENSE,
        "license_url": QWEN4B_AWQ_LICENSE_URL,
        "embedding_repo": MINILM_HF_REPO,
        "embedding_revision": MINILM_HF_REVISION,
        "embedding_dir": str(minilm_dir),
        "embedding_bytes": minilm_weights.stat().st_size if minilm_weights.exists() else 0,
        "embedding_license": MINILM_LICENSE,
    }


def _describe_director_awq(models_dir: Path) -> str:
    """Exact OK string for the GPU decider stack (byte-stable)."""
    weights = models_dir / QWEN4B_AWQ_SUBDIR / "model.safetensors"
    minilm_weights = models_dir / MINILM_SUBDIR / "model.safetensors"
    return (
        f"director-qwen4b-awq OK (Qwen3-4B-AWQ {weights.stat().st_size / 1024**3:.1f} GiB "
        f"+ MiniLM {minilm_weights.stat().st_size / 1024**2:.0f} MiB)"
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


def _describe_sfx(models_dir: Path) -> str:
    """Exact OK string for the SFX stack (byte-stable)."""
    large_weights = models_dir / MMAUDIO_SUBDIR / "weights" / "mmaudio_large_44k_v2.pth"
    return (
        f"sfx-mmaudio OK (large {large_weights.stat().st_size / 1024**3:.1f} GiB + VAE/sync/CLIP)"
    )


def _describe_causvid(models_dir: Path) -> str:
    """Exact OK string for the CausVid stack (byte-stable)."""
    gib = (models_dir / CAUSVID_SUBDIR / CAUSVID_CHECKPOINT_FILE).stat().st_size / 1024**3
    return f"causvid OK (DMD {gib:.1f} GiB + Wan2.1-1.3B base)"


def _record_film(models_dir: Path) -> dict[str, Any]:
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


def _record_realesrgan(models_dir: Path) -> dict[str, Any]:
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


def _describe_film(models_dir: Path) -> str:
    """Exact OK string for the FILM weights (byte-stable)."""
    size_mib = (models_dir / FILM_REPO_PATH).stat().st_size / 1024**2
    return f"film OK (FILM fp16 {size_mib:.0f} MiB)"


def _describe_realesrgan(models_dir: Path) -> str:
    """Exact OK string for the Real-ESRGAN anime weights (byte-stable)."""
    weights_path = models_dir / REALESRGAN_SUBDIR / REALESRGAN_ANIME_FILE
    size_mib = weights_path.stat().st_size / 1024**2
    return f"realesrgan-anime OK (anime 6B {size_mib:.0f} MiB)"


MODEL_SPECS: dict[str, ModelSpec] = {
    "longlive2-bf16": ModelSpec(
        name="longlive2-bf16",
        manifest_key="video",
        snapshots=(SnapshotSpec(WAN_HF_REPO, WAN_HF_REVISION, _WAN22_RELATIVE, tuple(WAN_ALLOW)),),
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
        expected_hashes=(ExpectedHash(f"longlive2/{LONGLIVE_HF_FILE}", EXPECTED_LONGLIVE_SHA256),),
        manifest_checkpoint=f"longlive2/{LONGLIVE_HF_FILE}",
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
    "director-qwen4b-awq": ModelSpec(
        name="director-qwen4b-awq",
        manifest_key="director-awq",
        snapshots=(
            SnapshotSpec(
                QWEN4B_AWQ_HF_REPO,
                QWEN4B_AWQ_HF_REVISION,
                QWEN4B_AWQ_SUBDIR,
                tuple(QWEN4B_AWQ_ALLOW),
            ),
            SnapshotSpec(MINILM_HF_REPO, MINILM_HF_REVISION, MINILM_SUBDIR, tuple(MINILM_ALLOW)),
        ),
        files=(),
        record_builder=_record_director_awq,
        checks=(
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/model.safetensors", QWEN4B_AWQ_MIN_BYTES),
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/config.json", 0),
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/generation_config.json", 0),
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/tokenizer.json", 0),
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/tokenizer_config.json", 0),
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/vocab.json", 0),
            RequiredFile(f"{QWEN4B_AWQ_SUBDIR}/merges.txt", 0),
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
        success_message=_describe_director_awq,
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
    "sfx-mmaudio": ModelSpec(
        name="sfx-mmaudio",
        manifest_key="sfx",
        snapshots=(
            SnapshotSpec(
                MMAUDIO_VOCODER_REPO,
                MMAUDIO_VOCODER_REVISION,
                MMAUDIO_VOCODER_SUBDIR,
                MMAUDIO_VOCODER_ALLOW,
            ),
            SnapshotSpec(
                MMAUDIO_CLIP_REPO,
                MMAUDIO_CLIP_REVISION,
                MMAUDIO_CLIP_SUBDIR,
                MMAUDIO_CLIP_ALLOW,
            ),
        ),
        files=(
            FileSpec(
                MMAUDIO_HF_REPO,
                MMAUDIO_HF_REVISION,
                "weights/mmaudio_small_44k.pth",
                "",
                MMAUDIO_SUBDIR,
            ),
            FileSpec(
                MMAUDIO_HF_REPO,
                MMAUDIO_HF_REVISION,
                "weights/mmaudio_medium_44k.pth",
                "",
                MMAUDIO_SUBDIR,
            ),
            FileSpec(
                MMAUDIO_HF_REPO,
                MMAUDIO_HF_REVISION,
                "weights/mmaudio_large_44k_v2.pth",
                "",
                MMAUDIO_SUBDIR,
            ),
            FileSpec(
                MMAUDIO_HF_REPO, MMAUDIO_HF_REVISION, "ext_weights/v1-44.pth", "", MMAUDIO_SUBDIR
            ),
            FileSpec(
                MMAUDIO_HF_REPO,
                MMAUDIO_HF_REVISION,
                "ext_weights/synchformer_state_dict.pth",
                "",
                MMAUDIO_SUBDIR,
            ),
        ),
        record_builder=_record_sfx,
        checks=(
            RequiredFile(
                f"{MMAUDIO_SUBDIR}/weights/mmaudio_small_44k.pth", MMAUDIO_SMALL_MIN_BYTES
            ),
            RequiredFile(
                f"{MMAUDIO_SUBDIR}/weights/mmaudio_medium_44k.pth", MMAUDIO_MEDIUM_MIN_BYTES
            ),
            RequiredFile(
                f"{MMAUDIO_SUBDIR}/weights/mmaudio_large_44k_v2.pth", MMAUDIO_LARGE_MIN_BYTES
            ),
            RequiredFile(f"{MMAUDIO_SUBDIR}/ext_weights/v1-44.pth", MMAUDIO_VAE_MIN_BYTES),
            RequiredFile(
                f"{MMAUDIO_SUBDIR}/ext_weights/synchformer_state_dict.pth",
                MMAUDIO_SYNCHFORMER_MIN_BYTES,
            ),
            RequiredFile(
                f"{MMAUDIO_VOCODER_SUBDIR}/bigvgan_generator.pt", MMAUDIO_VOCODER_MIN_BYTES
            ),
            RequiredFile(f"{MMAUDIO_VOCODER_SUBDIR}/config.json", 0),
            RequiredFile(
                f"{MMAUDIO_CLIP_SUBDIR}/open_clip_pytorch_model.bin", MMAUDIO_CLIP_MIN_BYTES
            ),
            RequiredFile(f"{MMAUDIO_CLIP_SUBDIR}/config.json", 0),
        ),
        success_message=_describe_sfx,
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
        expected_hashes=(
            ExpectedHash(f"{LTXV_SUBDIR}/{LTXV_DIT_FILE}", EXPECTED_LTXV_DIT_SHA256),
            ExpectedHash(f"{LTXV_SUBDIR}/{LTXV_UPSC_FILE}", EXPECTED_LTXV_UPSC_SHA256),
        ),
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
        manifest_checkpoint=f"{CAUSVID_SUBDIR}/{CAUSVID_CHECKPOINT_FILE}",
    ),
    "film": ModelSpec(
        name="film",
        manifest_key="film",
        snapshots=(),
        files=(FileSpec(FILM_HF_REPO, FILM_HF_REVISION, FILM_REPO_PATH, "", ""),),
        record_builder=_record_film,
        checks=(RequiredFile(FILM_REPO_PATH, FILM_MIN_BYTES),),
        success_message=_describe_film,
        expected_hashes=(ExpectedHash(FILM_REPO_PATH, EXPECTED_FILM_SHA256),),
        manifest_checkpoint=FILM_REPO_PATH,
    ),
    "realesrgan-anime": ModelSpec(
        name="realesrgan-anime",
        manifest_key="realesrgan",
        snapshots=(),
        files=(
            FileSpec(
                REALESRGAN_HF_REPO,
                REALESRGAN_HF_REVISION,
                REALESRGAN_ANIME_FILE,
                "",
                REALESRGAN_SUBDIR,
            ),
        ),
        record_builder=_record_realesrgan,
        checks=(
            RequiredFile(
                f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}", REALESRGAN_ANIME_MIN_BYTES
            ),
        ),
        success_message=_describe_realesrgan,
        expected_hashes=(
            ExpectedHash(
                f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}",
                EXPECTED_REALESRGAN_SHA256,
            ),
        ),
        manifest_checkpoint=f"{REALESRGAN_SUBDIR}/{REALESRGAN_ANIME_FILE}",
    ),
}


@dataclass(frozen=True)
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
    """Table-driven download: fetch, pre-verify expected hashes, merge record.

    The hash check runs BEFORE the manifest merge (071): whatever the hub
    returned must match the pinned baseline, otherwise the poisoned bytes
    are rejected and never become the attested-good record.
    """
    spec = _require_spec(spec_name)
    _run_spec_downloads(models_dir, spec)
    for expected in spec.expected_hashes:
        verify_checkpoint_sha256(models_dir / expected.relative_path, expected.expected_sha256)
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


def _manifest_hash_mismatches(models_dir: Path, spec: ModelSpec) -> list[str]:
    """Files whose bytes disagree with a sha the manifest records (071).

    Covers every shape the record builders write: per-file ``checkpoint_shas``
    dicts (ltxv) and single ``checkpoint_sha256`` + ``manifest_checkpoint``
    rows (longlive2/causvid/film/realesrgan). No manifest, no entry, or no
    sha for this key means no baseline exists — nothing to check (the
    ingest-time constants in ``download_model`` and the load-time
    ``verify_checkpoint_against_manifest`` are the closed gates; torn or
    unreadable manifests are validate_run's territory, not verify's).
    """
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(loaded, dict):
        return []
    entry = loaded.get(spec.manifest_key)
    if not isinstance(entry, dict):
        return []
    mismatches: list[str] = []

    def _check(relative_path: str, expected: str) -> None:
        candidate = models_dir / relative_path
        if not candidate.is_file() or sha256_file(candidate).lower() != expected.lower():
            mismatches.append(str(candidate))

    shas = entry.get("checkpoint_shas")
    if isinstance(shas, dict):
        for relative_path, expected in shas.items():
            if isinstance(relative_path, str) and isinstance(expected, str) and expected:
                _check(relative_path, expected)
        return mismatches
    recorded = entry.get("checkpoint_sha256")
    if isinstance(recorded, str) and recorded and spec.manifest_checkpoint is not None:
        _check(spec.manifest_checkpoint, recorded)
    return mismatches


def verify_model(models_dir: Path, spec_name: str) -> tuple[bool, str]:
    """Table-driven verify: checklist, then recorded hashes, then OK string."""
    spec = _require_spec(spec_name)
    missing = _collect_missing(models_dir, spec)
    if missing:
        return False, f"missing {len(missing)} files: {missing[:5]}"
    mismatched = _manifest_hash_mismatches(models_dir, spec)
    if mismatched:
        return False, f"hash mismatch {len(mismatched)} files: {mismatched[:5]}"
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


def download_director_awq_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the GPU decider stack (Qwen3-4B-AWQ + MiniLM).

    Merges under the `director-awq` manifest key (never the `director` key
    the 8B stack owns — overwrite semantics would clobber it).
    """
    return download_model(models_dir, "director-qwen4b-awq")


def verify_director_awq_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the GPU decider stack."""
    return verify_model(models_dir, "director-qwen4b-awq")


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


def download_sfx_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the SFX effects stack (SFX slice 2).

    All three 44 kHz variants (small/medium/large_v2) + shared VAE +
    synchformer from hkchengrex/MMAudio, the nvidia 44 kHz BigVGAN
    vocoder snapshot, and the DFN5B CLIP weights — the exact layout
    `voyage.audio.mmaudio_sfx` loads (no hub round-trip at run time).
    Merges into the shared manifest; returns the merged record.
    """
    return download_model(models_dir, "sfx-mmaudio")


def verify_sfx_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the SFX stack."""
    return verify_model(models_dir, "sfx-mmaudio")


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


def download_film_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the FILM interpolation weights (Track C).

    Single fp16 file into <models>/frame_interpolation/ (ComfyUI layout).
    Merges into the shared manifest; returns the merged record. Backs the
    `models download film` CLI target.
    """
    return download_model(models_dir, "film")


def verify_film_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the FILM weights."""
    return verify_model(models_dir, "film")


def download_realesrgan_models(models_dir: Path) -> dict[str, Any]:
    """Explicit download of the Real-ESRGAN anime upscaler (Track C).

    Single .pth into <models>/realesrgan/. Merges into the shared manifest;
    returns the merged record. Backs the `models download realesrgan-anime`
    CLI target.
    """
    return download_model(models_dir, "realesrgan-anime")


def verify_realesrgan_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the Real-ESRGAN anime weights."""
    return verify_model(models_dir, "realesrgan-anime")


def models_dir_layout(models_dir: Path) -> dict[str, str]:
    """Every models-tree root the registry downloads (086).

    Covers all shipped stacks: Wan2.2 + LongLive generator, Wan2.1 +
    CausVid DMD, LTXV DiT/upscaler + its PixArt text encoder, Qwen3-8B
    director, Qwen3.5-9B inspector, MiniLM embeddings, ACE-Step music,
    MMAudio SFX, FILM interpolation, Real-ESRGAN anime upscaler.
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
        "sfx_dir": str(models_dir / MMAUDIO_SUBDIR),
        "film_dir": str(models_dir / FILM_SUBDIR),
        "realesrgan_dir": str(models_dir / REALESRGAN_SUBDIR),
        "manifest": str(models_dir / "manifest.json"),
    }

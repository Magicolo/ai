"""Per-family model pins + manifest record builders (DESIGN §§84-85).

Family module of the issue-082 split: every weight-bundle pin (repo,
revision, allow-lists, size floors, licenses) and its manifest
`_record_*` / human `_describe_*` builders live here, once. The generic
machinery (spec dataclasses, `MODEL_SPECS` assembly, download / verify /
manifest core) stays in `voyage.model_registry`, which re-exports every
name below for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file
from voyage.registry_film import (
    EXPECTED_FILM_SHA256 as EXPECTED_FILM_SHA256,
)
from voyage.registry_film import (
    FILM_FILE as FILM_FILE,
)
from voyage.registry_film import (
    FILM_HF_REPO as FILM_HF_REPO,
)
from voyage.registry_film import (
    FILM_HF_REVISION as FILM_HF_REVISION,
)
from voyage.registry_film import (
    FILM_LICENSE as FILM_LICENSE,
)
from voyage.registry_film import (
    FILM_LICENSE_URL as FILM_LICENSE_URL,
)
from voyage.registry_film import (
    FILM_MIN_BYTES as FILM_MIN_BYTES,
)
from voyage.registry_film import (
    FILM_REPO_PATH as FILM_REPO_PATH,
)
from voyage.registry_film import (
    FILM_SUBDIR as FILM_SUBDIR,
)
from voyage.registry_film import (
    _describe_film as _describe_film,
)
from voyage.registry_film import (
    _record_film as _record_film,
)

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

# Data-only snapshot (issue 072): the loader resolves exactly these two files
# via `BigVGANv2.from_pretrained(vocoder_dir)` with the class already imported
# from the pinned `/opt/mmaudio` clone (`MMAUDIO_CODE_COMMIT`) — snapshot
# `.py` files are never imported (`trust_remote_code` is never set), so the
# old `*.py` + `alias_free_activation/*` globs only widened the executable
# surface for no runtime benefit. No activation file from the snapshot is
# consumed (activation code ships in the clone); keep the list exact.
MMAUDIO_VOCODER_ALLOW = (
    "config.json",
    "bigvgan_generator.pt",
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
# FILM pins live in `voyage.registry_film` (issue 082; re-exported at the
# top so existing importers keep working).
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

# EXPECTED_FILM_SHA256 lives in `voyage.registry_film` (issue 082;
# re-exported at the top so existing importers keep working).

EXPECTED_REALESRGAN_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"

_WAN22_RELATIVE = f"wan_models/{WAN_SUBDIR}"

_ACE_CHECKPOINTS_RELATIVE = f"{ACE_MAIN_SUBDIR}/{ACE_CHECKPOINTS_SUBDIR}"

_ACE_LM_RELATIVE = f"{ACE_MAIN_SUBDIR}/{ACE_CHECKPOINTS_SUBDIR}/{ACE_LM_SUBDIR}"


def _record_longlive2(models_dir: Path) -> dict[str, JsonValue]:
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


def _record_director(models_dir: Path) -> dict[str, JsonValue]:
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


def _record_inspector(models_dir: Path) -> dict[str, JsonValue]:
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


def _record_audio(models_dir: Path) -> dict[str, JsonValue]:
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


def _record_ltxv(models_dir: Path) -> dict[str, JsonValue]:
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


def _record_sfx(models_dir: Path) -> dict[str, JsonValue]:
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


def _record_director_awq(models_dir: Path) -> dict[str, JsonValue]:
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


# FILM builders live in `voyage.registry_film` (issue 082; re-exported
# at the top so existing importers keep working).


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


# FILM describe helper lives in `voyage.registry_film` (issue 082;
# re-exported at the top so existing importers keep working).


def _describe_realesrgan(models_dir: Path) -> str:
    """Exact OK string for the Real-ESRGAN anime weights (byte-stable)."""
    weights_path = models_dir / REALESRGAN_SUBDIR / REALESRGAN_ANIME_FILE
    size_mib = weights_path.stat().st_size / 1024**2
    return f"realesrgan-anime OK (anime 6B {size_mib:.0f} MiB)"

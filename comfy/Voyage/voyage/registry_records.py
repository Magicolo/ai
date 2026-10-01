"""Per-family model pins + manifest record builders (DESIGN §§84-85).

Family module of the issue-082 split: every weight-bundle pin (repo,
revision, allow-lists, size floors, licenses) and its manifest
`_record_*` / human `_describe_*` builders live here, once. The generic
machinery (spec dataclasses, `MODEL_SPECS` assembly, download / verify /
manifest core) stays in `voyage.model_registry`, which re-exports every
name below for backward compatibility.
"""

from __future__ import annotations

from voyage.registry_audio import (
    _ACE_CHECKPOINTS_RELATIVE as _ACE_CHECKPOINTS_RELATIVE,
)
from voyage.registry_audio import (
    _ACE_LM_RELATIVE as _ACE_LM_RELATIVE,
)
from voyage.registry_audio import (
    ACE_CHECKPOINTS_SUBDIR as ACE_CHECKPOINTS_SUBDIR,
)
from voyage.registry_audio import (
    ACE_LM17_MIN_BYTES as ACE_LM17_MIN_BYTES,
)
from voyage.registry_audio import (
    ACE_LM_ALLOW as ACE_LM_ALLOW,
)
from voyage.registry_audio import (
    ACE_LM_MIN_BYTES as ACE_LM_MIN_BYTES,
)
from voyage.registry_audio import (
    ACE_LM_REPO as ACE_LM_REPO,
)
from voyage.registry_audio import (
    ACE_LM_REVISION as ACE_LM_REVISION,
)
from voyage.registry_audio import (
    ACE_LM_SUBDIR as ACE_LM_SUBDIR,
)
from voyage.registry_audio import (
    ACE_MAIN_ALLOW as ACE_MAIN_ALLOW,
)
from voyage.registry_audio import (
    ACE_MAIN_LICENSE as ACE_MAIN_LICENSE,
)
from voyage.registry_audio import (
    ACE_MAIN_REPO as ACE_MAIN_REPO,
)
from voyage.registry_audio import (
    ACE_MAIN_REVISION as ACE_MAIN_REVISION,
)
from voyage.registry_audio import (
    ACE_MAIN_SUBDIR as ACE_MAIN_SUBDIR,
)
from voyage.registry_audio import (
    ACE_TURBO_MIN_BYTES as ACE_TURBO_MIN_BYTES,
)
from voyage.registry_audio import (
    _describe_audio as _describe_audio,
)
from voyage.registry_audio import (
    _record_audio as _record_audio,
)
from voyage.registry_causvid import (
    CAUSVID_CHECKPOINT_FILE as CAUSVID_CHECKPOINT_FILE,
)
from voyage.registry_causvid import (
    CAUSVID_CHECKPOINT_NAME as CAUSVID_CHECKPOINT_NAME,
)
from voyage.registry_causvid import (
    CAUSVID_CHECKPOINT_SUBDIR as CAUSVID_CHECKPOINT_SUBDIR,
)
from voyage.registry_causvid import (
    CAUSVID_CKPT_MIN_BYTES as CAUSVID_CKPT_MIN_BYTES,
)
from voyage.registry_causvid import (
    CAUSVID_COMMIT as CAUSVID_COMMIT,
)
from voyage.registry_causvid import (
    CAUSVID_COMMIT_SHORT as CAUSVID_COMMIT_SHORT,
)
from voyage.registry_causvid import (
    CAUSVID_HF_REPO as CAUSVID_HF_REPO,
)
from voyage.registry_causvid import (
    CAUSVID_HF_REVISION as CAUSVID_HF_REVISION,
)
from voyage.registry_causvid import (
    CAUSVID_LICENSE as CAUSVID_LICENSE,
)
from voyage.registry_causvid import (
    CAUSVID_LICENSE_URL as CAUSVID_LICENSE_URL,
)
from voyage.registry_causvid import (
    CAUSVID_SUBDIR as CAUSVID_SUBDIR,
)
from voyage.registry_causvid import (
    WAN21_ALLOW as WAN21_ALLOW,
)
from voyage.registry_causvid import (
    WAN21_DIT_MIN_BYTES as WAN21_DIT_MIN_BYTES,
)
from voyage.registry_causvid import (
    WAN21_HF_REPO as WAN21_HF_REPO,
)
from voyage.registry_causvid import (
    WAN21_HF_REVISION as WAN21_HF_REVISION,
)
from voyage.registry_causvid import (
    WAN21_LICENSE as WAN21_LICENSE,
)
from voyage.registry_causvid import (
    WAN21_LICENSE_URL as WAN21_LICENSE_URL,
)
from voyage.registry_causvid import (
    WAN21_SUBDIR as WAN21_SUBDIR,
)
from voyage.registry_causvid import (
    WAN21_T5_MIN_BYTES as WAN21_T5_MIN_BYTES,
)
from voyage.registry_causvid import (
    WAN21_VAE_MIN_BYTES as WAN21_VAE_MIN_BYTES,
)
from voyage.registry_causvid import (
    _describe_causvid as _describe_causvid,
)
from voyage.registry_causvid import (
    _record_causvid as _record_causvid,
)
from voyage.registry_director import (
    MINILM_ALLOW as MINILM_ALLOW,
)
from voyage.registry_director import (
    MINILM_HF_REPO as MINILM_HF_REPO,
)
from voyage.registry_director import (
    MINILM_HF_REVISION as MINILM_HF_REVISION,
)
from voyage.registry_director import (
    MINILM_LICENSE as MINILM_LICENSE,
)
from voyage.registry_director import (
    MINILM_MIN_BYTES as MINILM_MIN_BYTES,
)
from voyage.registry_director import (
    MINILM_SUBDIR as MINILM_SUBDIR,
)
from voyage.registry_director import (
    QWEN4B_AWQ_ALLOW as QWEN4B_AWQ_ALLOW,
)
from voyage.registry_director import (
    QWEN4B_AWQ_HF_REPO as QWEN4B_AWQ_HF_REPO,
)
from voyage.registry_director import (
    QWEN4B_AWQ_HF_REVISION as QWEN4B_AWQ_HF_REVISION,
)
from voyage.registry_director import (
    QWEN4B_AWQ_LICENSE as QWEN4B_AWQ_LICENSE,
)
from voyage.registry_director import (
    QWEN4B_AWQ_LICENSE_URL as QWEN4B_AWQ_LICENSE_URL,
)
from voyage.registry_director import (
    QWEN4B_AWQ_MIN_BYTES as QWEN4B_AWQ_MIN_BYTES,
)
from voyage.registry_director import (
    QWEN4B_AWQ_SUBDIR as QWEN4B_AWQ_SUBDIR,
)
from voyage.registry_director import (
    QWEN_ALLOW as QWEN_ALLOW,
)
from voyage.registry_director import (
    QWEN_HF_REPO as QWEN_HF_REPO,
)
from voyage.registry_director import (
    QWEN_HF_REVISION as QWEN_HF_REVISION,
)
from voyage.registry_director import (
    QWEN_LICENSE as QWEN_LICENSE,
)
from voyage.registry_director import (
    QWEN_LICENSE_URL as QWEN_LICENSE_URL,
)
from voyage.registry_director import (
    QWEN_MIN_BYTES as QWEN_MIN_BYTES,
)
from voyage.registry_director import (
    QWEN_SUBDIR as QWEN_SUBDIR,
)
from voyage.registry_director import (
    _describe_director as _describe_director,
)
from voyage.registry_director import (
    _describe_director_awq as _describe_director_awq,
)
from voyage.registry_director import (
    _record_director as _record_director,
)
from voyage.registry_director import (
    _record_director_awq as _record_director_awq,
)
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
from voyage.registry_inspector import (
    QWEN35_ALLOW as QWEN35_ALLOW,
)
from voyage.registry_inspector import (
    QWEN35_HF_REPO as QWEN35_HF_REPO,
)
from voyage.registry_inspector import (
    QWEN35_HF_REVISION as QWEN35_HF_REVISION,
)
from voyage.registry_inspector import (
    QWEN35_LICENSE as QWEN35_LICENSE,
)
from voyage.registry_inspector import (
    QWEN35_LICENSE_URL as QWEN35_LICENSE_URL,
)
from voyage.registry_inspector import (
    QWEN35_MIN_BYTES as QWEN35_MIN_BYTES,
)
from voyage.registry_inspector import (
    QWEN35_SUBDIR as QWEN35_SUBDIR,
)
from voyage.registry_inspector import (
    _describe_inspector as _describe_inspector,
)
from voyage.registry_inspector import (
    _record_inspector as _record_inspector,
)
from voyage.registry_ltxv import (
    EXPECTED_LTXV_DIT_SHA256 as EXPECTED_LTXV_DIT_SHA256,
)
from voyage.registry_ltxv import (
    EXPECTED_LTXV_UPSC_SHA256 as EXPECTED_LTXV_UPSC_SHA256,
)
from voyage.registry_ltxv import (
    LTXV_COMMIT as LTXV_COMMIT,
)
from voyage.registry_ltxv import (
    LTXV_COMMIT_SHORT as LTXV_COMMIT_SHORT,
)
from voyage.registry_ltxv import (
    LTXV_DIT_FILE as LTXV_DIT_FILE,
)
from voyage.registry_ltxv import (
    LTXV_DIT_MIN_BYTES as LTXV_DIT_MIN_BYTES,
)
from voyage.registry_ltxv import (
    LTXV_HF_REPO as LTXV_HF_REPO,
)
from voyage.registry_ltxv import (
    LTXV_HF_REVISION as LTXV_HF_REVISION,
)
from voyage.registry_ltxv import (
    LTXV_SUBDIR as LTXV_SUBDIR,
)
from voyage.registry_ltxv import (
    LTXV_TE_ALLOW as LTXV_TE_ALLOW,
)
from voyage.registry_ltxv import (
    LTXV_TE_REPO as LTXV_TE_REPO,
)
from voyage.registry_ltxv import (
    LTXV_TE_REVISION as LTXV_TE_REVISION,
)
from voyage.registry_ltxv import (
    LTXV_TE_SUBDIR as LTXV_TE_SUBDIR,
)
from voyage.registry_ltxv import (
    LTXV_UPSC_FILE as LTXV_UPSC_FILE,
)
from voyage.registry_ltxv import (
    LTXV_UPSC_MIN_BYTES as LTXV_UPSC_MIN_BYTES,
)
from voyage.registry_ltxv import (
    _describe_ltxv as _describe_ltxv,
)
from voyage.registry_ltxv import (
    _record_ltxv as _record_ltxv,
)
from voyage.registry_realesrgan import (
    EXPECTED_REALESRGAN_SHA256 as EXPECTED_REALESRGAN_SHA256,
)
from voyage.registry_realesrgan import (
    REALESRGAN_ANIME_FILE as REALESRGAN_ANIME_FILE,
)
from voyage.registry_realesrgan import (
    REALESRGAN_ANIME_MIN_BYTES as REALESRGAN_ANIME_MIN_BYTES,
)
from voyage.registry_realesrgan import (
    REALESRGAN_HF_REPO as REALESRGAN_HF_REPO,
)
from voyage.registry_realesrgan import (
    REALESRGAN_HF_REVISION as REALESRGAN_HF_REVISION,
)
from voyage.registry_realesrgan import (
    REALESRGAN_LICENSE as REALESRGAN_LICENSE,
)
from voyage.registry_realesrgan import (
    REALESRGAN_LICENSE_URL as REALESRGAN_LICENSE_URL,
)
from voyage.registry_realesrgan import (
    REALESRGAN_SUBDIR as REALESRGAN_SUBDIR,
)
from voyage.registry_realesrgan import (
    REALESRGAN_UPSTREAM_URL as REALESRGAN_UPSTREAM_URL,
)
from voyage.registry_realesrgan import (
    _describe_realesrgan as _describe_realesrgan,
)
from voyage.registry_realesrgan import (
    _record_realesrgan as _record_realesrgan,
)
from voyage.registry_sfx import (
    MMAUDIO_CLIP_ALLOW as MMAUDIO_CLIP_ALLOW,
)
from voyage.registry_sfx import (
    MMAUDIO_CLIP_LICENSE as MMAUDIO_CLIP_LICENSE,
)
from voyage.registry_sfx import (
    MMAUDIO_CLIP_MIN_BYTES as MMAUDIO_CLIP_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_CLIP_REPO as MMAUDIO_CLIP_REPO,
)
from voyage.registry_sfx import (
    MMAUDIO_CLIP_REVISION as MMAUDIO_CLIP_REVISION,
)
from voyage.registry_sfx import (
    MMAUDIO_CLIP_SUBDIR as MMAUDIO_CLIP_SUBDIR,
)
from voyage.registry_sfx import (
    MMAUDIO_CODE_COMMIT as MMAUDIO_CODE_COMMIT,
)
from voyage.registry_sfx import (
    MMAUDIO_CODE_COMMIT_SHORT as MMAUDIO_CODE_COMMIT_SHORT,
)
from voyage.registry_sfx import (
    MMAUDIO_EXT_FILES as MMAUDIO_EXT_FILES,
)
from voyage.registry_sfx import (
    MMAUDIO_HF_REPO as MMAUDIO_HF_REPO,
)
from voyage.registry_sfx import (
    MMAUDIO_HF_REVISION as MMAUDIO_HF_REVISION,
)
from voyage.registry_sfx import (
    MMAUDIO_LARGE_MIN_BYTES as MMAUDIO_LARGE_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_LICENSE as MMAUDIO_LICENSE,
)
from voyage.registry_sfx import (
    MMAUDIO_LICENSE_URL as MMAUDIO_LICENSE_URL,
)
from voyage.registry_sfx import (
    MMAUDIO_MEDIUM_MIN_BYTES as MMAUDIO_MEDIUM_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_SMALL_MIN_BYTES as MMAUDIO_SMALL_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_SUBDIR as MMAUDIO_SUBDIR,
)
from voyage.registry_sfx import (
    MMAUDIO_SYNCHFORMER_MIN_BYTES as MMAUDIO_SYNCHFORMER_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_VAE_MIN_BYTES as MMAUDIO_VAE_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_VOCODER_ALLOW as MMAUDIO_VOCODER_ALLOW,
)
from voyage.registry_sfx import (
    MMAUDIO_VOCODER_LICENSE as MMAUDIO_VOCODER_LICENSE,
)
from voyage.registry_sfx import (
    MMAUDIO_VOCODER_MIN_BYTES as MMAUDIO_VOCODER_MIN_BYTES,
)
from voyage.registry_sfx import (
    MMAUDIO_VOCODER_REPO as MMAUDIO_VOCODER_REPO,
)
from voyage.registry_sfx import (
    MMAUDIO_VOCODER_REVISION as MMAUDIO_VOCODER_REVISION,
)
from voyage.registry_sfx import (
    MMAUDIO_VOCODER_SUBDIR as MMAUDIO_VOCODER_SUBDIR,
)
from voyage.registry_sfx import (
    MMAUDIO_WEIGHT_FILES as MMAUDIO_WEIGHT_FILES,
)
from voyage.registry_sfx import (
    _describe_sfx as _describe_sfx,
)
from voyage.registry_sfx import (
    _record_sfx as _record_sfx,
)

# Phase 3 director LLM pins live in `voyage.registry_director` (issue 082;
# re-exported at the top so existing importers keep working).

# GPU decider pins live in `voyage.registry_director` (issue 082;
# re-exported at the top so existing importers keep working).

# Phase 5 inspector pins live in `voyage.registry_inspector` (issue 082;
# re-exported at the top so existing importers keep working).

# Phase 4 music-stack pins live in `voyage.registry_audio` (issue 082;
# re-exported at the top so existing importers keep working).

# Shared MiniLM embedding pins live in `voyage.registry_director`
# (issue 082; re-exported at the top so existing importers keep working —
# batch-15 decision: MINILM moves with the director triple as its single
# source for both `_record_director` builders; silent pin duplication is
# forbidden).

# Phase 7 LTXV pins live in `voyage.registry_ltxv` (issue 082;
# re-exported at the top so existing importers keep working).

# Stream D CausVid + Wan2.1 pins live in `voyage.registry_causvid`
# (issue 082; re-exported at the top so existing importers keep working).

# SFX effects-stack pins live in `voyage.registry_sfx` (issue 082;
# re-exported at the top so existing importers keep working).

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

# Real-ESRGAN pins live in `voyage.registry_realesrgan` (issue 082;
# re-exported at the top so existing importers keep working).

# Expected ingest hashes (issue 071): `download_model` verifies these BEFORE
# merging the manifest record, so a poisoned first fetch can never become the
# attested baseline. Provenance per row: the LTXV/FILM/Real-ESRGAN hashes
# were measured live 2026-09-30 from the provisioned volume (all FileSpec
# pinned-revision fetches). No constant exists for the CausVid DMD checkpoint:
# its manifest record carries no sha and the weight file was pruned
# 2026-09-24 — re-provision, measure, and add it here (residual).
# (A removed video backend's generator hash lived here until issue
# 079 deleted it with the backend.)

# EXPECTED_LTXV_* hashes live in `voyage.registry_ltxv` (issue 082;
# re-exported at the top so existing importers keep working).

# EXPECTED_FILM_SHA256 lives in `voyage.registry_film` (issue 082;
# re-exported at the top so existing importers keep working).

# EXPECTED_REALESRGAN_SHA256 lives in `voyage.registry_realesrgan`
# (issue 082; re-exported at the top so existing importers keep working).

# _ACE_* derived relatives live in `voyage.registry_audio` (issue 082;
# re-exported at the top so existing importers keep working).


# Director record builder lives in `voyage.registry_director`
# (issue 082; re-exported at the top so existing importers keep working).


# Inspector record builder lives in `voyage.registry_inspector`
# (issue 082; re-exported at the top so existing importers keep working).


# Audio record builder lives in `voyage.registry_audio`
# (issue 082; re-exported at the top so existing importers keep working).


# LTXV record builder lives in `voyage.registry_ltxv`
# (issue 082; re-exported at the top so existing importers keep working).


# CausVid record builder lives in `voyage.registry_causvid`
# (issue 082; re-exported at the top so existing importers keep working).


# SFX record builder lives in `voyage.registry_sfx`
# (issue 082; re-exported at the top so existing importers keep working).


# Director describe helper lives in `voyage.registry_director`
# (issue 082; re-exported at the top so existing importers keep working).


# Director AWQ record builder lives in `voyage.registry_director`
# (issue 082; re-exported at the top so existing importers keep working).


# Director AWQ describe helper lives in `voyage.registry_director`
# (issue 082; re-exported at the top so existing importers keep working).


# Inspector describe helper lives in `voyage.registry_inspector`
# (issue 082; re-exported at the top so existing importers keep working).


# Audio describe helper lives in `voyage.registry_audio`
# (issue 082; re-exported at the top so existing importers keep working).


# LTXV describe helper lives in `voyage.registry_ltxv`
# (issue 082; re-exported at the top so existing importers keep working).


# SFX describe helper lives in `voyage.registry_sfx`
# (issue 082; re-exported at the top so existing importers keep working).


# CausVid describe helper lives in `voyage.registry_causvid`
# (issue 082; re-exported at the top so existing importers keep working).


# FILM builders live in `voyage.registry_film` (issue 082; re-exported
# at the top so existing importers keep working).


# Real-ESRGAN record builder lives in `voyage.registry_realesrgan`
# (issue 082; re-exported at the top so existing importers keep working).


# FILM describe helper lives in `voyage.registry_film` (issue 082;
# re-exported at the top so existing importers keep working).


# Real-ESRGAN describe helper lives in `voyage.registry_realesrgan`
# (issue 082; re-exported at the top so existing importers keep working).

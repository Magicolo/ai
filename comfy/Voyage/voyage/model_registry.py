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

from voyage.atomic import JsonValue
from voyage.hashing import sha256_file
from voyage.registry_records import (
    _ACE_CHECKPOINTS_RELATIVE,
    _ACE_LM_RELATIVE,
    _WAN22_RELATIVE,
    ACE_CHECKPOINTS_SUBDIR,
    ACE_LM17_MIN_BYTES,
    ACE_LM_ALLOW,
    ACE_LM_MIN_BYTES,
    ACE_LM_REPO,
    ACE_LM_REVISION,
    ACE_LM_SUBDIR,
    ACE_MAIN_ALLOW,
    ACE_MAIN_LICENSE,
    ACE_MAIN_REPO,
    ACE_MAIN_REVISION,
    ACE_MAIN_SUBDIR,
    ACE_TURBO_MIN_BYTES,
    CAUSVID_CHECKPOINT_FILE,
    CAUSVID_CHECKPOINT_NAME,
    CAUSVID_CHECKPOINT_SUBDIR,
    CAUSVID_CKPT_MIN_BYTES,
    CAUSVID_COMMIT,
    CAUSVID_COMMIT_SHORT,
    CAUSVID_HF_REPO,
    CAUSVID_HF_REVISION,
    CAUSVID_LICENSE,
    CAUSVID_LICENSE_URL,
    CAUSVID_SUBDIR,
    EXPECTED_FILM_SHA256,
    EXPECTED_LONGLIVE_SHA256,
    EXPECTED_LTXV_DIT_SHA256,
    EXPECTED_LTXV_UPSC_SHA256,
    EXPECTED_REALESRGAN_SHA256,
    FILM_FILE,
    FILM_HF_REPO,
    FILM_HF_REVISION,
    FILM_LICENSE,
    FILM_LICENSE_URL,
    FILM_MIN_BYTES,
    FILM_REPO_PATH,
    FILM_SUBDIR,
    LONGLIVE_COMMIT,
    LONGLIVE_COMMIT_SHORT,
    LONGLIVE_HF_FILE,
    LONGLIVE_HF_REPO,
    LONGLIVE_HF_REVISION,
    LONGLIVE_LICENSE,
    LONGLIVE_LICENSE_URL,
    LTXV_COMMIT,
    LTXV_COMMIT_SHORT,
    LTXV_DIT_FILE,
    LTXV_DIT_MIN_BYTES,
    LTXV_HF_REPO,
    LTXV_HF_REVISION,
    LTXV_SUBDIR,
    LTXV_TE_ALLOW,
    LTXV_TE_REPO,
    LTXV_TE_REVISION,
    LTXV_TE_SUBDIR,
    LTXV_UPSC_FILE,
    LTXV_UPSC_MIN_BYTES,
    MINILM_ALLOW,
    MINILM_HF_REPO,
    MINILM_HF_REVISION,
    MINILM_LICENSE,
    MINILM_MIN_BYTES,
    MINILM_SUBDIR,
    MMAUDIO_CLIP_ALLOW,
    MMAUDIO_CLIP_LICENSE,
    MMAUDIO_CLIP_MIN_BYTES,
    MMAUDIO_CLIP_REPO,
    MMAUDIO_CLIP_REVISION,
    MMAUDIO_CLIP_SUBDIR,
    MMAUDIO_CODE_COMMIT,
    MMAUDIO_CODE_COMMIT_SHORT,
    MMAUDIO_EXT_FILES,
    MMAUDIO_HF_REPO,
    MMAUDIO_HF_REVISION,
    MMAUDIO_LARGE_MIN_BYTES,
    MMAUDIO_LICENSE,
    MMAUDIO_LICENSE_URL,
    MMAUDIO_MEDIUM_MIN_BYTES,
    MMAUDIO_SMALL_MIN_BYTES,
    MMAUDIO_SUBDIR,
    MMAUDIO_SYNCHFORMER_MIN_BYTES,
    MMAUDIO_VAE_MIN_BYTES,
    MMAUDIO_VOCODER_ALLOW,
    MMAUDIO_VOCODER_LICENSE,
    MMAUDIO_VOCODER_MIN_BYTES,
    MMAUDIO_VOCODER_REPO,
    MMAUDIO_VOCODER_REVISION,
    MMAUDIO_VOCODER_SUBDIR,
    MMAUDIO_WEIGHT_FILES,
    QWEN4B_AWQ_ALLOW,
    QWEN4B_AWQ_HF_REPO,
    QWEN4B_AWQ_HF_REVISION,
    QWEN4B_AWQ_LICENSE,
    QWEN4B_AWQ_LICENSE_URL,
    QWEN4B_AWQ_MIN_BYTES,
    QWEN4B_AWQ_SUBDIR,
    QWEN35_ALLOW,
    QWEN35_HF_REPO,
    QWEN35_HF_REVISION,
    QWEN35_LICENSE,
    QWEN35_LICENSE_URL,
    QWEN35_MIN_BYTES,
    QWEN35_SUBDIR,
    QWEN_ALLOW,
    QWEN_HF_REPO,
    QWEN_HF_REVISION,
    QWEN_LICENSE,
    QWEN_LICENSE_URL,
    QWEN_MIN_BYTES,
    QWEN_SUBDIR,
    REALESRGAN_ANIME_FILE,
    REALESRGAN_ANIME_MIN_BYTES,
    REALESRGAN_HF_REPO,
    REALESRGAN_HF_REVISION,
    REALESRGAN_LICENSE,
    REALESRGAN_LICENSE_URL,
    REALESRGAN_SUBDIR,
    REALESRGAN_UPSTREAM_URL,
    WAN21_ALLOW,
    WAN21_DIT_MIN_BYTES,
    WAN21_HF_REPO,
    WAN21_HF_REVISION,
    WAN21_LICENSE,
    WAN21_LICENSE_URL,
    WAN21_SUBDIR,
    WAN21_T5_MIN_BYTES,
    WAN21_VAE_MIN_BYTES,
    WAN_ALLOW,
    WAN_HF_REPO,
    WAN_HF_REVISION,
    WAN_LICENSE,
    WAN_SUBDIR,
    _describe_audio,
    _describe_causvid,
    _describe_director,
    _describe_director_awq,
    _describe_film,
    _describe_inspector,
    _describe_longlive2,
    _describe_ltxv,
    _describe_realesrgan,
    _describe_sfx,
    _record_audio,
    _record_causvid,
    _record_director,
    _record_director_awq,
    _record_film,
    _record_inspector,
    _record_longlive2,
    _record_ltxv,
    _record_realesrgan,
    _record_sfx,
)

# Backward-compat surface (issue 082): every name below is either defined
# in this seam (spec dataclasses, MODEL_SPECS assembly, download/verify
# core) or re-exported from the registry_records family module. mypy reads
# this list as the explicit export contract, so imports from
# voyage.model_registry keep working exactly as before the split.
__all__ = [
    "ACE_CHECKPOINTS_SUBDIR",
    "ACE_LM17_MIN_BYTES",
    "ACE_LM_ALLOW",
    "ACE_LM_MIN_BYTES",
    "ACE_LM_REPO",
    "ACE_LM_REVISION",
    "ACE_LM_SUBDIR",
    "ACE_MAIN_ALLOW",
    "ACE_MAIN_LICENSE",
    "ACE_MAIN_REPO",
    "ACE_MAIN_REVISION",
    "ACE_MAIN_SUBDIR",
    "ACE_TURBO_MIN_BYTES",
    "CAUSVID_CHECKPOINT_FILE",
    "CAUSVID_CHECKPOINT_NAME",
    "CAUSVID_CHECKPOINT_SUBDIR",
    "CAUSVID_CKPT_MIN_BYTES",
    "CAUSVID_COMMIT",
    "CAUSVID_COMMIT_SHORT",
    "CAUSVID_HF_REPO",
    "CAUSVID_HF_REVISION",
    "CAUSVID_LICENSE",
    "CAUSVID_LICENSE_URL",
    "CAUSVID_SUBDIR",
    "EXPECTED_FILM_SHA256",
    "EXPECTED_LONGLIVE_SHA256",
    "EXPECTED_LTXV_DIT_SHA256",
    "EXPECTED_LTXV_UPSC_SHA256",
    "EXPECTED_REALESRGAN_SHA256",
    "ExpectedHash",
    "FILM_FILE",
    "FILM_HF_REPO",
    "FILM_HF_REVISION",
    "FILM_LICENSE",
    "FILM_LICENSE_URL",
    "FILM_MIN_BYTES",
    "FILM_REPO_PATH",
    "FILM_SUBDIR",
    "FileSpec",
    "LONGLIVE_COMMIT",
    "LONGLIVE_COMMIT_SHORT",
    "LONGLIVE_HF_FILE",
    "LONGLIVE_HF_REPO",
    "LONGLIVE_HF_REVISION",
    "LONGLIVE_LICENSE",
    "LONGLIVE_LICENSE_URL",
    "LTXV_COMMIT",
    "LTXV_COMMIT_SHORT",
    "LTXV_DIT_FILE",
    "LTXV_DIT_MIN_BYTES",
    "LTXV_HF_REPO",
    "LTXV_HF_REVISION",
    "LTXV_SUBDIR",
    "LTXV_TE_ALLOW",
    "LTXV_TE_REPO",
    "LTXV_TE_REVISION",
    "LTXV_TE_SUBDIR",
    "LTXV_UPSC_FILE",
    "LTXV_UPSC_MIN_BYTES",
    "MINILM_ALLOW",
    "MINILM_HF_REPO",
    "MINILM_HF_REVISION",
    "MINILM_LICENSE",
    "MINILM_MIN_BYTES",
    "MINILM_SUBDIR",
    "MMAUDIO_CLIP_ALLOW",
    "MMAUDIO_CLIP_LICENSE",
    "MMAUDIO_CLIP_MIN_BYTES",
    "MMAUDIO_CLIP_REPO",
    "MMAUDIO_CLIP_REVISION",
    "MMAUDIO_CLIP_SUBDIR",
    "MMAUDIO_CODE_COMMIT",
    "MMAUDIO_CODE_COMMIT_SHORT",
    "MMAUDIO_EXT_FILES",
    "MMAUDIO_HF_REPO",
    "MMAUDIO_HF_REVISION",
    "MMAUDIO_LARGE_MIN_BYTES",
    "MMAUDIO_LICENSE",
    "MMAUDIO_LICENSE_URL",
    "MMAUDIO_MEDIUM_MIN_BYTES",
    "MMAUDIO_SMALL_MIN_BYTES",
    "MMAUDIO_SUBDIR",
    "MMAUDIO_SYNCHFORMER_MIN_BYTES",
    "MMAUDIO_VAE_MIN_BYTES",
    "MMAUDIO_VOCODER_ALLOW",
    "MMAUDIO_VOCODER_LICENSE",
    "MMAUDIO_VOCODER_MIN_BYTES",
    "MMAUDIO_VOCODER_REPO",
    "MMAUDIO_VOCODER_REVISION",
    "MMAUDIO_VOCODER_SUBDIR",
    "MMAUDIO_WEIGHT_FILES",
    "MODEL_SPECS",
    "ModelSpec",
    "QWEN35_ALLOW",
    "QWEN35_HF_REPO",
    "QWEN35_HF_REVISION",
    "QWEN35_LICENSE",
    "QWEN35_LICENSE_URL",
    "QWEN35_MIN_BYTES",
    "QWEN35_SUBDIR",
    "QWEN4B_AWQ_ALLOW",
    "QWEN4B_AWQ_HF_REPO",
    "QWEN4B_AWQ_HF_REVISION",
    "QWEN4B_AWQ_LICENSE",
    "QWEN4B_AWQ_LICENSE_URL",
    "QWEN4B_AWQ_MIN_BYTES",
    "QWEN4B_AWQ_SUBDIR",
    "QWEN_ALLOW",
    "QWEN_HF_REPO",
    "QWEN_HF_REVISION",
    "QWEN_LICENSE",
    "QWEN_LICENSE_URL",
    "QWEN_MIN_BYTES",
    "QWEN_SUBDIR",
    "REALESRGAN_ANIME_FILE",
    "REALESRGAN_ANIME_MIN_BYTES",
    "REALESRGAN_HF_REPO",
    "REALESRGAN_HF_REVISION",
    "REALESRGAN_LICENSE",
    "REALESRGAN_LICENSE_URL",
    "REALESRGAN_SUBDIR",
    "REALESRGAN_UPSTREAM_URL",
    "RequiredFile",
    "RequiredGlob",
    "ShardFloor",
    "SnapshotRef",
    "SnapshotSpec",
    "WAN21_ALLOW",
    "WAN21_DIT_MIN_BYTES",
    "WAN21_HF_REPO",
    "WAN21_HF_REVISION",
    "WAN21_LICENSE",
    "WAN21_LICENSE_URL",
    "WAN21_SUBDIR",
    "WAN21_T5_MIN_BYTES",
    "WAN21_VAE_MIN_BYTES",
    "WAN_ALLOW",
    "WAN_HF_REPO",
    "WAN_HF_REVISION",
    "WAN_LICENSE",
    "WAN_SUBDIR",
    "_ACE_CHECKPOINTS_RELATIVE",
    "_ACE_LM_RELATIVE",
    "_WAN22_RELATIVE",
    "_collect_missing",
    "_describe_audio",
    "_describe_causvid",
    "_describe_director",
    "_describe_director_awq",
    "_describe_film",
    "_describe_inspector",
    "_describe_longlive2",
    "_describe_ltxv",
    "_describe_realesrgan",
    "_describe_sfx",
    "_manifest_hash_mismatches",
    "_merge_manifest_record",
    "_record_audio",
    "_record_causvid",
    "_record_director",
    "_record_director_awq",
    "_record_film",
    "_record_inspector",
    "_record_longlive2",
    "_record_ltxv",
    "_record_realesrgan",
    "_record_sfx",
    "_require_spec",
    "_run_spec_downloads",
    "_vocoder_python_files",
    "download_audio_models",
    "download_causvid_models",
    "download_director_awq_models",
    "download_director_models",
    "download_film_models",
    "download_inspector_models",
    "download_longlive2_bf16",
    "download_ltxv_models",
    "download_model",
    "download_realesrgan_models",
    "download_sfx_models",
    "models_dir_layout",
    "resolve_snapshot",
    "snapshot_present",
    "verify_audio_models",
    "verify_causvid_models",
    "verify_checkpoint_against_manifest",
    "verify_checkpoint_sha256",
    "verify_director_awq_models",
    "verify_director_models",
    "verify_film_models",
    "verify_inspector_models",
    "verify_longlive2_bf16",
    "verify_ltxv_models",
    "verify_model",
    "verify_realesrgan_models",
    "verify_sfx_models",
]


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
        loaded: JsonValue = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def download_longlive2_bf16(models_dir: Path) -> dict[str, JsonValue]:
    """Explicit download (DESIGN §85). Returns a manifest-ready record dict."""
    return download_model(models_dir, "longlive2-bf16")


def verify_longlive2_bf16(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of every required weight file."""
    return verify_model(models_dir, "longlive2-bf16")


def _merge_manifest_record(
    models_dir: Path, key: str, value: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Merge one record into models_dir/manifest.json (DESIGN §85)."""
    manifest_path = models_dir / "manifest.json"
    record: dict[str, JsonValue] = {}
    if manifest_path.exists():
        loaded: JsonValue = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    record_builder: Callable[[Path], dict[str, JsonValue]]  # manifest value for the key
    checks: tuple[RequiredFile | RequiredGlob | ShardFloor, ...]  # in order
    success_message: Callable[[Path], str]  # exact OK string (byte-stable)
    expected_hashes: tuple[ExpectedHash, ...] = ()  # ingest pins (071, checked pre-merge)
    manifest_checkpoint: str | None = None  # checkpoint relpath carrying checkpoint_sha256


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


def download_model(models_dir: Path, spec_name: str) -> dict[str, JsonValue]:
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
        loaded: JsonValue = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def download_director_models(models_dir: Path) -> dict[str, JsonValue]:
    """Explicit download of the Phase 3 director stack (DESIGN §§8-9, 85).

    Qwen3-8B snapshot (bf16 shards + tokenizer) plus MiniLM-L6-v2
    (safetensors + tokenizer/configs, skips the onnx/openvino/tf extras).
    Merges into the shared manifest; returns the merged record.
    """
    return download_model(models_dir, "director-qwen8b")


def verify_director_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the director stack."""
    return verify_model(models_dir, "director-qwen8b")


def download_director_awq_models(models_dir: Path) -> dict[str, JsonValue]:
    """Explicit download of the GPU decider stack (Qwen3-4B-AWQ + MiniLM).

    Merges under the `director-awq` manifest key (never the `director` key
    the 8B stack owns — overwrite semantics would clobber it).
    """
    return download_model(models_dir, "director-qwen4b-awq")


def verify_director_awq_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the GPU decider stack."""
    return verify_model(models_dir, "director-qwen4b-awq")


def download_inspector_models(models_dir: Path) -> dict[str, JsonValue]:
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


def download_audio_models(models_dir: Path) -> dict[str, JsonValue]:
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


def download_sfx_models(models_dir: Path) -> dict[str, JsonValue]:
    """Explicit download of the SFX effects stack (SFX slice 2).

    All three 44 kHz variants (small/medium/large_v2) + shared VAE +
    synchformer from hkchengrex/MMAudio, the nvidia 44 kHz BigVGAN
    vocoder snapshot, and the DFN5B CLIP weights — the exact layout
    `voyage.audio.mmaudio_sfx` loads (no hub round-trip at run time).
    Merges into the shared manifest; returns the merged record.
    """
    return download_model(models_dir, "sfx-mmaudio")


def verify_sfx_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the SFX stack, rejecting vocoder code."""
    ok, message = verify_model(models_dir, "sfx-mmaudio")
    if not ok:
        return ok, message
    unexpected = _vocoder_python_files(models_dir)
    if unexpected:
        preview = ", ".join(unexpected[:5])
        return False, (
            f"unexpected executable .py under {MMAUDIO_VOCODER_SUBDIR}: {preview} "
            "— vocoder snapshot is data-only (config.json + bigvgan_generator.pt); "
            "remove the files and re-provision"
        )
    return True, message


def _vocoder_python_files(models_dir: Path) -> list[str]:
    """Every `.py` file under the vocoder snapshot dir (issue 072).

    Empty when the dir is absent (the presence checklist owns that case) —
    non-empty means an executable fetch landed where only data belongs.
    """
    vocoder_dir = models_dir / MMAUDIO_VOCODER_SUBDIR
    if not vocoder_dir.is_dir():
        return []
    return sorted(str(path) for path in vocoder_dir.rglob("*.py") if path.is_file())


def download_ltxv_models(models_dir: Path) -> dict[str, JsonValue]:
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


def download_causvid_models(models_dir: Path) -> dict[str, JsonValue]:
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


def download_film_models(models_dir: Path) -> dict[str, JsonValue]:
    """Explicit download of the FILM interpolation weights (Track C).

    Single fp16 file into <models>/frame_interpolation/ (ComfyUI layout).
    Merges into the shared manifest; returns the merged record. Backs the
    `models download film` CLI target.
    """
    return download_model(models_dir, "film")


def verify_film_models(models_dir: Path) -> tuple[bool, str]:
    """Check presence (+ size sanity) of the FILM weights."""
    return verify_model(models_dir, "film")


def download_realesrgan_models(models_dir: Path) -> dict[str, JsonValue]:
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

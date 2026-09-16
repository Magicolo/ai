"""Manifest coverage: every file the engine loads must be downloadable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from download_models import (
    BIGVGAN_REPOSITORY,
    HUGGINGFACE_SOURCES,
    manifest_entries,
)

from zoomy.family_catalog import FAMILY_CATALOG


def test_manifest_covers_every_engine_loaded_file() -> None:
    """Fail when the catalog references a file the provisioner cannot fetch.

    ``text_encoder_file`` and the Ernie ``autoencoder_file`` are Comfy-loader
    legacy metadata: the engine pulls those components from the official
    HuggingFace repositories instead, so they are intentionally unprovisioned.
    """
    provisioned = {file_name for _, file_name in manifest_entries()}
    for family in FAMILY_CATALOG:
        assert family.base_model_file in provisioned
        if family.key != "ernie_turbo":
            assert family.autoencoder_file in provisioned
        for lora in family.loras:
            assert lora.file_name in provisioned


def test_manifest_has_no_orphans() -> None:
    """Fail when the provisioner downloads a file no family references."""
    referenced = set()
    for family in FAMILY_CATALOG:
        referenced.add(family.base_model_file)
        # flux2-vae is Comfy-legacy metadata (see above), not a download.
        if family.autoencoder_file != "flux2-vae.safetensors":
            referenced.add(family.autoencoder_file)
        for lora in family.loras:
            referenced.add(lora.file_name)
    # The MMAudio stack is shared infrastructure, not catalog data.
    referenced.update(
        {
            "mmaudio_large_44k_v2_fp16.safetensors",
            "mmaudio_vae_44k_fp16.safetensors",
            "mmaudio_synchformer_fp16.safetensors",
            "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
        }
    )
    assert {file_name for _, file_name in manifest_entries()} == referenced


def test_bigvgan_repository_is_pinned() -> None:
    """Pin the vocoder repo so rebuilds cannot silently drift."""
    assert BIGVGAN_REPOSITORY == "nvidia/bigvgan_v2_44khz_128band_512x"


def test_huggingface_paths_match_upstream_layout() -> None:
    """Pin HF sub-paths: Comfy-Org/z_image nests blobs under split_files/."""
    assert HUGGINGFACE_SOURCES["ae.safetensors"] == (
        "Comfy-Org/z_image",
        "split_files/vae/ae.safetensors",
    )

"""SFX effects-stack family pins + builders (SFX slice 2, three-caption doctrine; DESIGN §§84-85).

Split from `voyage.registry_records` (issue 082): the MMAudio
code/vocoder/CLIP row as importable pins with no cross-family coupling.
`voyage.registry_records` re-exports every name below so existing
importers keep working; new code imports from here directly.

- `MMAUDIO_*`: upstream code commit + weights Hub pin + layout + floors +
  license (native .pth weights, 44 kHz variants only).
- `MMAUDIO_VOCODER_*`: pinned 44 kHz BigVGAN vocoder snapshot (data-only,
  issue 072 — the loader resolves exactly the two files in
  `MMAUDIO_VOCODER_ALLOW`).
- `MMAUDIO_CLIP_*`: registry-pinned DFN5B CLIP text tower.
- `_record_sfx` / `_describe_sfx`: manifest value + exact OK string.
- No EXPECTED ingest hash: this row carries none (same as the
  inspector/CausVid rows — the manifest record carries no sha).
"""

from __future__ import annotations

from pathlib import Path

from voyage.atomic import JsonValue

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


def _describe_sfx(models_dir: Path) -> str:
    """Exact OK string for the SFX stack (byte-stable)."""
    large_weights = models_dir / MMAUDIO_SUBDIR / "weights" / "mmaudio_large_44k_v2.pth"
    return (
        f"sfx-mmaudio OK (large {large_weights.stat().st_size / 1024**3:.1f} GiB + VAE/sync/CLIP)"
    )

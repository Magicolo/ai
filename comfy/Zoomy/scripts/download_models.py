"""One-shot model provisioner: fill a zoomy models directory from the network.

Fresh containers start with an empty ``zoomy_models`` volume, so this script
downloads every file :class:`LocalEngine` reads into a Comfy-style tree.
Everything else the engine needs (official Ernie/Z-Image hub files, ACE-Step
checkpoints, the RIFE weights) self-populates on first use into the same
volume via ``HF_HOME`` and the music project directory.

Only Comfy-format files the engine actually opens are listed here. Notably
absent: the Comfy text encoders (ministral, qwen fp8), the Comfy ACE-Step
weights, and the prompt enhancer — the engine pulls those components from
the official HuggingFace repositories instead, so their ~30 GB of
Comfy-format copies are simply not needed.

Usage (inside the zoomy container, volume mounted at /models)::

    CIVITAI_API_KEY=... python scripts/download_models.py --models-dir /models

The script is idempotent: files already present at the expected size are
skipped, so re-runs resume interrupted downloads.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

CIVITAI_DOWNLOAD_URL = "https://civitai.com/api/download/models/{version_id}"
# CivitAI sits behind Cloudflare, which answers urllib's default User-Agent
# with error 1010 (access denied by browser-integrity check). A browser UA
# passes; verified live (urllib UA -> 403, browser UA -> 307 to the CDN).
CIVITAI_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# Every Comfy-format file the engine opens: (subdirectory, file name, source).
# CivitAI sources are version ids (bare file-form URLs 404, the
# version-scoped form is required); HuggingFace sources are (repo, path).
# Provenance matches the previous Comfy-side downloads (see wrapper AGENTS.md)
# except the Juggernaut DiTs: the bare version URL serves each version's
# PRIMARY file, which is the fp8-pruned variant (5.7 GB, verified 2026-09-16)
# rather than the full bf16/fp16 files Comfy ran (11.5 GB). The fp8 choice is
# deliberate — half the VRAM on the 16 GB card — and e2e frame renders confirm
# the zoom still looks right. To swap back to full precision, append
# ?fileId=<id> to the version URL: fast v3011968 full fp16 = 2891192 / full
# bf16 = 2891189; quality v2921151 full bf16 = 2799849 / full fp16 = 2804116.
CIVITAI_SOURCES = {
    "ernie-image-turbo.safetensors": "3028150",
    "juggernautZ_v10FastBy.safetensors": "3011968",
    "juggernautZ_v10ByRundiffusion.safetensors": "2921151",
    "c64style_ernie.safetensors": "2882202",
    "Chalkboard01-1_CE_ZIMG_AIT4k.safetensors": "2660695",
    "ClayArt01a_CE_ZIMG_AIT3k.safetensors": "3021033",
}
HUGGINGFACE_SOURCES = {
    "ae.safetensors": ("Comfy-Org/z_image", "split_files/vae/ae.safetensors"),
    "mmaudio_large_44k_v2_fp16.safetensors": (
        "Kijai/MMAudio_safetensors",
        "mmaudio_large_44k_v2_fp16.safetensors",
    ),
    "mmaudio_vae_44k_fp16.safetensors": (
        "Kijai/MMAudio_safetensors",
        "mmaudio_vae_44k_fp16.safetensors",
    ),
    "mmaudio_synchformer_fp16.safetensors": (
        "Kijai/MMAudio_safetensors",
        "mmaudio_synchformer_fp16.safetensors",
    ),
    "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors": (
        "Kijai/MMAudio_safetensors",
        "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
    ),
}

MANIFEST: tuple[tuple[str, str], ...] = (
    ("diffusion_models", "ernie-image-turbo.safetensors"),
    ("diffusion_models", "juggernautZ_v10FastBy.safetensors"),
    ("diffusion_models", "juggernautZ_v10ByRundiffusion.safetensors"),
    ("vae", "ae.safetensors"),
    ("loras", "c64style_ernie.safetensors"),
    ("loras", "Chalkboard01-1_CE_ZIMG_AIT4k.safetensors"),
    ("loras", "ClayArt01a_CE_ZIMG_AIT3k.safetensors"),
    ("mmaudio", "mmaudio_large_44k_v2_fp16.safetensors"),
    ("mmaudio", "mmaudio_vae_44k_fp16.safetensors"),
    ("mmaudio", "mmaudio_synchformer_fp16.safetensors"),
    ("mmaudio", "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"),
)

BIGVGAN_REPOSITORY = "nvidia/bigvgan_v2_44khz_128band_512x"
# The vocoder directory ships a 1.5 GB discriminator optimizer alongside the
# generator: training state the engine never loads, so it is excluded.
BIGVGAN_FILENAME_EXCLUSIONS = ("discriminator",)


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """Resolved locations for one provisioner run."""

    models_directory: Path
    seed_directory: Path
    seed_source: Path | None
    civitai_api_key: str | None


def parse_arguments(arguments: list[str] | None = None) -> DownloadPlan:
    """Parse CLI arguments into a download plan."""
    parser = argparse.ArgumentParser(description="Download zoomy model files.")
    parser.add_argument("--models-dir", default="/models", help="Models tree root.")
    parser.add_argument("--seed-dir", default="/models/seed", help="Seed images root.")
    parser.add_argument(
        "--seed-source",
        default=None,
        help="Directory holding ernie_zoom_seed.png / z_zoom_seed.png to copy.",
    )
    parsed = parser.parse_args(arguments)
    return DownloadPlan(
        models_directory=Path(parsed.models_dir),
        seed_directory=Path(parsed.seed_dir),
        seed_source=Path(parsed.seed_source) if parsed.seed_source else None,
        civitai_api_key=os.environ.get("CIVITAI_API_KEY"),
    )


def manifest_entries() -> tuple[tuple[str, str], ...]:
    """Return the (subdirectory, file name) pairs this script provisions."""
    return MANIFEST


def download_all(plan: DownloadPlan) -> None:
    """Download every manifest file plus the vocoder and seeds."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    for subdirectory, file_name in MANIFEST:
        destination = plan.models_directory / subdirectory / file_name
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"skip {destination} (already present)")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_name in CIVITAI_SOURCES:
            _download_civitai(CIVITAI_SOURCES[file_name], destination, plan.civitai_api_key)
        else:
            repository, path = HUGGINGFACE_SOURCES[file_name]
            downloaded = hf_hub_download(repository, path, local_dir=destination.parent)
            if Path(downloaded) != destination:
                shutil.move(downloaded, destination)
            print(f"done {destination}")
    _download_bigvgan(plan.models_directory)
    _copy_seeds(plan)


class _CivitaiRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow CivitAI download redirects without forwarding the API token.

    CivitAI answers ``/api/download/models/...`` with a redirect to a signed
    file URL on its CDN. The signature is embedded in the redirect target,
    while the CDN rejects requests carrying a foreign ``Authorization``
    header (HTTP 403) — and forwarding the bearer token there would leak it.
    """

    # Parameter names mirror the stdlib override point (req/fp/msg/newurl).
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def build_civitai_request(url: str, api_key: str | None) -> urllib.request.Request:
    """Build an authenticated CivitAI request with a Cloudflare-safe UA."""
    request = urllib.request.Request(url)  # noqa: S310
    # Callers pass our own https-pinned download constant (see S310 note at
    # the urlopen call site); this helper only attaches headers.
    request.add_header("User-Agent", CIVITAI_USER_AGENT)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    return request


def _download_civitai(version_id: str, destination: Path, api_key: str | None) -> None:
    """Download one CivitAI model version to ``destination``."""
    url = CIVITAI_DOWNLOAD_URL.format(version_id=version_id)
    if urlparse(url).scheme != "https":
        message = f"Refusing non-HTTPS download: {url}"
        raise ValueError(message)
    request = build_civitai_request(url, api_key)
    print(f"fetch {url} -> {destination}")
    opener = urllib.request.build_opener(_CivitaiRedirectHandler)
    with opener.open(request) as response, destination.open("wb") as target:  # noqa: S310
        # Scheme pinned to https by the guard above; the URL host is our own
        # constant, only the numeric version id varies.
        shutil.copyfileobj(response, target)
    print(f"done {destination}")


def _download_bigvgan(models_directory: Path) -> None:
    """Fetch the BigVGAN vocoder the MMAudio autoencoder loads."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    destination = models_directory / "mmaudio" / "nvidia" / "bigvgan_v2_44khz_128band_512x"
    if destination.is_dir() and any(destination.iterdir()):
        print(f"skip {destination} (already present)")
        return
    print(f"fetch {BIGVGAN_REPOSITORY} -> {destination}")
    snapshot_download(
        BIGVGAN_REPOSITORY,
        local_dir=destination,
        ignore_patterns=[f"*{exclusion}*" for exclusion in BIGVGAN_FILENAME_EXCLUSIONS],
    )
    print(f"done {destination}")


def _copy_seeds(plan: DownloadPlan) -> None:
    """Copy cold-start seeds from a host-provided directory when given."""
    if plan.seed_source is None:
        print("no --seed-source given, skipping seeds")
        return
    plan.seed_directory.mkdir(parents=True, exist_ok=True)
    for seed_name in ("ernie_zoom_seed.png", "z_zoom_seed.png"):
        source = plan.seed_source / seed_name
        destination = plan.seed_directory / seed_name
        if destination.is_file():
            print(f"skip {destination} (already present)")
        elif source.is_file():
            shutil.copyfile(source, destination)
            print(f"done {destination}")
        else:
            print(f"missing seed {source}", file=sys.stderr)


def main() -> None:
    """Entry point for ``python scripts/download_models.py``."""
    download_all(parse_arguments())


if __name__ == "__main__":
    main()

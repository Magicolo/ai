"""`models` and `doctor` verbs (DESIGN §§8-9, 85).

Verb module of the issue-080 split: weight-bundle management plus
the hardware/environment probe. Thin wrappers over `model_registry`
and `doctor` — no generation logic. Registry/doctor entry points resolve
through the voyage.cli seam at call time (function-level imports below),
so patching the seam keeps intercepting them exactly as pre-split.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voyage.doctor import probe


def cmd_doctor(_args: argparse.Namespace) -> int:
    from voyage.cli import check_ffmpeg  # seam dispatch (issue 080)

    facts = probe()
    ok, message = check_ffmpeg()
    print(f"python: {facts['python']}")
    print(f"ffmpeg: {message}")
    if facts["nvidia_smi"]:
        for line in facts["gpus"]:
            print(f"gpu: {line}")
    else:
        print("gpu: no nvidia-smi data (CPU-only environment)")
    director_python = facts.get("director_python")
    if director_python and facts.get("director_python_exists"):
        print(f"director python: {director_python}")
    elif director_python:
        print(f"director python: {director_python} (missing — director falls back to CPU)")
    else:
        print("director python: unset (director runs in-process on CPU)")
    return 0 if ok else 1


def _models_dir(args: argparse.Namespace) -> Path:
    raw = getattr(args, "models_dir", None) or os.environ.get("VOYAGE_MODELS_DIR", "/models")
    return Path(raw)


def _download_director(models_dir: Path) -> int:
    """Download the Phase 3 director stack (DESIGN §§8-9, 85)."""
    print(f"downloading director-qwen8b into {models_dir} ...")
    from voyage.cli import download_director_models  # seam dispatch (issue 080)

    try:
        record = download_director_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    director = record["director"]
    assert isinstance(director, dict)
    print(f"qwen3-8b: {director.get('checkpoint_bytes')} bytes")
    print(f"minilm: {director.get('embedding_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_director_awq(models_dir: Path) -> int:
    """Download the GPU decider stack (Qwen3-4B-AWQ + MiniLM)."""
    print(f"downloading director-qwen4b-awq into {models_dir} ...")
    from voyage.cli import download_director_awq_models  # seam dispatch (issue 080)

    try:
        record = download_director_awq_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    director = record["director-awq"]
    assert isinstance(director, dict)
    print(f"qwen3-4b-awq: {director.get('checkpoint_bytes')} bytes")
    print(f"minilm: {director.get('embedding_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_director_gguf(models_dir: Path) -> int:
    """Download the llama-server sidecar GGUF (DESIGN §140 llama entry)."""
    print(f"downloading director-qwen35-gguf into {models_dir} ...")
    from voyage.cli import download_director_gguf_models  # seam dispatch, see issue 080

    try:
        record = download_director_gguf_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    director = record["director-gguf"]
    assert isinstance(director, dict)
    print(f"qwen3.5-4b Q4_K_M: {director.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_audio(models_dir: Path) -> int:
    """Download the Phase 4 music stack (DESIGN §§6, 37, 85)."""
    print(f"downloading audio-acestep into {models_dir} ...")
    from voyage.cli import download_audio_models  # seam dispatch (issue 080)

    try:
        record = download_audio_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    audio = record["audio"]
    assert isinstance(audio, dict)
    print(f"turbo dit: {audio.get('turbo_bytes')} bytes")
    print(f"planner lm: {audio.get('planner_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_inspector(models_dir: Path) -> int:
    """Download the Phase 5 inspector VLM (DESIGN §§43-44, 85)."""
    print(f"downloading inspector-qwen35 into {models_dir} ...")
    from voyage.cli import download_inspector_models  # seam dispatch (issue 080)

    try:
        record = download_inspector_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    inspector = record["inspector"]
    assert isinstance(inspector, dict)
    print(f"qwen3.5-9b: {inspector.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_sfx(models_dir: Path) -> int:
    """Download the SFX effects stack (SFX slice 2, three-caption doctrine)."""
    print(f"downloading sfx-mmaudio into {models_dir} ...")
    from voyage.cli import download_sfx_models  # seam dispatch (issue 080)

    try:
        record = download_sfx_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    sfx = record["sfx"]
    assert isinstance(sfx, dict)
    print(f"variants: {sfx.get('variants')}")
    print(f"large: {sfx.get('large_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_film(models_dir: Path) -> int:
    """Download the FILM interpolation weights (Track C: augment floors)."""
    print(f"downloading film into {models_dir} ...")
    from voyage.cli import download_film_models  # seam dispatch (issue 080)

    try:
        record = download_film_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    film = record["film"]
    assert isinstance(film, dict)
    print(f"checkpoint: {film.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def _download_realesrgan(models_dir: Path) -> int:
    """Download the Real-ESRGAN anime upscaler (Track C: augment floors)."""
    print(f"downloading realesrgan-anime into {models_dir} ...")
    from voyage.cli import download_realesrgan_models  # seam dispatch (issue 080)

    try:
        record = download_realesrgan_models(models_dir)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    realesrgan = record["realesrgan"]
    assert isinstance(realesrgan, dict)
    print(f"checkpoint: {realesrgan.get('checkpoint_bytes')} bytes")
    print(f"manifest: {models_dir / 'manifest.json'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    # Seam dispatch (issue 080): registry entry points resolve through the
    # voyage.cli namespace at call time, so patching them there (the
    # pre-split interception point) keeps working.
    from voyage.cli import (
        download_causvid_models,
        download_ltx23_models,
        download_ltx25_models,
        download_ltxv_models,
        verify_audio_models,
        verify_causvid_models,
        verify_director_awq_models,
        verify_director_gguf_models,
        verify_director_models,
        verify_film_models,
        verify_inspector_models,
        verify_ltx23_models,
        verify_ltx25_models,
        verify_ltxv_models,
        verify_realesrgan_models,
        verify_sfx_models,
    )

    action = args.models_action
    if action == "list":
        print("video: fake (built-in) | ltx25 (LTX-2.5 Q3 + TE + VAEs, joint A/V, default)")
        print("video: ltxv-2b (LTXV 2B distilled, Phase 7 alternative)")
        print("video: causvid (CausVid DMD causal generator + Wan2.1-1.3B base)")
        print("video: ltx23 (LTX-2.3 Q3 + Gemma3 TE + VAEs, joint A/V)")
        print("audio: fake (built-in) | audio-acestep (ACE-Step 1.5 turbo + 0.6B planner)")
        print("sfx: fake (built-in) | sfx-mmaudio (MMAudio 44k effects, CC-BY-NC-4.0)")
        print(
            "director: deterministic (built-in) | director-qwen8b (Qwen3-8B + MiniLM) "
            "| director-qwen4b-awq (Qwen3-4B-AWQ GPU decider) "
            "| director-qwen35-gguf (Qwen3.5-4B Q4_K_M llama-server sidecar)"
        )
        print("inspector: skipped (built-in) | inspector-qwen35 (Qwen3.5-9B VLM, experimental)")
        print(
            "augment: film (FILM interpolation weights) "
            "| realesrgan-anime (Real-ESRGAN anime upscaler)"
        )
        return 0
    if action == "verify":
        lok, lmessage = verify_ltxv_models(_models_dir(args))
        print(lmessage)
        cok, cmessage = verify_causvid_models(_models_dir(args))
        print(cmessage)
        l25ok, l25message = verify_ltx25_models(_models_dir(args))
        print(l25message)
        l23ok, l23message = verify_ltx23_models(_models_dir(args))
        print(l23message)
        dok, dmessage = verify_director_models(_models_dir(args))
        print(dmessage)
        dawq_ok, dawq_message = verify_director_awq_models(_models_dir(args))
        print(dawq_message)
        dgguf_ok, dgguf_message = verify_director_gguf_models(_models_dir(args))
        print(dgguf_message)
        aok, amessage = verify_audio_models(_models_dir(args))
        print(amessage)
        sok, smessage = verify_sfx_models(_models_dir(args))
        print(smessage)
        fok, fmessage = verify_film_models(_models_dir(args))
        print(fmessage)
        rok, rmessage = verify_realesrgan_models(_models_dir(args))
        print(rmessage)
        iok, imessage = verify_inspector_models(_models_dir(args))
        print(imessage)
        print("fake backends need no model files: OK")
        all_ok = (
            lok
            and cok
            and l25ok
            and l23ok
            and dok
            and dawq_ok
            and dgguf_ok
            and aok
            and sok
            and fok
            and rok
            and iok
        )
        return 0 if all_ok else 1
    if action == "download":
        target = getattr(args, "models_target", "ltxv-2b")
        if target == "ltxv-2b":
            models_dir = _models_dir(args)
            print(f"downloading ltxv-2b into {models_dir} ...")
            try:
                record = download_ltxv_models(models_dir)
            except Exception as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 1
            video = record["ltxv"]
            assert isinstance(video, dict)
            print(f"DiT: {video.get('checkpoint_bytes')} bytes")
            print(f"manifest: {models_dir / 'manifest.json'}")
            return 0
        if target == "causvid":
            models_dir = _models_dir(args)
            print(f"downloading causvid into {models_dir} ...")
            try:
                record = download_causvid_models(models_dir)
            except Exception as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 1
            video = record["causvid"]
            assert isinstance(video, dict)
            print(f"DMD checkpoint: {video.get('checkpoint_bytes')} bytes")
            print(f"manifest: {models_dir / 'manifest.json'}")
            return 0
        if target == "ltx25":
            models_dir = _models_dir(args)
            print(f"downloading ltx25 into {models_dir} ...")
            try:
                record = download_ltx25_models(models_dir)
            except Exception as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 1
            video = record["ltx25"]
            assert isinstance(video, dict)
            print(f"DiT: {video.get('checkpoint_bytes')} bytes")
            print(f"manifest: {models_dir / 'manifest.json'}")
            return 0
        if target == "ltx23":
            models_dir = _models_dir(args)
            print(f"downloading ltx23 into {models_dir} ...")
            try:
                record = download_ltx23_models(models_dir)
            except Exception as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 1
            video = record["ltx23"]
            assert isinstance(video, dict)
            print(f"DiT: {video.get('checkpoint_bytes')} bytes")
            print(f"manifest: {models_dir / 'manifest.json'}")
            return 0
        if target == "director-qwen8b":
            return _download_director(_models_dir(args))
        if target == "director-qwen4b-awq":
            return _download_director_awq(_models_dir(args))
        if target == "director-qwen35-gguf":
            return _download_director_gguf(_models_dir(args))
        if target == "audio-acestep":
            return _download_audio(_models_dir(args))
        if target == "sfx-mmaudio":
            return _download_sfx(_models_dir(args))
        if target == "film":
            return _download_film(_models_dir(args))
        if target == "realesrgan-anime":
            return _download_realesrgan(_models_dir(args))
        if target == "inspector-qwen35":
            return _download_inspector(_models_dir(args))
        print(f"unknown models target {target!r}", file=sys.stderr)
        known_targets = ", ".join(
            [
                "ltxv-2b",
                "causvid",
                "ltx25",
                "ltx23",
                "director-qwen8b",
                "director-qwen4b-awq",
                "director-qwen35-gguf",
                "audio-acestep",
                "sfx-mmaudio",
                "film",
                "realesrgan-anime",
                "inspector-qwen35",
            ]
        )
        print(f"known: {known_targets}", file=sys.stderr)
        return 2
    if action == "info":
        print("backends: `voyage models list` (video/audio/director/inspector)")
        print("weights: ltxv-2b (~7 GB) | causvid (~28 GB)")
        print("weights: ltx25 (~38 GB Q3 + TE + VAEs + upscaler) | ltx23 (~20 GB Q3 + TE + VAEs)")
        print("weights: director-qwen8b (~16 GB) | audio-acestep | inspector-qwen35 (~19 GB)")
        print("weights: sfx-mmaudio (~8 GB: 3 variants + VAE/sync/CLIP/vocoder)")
        print("weights: film (~66 MB interpolation) | realesrgan-anime (~2 MB upscaler)")
        print("note: MMAudio weights are CC-BY-NC-4.0 (non-commercial)")
        print("pins: voyage/model_registry.py (single source); human mirror docs/MODELS.md")
        print("note: CausVid DMD checkpoint is CC BY-NC-SA 4.0 (non-commercial)")
        print("check: `voyage models verify` for presence + size sanity")
        return 0
    return 2

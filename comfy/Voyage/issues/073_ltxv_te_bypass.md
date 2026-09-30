# 073 — LTXV text-encoder load bypasses the pinned local snapshot

- Severity: HIGH
- File: `Voyage/voyage/workers/video_ltxv.py:48` (`TE_REPO_ID`), `:341-342` (`T5Tokenizer/T5EncoderModel.from_pretrained(TE_REPO_ID, …)`); registry pin unused at load site: `Voyage/voyage/model_registry.py:185-187` (`LTXV_TE_REVISION`, `LTXV_TE_ALLOW`)
- Area: workers / registry — LTXV text encoder

## Description

The registry carefully downloads + pins the PixArt TE snapshot (`LTXV_TE_SUBDIR`, rev `b89adade…`), but `LTXVSession.__init__` ignores it and loads the tokenizer/encoder straight from the hub id with no `revision=`, no `local_files_only=`, and no `trust_remote_code=` argument. Every video-worker init therefore depends on hub/cache state, can drift across upstream pushes, and violates the offline-first contract the director worker honors (`workers/director.py:163-181` threads `local_files_only=offline` through every load).

## Rationale

Same class as issue 070 with a runtime multiplier: it is not a one-time provision fetch but a per-session network/cache dependency on the hot path, and the pinned revision already exists — it is simply not used.

## Live evidence

Re-verified 2026-09-30 live:

```
Voyage/voyage/workers/video_ltxv.py:48:
TE_REPO_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
Voyage/voyage/workers/video_ltxv.py:341-342:
        tokenizer = T5Tokenizer.from_pretrained(TE_REPO_ID, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(TE_REPO_ID, subfolder="text_encoder").to(
Voyage/voyage/workers/video_ltxv.py:338-339 (neighboring local-load pattern):
        vae = CausalVideoAutoencoder.from_pretrained(str(dit_path)).to(device, dtype=torch.bfloat16)
        scheduler = RectifiedFlowScheduler.from_pretrained(str(dit_path))
Voyage/voyage/model_registry.py:185-187:
LTXV_TE_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"
LTXV_TE_REVISION = "b89adadeccd9ead2adcb9fa2825d3fabec48d404"
LTXV_TE_SUBDIR = "PixArt-XL-2-1024-MS"
LTXV_TE_ALLOW = ["tokenizer/*", "text_encoder/*"]
```

`rg "LTXV_TE_REVISION" voyage/` → defined + recorded in manifest (`_record_ltxv`), never passed to any `from_pretrained`. `rg "TE_REPO_ID|from_pretrained" voyage/workers/video_ltxv.py` → hub-id loads only at `:341-342`.

## Repro

With a fully provisioned volume, `HF_HUB_OFFLINE=1 python -m voyage.workers.video_ltxv …init/generate…` (or disconnect network) → TE load fails or falls back to cache instead of the pinned snapshot; alternatively push-date A/B: `snapshot_download("PixArt-alpha/PixArt-XL-2-1024-MS")` before/after an upstream commit → different bytes, same voyage manifest.

## Fix candidates

1. Resolve `models_dir / LTXV_TE_SUBDIR` (mirroring `director._resolve_model_source`) and `from_pretrained(str(local_te_dir), …, local_files_only=True [, revision=LTXV_TE_REVISION])`.
2. Add a test asserting no bare hub-id `from_pretrained` remains in workers.

## Refs

- Overlaps with 070 (same floating-pin class, one-time provision fetch — Wan2.2 `revision=None`) — this issue owns the per-session hot-path leg.
- HF transformers SECURITY.md ("setting a revision… protect yourself from updates") — https://github.com/huggingface/transformers/blob/main/SECURITY.md

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

## Progress log

- 2026-09-30: premise re-verified against live code: `TE_REPO_ID` at
  `Voyage/voyage/workers/video_ltxv.py:48`, bare-hub loads at `:341-342`,
  registry pin `LTXV_TE_REVISION = b89adade…` (`model_registry.py:185-187`)
  never passed to any `from_pretrained` (rg confirms), while
  `workers/director.py:128-162` honors the offline-first contract via
  `_resolve_model_source` — holds verbatim.
- 2026-09-30: `revision=` support probe-verified CPU-only in the video image
  (`docker run --rm voyage-video:latest python3 -c inspect…` on transformers
  4.57.6): both `T5Tokenizer.from_pretrained` and
  `T5EncoderModel.from_pretrained` accept `revision` (default `'main'`) and
  `local_files_only` (default `False`) — the contract's "if supported"
  condition holds, both flags are passed.
- 2026-09-30: TDD — `test_ltxv_session_loads_te_from_local_snapshot`
  (stubbed torch/ltx_video/transformers; asserts local source +
  `local_files_only=True` + pinned revision), the two `_resolve_te_source`
  tests (present → no download; absent → fetch-then-local), and the AST
  grep-gate all failed pre-fix (hub-id loads at old `:341-342`), all green
  post-fix. Mid-pass fix: the absent-snapshot test briefly hit the REAL hub
  (5.5 min download) because `video_ltxv` used a `from`-import binding the
  test's `monkeypatch.setattr(model_registry, "download_model", …)` could not
  reach — the resolver now uses module-attribute access (director idiom),
  keeping the seam patchable; no tree pollution (tmp dir only).
- 2026-09-30: gates — ruff + format + mypy strict clean on
  `voyage/workers/video_ltxv.py`; full suite shows no new failures from this
  issue.

## Resolution

- Fix candidate 1 applied: new `LTXVSession`-adjacent `_resolve_te_source`
  (mirrors `director._resolve_model_source`, parameterized by `models_dir` —
  present snapshot → `<models_dir>/PixArt-XL-2-1024-MS`, absent → download the
  owning `ltxv-2b` spec with the `HF_HUB_OFFLINE` guard lifted, re-check, raise
  on still-incomplete) and `__init__` loads both classes from the resolved
  source with `local_files_only=True, revision=LTXV_TE_REVISION`. The
  `TE_REPO_ID` constant is deleted (single source: `LTXV_TE_REPO`); nothing
  else imported it (rg-verified).
- Fix candidate 2 applied: `test_no_bare_hub_id_from_pretrained_in_workers`
  AST-scans `voyage/workers/*.py` (docstrings ignored by construction) and
  fails on any `from_pretrained` rooted at a `/`-literal or `*_REPO_ID` /
  `*_HF_REPO` / `*_REPO` name — director's `source`-variable loads and the
  local-path DiT/VAE/scheduler loads pass.
- Residuals (GPU box required): a live `HF_HUB_OFFLINE=1` session-init probe
  against a provisioned volume (stub tests prove the wiring; only a real init
  proves end-to-end offline-first); the absent-snapshot download path is
  covered by a stubbed test only — a live re-provision of the TE snapshot
  would exercise the real fetch + re-check.

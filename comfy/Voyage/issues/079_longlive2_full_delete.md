# 079 — Full-delete `longlive2` backend (default is ltxv, LongLive is heaviest legacy)

- Severity: HIGH (structure / scope: 1264L worker + registry + image layers + tests)
- Area: backends — video `longlive2` removal
- Decision: Q&A round 1-2 locked — deprecate = full delete, not soft-hide
- Rank rationale: biggest single-file deletion; kills own `.pt` tape format, checkpoint-load OOM, LongLive-first doc presentation.

## Description

`ltxv` is the default since 2026-09-29 (`voyage/config.py:204-211` `_DEFAULT_ROW = BACKEND_REGISTRY["ltxv"]`, 768×512@24, `segment_frames=96`). `longlive2` (1280×704@24, 29f, `persistent_kv`) is live but non-default, biggest worker file, own torch `.pt` recovery tape, own `imageio.get_writer` path, known checkpoint-load ~38 GB anon-RSS OOM (filed `Voyage/issues/096` in history, now archived in git). User dropped it to fallback; Q&A chose full delete.

Live sites (verified 2026-09-30 sweeps):

- `voyage/workers/video_longlive.py:1` — 1264 lines, resident BF16→FP8 DiT + CPU T5 twin, `LongLiveStreamSession`, `local_attn_size=16/sink=8`, VAE-offload, chunked decode, `.pt` tape
- `voyage/config.py:135-198` — `BACKEND_REGISTRY` rows (`longlive2` 1280×704/29f + `longlive2-bf16` spec); `:140,155,170,188` `segment_frames 96/29/72`
- `voyage/supervisor.py:88-108` — `VIDEO_WORKER_MODULES` + `STREAMING_VIDEO_BACKENDS = ("longlive2","ltxv","causvid")`
- `voyage/model_registry.py` — longlive/wan/Qwen pins + `download_longlive2_bf16` manifest-clobber site
- `worker/Dockerfile.video:50-54,140-168` — LongLive@`6b36d20` clone + `wan_models` shims + `_enter_longlive_tree` CWD shim (`video_longlive.py:~20`)
- `voyage/cli.py:1174` — `_CUDA_BACKENDS` includes `longlive2`; `:1143-1163` `_LTXV_NOVEL_BLOCK_FRAMES=96/_LONGLIVE_*=8,4` duplicating registry
- `scripts/qualify.sh:2` — help hardcodes `--backend longlive2`; `scripts/run.sh:67` sniff includes it
- Tests: `tests/test_longlive.py` (11), `test_longlive_stages.py` (6), `test_longlive_init_validation.py` (9), `test_longlive_offload_fusion.py` (4), `test_precision.py` fp8/bf16 legs, `test_qualification.py` longlive2 leg
- Docs/reports: `reports/video-backends.md` longlive2 leg (keep as dated evidence, mark historical), `reports/longlive-audit.md` (keep), `docs/UPSTREAM_LONG_LIVE_PATCHES.md`, backend-first ordering in INSTALL/BACKENDS/MODELS/README

## Why this is an issue

- Keeps two recovery formats alive (LongLive `.pt` vs JSON `write_tape_atomic` for ltxv/causvid) — supervisor resume must understand both.
- Keeps heaviest image layers + `transformers==4.57.6` pin tension (LongLive `x_clip_loss` vs 5.x) for a non-default path.
- Every backend-matrix change (streaming sets, CUDA sets, presets, qual harness) pays a 4-backend tax instead of 3.

## Live evidence

Sweep inventories (2026-09-30): worker table (longlive 1264L largest), `grep -n "STREAMING_VIDEO_BACKENDS\|VIDEO_WORKER_MODULES\|BACKEND_REGISTRY"` showing longlive2 in all three sets, `reports/video-backends.md:85-117` longlive2 VALID 87f numbers.

## Reproduction

```bash
wc -l voyage/workers/video_*.py
grep -rn "longlive" voyage/config.py voyage/supervisor.py voyage/cli.py worker/Dockerfile.video scripts/*.sh | head -n 40
grep -rln "longlive" tests/ docs/ reports/
```

## Resolution candidates

1. Delete `voyage/workers/video_longlive.py` + registry longlive rows + `VIDEO_WORKER_MODULES["longlive2"]` + `STREAMING_VIDEO_BACKENDS` entry (derive remainder from registry — see 083) + Dockerfile LongLive clone/shims + `run.sh`/`qualify.sh` refs.
2. Delete or repoint longlive-only tests (stages/init-validation/offload-fusion/precision-longlive legs); keep `reports/longlive-audit.md`, mark `video-backends.md` longlive leg historical.
3. Reorder INSTALL/BACKENDS/MODELS/README ltxv-first (covered in 092 but lands with this delete).
4. Gate: `Voyage/scripts/gates.sh` green + `grep -rni longlive voyage/ worker/ scripts/ tests/` empty except historical reports/docs markers.

## Refs

- `voyage/config.py:117-211`, `voyage/supervisor.py:88-108`, `voyage/workers/video_longlive.py:1`, `worker/Dockerfile.video`, `reports/video-backends.md`, `reports/longlive-audit.md`
- Prior: issues 025 (quadruple registries), 036 (god modules), archived 096 (checkpoint OOM)

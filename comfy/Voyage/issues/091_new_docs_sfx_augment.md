# 091 — New operator docs: `docs/SFX.md` + `docs/AUGMENT.md` (both shipped, zero operator coverage)

- Severity: MEDIUM (docs — shipped features undocumented)
- Area: docs — new coverage
- Decision: Q&A locked — standalone docs (not extends)

## Description

SFX shipped slices 1–4 (three-caption doctrine 2026-09-29: video/music/sfx captions evolving with prompt; finalize-time MMAudio pass `voyage/workers/sfx_mmaudio.py` + fake, 8s windows/1s manual-fade joins, amix −6dB, stems + `sfx.jsonl` under `audio/sfx/`; standalone `voyage sfx` verb; CUDA pairs mmaudio/cuda:0, fake CPU-only; ladder small-2060 4.6GiB / large-4060 6.2GiB; CC-BY-NC-4.0 weights `models download sfx-mmaudio` ~13GB) with zero operator doc outside DESIGN §140 + weight rows in `MODELS.md` (`AUDIO.md` greps zero SFX/caption words; `BACKENDS.md` zero SFX rows; `PROMPTING.md` pre-SFX three-caption families). Augment floors shipped 2026-09-30 (every video ≥32fps + ≥1280×720 by default: `AugmentConfig`, `--min-fps/--min-resolution/--no-augment` on finalize/generate/run, `[augment]` TOML, TUI fields; `media.py plan_augmentation` pure + chunked `augment.py` + lazy-torch `augment_worker.py` stand-in with full FILM port open; 2-GPU video-aug-cuda:0/SFX-cuda:1 pairing) with zero operator doc outside §140 (`OPERATIONS.md` Finalization pre-floors; no TROUBLESHOOTING augment modes; BENCHMARKING e2e numbers pre-floors).

## Rationale

Operators following INSTALL/OPERATIONS under-provision SFX + augment stacks, miss caption pins (`--sfx-caption/--music-caption/--video-caption`), miss `--no-augment` escape, misread e2e timings (floors lift CausVid 832×480@16 + fake 320×180 testsrc ~2.7× px + 2× fps).

## Live evidence

- `rg -n "sfx|SFX|caption" docs/AUDIO.md docs/BACKENDS.md docs/PROMPTING.md` → zero SFX rows (sweep 2026-09-30)
- `rg -n "min-fps|min-resolution|no-augment|plan_augmentation" docs/ README.md` → zero operator hits
- `voyage/workers/sfx_mmaudio.py:382`, `voyage/sfx_finalize.py:560`, `voyage/augment.py:281`, `voyage/config.py:375-390`, `voyage/cli.py:1761-1767`

## Repro

```bash
rg -ni "sfx|mmaudio|caption" docs/ | head -n 20
rg -n "min-fps|min-resolution|no-augment|1280|32 ?fps" docs/ README.md | head -n 20
```

## Fix candidates

1. New `docs/SFX.md`: ladder + weights/license, three-caption doctrine + pins, finalize pass (windows/fades/amix/stems/`sfx.jsonl`), `voyage sfx` verb, CUDA pairing, failure modes (OOM ladder → MODELS pointer).
2. New `docs/AUGMENT.md`: floors + flags/TOML/TUI, `plan_augmentation` contract (fast-path vs re-encode), chunked runner + 2-GPU pairing, FILM stand-in caveat + weight-port follow-up, cost note (Causvid/fake lift math), effect on benchmark numbers.
3. Index both from README docs list + DESIGN §87 coverage list (see 092); keep captions reference in PROMPTING to one paragraph + pointer (no duplication).
4. Gate: `rg` verifiers above non-empty + `gates.sh` green (docs-only + index).

## Refs

- DESIGN §140 (SFX slices 1-4, finalize floors); `docs/MODELS.md:45-106` (film/realesrgan/SFX rows); `voyage/sfx_finalize.py`, `voyage/augment.py`

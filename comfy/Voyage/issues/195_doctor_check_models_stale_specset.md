# 195 — `doctor.check_models` verifies 6 of 10 registry specs; the 4 newest default-CUDA stacks are invisible

- Severity: LOW-MEDIUM
- Area: doctor preflight — spec-set staleness
- Files (as-read 2026-09-30):
  - `voyage/doctor.py:143-150` (hardcoded verifier tuple: 6 entries)
  - `voyage/model_registry.py:742-1041` (`MODEL_SPECS`: 10 entries)
  - `voyage/models_ensure.py:46-50,93-105` (ensure set: video spec + film + realesrgan-anime + acestep + director + sfx + inspector)
  - `voyage/cli.py:337-372` (`models verify` verb: verifies all 10 incl. awq/sfx/film/realesrgan/inspector)

## Description

`check_models` hardcodes six verifiers:

```python
# doctor.py:143-150 (as-read)
verifiers = (
    ("longlive2-bf16", model_registry.verify_longlive2_bf16),
    ("ltxv-2b", model_registry.verify_ltxv_models),
    ("causvid", model_registry.verify_causvid_models),
    ("director-qwen8b", model_registry.verify_director_models),
    ("audio-acestep", model_registry.verify_audio_models),
    ("inspector-qwen35", model_registry.verify_inspector_models),
)
```

Missing: `film`, `realesrgan-anime`, `sfx-mmaudio`, `director-qwen4b-awq` — all
four present in `MODEL_SPECS`, all four downloaded by default on CUDA runs
(`models_ensure.required_specs`: film + realesrgan-anime whenever
`augment_enabled`, sfx-mmaudio when the pass runs, `director-qwen4b-awq`
whenever the director device is not cpu — the default, `config.py:601`
`director_device = "cuda:1"`). Consequence: `doctor` reports `models_ok: true`
on a box missing ~16 GB of default-path weights (SFX 13 GB + AWQ 2.6 GB + FILM
66 MB + Real-ESRGAN 18 MB), and `generate` then downloads them mid-flight —
the exact late-fetch `models_ensure` was built to front-load. Conversely the
`models verify` CLI verb (`cli.py:337-372`) checks all ten, so the two
preflight surfaces disagree on what "ready" means.

## Rationale

Doctor is the provisioning preflight (`voyage doctor` before `generate`); a
`models_ok` that cannot see the default CUDA stacks misleads capacity planning
(disk, download time, offline readiness) on exactly the runs that need it. The
spec set grew (film/realesrgan/sfx/awq are all post-Qwen additions); the
hardcoded tuple did not follow. Table-driving the tuple off `MODEL_SPECS` (or
the ensure set) removes the skew class permanently.

## Live evidence (verified live 2026-09-30, host rg)

- `rg -n '"film"|"realesrgan-anime"|"sfx-mmaudio"|"director-qwen4b-awq"' voyage/doctor.py`
  → zero hits (the four specs appear nowhere in the file).
- `rg -n 'verify_(film|realesrgan|sfx|director_awq)' voyage/cli.py` → all four
  wired in `cmd_models verify` — the CLI surface is complete, doctor lags it.
- `sed -n '178,183p' voyage/doctor.py` — `models_ok` is the conjunction over
  `checks.values()` only, so absent specs default to ready rather than unknown.

## Repro

1. Provision a models dir with only the six doctor-covered stacks (no
   `frame_interpolation/`, no `realesrgan/`, no `sfx/`, no AWQ snapshot).
2. `voyage doctor` → `models_ok: true`.
3. `voyage generate --backend ltxv …` (CUDA) → ensure downloads film +
   realesrgan-anime + sfx-mmaudio + director-qwen4b-awq before the first
   segment — gigabytes doctor said were ready.

## Fix candidates

1. (Preferred) Derive the verifier list from `MODEL_SPECS` (spec name →
   `verify_model`) instead of the handwritten tuple, so new specs are covered
   by construction; keep per-spec friendly messages.
2. Alternatively append the four missing verifiers to the tuple now (small,
   matches the `models verify` verb one-to-one).
3. Decide the `models_ok` semantics for the fake path explicitly (fake needs no
   weights — `models_ok: false` on an empty dir is arguably correct for CUDA
   but noise for smoke runs; document whichever is chosen).
4. Tests: models dir with only the six legacy stacks → the four new checks
   report not-ok; full dir → `models_ok: true`.

## Refs

- In-tree: `voyage/doctor.py:117-162` (`check_models`); `voyage/model_registry.py:
  742-1041` (`MODEL_SPECS` keys), `:1201-1342` (per-spec verify wrappers);
  `voyage/models_ensure.py:67-139` (the authoritative need-set);
  `voyage/cli.py:337-372`.
- Not-a-duplicate: 066 is probe location (root disk) + conjunction shape +
  no-alerts — the *which-specs* set is a different axis (066 predates the four
  specs' default-on status); 146 is the `models list` *printout* omitting
  film/realesrgan (different verb, and it never names sfx/awq/doctor);
  065 is INSTALL/README download lists (docs, different files). Extend 065's
  mirror test to cover doctor if convenient.

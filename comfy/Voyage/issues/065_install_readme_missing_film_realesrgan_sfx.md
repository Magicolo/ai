# 065 — INSTALL.md / README model-download lists are stale: missing `film`, `realesrgan-anime`, `sfx-mmaudio`

**Severity:** MEDIUM

**File:line:** `docs/INSTALL.md:81-86`; `README.md:41-46` vs `docs/MODELS.md:45,57,67,81,94,106`, `voyage/doctor.py:113-120`, `voyage/model_registry.py` (FILM / RealESRGAN / SFX specs)
- **Description:** INSTALL and README list six `models download` targets (longlive2, ltxv, causvid, director-qwen8b, audio-acestep, inspector-qwen35). MODELS.md plus the registry additionally define `film` (~66 MB), `realesrgan-anime` (~18 MB), and `sfx-mmaudio` (~13 GB) — the finalize-augmentation pair (now default floors: ≥32 fps, ≥720p) and the SFX pass. An operator following INSTALL under-provisions exactly the stacks `generate` auto-ensures or finalize needs, then hits a late download-or-fail.
- **Rationale:** Missing runbooks / stale install docs were the explicit sweep target. The single-source-of-truth rule ("All pins live in code in `model_registry.py`; this file mirrors them") is already violated in the other direction — the two entry-point docs are a subset, not a mirror.
- **Evidence (re-verified live 2026-09-30):** `grep -n "models download" docs/INSTALL.md README.md` → 6 rows each, zero hits for `film|realesrgan|sfx`; same grep on `docs/MODELS.md` → rows incl. `film` (:45), `realesrgan-anime` (:57), `sfx-mmaudio` (:106). `doctor.check_models` verifies six stacks (`doctor.py:113-120`: longlive2/ltxv/causvid/director/audio/inspector) — film/realesrgan/SFX presence isn't even summarized, compounding the blind spot. (Sweep cited `INSTALL.md:80-87`; live is `:81-86` — same six rows.)
- **Repro:** Diff the three download lists; or follow INSTALL verbatim on a fresh `~/.cache/voyage-models`, then `generate --backend ltxv` with default augmentation/SFX → on-demand fetch or failure that INSTALL never mentioned.
- **Fix candidates:** Add the three rows to INSTALL + README (with sizes/licenses: BSD/MIT+Apache vs CC-BY-NC-4.0 for SFX), add them to `check_models`/doctor summary, and add a mirror-freshness test (every registry `MODEL_SPECS` key appears in INSTALL/README/MODELS).
- **Refs:** `docs/MODELS.md:45-106`; `docs/INSTALL.md`; `README.md:41-46`.

**Overlaps with:** 146 (models list omits film/realesrgan — same stale-list root cause; recommend merging into one).

## Progress log

- 2026-09-30 (surface-rank2 track): premise re-verified live in `voyage:latest` — INSTALL.md and README.md each list 6 `models download` rows (longlive2-bf16, ltxv-2b, causvid, director-qwen8b, audio-acestep, inspector-qwen35), zero hits for film/realesrgan/sfx-mmaudio; MODELS.md already carries film (:45), realesrgan-anime (:57) and sfx-mmaudio (:106); MODEL_SPECS has 10 keys. Verdict: premise CONFIRMED, plus one adjacent gap found live — `director-qwen4b-awq` (MODEL_SPECS key, download/verify supported) is mirrored nowhere (no doc, no list row; overlaps 196's MODELS-AWQ scope, download-row only here).
- Fix: added 4 rows (film ~66 MB MIT+Apache-2.0, realesrgan-anime ~18 MB BSD-3-Clause, sfx-mmaudio ~13 GB CC-BY-NC-4.0, director-qwen4b-awq ~2.6 GB) to `docs/INSTALL.md` + `README.md`; added a `director-qwen4b-awq` mirror subsection to `docs/MODELS.md`; rewrote the `models list` verb to print full download-target names plus the two 146 augment rows. New mirror test pins every MODEL_SPECS key in all three docs + list output (`tests/test_surface_rank2.py`).
- Evidence: `tests/test_surface_rank2.py` 20 passed; ruff check + format-check + mypy strict clean on all touched files; neighboring suites (phase3/feedback/tui/phase2/supervisor-hardening/cli-tui-split/backend-registry/registry-pins/augment-models/cuda-preflight) 198 passed. Full suite: 793 passed + 1 failure in the concurrent group's new `test_media_robustness_rank2.py` (media area, untouched by this track; passes in isolation — order-dependent flake).
- Not done (out of owned files): `doctor.check_models` still summarizes 6 stacks (film/realesrgan/SFX/AWQ absent) — owner of `voyage/doctor.py` to extend; see DESIGN proposal in the track report.

## Resolution

- FIXED 2026-09-30: INSTALL/README/MODELS now mirror all 10 MODEL_SPECS keys; `models list` discovers every `models download` target. 146 FOLDED into this fix (CLI surface of the same omission; its two rows landed in the list verb and its case is covered by the shared mirror test).

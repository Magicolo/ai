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

# 107 — `ARCHITECTURE.md` commit pipeline omits the inspector piggyback + feedback-amendment stage

- **Severity:** Low (docs — experimental/off-by-default path, but the pipeline is presented as the per-segment contract)
- **File:line:** `Voyage/docs/ARCHITECTURE.md:31-36`; `Voyage/voyage/supervisor.py:1439-1481` (step 1b), `:949`, `:1293-1301`
- **Description:** ARCHITECTURE's "Commit pipeline (per segment)" reads `director decide → staged prompt plan → video generate_blocks → audio coverage → validate_video + validate_audio + A/V drift check → metadata + sha256.json → DONE → state advance → resource_gauges event`. The live code runs an ordered synchronous inspect step *before* the director decision on every segment when `[experimental] visual_inspector` is on: `_inspect_previous_segment` (supervisor.py:1293, DESIGN §44) samples the previous segment's committed video, merges measured metrics into the director context, and returns `amendments` that are applied to the staged text (supervisor.py:949 via `apply_feedback_amendments`) and that *invalidate the prefetched proposal* (supervisor.py:1478-1481: `if amendments: prefetched_raw = None`). None of this — the extra worker RPC per segment, the `inspect` stage timing in `stage_seconds`, the prefetch-invalidation edge — appears in the pipeline. An operator enabling the inspector from BACKENDS.md ("Qwen3.5-9B reads the previous segment's middle frame… retry→skip, never blocks a commit") gets no picture of where it sits or what it can change.
- **Rationale:** 091 files the missing SFX/AUGMENT operator docs and 092 the ARCHITECTURE layout gaps (`audio/sfx/` stems, `sfx.jsonl`, `video_tail.mp4`, augment artifacts, 2-GPU pairing) — neither mentions the inspector's place in the commit pipeline. The `paths.py:28-31` line citations in the same file were re-verified live and are exact, so this is a content omission, not citation rot. Small fix (one pipeline line + one sentence on prefetch invalidation), keeps the architecture doc truthful as the experimental flag gets used.
- **Evidence (verified live 2026-09-30):**
  - `docs/ARCHITECTURE.md:33-36` — pipeline with no inspect/amend/style step; `grep -n "inspector\|style_similarity\|feedback" docs/ARCHITECTURE.md` → zero hits.
  - `voyage/supervisor.py:1439-1446` (docstring: "folds in the previous segment's visual inspect"), `:1470-1471` (step 1b piggyback inspect inside `self._stage("inspect", …)`), `:1478-1481` (amendments invalidate prefetch), `:949` (amendments applied to stage text).
  - `docs/OPERATIONS.md:194` confirms the inspector runs "only when enabled" — consistent with the flag, but no doc shows its pipeline position.
- **Repro:**
  ```bash
  sed -n '31,36p' docs/ARCHITECTURE.md
  grep -n "_inspect_previous_segment\|prefetched_raw = None" voyage/supervisor.py
  ```
- **Fix candidates:**
  1. Insert the stage: `… → [inspect previous segment + amendments →] director decide → …`, plus one sentence: amendments apply post-validation pre-style-check and invalidate the prefetched proposal; failures degrade to `inspect_skipped` and never block the commit.
  2. Cross-ref `docs/BACKENDS.md:94-98` (inspector backend) and DESIGN §44 from the new line.
  3. Gate: `grep -n "inspect" docs/ARCHITECTURE.md` non-empty; gates green (docs only).
- **Refs:** `Voyage/issues/091_new_docs_sfx_augment.md` (SFX/AUGMENT operator docs — adjacent gap, fix separately); `Voyage/issues/092_docs_oneline_batch_qual_leg.md` (ARCHITECTURE layout items — no pipeline item); DESIGN §44; `Voyage/voyage/supervisor.py:1285-1301`.

## Progress log (Group C, 2026-09-30)

- Verdict: CONFIRMED live on the docs side. `docs/ARCHITECTURE.md:31-36`
  pipeline had no inspect/amend/style step and `grep -n -i
  "inspector|feedback|amend"` on the file returned zero hits; the
  `docs/BACKENDS.md:96-100` inspector backend (retry→skip, never blocks a
  commit) confirms the flag-gated behavior the pipeline must place.
  Supervisor cites (`:1439-1481`, `:949`, `:1293-1301`) are as-recorded —
  the file is under concurrent migration (CLI split landed), so the exact
  lines were not re-read; the docs edit cites BACKENDS + DESIGN §44 only.
- Fix: pipeline line prefixed with the bracketed inspect stage + one
  paragraph (amendments apply post-validation pre-style-check, invalidate
  the prefetched proposal, degrade to `inspect_skipped`, never block;
  cross-refs BACKENDS inspector section + DESIGN §44), per fix candidate 1-2.
- Files changed: `docs/ARCHITECTURE.md`.
- Gates: `grep -n "inspect" docs/ARCHITECTURE.md` non-empty; docs-only.

## Resolution

- Done. Residual: none in this file's scope (supervisor line re-verification
  belongs to any future code-side change of the inspect step).

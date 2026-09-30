# 042 — Docs drift risk: README geometry vs finalize augmentation floors + `docs/` staleness surface

- Severity: LOW (docs, but user-facing)
- Files: `Voyage/README.md:59,81`, `Voyage/docs/*.md` (12 files), `Voyage/DESIGN.md`, `Voyage/TASK.md`
- Area: docs
- Overlaps with: 092 (docs one-liner batch incl. qual leg — adjacent, not a duplicate)

## Description

README pins `ltxv (the default) renders 768x512 @ 24 fps` (:59) and
`--draft (640x352 …)` (:81), while the finalize augmentation default
(AGENTS.md §11, 2026-09-30) ships every video at >=32 fps + >=1280x720
(`AugmentConfig`, `--min-fps/--min-resolution/--no-augment`). A reader
following README expects 768x512@24 out; they get upscaled/augmented video
unless they pass `--no-augment`. Whether `docs/OPERATIONS/BACKENDS/
ARCHITECTURE` already describe the floors was not verifiable in this pass —
the surface (12 docs + DESIGN + TASK + README) has no drift-check, and §12
requires DESIGN updates with every behavior change.

## Rationale

Stale geometry numbers cause real user confusion (resolution/fps expectations
drive downstream tooling). The repo already learnt this lesson once
(slice-4 "default shift breaks other agents' finalize-geometry expectations
— own the fix").

## Live evidence (re-verified 2026-09-30)

`Voyage/README.md:56-61,81` read live:

```bash
VOYAGE_GPUS=1 ./scripts/run.sh generate --backend ltxv --duration 5s \
  --style "pastel neon line-art, peaceful"
# -> ./output/voyage/final.mp4 (run dir defaults to ./output/<run-id>)
# ltxv (the default) renders 768x512 @ 24 fps; --backend fake needs no GPU.   # :59
Fast iteration: add `--draft` (640×352, 1 block/segment, 45 s takes).          # :81
```

AGENTS.md §11 finalize-floors entry (2026-09-30): every shipped video is
`>=32fps + >=1280x720` by default with `--no-augment` escape hatch.
`ls docs/` = 12 files. The sweep left `rg -n "768x512|24 fps|32fps|1280"
README.md docs/ DESIGN.md` as the verifier's one-liner (not run to keep the
pass read-bounded — flagged as the check to run).

## Repro

```bash
rg -n "768.?512|1280|24 ?fps|32 ?fps|min-fps|min-resolution|no-augment" README.md docs/ DESIGN.md voyage/*.py | head -n 40
```

## Fix candidates

(a) Update README run/finalize sections with the floors + `--no-augment`
escape hatch.
(b) Add a docs-drift checklist line to the behavior-change definition of
done (DESIGN + README + affected `docs/` file in the same commit).
(c) Pin legacy-test geometry notes (`min_*=0` fastpath) in OPERATIONS so
future default shifts do not re-break foreign suites.

## Refs

- AGENTS.md §12 ("Every behavior change updates DESIGN.md (+ AGENTS.md if
  non-obvious)"); §87 docs-tree contract.
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §12.

## Progress log (Group C, 2026-09-30)

- Verdict: premise FIXED already by a concurrent agent — `README.md:62-64`
  live reads "renders native 768x512 @ 24 fps; finalize lifts to
  >=1280x720 @ >=32 fps via the augmentation floors (see docs/AUGMENT.md;
  --no-augment keeps native geometry)". Fix candidate (a) is done.
- Live sweep `rg -n "768.?512|1280|24 ?fps|32 ?fps|min-fps|min-resolution|no-augment"
  README.md docs/` confirms floors documented in README:62-64,93,131,
  OPERATIONS:174,219-223, AUGMENT (full file), BENCHMARKING:49-53.
- One residual of the same class found in owned docs: `AUGMENT.md:124`
  still said fake testsrc "(320x180-class)" (live preset is 768x432
  fake-432p, `config.py:167-179`) — fixed to "(768x432 fake-432p)".
- Files changed: `docs/AUGMENT.md` (one line).

## Resolution

- README geometry drift: no action (already correct).
- `docs/AUGMENT.md:124`: fake-geometry label corrected to the live preset.
- No DESIGN change needed (design already records the floors entry).
- Residual: none in this file's scope (BACKENDS fake cell is 159's fix).

# 093 — TASK.md prune-merge + delete (1737L frozen brief → DESIGN, open items only)

- Severity: MEDIUM (docs — archive, don't extend)
- Files: `Voyage/TASK.md` (1737 lines, §§1-30), `Voyage/DESIGN.md` (7577 lines, §§1-140 + §§137A-D — sweep said 7538)
- Area: docs — brief retirement
- Decision: Q&A locked — prune and delete (migrate open items only, not verbatim)

## Description

TASK is a frozen 2026-09-23 backend-audit brief + §30 merge checklist (2026-09-24). §30.4 "artifacts never produced" is now false (both reports exist); §30.2 follow-ups partially resolved (LTXV drift, adapter, artifacts, xrefs). Structure: §§1-4 background/baselines/rules → §§5-16 phased audit A–M → §§17-20 LTXV/Causvid/cross-backend → §§21-29 benchmarks/targets/acceptance → §30 transition checklist (30.1 Causvid, 30.2 LTXV drift, 30.3 adapter, 30.4 artifacts, 30.5 xrefs). DESIGN already merged the backend-neutral proposal (+613/-64: §§5/11/14/24/48/59/85/86/118-120/128/136-139 + §§137A-D, §22.5 + §140 verbatim). Remaining relation is checklist-vs-log (§30 vs §140) and brief-vs-spec (TASK §§2/17-19 vs DESIGN §5/§§137A-D) — historical duplication, not live spec.

## Rationale

Two authorities (brief + spec) drift by construction; §30's false "never produced" + stale follow-ups mislead. Deletion without migration loses the open tail; verbatim merge preserves done noise. Prune-merge keeps the open tail exactly once.

## Live evidence

- `wc -l TASK.md DESIGN.md` → 1737 / 7577; `grep -n "§30\|30\." TASK.md | head -n 20`
- `grep -n "artifacts never produced\|pending Stream A" TASK.md reports/video-backends.md`

## Repro

```bash
grep -n "^#\|^## " TASK.md | head -n 40
grep -n "137A\|137B\|137C\|137D\|§30" DESIGN.md TASK.md | head -n 30
```

## Fix candidates

1. Migrate open §30 items only into DESIGN (§140 entry + appropriate §§ for any surviving normatives); prune done/resolved; record mapping (TASK §30.x → DESIGN §y) in the migration commit message.
2. Delete `Voyage/TASK.md`; repair §30 cross-refs (`grep -rn "TASK" docs/ DESIGN.md README.md`); keep history in git (`git log --oneline -- Voyage/TASK.md`).
3. Gate: `grep -rni "TASK.md" Voyage/ --include="*.md" --include="*.py"` only historical pointers + `gates.sh` green.

## Refs

- DESIGN §§137A-D, §140; `reports/video-backends.md`, `reports/longlive-audit.md`; AGENTS.md §9 (per-commit approval)

## Progress log (2026-09-30, resolution pass)

- As-read drift confirmed: `wc -l TASK.md DESIGN.md` → **1737 / 7823**
  (issue said 7577 — DESIGN grew). Both reports exist
  (`reports/longlive-audit.md`, `reports/video-backends.md`), and §30.2
  follow-ups kept resolving after the issue was filed
  (`UPSTREAM_LTXV_NOTES.md` landed 2026-09-29 per §30.2's own text).
- TASK-only slice applied: §30.4 heading corrected from "never
  produced" to "produced (2026-09-24+…)" with a body noting which
  artifacts exist, what stays open (81/97/121 matrix, TeaCache/Q8/FP8,
  extension-throughput), and that full prune-merge + delete is tracked
  here, not done. Zero `TASK.md` cross-refs elsewhere
  (`rg TASK.md` outside issues/ → no hits), so no ref repair was needed.
- Full fix deliberately not attempted: migration needs writes to
  `DESIGN.md` (frozen for this pass) and deletion needs per-commit
  approval (AGENTS §9) — both out of scope.

## Resolution

- Partially resolved (stale "never produced" claim fixed in place).
  Residual: migrate open §30 items into DESIGN §140 + appropriate §
  normatives, record the §30.x → §y mapping in the migration commit
  message, delete `Voyage/TASK.md`, then run the issue's gate (`TASK.md`
  mentions only historical + `gates.sh` green).

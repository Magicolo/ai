# Voyage — Video Backend Audit and Generator Substitution Task (RETIRED 2026-09-30, batch 12)

This brief is fully merged and carries no live content. Do not extend it —
`DESIGN.md` is the single authority; this file stays only until its
`git rm` is user-approved (AGENTS.md §9), tracked in
`Voyage/issues/093_task_prune_merge_delete.md`.

- Full text lives in git history: `git log --oneline -- Voyage/TASK.md`
  (frozen 2026-09-23 brief, §§1-29 audit phases A–M + LTXV/Causvid build
  phases; §30 transition checklist added 2026-09-24, §30.4 heading
  corrected 2026-09-30).
- Open-item homes (all live content moved exactly once):
  - §30.1 Causvid opens (overlap sweep, resume A/B, 24 fps finalize
    stage, eyeball) → `DESIGN.md` §§137D/128 + the Causvid worker slice.
  - §30.2 LTXV opens (81/97/121 matrix, TeaCache/Q8/FP8 study,
    extension-throughput, eyeball) → `DESIGN.md` §5.3 deferred +
    the §137A qualification harness (`scripts/qualify.sh`,
    `docs/BENCHMARKING.md` "Backend qualification").
  - §30.3 config/interface duality → resolved via the
    `VideoBackendAdapter` contract (`voyage/backends.py`, DESIGN §140
    Stream C entry).
  - §30.4 artifacts → produced: `reports/longlive-audit.md`,
    `reports/video-backends.md`; remaining breadth = the §30.2 opens.
  - §30.5 cross-references → pointers only, no action.
- Dangling refs: none — `rg TASK.md` outside `Voyage/issues/` shows only
  this file's self-pointer (retired) and historical DESIGN §140 entries.
- Pending user approval: `git rm Voyage/TASK.md` (deletion needs
  per-commit approval; nothing is lost — history keeps every line).

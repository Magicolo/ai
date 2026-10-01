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

## Progress log — 2026-09-30 (this pass, DESIGN owner)

- Re-verified live: `TASK.md` 1746 lines (§§1-30 intact);
  `DESIGN.md` 8010 lines; §30.4 heading already corrected (batch 7);
  `rg TASK.md` outside `issues/` shows zero non-historical refs (no ref
  repair owed). Premise HOLDS — DESIGN-side write still open, TASK-only
  slice already landed.
- Files changed: `Voyage/DESIGN.md` append-only (new `## 2026-09-30 —
  Issue 093` §140 entry recording the §30.x to DESIGN §y mapping;
  zero deletions).

## Resolution — 2026-09-30 (this pass)

- Verdict: DESIGN-PART FIXED, DELETION DEFERRED. Mapping recorded in
  the DESIGN entry (§30.1 to §§137D/128 + worker slice; §30.2 to §5.3
  deferred + §137A harness; §30.3 to Stream C adapter entry; §30.4 open
  breadth; §30.5 pointers only).
- Deletion approval stays PENDING — `Voyage/TASK.md` NOT deleted by
  this pass (per-commit approval per AGENTS §9); recorded explicitly,
  nothing deleted.
- Test evidence: docs-only; gate is `rg TASK.md` (zero hits outside
  `issues/`) + full `gates.sh` by the committer at delete time.

## Progress log (2026-09-30, batch 12 — merge owning pass)

- Re-verified live: `TASK.md` was 1746 lines (§§1-29 frozen brief +
  §30 checklist); `DESIGN.md` 8010+ lines with the batch-9 §140 093
  entry intact (`DESIGN.md:8052-8054`, read-only check — DESIGN.md not
  touched per contract). Every §30 live item confirmed homed (read-only
  greps): §30.1 opens (overlap sweep, resume A/B, 24 fps finalize,
  eyeball) in §§137D/128 (`DESIGN.md:7274`); §30.2 opens (81/97/121
  matrix, TeaCache/Q8/FP8, extension-throughput, eyeball) in §5.3
  deferred (`DESIGN.md:554,487`) + §137A harness (`DESIGN.md:5984`,
  `scripts/qualify.sh`); §30.3 adapter resolved
  (`voyage/backends.py:VideoBackendAdapter`, §140 Stream C entry);
  §30.4 artifacts produced (`reports/longlive-audit.md`,
  `reports/video-backends.md`); §30.5 pointers only. §§1-29 are
  one-shot audit history (phases A–M ran, reports exist) — no live
  normative content; nothing there belongs in `docs/` (the §87 docs
  tree + `docs/BENCHMARKING.md` already carry the durable protocol).
- Dangling refs re-verified: `rg TASK.md` outside `Voyage/issues/`
  shows only the stub's own self-rows (history pointer + git-rm
  note — retired, not live) and the
  historical DESIGN §140 entries (`DESIGN.md:6924,8052-8054`) — no ref
  repair owed. (Full inventory for the record: `DESIGN.md:527,554,7169`
  cite TASK §§4.2/4.3 as executed-methodology history, `:6924`
  chronicles the §30 checklist merge, `:8052-8062` chronicle this
  issue's DESIGN write — all historical, zero live-spec refs.)
- Merge executed (owned files only): (1) TASK §30.4's procedural
  "run the §137A qualification" instruction had no `docs/` home, so
  `docs/BENCHMARKING.md` gains a "Backend qualification" section
  (driver usage, absolute-path rule, manual recovery/eyeball, report
  locations, contention rule — all restated from
  `scripts/qualify.sh:1-20`, no new claims); (2) `TASK.md` replaced
  by a pointer stub (title + git-history pointer + §30.x→DESIGN §y
  mapping + zero-dangling-refs note + deletion-approval note) —
  the file stays (no `git rm`), history keeps every line, and the
  dual-authority drift hazard ("two authorities drift by
  construction") is gone. No test reads `TASK.md` content (verified:
  only `tests/test_causvid_prep.py:1` docstring mentions TASK §30.1 —
  historical, left intact), so the stub is test-safe. No behavior
  change → no TDD failing test (docs-only, same gate as batch 9).
- Gates: `docs/BENCHMARKING.md` prose-only (no python → ruff/mypy
  N/A); related evidence — owned longlive suites green in-container
  (51 passed, 3 skipped, shared baseline run); `rg TASK.md` outside
  `issues/` shows only the retired self-pointer + historical DESIGN
  entries as predicted.

## Resolution (2026-09-30, batch 12)

- Verdict: FIXED (merge complete) + DELETION PENDING (user approval).
  Files changed: `Voyage/TASK.md` (1746L brief → pointer stub, no live
  content left), `Voyage/docs/BENCHMARKING.md` (+"Backend
  qualification" section — the §30.4 procedural home).
- DESIGN proposals (text only, no DESIGN.md write per contract): none
  outstanding — the batch-9 §140 entry already homes every §30 item;
  no further DESIGN edit is owed by this merge. (If a future pass
  wants it: append one §140 line noting `TASK.md` is a retired stub
  pending `git rm` approval — optional, cosmetic.)
- Residuals: exactly one — `git rm Voyage/TASK.md` needs explicit
  user approval per AGENTS.md §9 (recorded in the stub itself, so the
  approval can land without re-reading this issue). Nothing else is
  open: no live content unhomed, no dangling refs, no follower
  edits.

## Progress log (2026-09-30, batch 13 — verify-only, this pass)

- Re-verified live (host reads, no edits — deletion forbidden without
  approval): `TASK.md` is the 28-line pointer stub (title + RETIRED
  note + git-history pointer + §30.x→DESIGN §y mapping + rm-pending
  note — zero live content, unchanged since batch 12);
  `docs/BENCHMARKING.md` "Backend qualification" section intact
  (`:69`, the §30.4 procedural home); `rg TASK.md` outside
  `Voyage/issues/` shows only the stub's self-rows + historical
  DESIGN §140 entries (`DESIGN.md:6926,8092` — methodology history
  + batch-12 as-built, zero live-spec refs) — no ref repair owed.
- No test reads `TASK.md` content (only the historical
  `test_causvid_prep.py:1` docstring mention — left intact).
  No deletion performed (`git rm` stays pending).

## Resolution (2026-09-30, batch 13)

- Verdict: VERIFIED (stub + BENCHMARKING home intact) + DELETION STILL
  PENDING (user approval). Files changed: none (this issue file only).
  Gate evidence: `rg TASK.md` outside `issues/` — only retired
  self-pointer + historical DESIGN entries as predicted.
  DESIGN proposals: none.
- Residuals: exactly one, unchanged — `git rm Voyage/TASK.md` needs
  explicit user approval per AGENTS.md §9.

## Progress log (2026-10-01, this pass — verify-only, no deletion)

- Re-verified live (host reads + `git ls-files`, no edits — `git rm`
  forbidden without user approval per AGENTS §9): `TASK.md` is the
  28-line pointer stub (unchanged since batch 12; tracked, not
  ignored); `docs/BENCHMARKING.md:69` "Backend qualification" section
  intact (the §30.4 procedural home); DESIGN historical entries
  `:6929` + `:8095` intact (methodology history + batch-12 as-built,
  zero live-spec refs).
- Dangling-ref sweep: `grep -rn "TASK\.md"` over `tests/` +
  `voyage/` (excl. `issues/`) → only the stub's self-rows
  (title/history-pointer/rm-pending note) + the 2 historical DESIGN
  entries — no ref repair owed. `VALID_TASK_TYPES` hits in
  `test_acestep_contract.py` are ACE-Step task types, unrelated to
  `TASK.md`. `test_causvid_prep.py:1` docstring cites "TASK §30.1"
  (historical, left intact).
- No test reads `TASK.md` content: no `open`/`Path`/`read` of
  `TASK.md` anywhere in `tests/` or `voyage/` (grep clean) — the stub
  is test-safe.
- Ready-to-execute state for the orchestrator (NOT run here): from
  repo root `/home/goulade/Projects/ai`, `git rm
  comfy/Voyage/TASK.md` (deletion needs explicit user approval per
  AGENTS §9; the standing continue-until-clean directive covers
  execution once approved). Post-rm verification: `grep -rn
  "TASK\.md" Voyage/ --include="*.md" --include="*.py" | grep -v
  "issues/"` must show only the 2 historical DESIGN entries, then
  `Voyage/scripts/gates.sh` green. Nothing is lost — full text in
  `git log --oneline -- Voyage/TASK.md` (recorded in the stub).

## Resolution (2026-10-01, this pass)

- Verdict: VERIFIED (stub + BENCHMARKING home intact) + DELETION
  READY (exact command + post-conditions above; approval still
  pending). Files changed: none (this issue file only). DESIGN
  proposals: none.
- Residuals: exactly one — `git rm` execution by the orchestrator
  under approval. Nothing else is open.

## Progress log (2026-10-01, this pass — verify-only, no deletion)

- Re-verified live (host reads, no edits — `git rm` forbidden without
  user approval per AGENTS §9): `TASK.md` is the 28-line pointer stub
  (unchanged since batch 12; tracked as `comfy/Voyage/TASK.md` from repo
  root `/home/goulade/Projects/ai`); `docs/BENCHMARKING.md:69` "Backend
  qualification" section intact (the §30.4 procedural home);
  historical DESIGN entries `:6930` + `:8058-8062` + `:8096` intact
  (methodology history + batch-12 as-built, zero live-spec refs;
  `DESIGN.md:528,555,7175` cite TASK §§4.2/4.3 as executed-methodology
  history only).
- Dangling-ref sweep: `grep -rn "TASK\.md" --include="*.md"
  --include="*.py" . | grep -v "issues/"` → only the stub's self-rows
  (`TASK.md:8` history pointer, `TASK.md:27` rm-pending note) + the 3
  historical DESIGN entries — no ref repair owed. `VALID_TASK_TYPES`
  hits (if any) are ACE-Step task types, unrelated. No test reads
  `TASK.md` content (no `open`/`Path`/`read` of `TASK.md` in `tests/` or
  `voyage/` — grep clean; only historical docstring mentions remain).
- Ready-to-execute state for the orchestrator (NOT run here): from repo
  root `/home/goulade/Projects/ai`, `git rm comfy/Voyage/TASK.md`
  (deletion needs explicit user approval per AGENTS §9). Post-rm
  verification: `grep -rn "TASK\.md" Voyage/ --include="*.md"
  --include="*.py" | grep -v "issues/"` must show only the historical
  DESIGN entries, then `Voyage/scripts/gates.sh` green. Nothing is lost
  — full text in `git log --oneline -- Voyage/TASK.md` (recorded in the
  stub).

## Resolution (2026-10-01, this pass — re-verified)

- Verdict: VERIFIED (stub + BENCHMARKING home intact) + DELETION READY
  (exact command + post-conditions above; approval still pending).
  Files changed: none (this issue file only). DESIGN proposals: none.
- Residuals: exactly one — `git rm` execution by the orchestrator under
  approval. Nothing else is open.

## Progress log (2026-10-01, record-only maintenance pass — verify-only, no deletion)

- Re-verified live (host reads, no edits — `git rm` NOT run, needs user
  approval per AGENTS §9): `TASK.md` is the 28-line pointer stub
  (unchanged since batch 12); `docs/BENCHMARKING.md:69` "Backend
  qualification" section intact (the §30.4 procedural home); historical
  DESIGN entries `:6931` + `:8113` intact (methodology history +
  batch-12 as-built, zero live-spec refs).
- Dangling-ref sweep: `grep -rn "TASK\.md" --include="*.md"
  --include="*.py" Voyage/ | grep -v "issues/"` → only the stub's
  self-rows (`TASK.md:8` history pointer, `TASK.md:27` rm-pending note)
  + the historical DESIGN entries — no ref repair owed.
- No test reads `TASK.md` content: no `open`/`Path`/`read` of `TASK.md`
  anywhere in `tests/` or `voyage/` (grep clean) — the stub is test-safe.
- Ready-to-execute state for the orchestrator (NOT run here): from repo
  root `/home/goulade/Projects/ai`, `git rm comfy/Voyage/TASK.md`
  (deletion needs explicit user approval per AGENTS §9). Post-rm
  verification: `grep -rn "TASK\.md" Voyage/ --include="*.md"
  --include="*.py" | grep -v "issues/"` must show only the historical
  DESIGN entries, then `Voyage/scripts/gates.sh` green. Nothing is lost
  — full text in `git log --oneline -- Voyage/TASK.md` (recorded in stub).

## Resolution (2026-10-01, record-only maintenance pass)

- Verdict: VERIFIED (stub + BENCHMARKING home intact) + DELETION READY
  (exact command + post-conditions above; approval still pending).
  Files changed: none (this issue file only). DESIGN proposals: none.
- Residuals: exactly one — `git rm` execution by the orchestrator under
  approval. Nothing else is open.

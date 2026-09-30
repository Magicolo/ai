# 108 — Stray root-owned empty `Voyage/Voyage/` residue dir: invisible to git, provenance unknown

- **Severity:** Low (repo-root hygiene — empty today, same class as the 053 root-ownership residue)
- **File:line:** `Voyage/Voyage/` (directory, not a file)
- **Description:** An empty directory `Voyage/Voyage/` sits in the tree, owned by `root:root` (mode 755, mtime 2026-09-29 22:03). It is empty (nothing to keep), untracked yet invisible (`git status` never reports empty dirs; `git check-ignore` reports no rule), and uncovered by `.gitignore` (which lists only `output/`) and `.dockerignore`. The only other root-owned entry under `Voyage/` is nothing — `find Voyage -user root` returns just this dir — so it is a lone anomaly, not a pattern. Provenance is open: root ownership points at a container run (pre-`--user` rollout or a bare `docker run` without it, same evening as the 2026-09-29 container-user rollout and excerpt runs), plausibly a CWD/path-doubling slip writing a relative `Voyage/…` path — but no log names it, so this file records the residue, not a mechanism.
- **Rationale:** 089 covers in-tree cache residue (`.coverage`/`.hypothesis`/`.mypy_cache`/`.ruff_cache`) and 069/090 the `.dockerignore` gaps, but neither names a stray root-owned directory or the git-invisibility of empty dirs. The host user *can* remove it (removal is governed by the parent: `Voyage/` is goulade-owned `drwxrwxr-x`), so the fix is one `rmdir` — the value here is recording the anomaly class (container-created, git-invisible residue) so the next one gets noticed instead of accumulating.
- **Evidence (verified live 2026-09-30):**
  - `ls -la Voyage/Voyage/` → `total 8, drwxr-xr-x 2 root root … Sep 29 22:03 ., ..` (empty).
  - `find Voyage -user root` → only `Voyage/Voyage`.
  - `git status --porcelain Voyage/Voyage/` → empty (git does not report empty dirs); `git check-ignore -v Voyage/Voyage` → empty (no ignore rule); `grep -n "output/" Voyage/.gitignore` → line 6 `output/` only.
  - `stat -c "%U %a %y"` → `root 755 2026-09-29 22:03:33`.
- **Repro:**
  ```bash
  ls -la Voyage/Voyage/; find Voyage -user root; git status --porcelain Voyage/Voyage/; echo "(empty = git-invisible)"
  ```
- **Fix candidates:**
  1. `rmdir Voyage/Voyage/` (host-owned parent permits it; verify `find Voyage -user root` returns nothing after).
  2. Optional: extend the 089-proposed post-gate cleanliness check with `find . -user root -not -path "./output/*"` so container-created residue fails loudly instead of sitting git-invisible.
  3. Do NOT add an empty-dir placeholder (no `.gitkeep` policy in this repo) — removal is the fix.
- **Refs:** `Voyage/issues/089_test_hygiene_lock_markers.md` (in-tree residue class); `Voyage/issues/090_scripts_containers_cleanup.md` (dockerignore gaps); DESIGN §140 "Containers run as the host user (issue 053 follow-up)" (the rollout this residue likely predates).

## Progress log (Group C, 2026-09-30)

- Verdict: ALREADY ABSENT — `rmdir Voyage/Voyage` on host →
  "No such file or directory" (exit 1); `find Voyage -maxdepth 2 -name
  Voyage` returns only the tree root itself; `ls Voyage/` shows no nested
  dir. A concurrent agent (or the 053-follow-up rollout) already removed
  it; no ownership block was hit because there is nothing to remove.
- No `rmdir` was needed and none was forced (contract: only if empty —
  vacuously satisfied).
- Recurrence guard (my scope): 167 added `Voyage/` to both
  `.dockerignore` and `.gitignore`, so a future misdirected output there
  stays uncommitted and out of the build context.
- Files changed: none.

## Resolution

- Closed (residue gone, guard in place via 167). Residual: none (the
  089-proposed post-gate root-ownership check stays with 089).

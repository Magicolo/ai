# 105 — `qualify.sh` idle gate aborts on idle GPUs: `pipefail` + `grep` no-match exits before the comparison

- **Severity:** Medium (scripts — qual harness unusable on exactly the condition it should admit)
- **File:line:** `Voyage/scripts/qualify.sh:16-21`
- **Description:** The contention gate pipes `nvidia-smi --query-compute-apps=used_memory` through `grep -oE '[0-9]+' | awk …`. On an idle GPU the query prints nothing, `grep` finds no digits and exits 1, and under `set -euo pipefail` the command-substitution assignment fails — the script aborts with exit 1 and **no diagnostic** before ever reaching the `[ "$held_mib" -gt 2048 ]` comparison. The same abort fires when `nvidia-smi` is absent. So the gate can never admit an idle box; it fails closed *and* silent, with no message distinguishing "GPU busy" from "gate script broke".
- **Rationale:** 064 analyzes this gate as fail-*open* ("yields 0 when `nvidia-smi` is absent … gate passes with `held_mib=0`, then `benchmark video` fails late") and proposes `command -v nvidia-smi || exit 4`. The live behavior is the opposite failure mode: the script dies at the assignment, so 064's repro (`PATH=/usr/bin:/bin ./scripts/qualify.sh …` → "gate passes") does not reproduce — it exits 1 with empty output. Both files agree the gate is broken, but the mechanism and the user-visible symptom here (silent abort on a healthy idle GPU, exit 1, no "GPU busy" line, no late failure — just nothing) are distinct and need a different fix (`|| true` / match-count instead of match-extract, not just an existence check).
- **Evidence (verified live 2026-09-30, host bash):**
  - `scripts/qualify.sh:15-21` as read (assignment + `-gt 2048` + `exit 3`).
  - Exact-pipeline probe with empty input (what idle `nvidia-smi` produces):
    `bash -c 'set -euo pipefail; held_mib="$(printf "" | grep -oE "[0-9]+" | awk "{s+=$1} END {print s+0}")"; echo "held=[$held_mib]"'` → prints nothing, `exit=1` (the `echo` never runs). `bash -n scripts/*.sh` all pass — syntax ≠ correctness.
- **Repro:**
  ```bash
  bash -c 'set -euo pipefail; held_mib="$(printf "" | grep -oE "[0-9]+" | awk "{s+=\$1} END {print s+0}")"; echo "held=[$held_mib]"'; echo "exit=$?"
  # → exit=1, no held= line. On a real idle-GPU box, ./scripts/qualify.sh <run-dir>
  # dies the same way before printing anything.
  ```
- **Fix candidates:**
  1. Neutralize the grep exit status: append `|| true` inside the substitution, or use `grep -c` / `awk`-only digit summation (`nvidia-smi … | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+$/) s+=$i} END {print s+0}'`), which exits 0 on empty input.
  2. Keep 064's `command -v nvidia-smi || exit 4` (fail closed with a message when the tool is absent) *in addition* — the two fixes compose.
  3. Echo the admitted path too (`qualify: GPU idle (0 MiB held) — proceeding`) so silent-abort regressions are visible.
  4. Gate: run the gate on an idle GPU (real `nvidia-smi`, zero compute apps) and assert exit 0 past the gate; run with `PATH` lacking `nvidia-smi` and assert exit 4 with a message.
- **Refs:** `Voyage/issues/064_qualify_sh_fragile_gate_paths_no_artifact.md` (overlapping file, opposite mechanism — fix once, keep both as record); `Voyage/scripts/qualify.sh:16-21`; repo GPU-contention rule (AGENTS.md §11).

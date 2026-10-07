# 242 — `boundary_metrics.main` conflates verdict-FAIL and tool-error on exit 1

Severity: MEDIUM (tracks E-07 + C-13a, joint entry).

## Technical description

`voyage/boundary_metrics.py:369-391`: `except (ValueError, FileNotFoundError, MediaError,
OSError) → return 1`; final line `return 1 if verdict == "FAIL" else 0`. Misuse is 2. So
a `FAIL` continuity gate, a missing `video.mp4`, and a torn manifest all exit 1 with only
stderr text differing.

## Rationale

CLI checklist guidance (exit 0 success / 1 error / 2 misuse, plus distinct codes for
distinct failures) exists so `qualify.sh`-style drivers and CI can gate on cuts without
tripping on infra errors. The module is explicitly the §137A gate
(`BOUNDARY_RATIO_LIMIT = 3.0`, `:44-50`) — conflating its two failure modes weakens the
gate. Two independent tracks converged on this finding.

## Live evidence

Read `main()` tail (`:378-391`): both `sys.stderr.write(…); return 1` and `return 1 if
summary.get("verdict") == "FAIL"` yield 1. `--json` still prints the document on the
verdict path but not on the error path — same code, different payloads.

Repro: run against a 2-segment PASS run with one `video.mp4` deleted (tool error → 1) vs
an intact run with ratio ≥ 3 (FAIL → 1) — identical exit codes. Also:
`python -m voyage.boundary_metrics --run /nonexistent; echo $?` → 1; vs a FAIL run → also
1.

## Source refs

`voyage/boundary_metrics.py:18-21,44-50,114-174,369-395`.

## Online sources

- CLI exit-code conventions (0 success / 1 general error / 2 misuse — jagrat7 CLI skill;
  Cody Ray "different exit codes … nice-to-have for scripting").

## Fix candidates

- `0` PASS/N/A, `1` FAIL verdict, `3` tool error (keep `2` misuse); print the JSON
  document on FAIL (already does with `--json`), keep stderr-only on tool error; update
  `qualify.sh`/tests pins. Or emit a machine-readable `{"error": …}` envelope on the json
  path.

## Log

- 2026-10-07: filed from read-only Tracks C + E sweeps (both tracks independently found
  it); no code touched.

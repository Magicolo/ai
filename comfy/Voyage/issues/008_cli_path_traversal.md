# 008 — CLI path traversal: `--run-id`/`--output`/`--run`/`--final-video` unvalidated

- Status: resolved in live tree (flat-name check + `resolve_run_dir` on every verb)
- Severity: HIGH (security / robustness; resolved, record only)
- Group: security/CLI — Rank: 2/5 (fixed; consumer-side remainder → 015)
- Area: supply chain + CLI input validation
- Rank rationale: writes outside the run tree from a crafted flag; the TUI already
  had the right check, the CLI didn't use it.

## Technical description

Pre-fix, the TUI validated `name` with `_flat_folder_name` (rejects `/`, `\`, `..`
— `voyage/tui_state.py:118-121` at pass 1), but CLI `cmd_init`
(`Path(args.output)` raw, `voyage/cli.py:101-102`) and `cmd_generate`
(`Path("output")/args.run_id`, no check) didn't. `--run-id ../../tmp/evil-run`
escaped; `--output /etc/cron.d/x` + `--force` wrote `voyage.toml`/`state.json`
anywhere the container user could write; `--final-video` in `cmd_finalize`
likewise unchecked.

Live state (re-verified 2026-09-30): `voyage/cli.py:83-130` —
`resolve_run_dir()` central helper, `is_flat_folder_name()` mirroring the TUI
check, `_check_run_id()` rejecting traversal at the CLI layer, wired into `init`
(`:172-175`) and `generate` (`:1249-1255`); finalize resolves through
`resolve_run_dir` (`:773`).

## Why this is an issue

A crafted `--run-id` or `--output` writes `voyage.toml` and `state.json` outside
the run tree — anywhere the container user can write — so one unsanitized variable
in a wrapper script escapes the output directory and can clobber unrelated files.
The TUI already rejected `/`, `\` and `..`, which made the CLI strictly weaker
than its sibling interface for the same operation. Since runs are routinely
created programmatically (`generate --run-id` from scripts and launchers), this
was a live confused-deputy gap rather than a theoretical one.

## Evidence

Live verification 2026-09-30:

```
$ python3 -c "from pathlib import Path; print((Path('output')/'../../tmp/evil-run').resolve())"
/home/goulade/Projects/ai/comfy/tmp/evil-run
```

Escapes `Voyage/output/` to a sibling of the checkout (traversal mechanics
unchanged — the guard is the flat-name check, not the path algebra):

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c \
    "from voyage.cli import is_flat_folder_name; ..."
flat evil: False
flat voyage: True
flat dot: False
```

```
$ rg -n "is_flat_folder_name|_check_run_id|resolve_run_dir" voyage/cli.py | head
81:def resolve_run_dir(value: str) -> Path:
99:def is_flat_folder_name(value: str) -> bool:
116:def _check_run_id(run_id: str) -> int:
118:    if not is_flat_folder_name(run_id):
170:    if _check_run_id(run_id) != 0:
1210:    if _check_run_id(run_id) != 0:
```

## Reproduction

1. Pre-fix: `voyage init --output output/../../tmp/evil-run --run-id x ... --force`
   → wrote outside the project tree. Now: rejected exit 2.
2. Pre-fix: `voyage generate --run-id ../../tmp/evil-run ...` → run dir outside
   `output/`. Now: `error: --name/--run-id must be a flat folder name`.

## Source references

- `voyage/cli.py:78-132` (helpers), `:169-172` (init), `:734` (finalize),
  `:1209-1215` (generate); `voyage/tui_state.py:141-183` (TUI good pattern);
  `voyage/supervisor.py:383-414` (run-relative consumers).

## Resolution candidates

1. (Landed) Apply the flat-folder check to `--run-id`/TUI `name` at the CLI layer;
   resolve + constrain `--output/--run/--final-video` (warn/refuse escapes).
2. (Landed) Centralize in one `resolve_run_dir()` helper used by every verb (also
   fixes 057).
3. Tests: traversal `--run-id` rejected; absolute `--output` inside tree accepted.

## Online references

- OWASP Path Traversal — "This attack ... allows access to files ... outside of
  the intended folder":
  https://owasp.org/www-community/attacks/Path_Traversal
- CWE-22 Improper Limitation of a Pathname to a Restricted Directory:
  https://cwe.mitre.org/data/definitions/22.html

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; traversal probe re-verified live.
- Resolution batch 2: traversal helper + verb wiring landed.
- 2026-09-30: re-verified live (probe + helpers present); reconstructed from
  archived pass-1 text (commit `b5d7dda`). Status → resolved.

# 080 — TUI `name="."` passes the flat-folder check, targets the shared `output/` dir

- Status: open
- Severity: low (run initializes inside the shared directory; `init --force`
  writes run files among all runs)
- Area: TUI validation — `voyage/tui_state.py:118-121,202-203`
- Rank rationale: pass-2 finding; distinct input class (single dot, no traversal
  substring) and consequence (collision, not escape) from 008.

## Technical description

```python
def _flat_folder_name(raw: str) -> bool:
    text = raw.strip()
    return bool(text) and "/" not in text and "\\" not in text and ".." not in text
```

`"."` contains neither `/` nor `..` → passes → `output = "output/."` →
resolves to shared `output/`.

## Why this is an issue

A single-dot name passes validation and initializes the run inside the shared
`output/` directory itself, writing run files among every other run's
artifacts — with `--force` potentially colliding with or clobbering them. It
is a different input class from traversal (no escape, but a collision), and the
same one-character gap exists at the CLI layer (`--run-id .`).

## Evidence (live probes by pass-2 sweep)

`name='.' → flat=True, err=None`; `Path('output')/'.'` → `'output'`; CLI
`--run-id .` → `output='output'` → resolved `<cwd>/output`.

## Reproduction

TUI name `.` → Generate → run initializes inside `output/` itself.

## Source references

- Files/lines above.

## Resolution candidates

Reject `"."` explicitly (e.g. `text not in (".",)` plus a reserved-name check);
apply the same hardened check at the CLI layer (see 008/057).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`tui_state.py:118-121` `_flat_folder_name`,
  `:202-203` output construction — both match; `"."` passes the check).
- Open: implement + tests.

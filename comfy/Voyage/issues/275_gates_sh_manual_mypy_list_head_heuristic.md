# 275 — `gates.sh` quality gates rot: explicit 150-file mypy list + `head -15 | grep DESIGN` heuristic

Severity: LOW (track B-15).

## Technical description

`gates.sh` typechecks `mypy voyage` plus ~150 explicitly listed `tests/test_*.py` files
("Untracked/in-flight test files stay out until committed and clean"). Any new test file
is silently un-typechecked until someone edits the script. The DESIGN-ref ratchet
(`head -n 15 | grep -q DESIGN`) false-fails any future file with a longer header and
false-passes a `DESIGN` mention in a comment that isn't a section ref.

## Rationale

Gates that require manual list maintenance decay — the exact failure mode already
documented for the TUI ignores (#264).

## Live evidence

```
missing_refs="$(for f in voyage/*.py; do head -n 15 "$f" | grep -q DESIGN || echo "$f"; done)"  # scripts/gates.sh:17
mypy voyage tests/conftest.py tests/test_seeds_properties.py ... (≈150 files)  # :41-79
grep -c "tests/test_" scripts/gates.sh → 37 (matches collapsed lines)
```

Repro: add a new `tests/test_foo.py` → not typechecked until `gates.sh` is edited; put a
`DESIGN` mention in a line-20 comment → ratchet misses it.

## Source refs

`scripts/gates.sh:13-21,28-42`; `scripts/lib/common.sh:46-70` (cache guard is fine).

## Online sources

- None (in-tree gate maintenance is the subject).

## Fix candidates

- Generate the mypy test list (`git ls-files 'tests/test_*.py'` minus a checked-in
  exclusion list) or invert to `mypy voyage tests` with per-file `ignore_errors` in
  `pyproject`; replace the `head` heuristic with an AST docstring check (stdlib
  `ast.get_docstring` contains `DESIGN §`).

## Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.

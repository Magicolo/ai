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

## Evaluation
- Claim CONFIRMED live 2026-10-07: gates.sh carried a hand-maintained
  ~150-file mypy list while `git ls-files 'tests/test_*.py'` reports 226
  tracked test files — 76 tracked files silently un-typechecked (any new
  test file joined them by default). The `head -n 15 | grep DESIGN` ratchet
  demonstrably misses long headers and passes stray comments. All 72
  `voyage/*.py` files currently pass the head check, so the AST replacement
  is behavior-preserving today.

## Progress log
- 2026-10-07 (Group J): chose list-generation over `mypy voyage tests`
  inversion (inversion would newly surface the 76 files' pre-existing
  errors and break gates). New checked-in
  `scripts/gates-mypy-excludes.txt` (the 76, sorted, with contract header);
  gates.sh resolves the list as `git ls-files 'tests/test_*.py'` (tracked
  only — untracked/in-flight files stay out, as before) minus the excludes
  (`LC_ALL=C` both sides for `comm`), with a fail-loud empty-list guard;
  the generated list is byte-identical in content to the old explicit list
  (diff-verified), so gate behavior is unchanged and new files are checked
  by default. DESIGN ratchet is now stdlib `ast.get_docstring` contains
  `DESIGN` (checked `DESIGN §`: 3 files legitimately cite "DESIGN task
  group/Phase" without § — `config.py` (out of scope), `director.py`,
  `registry_ltxv.py` — so strict-§ would newly fail gates; `DESIGN` keeps
  them green while fixing the header/comment blind spots; the § question
  belongs to the owning doc pass). Verified: `bash -n` clean, DESIGN check
  empty, generated list 150 files identical to old.

## Resolution (2026-10-07)
- RESOLVED. Files: `Voyage/scripts/gates.sh`,
  `Voyage/scripts/gates-mypy-excludes.txt` (new). Open, deliberately:
  full `gates.sh` is red at HEAD on 2 mypy errors in
  `test_rpc_deadline_finite.py` + `test_adapter_contract.py` — both from
  concurrent agents' uncommitted in-flight edits (issue-282 hunks), outside
  this scope and untouched; the gates-equivalent full-list mypy run was
  otherwise clean (250 files, only those 2 errors).

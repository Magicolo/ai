# 206 — `resolve_run_ref --run` is unvalidated while `--name` is traversal-guarded; `output_root()` is cwd-coupled; scripts ignore `$XDG_CACHE_HOME` / `qualify.sh` day-collision artifacts (HIGH)

## Technical description
`--name` goes through `is_flat_folder_name`/`_check_run_id` (no slashes, no
`..`, no `.`, no Windows-reserved). `--run` does `Path(run).resolve()` with
no flatness, absoluteness, or containment check
(`voyage/cli_paths.py:45-68`) — the comment even notes relative dirs
"double up inside payload paths" but doesn't reject them. `output_root()`
is `Path.cwd()/output` (`:71-78`), so the same NAME resolves differently per
cwd. `warn_if_outside_output_dir` fires only in `cmd_finalize` for `--output`
(`cli_finalize.py:44`); `configure --final-video /elsewhere/final.mp4` never
warns.

```python
def resolve_run_ref(...):              # voyage/cli_paths.py:45-68
    if name: ... _check_run_id ... return (output_root()/stripped).resolve()
    if run:  return Path(run).resolve()   # no _check_run_id, no absolute gate
def output_root() -> Path: ... return (Path.cwd()/"output").resolve()  # :71-78
```

## Rationale
Asymmetric validation + cwd-coupled root = two names for one run and one
name for two runs. The traversal guard (008) and containment guard (024)
each cover half the surface.

## Live evidence
`warn_if_outside_output_dir` callers: `voyage/cli_finalize.py:44` only.

## Repro
- `--run ../../evil` resolves outside `output/` silently.
- `cd /tmp && voyage generate NAME` vs `cd Voyage && voyage generate NAME`
  address different dirs.
- `configure --final-video /tmp/x.mp4` prints no containment warning while
  `finalize --output /tmp/x.mp4` does.

## Source refs
- `Voyage/voyage/cli_paths.py:20-111`
- `Voyage/voyage/cli_finalize.py:41-44`
- `Voyage/voyage/cli_configure.py:432-479` (no warn call)

## Online sources
- Absolute-path invariant + warn-never-silently-escape:
  https://specifications.freedesktop.org/basedir/latest/

## Fix candidates
1. Single `resolve_run_dir` enforcing absolute + warn-on-outside for both
   flags.
2. `output_root()` honoring `VOYAGE_OUTPUT` env with cwd fallback
   (documented).
3. Add `--final-video` containment warn in `configure`; tests for
   `../../evil`, relative `--run`, cwd-shift.

## Log
- Track B sweep, 2026-10-07. Read-only; nothing fixed.

## Consolidated from 274_xdg_cwd_coupled_output_unwarned_final_video_date_collision (2026-10-07)

Severity: LOW (track B-14). The cwd-coupled `output_root()` and unwarned
`configure --final-video` half of that issue is subsumed by the canonical findings above;
the script-level unique parts follow.

### Technical description

`run.sh:139` uses `${VOYAGE_MODELS:-$HOME/.cache/voyage-models}`, ignoring
`$XDG_CACHE_HOME` (spec default is `$HOME/.cache` only when unset — Voyage never reads
it). `qualify.sh:135` names artifacts `qual-<base>-$(date +%F).json` (day granularity —
two runs same day collide/overwrite via `tee`).

### Rationale

XDG non-compliance breaks non-standard homes; colliding report names are the small paper
cuts that become data loss at scale.

### Live evidence

```
models="${VOYAGE_MODELS:-$HOME/.cache/voyage-models}"  # scripts/run.sh:139, no XDG_CACHE_HOME
artifact="reports/qual-$(basename "$run_dir")-$(date +%F).json"  # scripts/qualify.sh:135
```

Repro: run `qualify.sh` twice same day → second `tee` overwrites the first artifact.

### Source refs

`scripts/run.sh:139`; `scripts/qualify.sh:135`.

### Online sources

- `https://specifications.freedesktop.org/basedir/latest/` (`$XDG_CACHE_HOME` default
  `$HOME/.cache`).
- `https://pyxdg.readthedocs.io/en/latest/basedirectory.html`.

### Fix candidates

- `${VOYAGE_MODELS:-${XDG_CACHE_HOME:-$HOME/.cache}/voyage-models}`; `date +%FT%H%M%S`
  artifact names (the `VOYAGE_OUTPUT` override + `configure --final-video` warn live in
  the canonical fix list above).

### Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.
- 2026-10-07: consolidated into 206 (cwd/output-root family; cwd-coupled and unwarned
  halves subsumed).

## Progress (2026-10-07, group O)

- `voyage/cli_paths.py`: new single `resolve_run_dir(*, run, name)` —
  both/neither → None + stderr (unchanged), `--name` flat-guard
  (unchanged), `--run` must be absolute (relative → None + "must be an
  absolute path") and warns via `warn_if_outside_output_dir(..., "--run")`
  when it escapes the tree; `resolve_run_ref` kept as a delegating alias
  (sole caller `cli_finalize.cmd_finalize` untouched). `output_root()`
  honors `VOYAGE_OUTPUT` (expanduser + resolve) with the cwd default
  unchanged (the `test_output_root_is_cwd_output` pin still passes).
- `voyage/cli_configure.py`: `_commit_manifest` warns on an explicit
  `--final-video` outside the tree (create + update share the site;
  inherited values do not re-warn).
- `scripts/run.sh:198`: `models="${VOYAGE_MODELS:-${XDG_CACHE_HOME:-$HOME/.cache}/voyage-models}"`
  (explicit > XDG > HOME default, verified live in all three shapes).
- `scripts/qualify.sh:182`: artifact `qual-${name}-$(date +%FT%H%M%S).json`
  (second granularity — same-day reruns no longer `tee`-overwrite).
- New `tests/test_issue_206_runref.py` (12 tests: both/neither/empty
  misuse, name resolve + traversal reject, relative-run reject,
  outside-run warns, inside-run quiet, alias parity, VOYAGE_OUTPUT
  override + default, configure outside-warns-but-proceeds with manifest
  pin, configure inside quiet).
- Verified in-container: `ruff check` clean, `ruff format --check` clean
  (after one reflow), `mypy` strict clean on `cli_paths` +
  `cli_configure`, scoped pytest 12/12 green; neighbors
  (`test_output_containment`, `test_configure`, `test_run_sh`,
  `test_definition_tiers`, `test_scoreboard`, ...) green.

## Resolution (2026-10-07)

FIXED. All five surfaces closed with warn-only semantics preserved
(outside-tree paths stay legal for `/tmp` flows and tmp_path suites).
`--run` had no live CLI parser surface (only the finalize library path,
which always passes an absolute dir), so the new absolute gate breaks
no production caller. Nothing left open.

## Evaluation (2026-10-07, group O)

Re-verified live against current tree (all claims hold):
- `--run` unvalidated: `voyage/cli_paths.py:63-66` is still a bare
  `Path(run).resolve()` (no flatness/absolute/containment check) while
  `--name` goes through `_check_run_id` (`:58-62`). No live parser
  defines `--run`/`--name` flags (configure/generate take positional
  NAME); the only caller is `cli_finalize.cmd_finalize` (`:27`, legacy
  library path, defensive getattr) plus `cli_generate._finalize_run_dir`
  which always passes an absolute `str(run_dir)`.
- `output_root()` cwd-coupled: `:71-78` unchanged; `tests/test_output_containment.py:26-29`
  pins cwd-coupled as the intentional (024) default, so a `VOYAGE_OUTPUT`
  override must keep cwd/output as the default.
- Containment half-warned: `warn_if_outside_output_dir` has exactly one
  production caller (`cli_finalize.py:44`); `cli_configure.py:435-483`
  (`_commit_manifest`, create + update) never warns on `--final-video`.
- 274 script halves: `scripts/run.sh:198`
  `models="${VOYAGE_MODELS:-$HOME/.cache/voyage-models}"` (no
  `$XDG_CACHE_HOME`); `scripts/qualify.sh:182`
  `artifact="reports/qual-${name}-$(date +%F).json"` (day granularity,
  same-day reruns `tee`-overwrite).
- Reviewer note (222): multi-uid harm correctly out of scope — not
  evaluated here.

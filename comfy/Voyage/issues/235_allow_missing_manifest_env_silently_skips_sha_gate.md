# 235 — `VOYAGE_ALLOW_MISSING_MANIFEST=1` env (or `allow_missing_manifest=True`) silently skips the only load-time sha gate

Severity: MEDIUM (track F-09).

## Technical description

`verify_checkpoint_against_manifest` fails closed by default (good), but any
`VOYAGE_ALLOW_MISSING_MANIFEST=1/true/yes` in the worker environment — or one
`allow_missing_manifest=True` call — returns early with no warning
(`model_registry.py:537-544`). The env is inherited through `rpc._spawn_env`,
`run.sh -e` passthrough, and the HF-offline dance, so a stale export or a debugging
session permanently downgrades every CausVid load (and any future caller) without a trace
in logs/metrics.

## Rationale

One env var disables fail-closed verification process-wide, no log.

## Live evidence

`voyage/model_registry.py:527-551`; sole production caller passes no flag
(`video_causvid.py:567`), so the env branch is the live bypass.

Repro: `VOYAGE_ALLOW_MISSING_MANIFEST=1 python3 -c "from voyage.model_registry import
verify_checkpoint_against_manifest;
verify_checkpoint_against_manifest(Path('/models'),'causvid',Path('/tmp/evil.pt'))"`
→ returns instead of raising.

## Source refs

`voyage/model_registry.py:537-544`.

## Online sources

- Fail-closed design doctrine (in-tree 071: "fails closed by default").
- pip `--require-hashes` principle (no silent opt-out of verification).

## Fix candidates

- Remove the env branch (explicit kwarg only), or emit a loud `stderr` +
  `hash_verification_skipped` metric on every bypass; scope the opt-in to named
  external-volume keys, never global.

## Log

- 2026-10-07: filed from read-only Track F sweep; no code touched.

## Evaluation

- 2026-10-07 (Group L): re-read live — `voyage/model_registry.py:580-587`
  still returns silently on either opt-in; sole production caller
  `voyage/workers/video_causvid.py:567` passes no flag, so the env branch
  is the live bypass; all other references are tests. Issue is LIVE.
  Fix choice: the loud-bypass candidate, NOT env-branch removal —
  `tests/test_registry_pins.py` (concurrent scope, untouchable) pins the
  env opt-in as returning, so removal would break a forbidden file. Loud
  stderr preserves its behavior while ending the silence. No metric sink
  exists at this layer (`model_registry` never imports metrics logging),
  so stderr is the loud channel; callers' logs capture it.

## Progress log

- 2026-10-07 (Group L): bypass branch now writes a `WARNING` to stderr
  naming key + checkpoint + opt-in source (env vs kwarg) before returning;
  docstring updated; fail-closed default untouched; `_merge_manifest_record`
  lines untouched. Tests added in `tests/test_checkpoint_safety.py` (in
  scope): kwarg-bypass-loud + env-bypass-loud, both asserting WARNING +
  key + source on stderr.

## Resolution (2026-10-07)

- RESOLVED. Files changed: `voyage/model_registry.py` (bypass branch +
  `import sys` + docstring only), `tests/test_checkpoint_safety.py`
  (2 new tests). Verification: scoped pytest below. Left open: per-key
  scoping of the env opt-in (e.g. `VOYAGE_ALLOW_MISSING_MANIFEST_KEYS`)
  and a `hash_verification_skipped` metric emission at the supervisor
  layer — both are enhancements, not gaps: every bypass is now loud.

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

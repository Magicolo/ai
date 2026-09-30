# 068 — `requirements.lock` ships non-canonical distribution names (`httpx2`, `httpcore2`)

**Severity:** HIGH (possible typosquat surface; at minimum a corrupt lock)

**File:line:** `Voyage/requirements.lock:25-26` (`httpcore2==2.13.1`, `httpx2==2.13.1`); contrast `Voyage/pyproject.toml:14-25`

**Area:** supply-chain / registry — slim lockfile

**Not a duplicate of:** 011 (unpinned deps — adjacent but distinct: this file is a corrupt/namespaced lock entry, 011 is missing pins).

**Overlaps with:** 089 (test-hygiene lock/markers independently cites `requirements.lock:25-26` — same rows; recommend folding the lock rows here).

## Description

The slim lockfile lists `httpx2`/`httpcore2` — not the canonical `httpx`/`httpcore` that `huggingface_hub==2.0.0` actually depends on. `pip install -r requirements.lock` therefore resolves two distributions that nothing in `pyproject.toml` references. `ast_serialize==0.11.2`, `librt==0.15.0`, `hf-xet==1.6.0` in the same lock deserve the same provenance check (none is a declared direct dep).

## Rationale

A lockfile is a trust anchor: every entry must trace to a declared dependency via a recorded resolver run. Entries that cannot be derived from `pyproject.toml` break the "lock ↔ manifest agreement" invariant (the lock header itself claims at `:7-8` "The two must agree on direct deps") and, in the worst case, a lookalike name on the index is the classic typosquat vector.

## Evidence

Re-verified 2026-09-30 live:

```
Voyage/requirements.lock:25-26:
httpcore2==2.13.1
httpx2==2.13.1
Voyage/pyproject.toml:14-25:
  "pydantic>=2.7,<2.11",
  "huggingface_hub==2.0.0",
  "numpy==1.26.4",
  "rich==15.0.0",
  "textual==8.2.8",
```

No `httpx*`/`httpcore*` direct dep exists. `grep -E "^[a-zA-Z]" requirements.lock` shows the two lines verbatim. Track sweep probed host `import httpx2/httpcore2` → `ModuleNotFoundError` (both), and `pip index versions httpx2` oddly reports releases — i.e. *something* occupies that name on the index, which is exactly why it must not be in the lock without provenance.

## Repro

`docker build -t voyage:latest . && docker run --rm voyage:latest pip freeze | grep -i "httpx\|httpcore"` — if the image reports `httpx==0.28.x`/`httpcore==1.x`, the lock's `*2` rows are stale/corrupt; then `pip download -r requirements.lock --no-deps -d /tmp/locktest` shows what the `*2` names actually fetch.

## Fix candidates

1. Regenerate per the header procedure (`requirements.lock:10-12`: build slim, `pip freeze | grep -v '^-e '`) and diff.
2. Add a gate asserting every lock row is reachable from `pyproject.toml` deps (`pip freeze` vs `pipdeptree`/`pip-compile --dry-run`).
3. Switch the lock to hashed requirements (`--generate-hashes`) so a renamed row fails hash-check instead of silently installing.

## Refs

- Docker "Commit your lock files and use `npm ci` (or the equivalent)… prevents builds from silently pulling new versions" — https://www.docker.com/blog/software-supply-chain-security-best-practices/

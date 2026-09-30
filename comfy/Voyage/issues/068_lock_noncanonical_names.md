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

## Progress log (2026-09-30)

Premise re-verified against live code/metadata — REFUTED for the `*2` rows:
- Installed `huggingface_hub==2.0.0` Requires-Dist contains `httpx2<3,>=2.0.0` (unconditional, marker-free); `httpx2==2.13.1` Requires `httpcore2`; both show intact `Required-by` chains (`httpx2 ← huggingface_hub`, `httpcore2 ← httpx2`); all three import with versions in `voyage:latest`.
- `pip download httpx2==2.13.1 --no-deps` yields a 36-entry wheel with a real `httpx2/` package (not a stub/shadow).
- Conclusion: the rows are GENUINELY INSTALLED, not stale/corrupt — the issue's "canonical httpx/httpcore" assumption predates this `huggingface_hub` release. Renaming them to `httpx`/`httpcore` would BREAK the image (old names are unreachable from every root). No lock change made.
- Secondary rows from the issue all check out reachable: `hf-xet` (hf_hub, platform markers), `ast_serialize` (mypy, PEP 503 separator equivalence), `librt` (mypy, CPython marker).
- Two subtleties the gate had to handle (found by BFS over installed metadata): `linkify-it-py` is reachable ONLY via the extras edge `textual → markdown-it-py[linkify] → linkify-it-py` (naive marker eval false-flags it); `typing-inspection` is declared by NOTHING (`Required-by` empty) yet `import pydantic` loads it at runtime — pydantic 2.10.6 metadata gap, row is load-bearing and stays.

## Resolution (2026-09-30) — PREMISE REFUTED, GATE ADDED (no lock change)

New `tests/test_lock_manifest_agreement.py` (5 tests, all green in-container): extras-aware BFS over installed Requires-Dist; asserts both directions (marker-applicable direct deps ⊆ lock; lock rows ⊆ reachable modulo a documented `METADATA_ORPHAN_ALLOWLIST` holding only `typing-inspection`, ratcheted to stay minimal — an entry that becomes reachable fails until removed); pins the 068 verdict (`httpx2`/`httpcore2` reachable); synthetic test proves discrimination.

Evidence: `evil-typo==1.0.0` appended to a scratch copy of the real lock → gate FAILS naming the offender (real code path, `/tmp` only, tree untouched); ruff + format green (file outside mypy scope, fully annotated regardless); the 5-test module passes inside the full-suite run (1129 passed; the only failures are the foreign `paths.py` ones noted in 052).

Residual: index-trust (a compromised `huggingface_hub` release could bless any rename) is NOT closed by a reachability gate — hashed requirements (fix candidate 3) remain the follow-up; `typing-inspection` allowlist entry should dissolve once pydantic metadata declares it.

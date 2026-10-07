# 207 — `EXPECTED_RIFE_SHA256` is 63 hex chars (truncated, can never match) (HIGH)

## Technical description
The RIFE ingest pin is one nibble short. `download_model("rife")` calls
`verify_checkpoint_sha256` → exact `lower()` compare against a 64-hex file
digest. A 63-char constant can never equal a real digest, so every honest
RIFE fetch fails closed; any path that provisioned RIFE outside
`download_model` never ran the gate at all.

## Rationale
Hash pins are only useful if syntactically valid. A truncated pin converts a
security gate into a liveness bug, which then pressures operators to bypass
the gate (exactly what the AGENTS log records: "provisioned outside
download_model so the ingest gate never ran").

## Live evidence
Verified live by orchestrator 2026-10-07 (all 15 `EXPECTED_*` measured;
only RIFE is short):

```
Voyage/voyage/registry_rife.py EXPECTED_RIFE_SHA256 = "8d0f6be4655a7c18 len= 63
# all other 14 EXPECTED_* across registry_film/ltx23/ltx25/ltxv/realesrgan: len= 64
```

## Repro
`python3 -c "import re; assert re.fullmatch(r'[0-9a-f]{64}',
__import__('voyage.registry_rife', fromlist=['EXPECTED_RIFE_SHA256']).EXPECTED_RIFE_SHA256)"`
→ AssertionError.

## Source refs
- `Voyage/voyage/registry_rife.py:52`
- In-tree contract `Voyage/voyage/model_registry.py:1298-1299`
  (`download_model` verifies `expected_hashes` pre-merge)

## Online sources
- pip `secure-installs`: hashes required, exact match.
- HF `revision=` must be the full hash.

## Fix candidates
1. Re-measure `sha256_file` of the pinned `rife_v4.25_heavy.safetensors` at
   `RIFE_HF_REVISION` and replace with full 64 hex.
2. Add a unit test asserting every `EXPECTED_*` matches `[0-9a-f]{64}`.
3. Reconcile the on-disk `40aa1838…` vs registry `8d0f6be4…` fork before
   re-downloading (AGENTS §11 RIFE entry).

## Log
- Track F sweep, 2026-10-07. Verified live by orchestrator 2026-10-07.
  Read-only; nothing fixed.

## Evaluation (2026-10-07, resolution pass)
- Still relevant: pin still 63 chars live (`python3 -c` length check:
  RIFE len=63, all other 14 EXPECTED_* len=64); fork unreconciled
  (on-disk `40aa1838…` vs registry `8d0f6be4…`, different prefixes =
  different files, not just truncation); `test_rife_pin_is_best_quality_heavy`
  (`tests/test_interp_backend.py:89-96`) pins the bogus 63-char value, so the
  suite was green over a broken gate.
- True digest WAS determinable live: HEAD `curl` at pinned revision
  `219da3c9…` returns `X-Linked-ETag: "40aa1838b91531f8…703191"` +
  `X-Linked-Size: 86669816` (= both on-disk sizes); full 86,669,816-byte
  download to `/tmp` hashes to
  `40aa1838b91531f829caaac026f40d9d2e2f1eb12b65d1d6029a58ae4c703191`
  (64 chars, matches both on-disk copies byte-for-byte). The old `8d0f6be4…`
  value matched nothing at the pinned revision — doubly wrong, not just
  truncated — so fix path (a) applies: replace the constant, no guessing.
  Scratch download removed after hashing.

## Resolution (2026-10-07)
- 13:32 UTC: verified pin still truncated (RIFE len=63, others len=64).
- 13:33 UTC: `sha256sum` both on-disk copies → `40aa1838…703191` (identical).
- 13:34 UTC: pinned-revision HEAD → ETag/size match on-disk; downloaded
  pinned bytes → same sha256. True digest established; no bytes guessed.
- Code changes: `voyage/registry_rife.py` pin → `40aa1838…703191` + provenance
  comment (never re-pin by hand) + corrected module docstring;
  `tests/test_interp_backend.py` pin test updated to the true digest;
  `tests/test_registry_pins.py` gained `test_all_expected_sha_pins_are_64_lower_hex`
  (every EXPECTED_* across all six registry modules must fullmatch
  `[0-9a-f]{64}` — the guard this issue asked for).
- Status: RESOLVED. Fork reconciled as a side effect: the on-disk files ARE
  the pinned-revision bytes, so no re-provisioning is needed. Verification:
  scoped pytest + ruff in-container (see pass-2 verification below).
- Deliberately left open: `Voyage/DESIGN.md:8704` still cites `sha 8d0f6be4`
  (shared doc, out of scope — orchestrator to correct); `test_interp_backend`
  exact-digest pin remains a second layer over the new syntactic gate.

## Verification (2026-10-07, post-fix)
- `python3 -c` length sweep: all 15 EXPECTED_* len=64 (RIFE now OK).
- `./scripts/test.sh -m "not gpu" tests/test_registry_pins.py
  tests/test_interp_backend.py tests/test_backend_registry.py` →
  59 passed, 1 skipped (skip is pre-existing: torch-bearing leg, slim image).
- In-container `ruff check` + `ruff format --check` on all 4 touched
  code/test files: clean. In-container `mypy voyage`: no issues in 97 files.
- No GPU workloads run (CPU-only pins/tests).

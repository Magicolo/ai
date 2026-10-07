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

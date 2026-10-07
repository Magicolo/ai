# 285 — ltx23 recovery tape `vae_revision` aliases the DiT revision — VAE swaps resume silently

Severity: MEDIUM (pass-2 worker-tails sweep).

## Technical description

`video_ltx23.build_recovery_tape` records `"vae_revision": LTX23_DIT_REVISION`. The
ltx23 registry has no VAE revision constant at all (DiT and TE revisions only), while the
ltx25 twin correctly records `LTX25_VAE_REVISION`. A swapped/upgraded ltx23 VAE produces
a byte-identical tape field, so no resume-time check can ever catch it.

## Rationale

The field's documented purpose is "so a tape never resumes across numerics"
(`video_ltx25.py:335-347`). For ltx23 that guarantee is void for the VAE leg — the four
files' shas exist as `EXPECTED_LTX23_*` but the tape carries the wrong revision string.

## Live evidence

```
$ PYTHONPATH=Voyage python3 -c "...grep..."
ltx23:251:"vae_revision": LTX23_DIT_REVISION,
ltx25:346:"vae_revision": LTX25_VAE_REVISION,
---registry_ltx23 has VAE_REVISION: False
---registry_ltx25 has VAE_REVISION: True
$ grep -n "REVISION" voyage/registry_ltx23.py
36:LTX23_DIT_REVISION = "96e8ed49..."
51:LTX23_TE_REVISION = "858acec7..."
```

No `LTX23_*VAE*REVISION` symbol exists; both ltx23 VAEs come from the same
`unsloth/LTX-2.3-GGUF` repo as the DiT, so the DiT revision happens to version them today
— but the tape field still asserts a false provenance.

Repro: inspect `build_recovery_tape(...).["vae_revision"] == LTX23_DIT_REVISION` always,
independent of the VAE files on disk (host import of the worker fails — imports
`loop`→pydantic, absent by design; source inspection instead).

## Source refs

`voyage/workers/video_ltx23.py:240-263` (tape builder, line 251);
`voyage/registry_ltx23.py:34-77` (pins, no VAE rev); contrast
`voyage/workers/video_ltx25.py:335-359` + `voyage/registry_ltx25.py` (has
`LTX25_VAE_REVISION`).

## Online sources

- None (in-tree tape-numerics contract is the anchor).

## Fix candidates

- (a) Add `LTX23_VAE_REVISION` (+ audio-VAE rev if distinct) to `registry_ltx23.py` and
  wire it into the tape; (b) alternatively record the VAE file shas in the tape
  (stronger than a revision string, matches the `EXPECTED_*_SHA256` ingest gates). Note
  `parse_recovery_tape` currently checks only backend/state_mode/tail-path, so any
  enforcement also needs a resume-time comparison — today the field is write-only.

## Log

- 2026-10-07: filed from read-only pass-2 worker-tails sweep; no code touched.

## Evaluation (2026-10-07)
- Claim CURRENT on re-read: no `LTX23_*VAE*REVISION` symbol existed and the tape
  carried the DiT revision in `vae_revision`. Fix candidate (a) adopted, plus a
  resume-time trust check.

## Progress log
- Batch-6 Group Q added `LTX23_VAE_REVISION` to `voyage/registry_ltx23.py` and
  wired it into `build_recovery_tape` in `voyage/workers/video_ltx23.py`, plus a
  resume-time tape-trust comparison so a VAE swap no longer resumes silently. New
  `tests/test_issue_285_ltx23_vae_revision.py` pins the tape field and the trust
  check. Scoped gates green (ruff + format + mypy strict + pytest).

## Resolution (2026-10-07)
- RESOLVED. The ltx23 tape now carries true VAE provenance and resume enforces
  it, matching the ltx25 numerics contract.

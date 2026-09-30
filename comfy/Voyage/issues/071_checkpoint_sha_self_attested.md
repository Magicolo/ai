# 071 — Checkpoint SHA is self-attested post-download, never pre-verified

- Severity: HIGH
- File: `Voyage/voyage/model_registry.py:484` + `:578` (records `sha256_file(...)` after fetch), `:358-380` (`verify_checkpoint_against_manifest` pass-through), `:1192-1198` (`verify_model` = presence + byte floors only)
- Area: model registry / security — hash verification order

## Description

No registry row carries an expected hash. The flow is download → hash whatever arrived → write it into `manifest.json` as `checkpoint_sha256`. `verify_checkpoint_against_manifest` returns silently when there is no manifest or no sha for the key ("volumes provisioned outside `voyage models download`" pass through by design), and `verify_model` — the gate `models_ensure` actually enforces — checks only existence and `*_MIN_BYTES` floors. A poisoned/truncated-but-padded first fetch becomes the attested-good baseline for all later loads.

## Rationale

Hash-after-download is tamper-*evident* against later mutation, not tamper-*preventing* at ingest. Best practice (Docker supply-chain guidance; HF pickle-scanning docs) is: pin the expected digest out-of-band, verify before first use, fail closed when no baseline exists for a known key.

## Live evidence

Re-verified 2026-09-30 live (line numbers as-read; registry grew 1267→1368L since the sweep):

```
Voyage/voyage/model_registry.py:476-484 (_record_longlive2):
        "checkpoint_sha256": sha256_file(generator_path),   # hashed AFTER fetch (:484)
Voyage/voyage/model_registry.py:570-578 (_record_causvid):
        "checkpoint_sha256": sha256_file(checkpoint_path),  # same pattern (:578)
Voyage/voyage/model_registry.py:358-380:
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        return                                        # manifest-missing pass-through
    ...
    recorded = entry.get("checkpoint_sha256")
    if not isinstance(recorded, str) or not recorded:
        return                                        # hash-less pass-through
Voyage/voyage/model_registry.py:1192-1198 (verify_model):
    missing = _collect_missing(models_dir, spec)      # presence + size floors only
    if missing: return False, ...
    return True, spec.success_message(models_dir)     # no hash comparison
```

Only `longlive2` (`"video"`) and `causvid` manifests even carry a `checkpoint_sha256` — Qwen/ACE/LTXV/MMAudio/FILM/ESRGAN rows record `*_bytes` only.

## Repro

Provision a volume, delete `manifest.json`, restart worker → `verify_checkpoint_against_manifest` no-ops → `torch.load` proceeds on manifest-less weights (pass-through branch, `:358-380`). More practically: replace `longlive2/model_bf16.pt` with same-size random bytes before first `download_model` returns — the random bytes become the attested baseline.

## Fix candidates

1. Add `expected_sha256` (or per-file shas) constants per spec and verify in `download_model` before `_merge_manifest_record`, reusing the existing `verify_checkpoint_sha256` (`:341-356`).
2. Make unknown-manifest + known-key fail closed (flag-gated if the "external volume" use case must survive).
3. Extend `verify_model` with a hash check wherever a sha is recorded.

## Refs

- Overlaps with 077 (manifest-repair gap silently blesses the hash-less state this issue describes) and 074 (augment hash leg) — ownership stays here (ingest-time verification).
- Docker "If your build system pulls a dependency and the hash does not match what was committed, the build should fail" — https://www.docker.com/blog/software-supply-chain-security-best-practices/

## Progress log

- 2026-09-30: premise re-verified against live `Voyage/voyage/model_registry.py`
  (registry since grew to 1368 lines; cited sites confirmed at `:476-484`
  longlive2 sha-after-fetch, `:570-578` causvid, `:358-380` pass-through
  branches, `:1192-1198` presence-only `verify_model`). Extra finding: the
  provisioned volume's `manifest.json` "causvid" record carries NO
  `checkpoint_sha256` (old provision) and the weight file is pruned — so no
  CausVid baseline is obtainable anywhere; the "ltxv" key is missing entirely
  (files provisioned without a record).
- 2026-09-30: TDD — four failing-first tests in `tests/test_registry_pins.py`
  (ingest tamper rejected pre-merge, ensure fails on recorded-sha mismatch,
  load closed-by-default without manifest, all failed pre-fix), plus two
  passing controls (matching-sha ensure stays green; explicit opt-in passes
  through). All green post-fix.
- 2026-09-30: gates — ruff + format + mypy strict clean on
  `voyage/model_registry.py`; full suite 1181 passed / 2 failed, both failures
  the intended behavior changes below (foreign file, exact updates specified).

## Resolution

- Fix candidate 1 applied: `EXPECTED_*_SHA256` constants (LongLive generator
  manifest-attested `ec9063…`; LTXV DiT `76aa8c…` + upscaler `5b0760…`, FILM
  `f226e5…`, Real-ESRGAN `f872d8…` measured live 2026-09-30 from pinned-revision
  volume bytes — provenance per constant in code). New `ExpectedHash` rows on
  the `longlive2-bf16` / `ltxv-2b` / `film` / `realesrgan-anime` specs;
  `download_model` verifies them via the existing `verify_checkpoint_sha256`
  BEFORE `_merge_manifest_record` (raises — poisoned bytes never attest).
- Fix candidate 2 applied: `verify_checkpoint_against_manifest` gains
  `allow_missing_manifest: bool = False` — unknown-manifest + known-key now
  raises by default (provisioned workers unchanged at call sites, therefore
  closed). External-volume opt-in is explicit: the flag, or
  `VOYAGE_ALLOW_MISSING_MANIFEST=1` (runtime opt-in with unmodified workers);
  the error message names both.
- Fix candidate 3 applied: `verify_model` runs a manifest-conditional hash leg
  after presence (`_manifest_hash_mismatches` covers `checkpoint_shas` dicts
  and `checkpoint_sha256` + new `ModelSpec.manifest_checkpoint` rows).
  `_record_ltxv/_record_film/_record_realesrgan` now write the shas so every
  recorded sha is checked at ensure time; fixture dirs without manifests skip
  the leg (zero collateral in `test_ltxv`/`test_longlive`/`test_augment_models`/
  `test_causvid_prep`).
- KNOWN COLLATERAL (2 foreign tests in `tests/test_checkpoint_safety.py`,
  owned by another track — one-line updates, behavior changes intended):
  (a) `test_longlive_download_preserves_foreign_manifest_keys` — the stubbed
  `b"fake-weights"` download now hits the ingest gate; fix by asserting the
  raise, or stubbing `EXPECTED_LONGLIVE_SHA256`-matching bytes;
  (b) `test_verify_against_manifest_passes_through_without_manifest` — the
  pass-through it pins is exactly what this issue removes; fix by passing
  `allow_missing_manifest=True` (or set the env var) to keep testing the
  escape hatch.
- Residuals: no CausVid DMD baseline exists (re-provision → measure → add
  `EXPECTED_CAUSVID_SHA256` + spec row); multi-file snapshots (Qwen/ACE/MMAudio)
  have per-file shas as future work; old-provision manifests without shas skip
  the ensure leg until re-download (presence-only, as before).

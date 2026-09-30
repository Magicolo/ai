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

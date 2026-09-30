# 072 — MMAudio vocoder allow-list fetches executable `*.py` from the Hub

- Severity: MEDIUM (downgraded from Medium-High: vocoder revision is pinned, and execution of the fetched `.py` via the redirect is unproven — `trust_remote_code` is never set)
- File: `Voyage/voyage/model_registry.py:275-285` (`MMAUDIO_VOCODER_ALLOW`), consumed at `:911`; loader shim `Voyage/voyage/audio/mmaudio_sfx.py:119-157`
- Area: model registry / security — vocoder snapshot

## Description

The nvidia BigVGAN snapshot spec explicitly downloads `*.py` + a whole `alias_free_activation/*` subtree from `nvidia/bigvgan_v2_44khz_128band_512x` into `/models/mmaudio/vocoder/`. `_load_feature_utils` then monkeypatches `BigVGANv2.from_pretrained` to redirect to that directory. Downloaded code sitting in a persistent, reused volume next to weights is one `sys.path`/`import` confusion away from execution — and the allow-pattern is unbounded (any present or future `.py` in that repo, including `alias_free_activation/` helpers).

## Rationale

Weight snapshots should be data-only. HF's pickle guidance stresses vetting executable content and minimizing what you fetch; a `*.py` glob is the opposite — it pre-approves arbitrary code files from a third-party repo into the runtime's neighborhood.

## Live evidence

Re-verified 2026-09-30 live (line numbers as-read):

```
Voyage/voyage/model_registry.py:275-285:
MMAUDIO_VOCODER_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"
MMAUDIO_VOCODER_REVISION = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"
MMAUDIO_VOCODER_SUBDIR = f"{MMAUDIO_SUBDIR}/vocoder/bigvgan_v2_44khz_128band_512x"
MMAUDIO_VOCODER_ALLOW = (
    "*.py",
    "config.json",
    "bigvgan_generator.pt",
    "alias_free_activation/*",
)
Voyage/voyage/audio/mmaudio_sfx.py:143-147:
    def _pinned_vocoder(cls: Any, name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(name_or_path) == "nvidia/bigvgan_v2_44khz_128band_512x":
            name_or_path = str(vocoder_dir)
        return original_from_pretrained(name_or_path, *args, **kwargs)
```

The `_pinned_vocoder` redirect proves the `.py` files land where a HubMixin loader resolves them.

## Repro

`voyage models download sfx-mmaudio && find ~/.cache/voyage-models/mmaudio/vocoder -name "*.py" | head`; compare against the repo file list at `MMAUDIO_VOCODER_REVISION` — any upstream-added `.py` arrives automatically.

## Fix candidates

1. Narrow the allow-list to the exact consumed files (`config.json`, `bigvgan_generator.pt`, enumerated activation files).
2. Fetch code via the pinned git-clone + `rev-parse --verify` pattern already used for `/opt/mmaudio` (`worker/Dockerfile.video:102-105`) instead of the weight snapshot.
3. Assert in `verify_sfx_models` that no `.py` exists under the vocoder dir.

## Refs

- HF "Pickle Scanning… displaying/vetting the list of imports… do not unpickle data from untrusted sources" — https://huggingface.co/docs/hub/security-pickle

## Progress log

- 2026-09-30: premise re-verified against live `Voyage/voyage/model_registry.py`
  (`MMAUDIO_VOCODER_ALLOW` still carries `*.py` + `alias_free_activation/*`;
  `verify_sfx_models` is presence + size-floors only via `verify_model`).
  071-seam check: `ExpectedHash` + `download_model` pre-merge verification
  technically fits snapshot files (any `relative_path` under the snapshot
  subdir verifies the same way), but NO baseline hash exists for either
  vocoder file — 071 resolved multi-file snapshots (Qwen/ACE/MMAudio) as
  explicit future work, and inventing a hash is forbidden (a wrong pin fails
  every provision loudly). Verdict: standalone gate now, per-file vocoder
  shas join the 071 follow-up when provisioned bytes are measurable.
  Enumeration: `_load_feature_utils` (`mmaudio_sfx.py:224-274`) imports
  `BigVGANv2` from the pinned `/opt/mmaudio` clone (`MMAUDIO_CODE_COMMIT`)
  and calls `from_pretrained(vocoder_dir)` with `trust_remote_code` never
  set — HubMixin resolves only `config.json` + `bigvgan_generator.pt`;
  zero snapshot activation files are consumed (activation code ships in the
  clone). Enumerated activation files = none; the allow-list is exactly the
  two data files.
- 2026-09-30: TDD — four failing-first tests in
  `tests/test_vocoder_allowlist.py` (allow-list data-only, top-level `.py`
  rejected, nested `alias_free_activation/*.py` rejected; 3 failed pre-fix,
  1 clean-tree control passed pre-fix). All green post-fix.
- 2026-09-30: gates (scoped) — ruff + format + mypy strict clean on
  `voyage/model_registry.py` + `tests/test_vocoder_allowlist.py`; scoped
  suite 107 passed / 3 skipped (torch arch-smoke skips in slim):
  `test_vocoder_allowlist` + `test_augment_weight_loading` +
  `test_augment_models` + `test_augment_runner` + `test_registry_pins` +
  `test_sfx_contract` + `test_checkpoint_safety`. Full `gates.sh` is the
  orchestrator's job (not run here per scope contract).

## Resolution

- Fix candidate 1 applied (narrow): `MMAUDIO_VOCODER_ALLOW` is now exactly
  `("config.json", "bigvgan_generator.pt")` — the two files
  `_load_feature_utils` resolves. The `*.py` glob and the
  `alias_free_activation/*` subtree are gone; a code comment records the
  provenance (class from the pinned clone, snapshot is data-only).
- Fix candidate 3 applied (fail-loud gate): new `_vocoder_python_files`
  helper lists every `.py` under the vocoder snapshot dir; `verify_sfx_models`
  runs it after the presence checklist and returns `(False, ... .py ...)`
  naming the offenders when non-empty. Old-provision volumes carrying snapshot
  `.py` files now fail verify until the files are removed and re-provisioned
  under the narrowed allow-list.
- Fix candidate 2 (git-clone for code) needs no change: vocoder CODE already
  ships via the verified-fetch `/opt/mmaudio` clone at `MMAUDIO_CODE_COMMIT`
  (`worker/Dockerfile.video:113-118`); the snapshot never supplied code.
- `voyage/audio/mmaudio_sfx.py` untouched (no vocoder-load-side change needed:
  the redirect already passes a data-only dir to an already-imported class).
- Residuals: per-file vocoder `ExpectedHash` rows await measurable provisioned
  bytes (071 follow-up — do NOT invent hashes); existing volumes provisioned
  under the old glob keep their `.py` files until re-provisioned (the new
  verify gate surfaces them).

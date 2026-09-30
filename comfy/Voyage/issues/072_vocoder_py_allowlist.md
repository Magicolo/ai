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

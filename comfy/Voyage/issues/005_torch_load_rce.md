# 005 — `torch.load` on checkpoints without `weights_only=True` (arbitrary code execution)

- Status: resolved in live tree (`weights_only=True` everywhere + sha256 pre-verify)
- Severity: HIGH (security — RCE via pickle; resolved, record only)
- Group: security/supply-chain — Rank: 1/5 (only RCE in set, fixed)
- Area: supply chain / worker model loading
- Rank rationale: the only RCE in the list; `.pth`/`.pt` are pickle, sources include
  HF mirrors, shared caches, and run dirs.

## Technical description

Pass-1 `rg -n "torch.load" voyage/` found four unflagged sites:

```
voyage/workers/video_longlive.py:111:  state = torch.load(          # T5 .pth, weights_only=False
voyage/workers/video_longlive.py:658:  generator_container = torch.load(str(generator_ckpt), map_location="cpu")  # no flag
voyage/workers/video_longlive.py:1036: tape = torch.load(handle, map_location="cpu", weights_only=False)
voyage/workers/video_longlive.py:1079: tape = torch.load(handle, map_location="cpu", weights_only=False)
voyage/workers/video_causvid.py:502:   loaded = torch.load(str(checkpoint), map_location="cpu")
```

A compromised mirror, cache poisoning (`~/.cache/voyage-models`), or a crafted
shared run dir yielded code execution at video-worker privilege. Notably the
CausVid 10.6 GiB snapshot — whose registry comment says "full training snapshot,
not params-only" — loaded with no `weights_only` flag at all.

Live state (re-verified 2026-09-30): every load is `weights_only=True` with a
sha256 pre-check — longlive T5 (`voyage/workers/video_longlive.py:167-170`),
generator container (`:706-717`, manifest hash checked before any `torch.load`),
recovery tapes (`:1189`), CausVid snapshot
(`voyage/workers/video_causvid.py:555-557`), MMAudio
(`voyage/audio/mmaudio_sfx.py:194`), augment worker (`:307,323`).

## Why this is an issue

- Pickle deserialization = RCE by design; model weights are the highest-risk input.
- Recovery tapes live under the run dir (shared/copied between machines) and were
  loaded with `weights_only=False` twice per resume path.
- LTXV tapes were already JSON (`video_ltxv.py:210` rejects pickles loudly) —
  LongLive was the outlier, so precedent for the safe pattern existed in-tree.

## Evidence

Live verification 2026-09-30:

```
$ rg -n "torch\.load|weights_only|quant_api" voyage/ | head
voyage/workers/video_longlive.py:167:        state = torch.load(
voyage/workers/video_longlive.py:170:             weights_only=True,  # 005: plain state dict — never unpickle code
voyage/workers/video_longlive.py:717:     return torch.load(str(generator_ckpt), map_location="cpu", weights_only=True)
voyage/workers/video_longlive.py:1189:         tape = torch.load(handle, map_location="cpu", weights_only=True)
voyage/workers/video_causvid.py:557:         loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
voyage/model_registry.py:333:             f"got {actual} — refusing to torch.load an untrusted file"
```

No `torch.load` without `weights_only` remains in `voyage/`.

## Reproduction

Static: `rg -n "torch.load" voyage/workers/video_longlive.py
voyage/workers/video_causvid.py` — pre-fix four sites lacked the flag; now all
carry it. Dynamic PoC (do NOT run outside a sandbox): craft a `.pt` whose pickle
payload touches a sentinel file; call the worker load path.

## Source references

- `voyage/workers/video_longlive.py:167-170,706-717,1178-1189`;
  `voyage/workers/video_causvid.py:555-557`; `voyage/model_registry.py:329-342`
  (sha pre-verify); `voyage/workers/video_ltxv.py:210` (JSON-tape good pattern).

## Resolution candidates

1. (Landed) `weights_only=True` everywhere state-dicts allow; safetensors path
   where possible.
2. (Landed) sha256-verify checkpoints against the registry before load.
3. (Partially landed) Migrate LongLive recovery tapes to JSON like LTXV/CausVid —
   strict `weights_only=True` + comment now contains the risk; full JSON migration
   remains an optional hardening.
4. Gates check: `rg "torch.load" voyage | grep -v weights_only` must be empty
   (or allow-listed with a security comment).

## Online references

- Python `pickle` security warning — "Warning: The pickle module is not secure.
  Only unpickle data you trust":
  https://docs.python.org/3/library/pickle.html
- PyTorch `torch.load` — "`weights_only=True` ... restricts the unpickler to only
  load tensors, primitive types, dicts and lists":
  https://pytorch.org/docs/stable/generated/torch.load.html
- Deserialization of untrusted data (CWE-502):
  https://cwe.mitre.org/data/definitions/502.html

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; `rg` re-verified by orchestrator.
- Resolution batch 3 + issue 056: `weights_only=True` + sha pre-verify landed.
- 2026-09-30: re-verified live (all sites flagged); reconstructed from archived
  pass-1 text (commit `b5d7dda`). Status → resolved.

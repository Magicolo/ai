# 005 — `torch.load` on checkpoints without `weights_only=True` (arbitrary code execution)

- Status: open
- Severity: critical (security — RCE via pickle)
- Area: supply chain / worker model loading
- Rank rationale: the only RCE in the list; `.pth`/`.pt` are pickle, sources
  include HF mirrors, shared caches, and run dirs.

## Technical description

```
$ rg -n "torch.load" Voyage/voyage/
Voyage/voyage/model_registry.py:181:# strict-load `torch.load(<checkpoint>)['generator']` ...
Voyage/voyage/workers/video_causvid.py:502:        loaded = torch.load(str(checkpoint), map_location="cpu")
Voyage/voyage/workers/video_longlive.py:111:        state = torch.load(          # T5 .pth, weights_only=False
Voyage/voyage/workers/video_longlive.py:658:        generator_container = torch.load(str(generator_ckpt), map_location="cpu")  # no flag
Voyage/voyage/workers/video_longlive.py:1036:        tape = torch.load(handle, map_location="cpu", weights_only=False)
Voyage/voyage/workers/video_longlive.py:1079:        tape = torch.load(handle, map_location="cpu", weights_only=False)
```

(`rg` output above verified live by orchestrator 2026-09-25.) A compromised
mirror, cache poisoning (`~/.cache/voyage-models`), or a crafted shared run dir
yields code execution at video-worker privilege (root in container, GPU host).

Notably `video_causvid.py:502` loads the 10.6 GiB `tianweiy/CausVid` DMD snapshot
— whose own registry comment says "full training snapshot, not params-only" —
with no `weights_only` flag at all.

## Why this is an issue

- Pickle deserialization = RCE by design; model weights are the highest-risk input.
- Recovery tapes live under the run dir (shared/copied between machines) and are
  loaded with `weights_only=False` twice per resume path.
- LTXV/CausVid tapes are already JSON (`video_ltxv.py:210` rejects pickles loudly)
  — LongLive is the outlier, so precedent for the safe pattern exists in-tree.

## Evidence

- `rg` output above. No `weights_only=True` anywhere in `voyage/`.

## Reproduction

Static: `rg -n "torch.load" Voyage/voyage/workers/video_longlive.py
Voyage/voyage/workers/video_causvid.py`. Dynamic PoC (do NOT run outside a
sandbox): craft a `.pt` whose pickle payload touches a sentinel file; call the
worker load path.

## Source references

- `voyage/workers/video_longlive.py:111,658,1036,1079`,
  `voyage/workers/video_causvid.py:502`, `voyage/model_registry.py:180-181`,
  `voyage/workers/video_ltxv.py:210` (good pattern).

## Resolution candidates

1. `weights_only=True` everywhere state-dicts allow; strict-load safetensors path
   where possible.
2. sha256-verify `model.pt`/`model_bf16.pt` against the registry before load.
3. Migrate LongLive recovery tapes to JSON like LTXV/CausVid (kills two
   `weights_only=False` sites permanently).
4. Add a gates check: `rg "torch.load" voyage | grep -v weights_only` must be empty
   (or allow-listed with a security comment).

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep; `rg` re-verified by orchestrator.
- Open: implement 1-4; never load an untrusted checkpoint in the PoC.
- 2026-09-25 (repair pass): `rg torch.load` re-run live — all four unflagged
  sites still present (`video_longlive.py:111,658,1036,1079`,
  `video_causvid.py:502`); `## Why this is an issue` already present, no change.

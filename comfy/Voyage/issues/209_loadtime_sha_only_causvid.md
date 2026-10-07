# 209 — Hash-verification coverage gaps: load-time sha only for CausVid; six weight families have zero ingest hash (size-floor only) (HIGH)

## Technical description
Only `video_causvid` calls `verify_checkpoint_against_manifest` before
`torch.load` (`video_causvid.py:567-568`). `LTX25Session.__init__`,
`LTX23Session`, `LTXVSession` (`video_ltxv.py:357-363`),
`mmaudio_sfx.initialize` (`voyage/audio/mmaudio_sfx.py:299`), ACE-Step init,
and the ComfyUI GGUF graph path check `exists()` only.
`augment_worker._verify_weights_manifest` is best-effort passthrough (no
manifest → return, no error).

```
$ grep -rn "verify_checkpoint_against_manifest" voyage/ --include="*.py"
voyage/workers/video_causvid.py:567: verify_checkpoint_against_manifest(models_dir, "causvid", checkpoint)
voyage/model_registry.py:507: def verify_checkpoint_against_manifest(...   # def only, no other caller
$ grep -rn "torch\.load" voyage/ --include="*.py"
voyage/audio/mmaudio_sfx.py:299: weights = torch.load(..., weights_only=True)   # no verify above
voyage/workers/video_causvid.py:568: loaded = torch.load(...)                    # preceded by verify (good)
voyage/workers/augment_worker.py:1100: torch.load(..., weights_only=True)       # manifest check is best-effort
```

## Rationale
Ingest (`download_model`) + periodic `verify_model` do not cover the actual
`torch.load` / `load_file` moment. A file swapped after `ensure` (shared
`/models` mount, multi-agent box, disk error) executes with no check. The one
correct call site proves the pattern was intended everywhere.

## Live evidence
See grep outputs above; `augment_worker.py:1124-1133`: "No manifest, no
entry, or an unreadable manifest passes through".

## Repro
Provision via `download_model`, flip one byte in `ltx25/*.gguf` or
`mmaudio/weights/*.pth`, call worker `handle_init`/`initialize` → loads
(only causvid raises).

## Source refs
- `Voyage/voyage/workers/video_causvid.py:567` (positive example)
- Gaps: `Voyage/voyage/workers/video_ltx25.py:~630-645`,
  `Voyage/voyage/workers/video_ltxv.py:357-363`,
  `Voyage/voyage/audio/mmaudio_sfx.py:292-299`,
  `Voyage/voyage/workers/augment_worker.py:1124-1187`

## Online sources
- HF guidance: pin `revision=` to immutable SHA + verify bytes out-of-band.
- pip `secure-installs` hash-checking doctrine.
- In-tree `verify_checkpoint_sha256` docstring ("refusing to torch.load an
  untrusted file").

## Fix candidates
1. Call `verify_checkpoint_against_manifest(models_dir, <key>, path)`
   (fail-closed, no `allow_missing_manifest`) at every session `__init__`
   before first `torch.load`/`load_file`/`create_transformer`.
2. Make `_verify_weights_manifest` fail-closed when a manifest is expected
   (or delete the passthrough and route through the registry helper).

## Log
- Track F sweep, 2026-10-07. Read-only; nothing fixed.

## Consolidated from 211_zero_hash_weight_families (2026-10-07)

Merged scope: the verification-coverage gap has two halves — no load-time check
anywhere but CausVid (above), and no ingest hash at all for six weight families
(below).

### Technical description (from 211)
`expected_hashes=()` for `director-qwen8b`, `director-qwen4b-awq`,
`inspector-qwen35`, `audio-acestep`, `sfx-mmaudio`, `causvid` (DMD +
`Wan2.1-T2V-1.3B` base). Verification is `RequiredFile.min_bytes` + globs.

```
audio-acestep: expected=0 checks=16
causvid: expected=0 checks=9
director-qwen35-gguf: expected=0 ...
director-qwen4b-awq: expected=0 ...
director-qwen8b: expected=0 ...
inspector-qwen35: expected=0 ...
sfx-mmaudio: expected=0 ...
```

In-tree TODOs admit it: `registry_records.py:680-682` ("No constant exists
for the CausVid DMD checkpoint … re-provision, measure, and add it here
(residual)"), `registry_director.py:17-18`, `registry_audio.py:12-13`,
`registry_sfx.py:15-16`, `registry_inspector.py:11-13`.

### Rationale (from 211)
The largest blobs — 8B/19B LLMs, ACE 7 GB+, MMAudio 10 GB+, Wan 17 GB —
accept any bytes of the right size. A truncated, swapped, or malicious file
above the floor passes `verify_model` and loads.

### Live evidence (from 211)
See spec-table probe output above
(`len(s.expected_hashes)` per `MODEL_SPECS` row).

### Repro (from 211)
`download_model` with a stubbed `hf_hub_download` writing `min_bytes` of
zeros → merges manifest, `verify_model` returns True.

### Source refs (from 211)
- `Voyage/voyage/model_registry.py:699-996` (spec table)
- Per-family `Voyage/voyage/registry_{director,audio,sfx,inspector,causvid}.py`

### Online sources (from 211)
- HF content-addressable cache (files stored by hash — pinning the LFS oid /
  `EXPECTED_*` is the documented equivalent).
- pip `--require-hashes` doctrine (all artifacts hashed, not sized).

### Fix candidates (from 211)
Measure live `sha256_file` at pinned revisions into `EXPECTED_*` +
`expected_hashes` (same 071 pattern as LTXV/FILM/RIFE/RealESRGAN); for
sharded snapshots (Qwen 8B/3.5) pin per-shard or a `checkpoint_shas` dict
like ltxv.

### Log (from 211)
- Track F sweep, 2026-10-07. Read-only; nothing fixed.

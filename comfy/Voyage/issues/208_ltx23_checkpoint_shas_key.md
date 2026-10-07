# 208 — LTX `checkpoint_shas`: ltx23 DiT key omits `distilled/` (verify always mismatches) and only 2 of 5 pinned files are recorded (60% presence-only) (HIGH)

## Technical description
The ltx23 DiT lives at `ltx23/distilled/<file>` (checks + `expected_hashes`
agree), but `_record_ltx23` records its ongoing sha under `ltx23/<file>`
(no subfolder). `_manifest_hash_mismatches` iterates recorded keys, so it
stats a nonexistent path → `not is_file()` → mismatch on every provisioned
volume.

```
$ python3 -c "from voyage.model_registry import MODEL_SPECS; ..."
ltx23/distilled/ltx-2.3-22b-distilled-Q3_K_M.gguf   # checks + expected agree
$ grep -n "checkpoint_shas" -A3 voyage/registry_ltx23.py
122-124: f"{LTX23_SUBDIR}/{LTX23_DIT_FILE}": sha256_file(dit_path),  # → ltx23/ltx-2.3… (wrong)
```
`dit_path = ltx23_dir / LTX23_DIT_SUBFOLDER / LTX23_DIT_FILE`
(`registry_ltx23.py:102`) — hashed bytes are right, recorded key is wrong.

## Rationale
A verify path that false-positives on good volumes trains the exact bypass
behavior the hash system exists to prevent.

## Live evidence
See probe output above (spec paths vs recorded key).

## Repro
Build a fake `models_dir` with correct files + `download_model`-style
manifest from `_record_ltx23`, then `verify_model(dir, "ltx23")` →
`hash mismatch … ltx23/ltx-2.3…` despite bytes being correct.

## Source refs
- `Voyage/voyage/registry_ltx23.py:122-124` (key), vs `:40-43` (layout) and
  `Voyage/voyage/model_registry.py:1127-1130` (expected)
- Reader `Voyage/voyage/model_registry.py:1360-1369` (`_check` fails on non-file)

## Online sources
- In-tree `_manifest_hash_mismatches` contract (recorded keys must be
  stat-able paths).

## Fix candidates
1. Key as `f"{LTX23_SUBDIR}/{LTX23_DIT_SUBFOLDER}/{LTX23_DIT_FILE}"`.
2. Add a cross-test asserting every `checkpoint_shas` key ∈
   `expected_hashes` paths ∪ `manifest_checkpoint`.

## Log
- Track F sweep, 2026-10-07. Read-only; nothing fixed.

## Consolidated from 210_partial_checkpoint_shas (2026-10-07)

Merged scope: the same recording path is not only keyed wrong (above), it is also
incomplete — `_record_ltx25` / `_record_ltx23` persist shas for DiT+TE only, while
`expected_hashes` pins 5 files per LTX stack (ingest gate).

### Technical description (from 210)
`expected_hashes` pins 5 files per LTX stack (ingest gate), but
`_record_ltx25` / `_record_ltx23` persist shas for DiT+TE only.
`_manifest_hash_mismatches` only checks recorded keys, so video/audio VAEs,
latent upscaler, and ltx23 connectors have no ongoing check in
`verify_model`/`ensure`.

```
$ python3 -c "from voyage.model_registry import MODEL_SPECS; ..."
5 ['ltx25/LTX-2.5...gguf', 'ltx25/gemma4...gguf', 'ltx25/vae/...video...',
   'ltx25/vae/...audio...', 'ltx25/latent_upscale_models/...']
$ grep -n "checkpoint_shas" -A4 voyage/registry_ltx25.py
121-124: only DIT_FILE + TE_FILE (2 entries)
```

Same shape in `registry_ltx23.py:121-125`. `ltxv-2b` is the correct example
(2/2 covered).

### Rationale (from 210)
Partial ongoing integrity — 60% of each LTX stack is presence-only after
ingest.

### Live evidence (from 210)
See probe output above.

### Repro (from 210)
Corrupt `ltx25/vae/ltx-2.5-video-vae-conv-bf16.safetensors` post-provision →
`verify_model(dir,"ltx25")` still `(True, "ltx25 OK …")`.

### Source refs (from 210)
- `Voyage/voyage/registry_ltx25.py:97-125`
- `Voyage/voyage/registry_ltx23.py:99-126`
- Reader `Voyage/voyage/model_registry.py:1365-1374`

### Online sources (from 210)
- Content-addressable cache practice (pin every artifact, not just the
  headline blobs).

### Fix candidates (from 210)
1. Record all 5 shas (hash cost is one-time, CPU-only, already paid at
   ingest).
2. Or split `manifest_checkpoint` per file.
3. Add a test asserting `set(checkpoint_shas) ⊇ set(expected_hashes paths)`.

Merged regression-test requirement (208 candidate 2 + 210 candidate 3): assert
every `checkpoint_shas` key resolves to a real file (key ∈ `expected_hashes`
paths ∪ `manifest_checkpoint`) AND the recorded set is a superset of the
`expected_hashes` paths.

### Log (from 210)
- Track F sweep, 2026-10-07. Read-only; nothing fixed.

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

## Evaluation (2026-10-07, resolving track)

All load-bearing claims re-verified live against the working tree before
fixing (concurrent agents have uncommitted changes elsewhere in the tree;
none touched the files below):

- DiT key bug TRUE: pre-fix `voyage/registry_ltx23.py:123` recorded
  `f"{LTX23_SUBDIR}/{LTX23_DIT_FILE}"` (`ltx23/ltx-2.3…`) while `checks`
  (`model_registry.py:1101-1104`), `expected_hashes`
  (`model_registry.py:1127-1130`) and the session's own `dit_path`
  (`registry_ltx23.py:102`, `workers/video_ltx23.py:676`) all resolve
  `ltx23/distilled/ltx-2.3…`. `_manifest_hash_mismatches`
  (`model_registry.py:1360-1363`) stats each recorded key, so every
  provisioned ltx23 volume mismatched on the DiT entry. Confirmed by
  reading all three sites, not just the issue text.
- 2-of-5 incompleteness TRUE for both stacks: pre-fix `_record_ltx25`
  (`registry_ltx25.py:121-124`) and `_record_ltx23` recorded DiT+TE only,
  while `expected_hashes` pins 5 files per stack (`model_registry.py:1041-1056`
  ltx25, `:1126-1144` ltx23). The reader (`model_registry.py:1365-1370`)
  only checks recorded keys, so VAEs/upscaler/connectors were
  presence-only after ingest — the repro direction (corrupt a VAE →
  `verify_model` still True) follows directly from the reader code.
- ltxv 2/2 correct TRUE: `_record_ltxv` (`registry_ltxv.py:74-77`) records
  both files its `expected_hashes` pins (`model_registry.py:1951-1954`
  shape, verified at `model_registry.py:1188-1195` in this tree). No
  change needed there.
- No live-weight measuring needed: all 5 `EXPECTED_LTX2*` pins already
  exist (207-pattern), so the fix records against existing pins — no
  digests invented.

## Progress log (2026-10-07, resolving track)

- `voyage/registry_ltx23.py`: DiT key now
  `f"{LTX23_SUBDIR}/{LTX23_DIT_SUBFOLDER}/{LTX23_DIT_FILE}"`; `checkpoint_shas`
  extended to all 5 `expected_hashes` paths (DiT+TE+connectors+video/audio
  VAE; the shared Mode-A upscaler stays pinned once in `registry_ltx25`
  by design, documented in the comment).
- `voyage/registry_ltx25.py`: `checkpoint_shas` extended to all 5
  `expected_hashes` paths (DiT+TE+video/audio VAE+upscaler). Also hosts
  the new shared `verify_recorded_shas` helper (the 209 load-time gate;
  see issue 209 for why the causvid helper could not be reused verbatim).
- `tests/test_registry_pins.py`: new parametrized
  `test_checkpoint_shas_cover_all_expected_hashes` (ltx25/ltx23/ltxv)
  asserting recorded keys EQUAL the spec's `expected_hashes` paths —
  covers both the distilled-key fix and the superset requirement
  (208-candidate-2 + 210-candidate-3 merged). Pre-existing I001 import
  sort failure on this file (present at HEAD, ruff-version drift) fixed
  with `ruff check --fix` as fix-on-sight.
- Gates (all in-container per `scripts/gates.sh` conventions):
  `ruff check` + `ruff format --check` clean on all 7 touched/new files;
  `mypy` clean on the 5 touched source modules;
  `pytest tests/test_registry_pins.py tests/test_ltx_session_verify.py`
  32 passed; neighbor scope
  (`test_registry_ltxv_split`/`test_registry_split`/`test_tail_derive`/
  `test_ltxv*`/`test_169_block_zero_fresh`/`test_perf_regressions`/
  `test_checkpoint_safety`) 138 passed.
- Not committed (per task instructions; concurrent agents hold
  uncommitted changes in adjacent files — own hunks only, no staging).

## Resolution (2026-10-07)

Fixed. `checkpoint_shas` now equals `expected_hashes` paths for
ltx25/ltx23 (ltxv was already exact), so every recorded key resolves and
`verify_model` checks the whole stack.

Migration note: volumes provisioned before this change carry 2-key
manifests. `verify_model` on an old manifest keeps checking just those 2
keys (reader only iterates recorded keys — no false positives), but the
new session load-time gates (issue 209) fail closed on the 3 missing
shas. Re-run `download_model` for the affected spec to re-record all 5.

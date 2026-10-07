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

## Evaluation (2026-10-07, resolving track)

Load-bearing claims re-verified live against the working tree before
fixing:

- Load-time gap TRUE for the three in-scope stacks: pre-fix
  `LTX25Session.__init__` (`workers/video_ltx25.py:756-776`),
  `LTX23Session.__init__` (`workers/video_ltx23.py:670-691`) and
  `LTXVSession.__init__` (`workers/video_ltxv.py:338-363`) only
  `exists()`-checked their weights; the sole
  `verify_checkpoint_against_manifest` call site in workers is
  `video_causvid.py:567`, confirmed via tree-wide grep (only other hit
  is the def at `model_registry.py:507`).
- ADAPTATION REQUIRED (stale assumption in the fix sketch): the sketch
  says to "mirror the existing correct video_causvid.py:567 call site",
  but `verify_checkpoint_against_manifest` (`model_registry.py:507-551`,
  verified live) only covers the single-`checkpoint_sha256` shape — it
  reads `entry.get("checkpoint_sha256")` and has no `checkpoint_shas`
  branch. Calling it with `("ltx25"|"ltx23"|"ltxv", …)` would fail closed
  on EVERY load (no such single-sha entry exists), turning the gate into
  a liveness bug. So the fix adds the dict-shape sibling
  `verify_recorded_shas` in `voyage/registry_ltx25.py` (in-scope file)
  instead of touching `voyage/model_registry.py` (concurrent-agent
  owned — that file has their uncommitted changes). Semantics mirror the
  causvid posture exactly: fail closed, no `allow_missing_manifest`
  parameter, no `VOYAGE_ALLOW_MISSING_MANIFEST` opt-in.
- `verify_checkpoint_sha256` (the per-file primitive) confirmed still
  exported at `model_registry.py:491` in the working tree despite the
  concurrent edits; the new helper lazy-imports it (avoids a
  registry↔registry import cycle: `model_registry` already imports
  `_record_ltx25` at its top).
- Six zero-ingest-hash families (director-qwen8b/awq, inspector-qwen35,
  audio-acestep, sfx-mmaudio, causvid) plus the `augment_worker`
  passthrough and the mmaudio/ACE load-time gaps are REAL but OUT OF
  SCOPE for this track: every file they live in
  (`voyage/model_registry.py`, `voyage/registry_{director,audio,sfx,
  inspector,causvid}.py`, `voyage/workers/augment_worker.py`,
  `voyage/audio/*`, `voyage/workers/loop.py`, `scripts/`, `docs/`) is
  owned by the concurrent agent per the task brief. Explicitly NOT
  addressed here — no per-shard pins added, no digests invented, no
  foreign files touched. These remain open (see Resolution).
- No new digests needed in scope: all 5+2 LTX pins already exist
  (207-pattern); the "measure live" instruction had nothing to measure.

## Progress log (2026-10-07, resolving track)

- `voyage/registry_ltx25.py`: new `verify_recorded_shas(models_dir,
  manifest_key, paths)` — fail-closed per-file check of a manifest
  `checkpoint_shas` dict (missing manifest / torn JSON / missing entry /
  missing per-file sha / byte mismatch all raise `ValueError`; TRY004
  `noqa`s are targeted with contract justification, matching the
  single-`ValueError`-type fail-closed convention of the causvid helper).
- `voyage/workers/video_ltx25.py`, `video_ltx23.py`, `video_ltxv.py`:
  new thin `_verify_stack_manifest(models_dir)` per worker (exact
  `checkpoint_shas`-keyed path maps: 5 files for ltx25/ltx23, 2 for
  ltxv; ltx23's shared upscaler stays presence-gated — pinned once in
  the ltx25 entry; ltxv's TE snapshot stays presence-gated via
  `_resolve_te_source` — no per-file pins exist for it), each called as
  the FIRST statement of the session `__init__` (before torch/ComfyUI
  imports) so swapped bytes fail fast with stdlib only.
- `tests/test_registry_pins.py`: existing
  `test_ltxv_session_loads_te_from_local_snapshot` now provisions a
  matching `checkpoint_shas` manifest (it constructs a real
  `LTXVSession`, which fail-closes without one).
- New `tests/test_ltx_session_verify.py` (13 tests): helper accept /
  tamper / missing-manifest / missing-entry / missing-per-file-sha /
  torn-manifest legs; all three sessions refuse construction without a
  manifest (no GPU stubs needed — the gate runs before heavy imports);
  parametrized per-stack coverage (gate verifies exactly the
  `expected_hashes` set, via a stubbed primitive) + tamper-trip legs.
- Gates (all in-container per `scripts/gates.sh` conventions):
  `ruff check` + `ruff format --check` clean on all 7 touched/new files;
  `mypy` clean on the 5 touched source modules;
  `pytest tests/test_registry_pins.py tests/test_ltx_session_verify.py`
  32 passed; neighbor scope 138 passed (see issue 208 log for the list).
- Not committed (per task instructions; own hunks only).

## Resolution (2026-10-07)

Partially resolved — the in-scope half is done: ltx25/ltx23/ltxv now
verify every recorded weight at session `__init__`, fail closed, with no
bypass path (208's `checkpoint_shas == expected_hashes` equality test
pins the recorded set so the gate cannot silently narrow).

Explicitly NOT resolved here (out-of-scope files, concurrent owner):
per-shard/measured pins for the six zero-`expected_hashes` families
(director-qwen8b/awq, inspector-qwen35, audio-acestep, sfx-mmaudio,
causvid DMD+Wan base); fail-closed load-time checks for mmaudio/ACE-Step
`torch.load`s and the `augment_worker` manifest passthrough; any
`model_registry.py` change (e.g. teaching
`verify_checkpoint_against_manifest` the dict shape, which would let the
three workers drop their local sibling). A follow-up track owning those
files should take these.

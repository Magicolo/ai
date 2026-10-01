# 070 — Largest video weight base floats: Wan2.2 snapshot has `revision=None`

**Severity:** HIGH

**File:line:** `Voyage/voyage/model_registry.py:418` (`revision: str | None  # None = floating …`), `:746` (`SnapshotSpec(WAN_HF_REPO, None, …)`), `:1137-1138` (revision kwarg skipped for `None`)

**Area:** model registry / pins — LongLive base weights

## Description

Every other snapshot pins a full 40-char revision; the Wan2.2-TI2V-5B base (3 diffusion shards + VAE + 11 GB T5 + tokenizer — the foundation under the LongLive generator) downloads `main` at whatever it points to on provision day. `_run_spec_downloads` (`:1050-1051`) skips the `revision=` kwarg entirely for it, and the manifest records no `wan_revision`, so two provisions a month apart can yield different base weights with identical manifests.

## Rationale

HF's own security guidance: "always verify the content… We recommend setting a revision in order to ensure you protect yourself from updates on the repository." A floating base also defeats the run-manifest provenance claim (`DESIGN §§84-85`: "records provider/repo/revision…") for the heaviest files in the volume.

## Evidence

Re-verified 2026-09-30 live:

```
Voyage/voyage/model_registry.py:33: WAN_HF_REPO = "Wan-AI/Wan2.2-TI2V-5B"
Voyage/voyage/model_registry.py:418: revision: str | None  # None = floating (Wan2.2 has no pinned revision yet — see 011)
Voyage/voyage/model_registry.py:746: snapshots=(SnapshotSpec(WAN_HF_REPO, None, _WAN22_RELATIVE, tuple(WAN_ALLOW)),),
Voyage/voyage/model_registry.py:1137-1138:
        if snapshot.revision is not None:
            snapshot_kwargs["revision"] = snapshot.revision
Voyage/voyage/model_registry.py:~455-471 (_record_longlive2): records repo/revision/sha for the
  LongLive file but only "wan_repo"/"wan_dir"/"wan_license" — no wan_revision.
```

`rg -n "WAN_HF_REVISION" voyage/model_registry.py` → zero hits (no such constant); all sibling specs (`LONGLIVE/QWEN/ACE/LTXV/CAUSVID/…`: `:24`, `:52`, `:104`, `:118`, `:157`, `:164`, `:187`, `:206`) carry `*_REVISION` constants.

## Repro

`voyage models download longlive2-bf16` on two dates (or `snapshot_download(repo_id="Wan-AI/Wan2.2-TI2V-5B", allow_patterns=[…])` twice across an upstream push) → differing shard hashes, identical manifest shape.

## Fix candidates

1. Pin `WAN_HF_REVISION = "<40-hex>"` like the rest.
2. Include `wan_revision` in `_record_longlive2`.
3. Add a gate test asserting no `SnapshotSpec(..., None, ...)` remains.

## Refs

- HF transformers SECURITY.md — "please **always** verify the content… We recommend setting a revision" — https://github.com/huggingface/transformers/blob/main/SECURITY.md

## Progress log

- 2026-09-30: premise re-verified against live `Voyage/voyage/model_registry.py`:
  `WAN_HF_REPO` at `:33`, `SnapshotSpec(WAN_HF_REPO, None, …)` at `:746`,
  revision-kwarg skip at `:1137-1138`, `_record_longlive2` (`:476-492`) with no
  `wan_revision`, and zero `WAN_HF_REVISION` hits — holds verbatim.
- 2026-09-30: resolution source checked — unresolvable CPU-only by design: the
  provisioned volume (`~/.cache/voyage-models/`) was pruned 2026-09-24
  (`wan_models/` + `longlive2/` absent; `ls` confirms), so no verified bytes
  exist to pin against. Per the contract, NO hash was invented.
- 2026-09-30: TDD — `test_wan_revision_constant_exists_and_is_used` and
  `test_wan_revision_is_recorded_in_manifest` (both in
  `tests/test_registry_pins.py`) failed pre-fix (no constant, no record key),
  green post-fix. `test_floating_snapshot_set_is_exactly_wan22` is labeled
  characterization (passes pre/post) — the regression gate against new floats.
- 2026-09-30: gates — ruff + format + mypy strict clean on
  `voyage/model_registry.py`; full suite shows no new failures from this issue.

## Resolution

- Honest placeholder, not a pin: new `WAN_HF_REVISION: str | None = None`
  constant with the exact pin procedure in its comment (resolve the verified
  main commit via `HfApi().model_info('Wan-AI/Wan2.2-TI2V-5B').sha`,
  re-provision the subset at that revision, sha256 the shards, set the 40-hex,
  shrink the floating-set test to empty). The `longlive2-bf16` SnapshotSpec
  now reads the constant (no more inline `None`), and `_record_longlive2`
  records `"wan_revision": WAN_HF_REVISION` (null until pinned) so the
  provenance gap is visible in every manifest instead of absent.
- Fix candidates 1+2 are therefore half-done (plumbing, no value); candidate 3
  is done as the floating-set gate, which encodes the still-open state
  honestly: it asserts the floating set is EXACTLY `{('longlive2-bf16',
  WAN_HF_REPO)}` — a new floating row fails, and pinning Wan forces the test
  to shrink to empty.
- Residual (GPU box + network required): the actual 40-hex pin. Do NOT pick it
  from the Hub API alone — the pinned bytes must be verified against a
  provisioned volume (re-provision first: the current volume has no Wan
  subset to compare against).

## Progress log (2026-09-30, tests-only pass — CHECK)

- Probe executed live (no GPU needed, no host pip): `$VOYAGE_MODELS` unset;
  `~/.cache/voyage-models/` contains 12 entries (`PixArt-XL-2-1024-MS`,
  `Qwen3-4B-AWQ`, `Qwen3-8B`, `Qwen3.5-9B`, `acestep`, `all-MiniLM-L6-v2`,
  `frame_interpolation`, `ltx25-gguf`, `ltxv-2b`, `manifest.json`,
  `mmaudio`, `realesrgan`) — `wan_models/` ABSENT, `longlive2/` ABSENT
  (both `ls` → "No such file or directory"). Prune date 2026-09-24
  confirmed still in effect; no Wan bytes were re-provisioned since.
- Pin state re-verified (read-only): `WAN_HF_REVISION: str | None = None`
  (`voyage/registry_records.py`, honest placeholder with pin procedure in
  its comment); `SnapshotSpec(WAN_HF_REPO, WAN_HF_REVISION, …)` reads the
  constant (no inline `None`); manifest records `wan_revision: null`
  (visible gap, not absent). Floating-set gate still encodes exactly
  `{('longlive2-bf16', WAN_HF_REPO)}`.
- Verdict: STILL-BLOCKED — no verified bytes exist to measure a revision
  against. Per the contract NO hash was invented (a wrong pin fails every
  provision loudly). No files changed in this pass.

## Resolution (2026-09-30, tests-only pass)

- Verdict: blocked (still-blocked with exact probe output above). Files
  changed: none. DESIGN proposals: none.
- Residuals (exact handoff, GPU+network owner): re-provision the Wan2.2
  subset at the verified main commit
  (`HfApi().model_info('Wan-AI/Wan2.2-TI2V-5B').sha`), sha256 the shards
  against the provisioned bytes, set the 40-hex `WAN_HF_REVISION`, shrink
  `tests/test_registry_pins.py` floating-set to empty. Do NOT pick the
  revision from the Hub API alone without byte verification.

## Progress log (2026-09-30, this pass — CHECK)

- Probe executed live (no GPU needed, no host pip): `$VOYAGE_MODELS`
  unset; `~/.cache/voyage-models/` holds 12 entries (`PixArt-XL-2-1024-MS`,
  `Qwen3-4B-AWQ`, `Qwen3-8B`, `Qwen3.5-9B`, `acestep`, `all-MiniLM-L6-v2`,
  `frame_interpolation`, `ltx25-gguf`, `ltxv-2b`, `manifest.json`,
  `mmaudio`, `realesrgan`) — `wan_models/` ABSENT and `longlive2/`
  ABSENT (both `ls` → "No such file or directory", dated 2026-09-30).
  Prune date 2026-09-24 still in effect; no Wan bytes re-provisioned
  since (the two new entries vs the last probe, `frame_interpolation`
  + `realesrgan`, are the augment weights for issue 166 — not Wan).
- Pin state re-verified in-container (`voyage:latest`, read-only):
  `WAN_HF_REVISION` still `None` (`voyage/registry_records.py:52`,
  honest placeholder with pin procedure); `SnapshotSpec(WAN_HF_REPO,
  WAN_HF_REVISION, …)` reads the constant (`voyage/model_registry.py:557`,
  no inline `None`); manifest records `"wan_revision": None`
  (`registry_records.py:503`, visible gap, not absent).
- Verdict: STILL-BLOCKED — no verified bytes exist to measure a revision
  against. Per the contract NO hash was invented (a wrong pin fails every
  provision loudly). No code change (`model_registry.py`/`registry_records.py`
  may only change if weights unblock — they did not). Gate evidence: n/a
  (no files changed); adjacent `tests/test_registry_pins.py` green in the
  166 neighbor run (145 passed, 4 skipped — shared line, same container).

## Resolution (2026-09-30, this pass)

- Verdict: blocked (still-blocked with dated probe output above). Files
  changed: none (this issue file only). DESIGN proposals: none.
- Residuals (exact handoff, GPU+network owner): unchanged — re-provision
  the Wan2.2 subset at the verified main commit, sha256 the shards
  against the provisioned bytes, set the 40-hex `WAN_HF_REVISION`, shrink
  the floating-set gate to empty. Do NOT pick the revision from the Hub
  API alone without byte verification.

## Progress log (2026-09-30, close-out pass — RE-PROBE ONLY)

- Probe executed live, no host pip (host `ls`, in-container
  `voyage:latest` read-only):
  `ls ~/.cache/voyage-models/` → 12 entries (`PixArt-XL-2-1024-MS`,
  `Qwen3-4B-AWQ`, `Qwen3-8B`, `Qwen3.5-9B`, `acestep`,
  `all-MiniLM-L6-v2`, `frame_interpolation`, `ltx25-gguf`, `ltxv-2b`,
  `manifest.json`, `mmaudio`, `realesrgan`) — `wan_models/` ABSENT and
  `longlive2/` ABSENT (both `ls` → "No such file or directory",
  `$VOYAGE_MODELS` unset). Prune date 2026-09-24 still in effect; the
  two entries vs the last probe (`frame_interpolation` + `realesrgan`)
  are the augment weights, not Wan.
- In-container value: `python3 -c "import
  voyage.registry_records as r; print(repr(r.WAN_HF_REVISION))"` →
  `None` (honest placeholder with pin procedure,
  `voyage/registry_records.py:52`); `voyage/model_registry.py:551`
  `SnapshotSpec(WAN_HF_REPO, WAN_HF_REVISION, …)` reads the constant;
  manifest records `"wan_revision": None`
  (`registry_records.py:503`, visible gap). Concurrent uncommitted hunks
  in `model_registry.py` are JsonValue typing only (verified `git diff`
  — no revision/registry-record hunks), so the probe is uncontaminated.
- Verdict: STILL-BLOCKED — no verified bytes exist to measure a shard
  sha256 against. Per the contract NO hash was invented (a wrong pin
  fails every provision loudly). No code change. Gate evidence: n/a
  (no files changed).

## Resolution (2026-09-30, close-out pass)

- Verdict: blocked (still-blocked with fresh probe output above). Files
  changed: none (this issue file only). DESIGN proposals: none.
- Residuals (exact handoff, GPU+network owner): unchanged — re-provision
  the Wan2.2 subset at the verified main commit
  (`HfApi().model_info('Wan-AI/Wan2.2-TI2V-5B').sha`), sha256 the shards
  against the provisioned bytes, set the 40-hex `WAN_HF_REVISION`, shrink
  `tests/test_registry_pins.py` floating-set to empty. Do NOT pick the
  revision from the Hub API alone without byte verification.

## Progress log (2026-09-30, batch 12 — RE-PROBE ONLY)

- Probe executed live, no host pip (host `ls`, in-container
  `voyage:latest` read-only), dated 2026-09-30T21:29:21Z:
  `ls ~/.cache/voyage-models/` → 12 entries (`PixArt-XL-2-1024-MS`,
  `Qwen3-4B-AWQ`, `Qwen3-8B`, `Qwen3.5-9B`, `acestep`,
  `all-MiniLM-L6-v2`, `frame_interpolation`, `ltx25-gguf`, `ltxv-2b`,
  `manifest.json`, `mmaudio`, `realesrgan`) — `wan_models/` ABSENT and
  `longlive2/` ABSENT (both `ls` → "No such file or directory",
  `$VOYAGE_MODELS` unset). Prune date 2026-09-24 still in effect; no Wan
  bytes re-provisioned since.
- In-container values: `WAN_HF_REVISION` → `None` (both
  `voyage.registry_records` and the `voyage.model_registry` re-export);
  `MODEL_SPECS['longlive2-bf16'].snapshots[0]` →
  `('Wan-AI/Wan2.2-TI2V-5B', None)` (reads the constant, no inline
  `None`); manifest records `"wan_revision": None` (visible gap).
- Verdict: STILL-BLOCKED — no verified bytes exist to measure a shard
  sha256 against. Per the contract NO hash was invented (a wrong pin
  fails every provision loudly). No code change (`model_registry.py` /
  `registry_records.py` / the floating-set gate test may only change if
  weights unblock — they did not). Gate evidence: adjacent
  `tests/test_registry_pins.py` 14 passed in-container (no new failures).

## Resolution (2026-09-30, batch 12)

- Verdict: blocked (still-blocked with fresh probe output above). Files
  changed: none (this issue file only). DESIGN proposals: none.
- Residuals (exact handoff, GPU+network owner): unchanged — re-provision
  the Wan2.2 subset at the verified main commit
  (`HfApi().model_info('Wan-AI/Wan2.2-TI2V-5B').sha`), sha256 the shards
  against the provisioned bytes, set the 40-hex `WAN_HF_REVISION`, shrink
  `tests/test_registry_pins.py` floating-set to empty. Do NOT pick the
  revision from the Hub API alone without byte verification.

## Progress log (2026-10-01, batch 13 — RE-PROBE ONLY)

- Probe executed live, no host pip (host `ls`, in-container
  `voyage:latest` read-only), dated 2026-10-01T01:49:31Z:
  `ls ~/.cache/voyage-models/` → 12 entries (`PixArt-XL-2-1024-MS`,
  `Qwen3-4B-AWQ`, `Qwen3-8B`, `Qwen3.5-9B`, `acestep`,
  `all-MiniLM-L6-v2`, `frame_interpolation`, `ltx25-gguf`, `ltxv-2b`,
  `manifest.json`, `mmaudio`, `realesrgan`) — `wan_models/` ABSENT and
  `longlive2/` ABSENT (both `ls` → "No such file or directory",
  `$VOYAGE_MODELS` unset). Prune date 2026-09-24 still in effect; no Wan
  bytes re-provisioned since.
- In-container values: `WAN_HF_REVISION` → `None`
  (`voyage.registry_records`); `MODEL_SPECS['longlive2-bf16'].snapshots[0]`
  → `('Wan-AI/Wan2.2-TI2V-5B', None)` (reads the constant, no inline
  `None`); manifest records `"wan_revision": None` (visible gap).
- Verdict: STILL-BLOCKED — no verified bytes exist to measure a shard
  sha256 against. Per the contract NO hash was invented (a wrong pin
  fails every provision loudly). No code change. Gate evidence:
  adjacent `tests/test_registry_pins.py` green in the batch-13 neighbor
  run (shared line — 117 passed incl. the 070 pin/gate tests).

## Resolution (2026-10-01, batch 13)

- Verdict: blocked (still-blocked with fresh probe output above). Files
  changed: none (this issue file only). DESIGN proposals: none.
- Residuals (exact handoff, GPU+network owner): unchanged — re-provision
  the Wan2.2 subset at the verified main commit
  (`HfApi().model_info('Wan-AI/Wan2.2-TI2V-5B').sha`), sha256 the shards
  against the provisioned bytes, set the 40-hex `WAN_HF_REVISION`, shrink
  the floating-set gate to empty. Do NOT pick the revision from the Hub
  API alone without byte verification.

## Correction (2026-10-01, same pass — SUPERSEDED BY 079 DELETE)

- The probe above was accurate at 2026-10-01T01:49:31Z and is now stale:
  mid-pass the 079 group landed the longlive2 delete in the live tree
  (`voyage/workers/video_longlive.py` deleted, `longlive2-bf16` gone
  from `MODEL_SPECS`, `WAN_HF_REPO` / `WAN_HF_REVISION` /
  `LONGLIVE_HF_FILE` gone from the registry namespace — verified live
  in-container `voyage:latest`, CPU-only, no host pip).
- Fresh live facts (same container): `sorted(MODEL_SPECS)` = 9 rows
  (`audio-acestep`, `causvid`, `director-qwen4b-awq`,
  `director-qwen8b`, `film`, `inspector-qwen35`, `ltxv-2b`,
  `realesrgan-anime`, `sfx-mmaudio`); floating-snapshot scan = `[]`
  (ZERO rows with `revision is None`); `hasattr(model_registry,
  'WAN_HF_REVISION')` = `False`. There is no floating Wan2.2 revision
  left to pin — nothing to invent, nothing to gate.
- Consequence: the 070 pin/gate tests in `tests/test_registry_pins.py`
  (constant-exists, manifest-record, floating-set) now fail with
  `AttributeError` on the deleted names (3 failed in-container) —
  FOREIGN fallout of the 079 delete, not this leg. Updating or removing
  those tests belongs to the 079 owner (longlive2 lines are explicitly
  out of this leg's scope per the batch brief); this leg touches
  neither the tests nor any longlive2 line.
- Verdict: SUPERSEDED — resolved-by-deletion. Files changed: none (this
  issue file only). DESIGN proposals: none.
- Residual (exact handoff, 079 owner): complete the delete consistently
  — remove or repurpose the orphaned 070 gate tests
  (`test_wan_revision_constant_exists_and_is_used`,
  `test_wan_revision_is_recorded_in_manifest`,
  `test_floating_snapshot_set_is_exactly_wan22`) and confirm no
  `wan_revision` / `longlive2` references survive in manifests, docs,
  or gates.

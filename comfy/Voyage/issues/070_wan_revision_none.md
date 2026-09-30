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

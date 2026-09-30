# 165 — Stale `(issue NNN)` citations across container/registry/worker comments resolve to unrelated live topics

- **Severity:** LOW (docs-only traceability — no runtime effect; spans all Dockerfiles plus registry/worker/vision/audio comments)
- **File:line:** `Voyage/worker/Dockerfile.video:1,29,36,75,101,120,139,170,205,213`; `Voyage/worker/Dockerfile.director:1,42,51,70` (ABSENT live 2026-09-30 — see note); `Voyage/Dockerfile:1,11,23,49`; `Voyage/.dockerignore:1`; `Voyage/.gitignore:7`; `Voyage/voyage/hashing.py:1`; `Voyage/voyage/model_registry.py:313,385`; `Voyage/voyage/workers/director.py:89,95,193,463`; `Voyage/voyage/workers/video_ltxv.py:120,185,235,738,808`; `Voyage/voyage/vision/metrics.py:43,60,179`; `Voyage/voyage/audio/acestep.py:41,47,58,64,149`
- **Area:** supply-chain / containers / registry / worker-tails — comment traceability

## Description

~25 in-tree comments cite `(issue NNN)` for supply-chain topics, but the numbers do not match `000_INDEX.md` live topics. Spot-verified cite-vs-live mapping (cited → live, re-verified 2026-09-30):

- 012 digest/apt/pip pins (`worker/Dockerfile.video:1`, `:29`; `Dockerfile:1,11`) → live 012 is `start_workers()` leak
- 048 doctor gate (`Dockerfile:49`; `worker/Dockerfile.video` doctor gate) → live 048 is LongLive full-segment copies
- 053 voyager-user/WORKDIR (`Dockerfile:34-47`; `worker/Dockerfile.video:200-230`) → live 053 is concat quoting/ffmpeg hygiene
- 054 rich/textual layer order + tests-not-baked (`worker/Dockerfile.video:170`; `Dockerfile:23`) → live 054 is SFX two-worker ledger race
- 055 verified-fetch clones (×5 in `worker/Dockerfile.video:36,75,101,120,139`) → live 055 is inspect-metrics rotation blindness
- 084 (`.dockerignore:1` header) → live 084 is worker validators + resident fakes
- 085 (`.gitignore:7` comment) → live 085 is single-source collapse
- 021 hashing helpers (`voyage/hashing.py:1`; `video_ltxv.py:235`; `model_registry.py:333-338` near `:313,385`) → live 021 is `_CUDA_BACKENDS` pollution
- 026 table-driven registry (near `model_registry.py:313`) → live 026 is prompt-staging truncation
- 056 VLM `trust_remote_code=False` (`workers/director.py:193` region) → live 056 is worker-log rotation
- 075 Qwen token/reload contract (`workers/director.py:89,95,463` region) → live 075 is `chmod 777 /opt`
- 019/028/032/038/045/063/064 cites (`video_ltxv.py:120,185,738,808`; `vision/metrics.py:43,60,179`; `audio/acestep.py:41,47,58,64,149`) → each resolves to a different live topic than the comment describes (same stale-number class)

Only the 011-family cites match (lockfile/pins — live 011 is unpinned-deps).

Live live-line spot checks 2026-09-30:

- `Dockerfile:1` = `# Base pinned by digest (issue 012): ...`; `:11` = `# Versions pinned (issue 012): ...`; `:23` = `# tests/ are NOT baked in (issue 054): ...`; `:49` = `# Build-time doctor gate (issue 048): ...`
- `worker/Dockerfile.video:36` = `# Verified-fetch clones (issue 055): ...`; `:75,101,139` same class; `:120` = `# NOTE (issue 055): the pip-git fetch ...`; `:170` = `# Console progress (rich) + launcher TUI (textual) land first (issue 054):`
- `voyage/hashing.py:1` = `"""Shared SHA-256 helpers (issue 021).`
- `voyage/workers/director.py:89-95` = `_require_module` guard pinned by `test_director_request_validation`; `:193` = `trust_remote_code=False` site; `:463` = `except Exception ... chain must survive`
- `voyage/workers/video_ltxv.py:120` = `validate_fps ... (issue 064)`; `:185` = tensor-handoff `(issue 028)`; `:235` = `Delegates to ... (issue 021)`; `:738` = `:mod:\`video_common\` (issue 019)`; `:808` = `One validated struct (issue 045)`
- `voyage/vision/metrics.py:43,60,179` = three `(issue 032)` sites (select filter / frame-total / SegmentZeroAnchor)
- `voyage/audio/acestep.py:41,47,58,64` = four `(issue 063)` validators; `:149` = `(issue 038)` validate-before-import
- `model_registry.py:313` = `#` inside the FILM/Real-ESRGAN block (`:284-330`); `:385` = `return download_model(models_dir, "longlive2-bf16")`

**Live absence note (re-verified 2026-09-30):** `Voyage/worker/Dockerfile.director` does NOT exist — `ls Voyage/worker/` shows only `Dockerfile.video`; `find Voyage -maxdepth 3 -name "Dockerfile*"` returns only `Voyage/Dockerfile` + `Voyage/worker/Dockerfile.video`. The draft's `worker/Dockerfile.director:1,42,51,70` cites (base digest / pip pins / director venv) therefore have no live file at all — doubly stale (wrong number + missing file). The non-findings leg of the same sweep quotes `worker/Dockerfile.director:37` pins, so the file existed at sweep time and was removed/renamed by a concurrent edit before this filing.

## Rationale (non-overlap)

- Issue pointers are the tree's traceability mechanism (§12 fix-on-sight cites `file:line`; Dockerfiles lean on them for bump procedures). A reader following `(issue 055)` to verify the clone discipline lands on an unrelated rotation bug and concludes the control is unfiled — or "fixes" the wrong thing.
- None of 067/068/070/071/072/076/077/078 (live supply-chain issues) names the citation-number drift; they cover pins/hashes/allow-lists, not comment pointers.
- Low severity (docs-only, no runtime effect), but it spans all Dockerfiles plus registry/worker/vision/audio comments, so one batch fix beats per-file drift.

## Live evidence (verified 2026-09-30)

```
$ sed -n '1p;11p;23p;49p' Voyage/Dockerfile
# Base pinned by digest (issue 012): ...
# Versions pinned (issue 012): ...
# tests/ are NOT baked in (issue 054): ...
# Build-time doctor gate (issue 048): ...
$ sed -n '1p;29p;36p;75p;101p;120p;139p;170p;205p;213p' Voyage/worker/Dockerfile.video
# Base pinned by digest (issue 012): ...
# pip pinned exact (issue 012): ...
# Verified-fetch clones (issue 055): ... (×3 + NOTE ×1 + console ×1 ...)
$ ls Voyage/worker/; find Voyage -maxdepth 3 -name "Dockerfile*"
Dockerfile.video   # only file — Dockerfile.director MISSING
Voyage/Dockerfile / Voyage/worker/Dockerfile.video
$ grep -n "^- 055\|^- 012\|^- 054\|^- 048\|^- 053" Voyage/issues/000_INDEX.md
- 055 — inspect metrics rotation-blind (HIGH) / - 012 — start_workers() leak ...
- 054 — SFX two-worker ledger race (MEDIUM) / - 048 — LongLive full-segment ...
- 053 — concat quoting + ffmpeg hygiene (MEDIUM)
$ rg -n "\(issue 0" Voyage/Dockerfile Voyage/worker/Dockerfile.video Voyage/.dockerignore Voyage/.gitignore Voyage/voyage/ | head
... all pairs above mismatch; only (issue 011) resolves correctly
```

## Repro

```bash
rg -n "\(issue 0" Voyage/Dockerfile Voyage/worker/Dockerfile.* Voyage/.dockerignore Voyage/.gitignore Voyage/voyage/ | while read l; do echo "$l"; done
# resolve each number against Voyage/issues/000_INDEX.md — all pairs above mismatch
ls Voyage/worker/Dockerfile.director  # No such file (live 2026-09-30)
```

## Fix candidates

1. Rewrite each cite to the live number where one exists (e.g. digest-pins/verified-fetch/doctor-gate/voyager-user have no live issue — either file them or drop the number and keep prose).
2. Add a gate test asserting every `(issue NNN)` cite in `Dockerfile*`/`*.py` headers resolves to an existing `Voyage/issues/NNN_*.md` whose title keywords overlap the comment topic.
3. Cheapest: strip numbers from comments that describe procedures, keep numbers only where the issue is the design record.

## Refs

- `Voyage/worker/Dockerfile.video:1,29,36,75,101,120,139,170,205,213`; `Voyage/Dockerfile:1,11,23,49`; `Voyage/.dockerignore:1`; `Voyage/.gitignore:7`; `Voyage/voyage/hashing.py:1`; `Voyage/voyage/model_registry.py:313,385`; `Voyage/voyage/workers/director.py:89,95,193,463`; `Voyage/voyage/workers/video_ltxv.py:120,185,235,738,808`; `Voyage/voyage/vision/metrics.py:43,60,179`; `Voyage/voyage/audio/acestep.py:41,47,58,64,149`; `Voyage/issues/000_INDEX.md` (live-number authority)
- Docker "Pin dependencies and verify integrity" (https://www.docker.com/blog/software-supply-chain-security-best-practices/); SLSA provenance/attestation expectations (https://slsa.dev/provenance/v1)

(End of file)

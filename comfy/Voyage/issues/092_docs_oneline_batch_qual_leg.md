# 092 — Docs one-line batch + qual LTXV leg + ltxv-first ordering

- Severity: LOW (docs drift, user-facing)
- Files: `README.md:58,41-46,83-88,113-127`, `docs/INSTALL.md:81-86`, `docs/UPSTREAM_LONG_LIVE_PATCHES.md:§3`, `DESIGN.md:§5.1,§§56-57,§87`, `reports/video-backends.md`, `docs/BACKENDS.md`, `docs/MODELS.md`, `docs/OPERATIONS.md:Finalization`, `docs/TROUBLESHOOTING.md`, `docs/BENCHMARKING.md`
- Area: docs/reports — polish + evidence

## Description

One-line fixes (each single verifiable edit, one polish commit): `UPSTREAM_LONG_LIVE_PATCHES.md §3` table says `local_attn_size` ours = 8, live is 16 (continuity fix §22.5; `reports/video-backends.md` confirms 16/8). README:58 "ltxv renders 768×512 @ 24 fps" — output default is ≥1280×720@≥32fps via floors (2026-09-30); model prereqs + INSTALL download lists miss `film`/`realesrgan-anime`/`sfx-mmaudio` (same 3-stack omission in both files — one fix covers both); `generate` flags miss `--min-fps/--min-resolution/--no-augment/--sfx-caption/--music-caption/--video-caption/--no-download`. DESIGN §5.1 as-built "CausVid has no worker/registry/preset entry" is false (worker landed 2026-09-24, VALID 144f); §§56-step-10/57 normative 768×432/24fps is two generations behind (keep-generation-resolution as-built + floors supersede — needs one rewritten paragraph, not a third as-built); §87 lists 11 docs but not 3 `UPSTREAM_*_NOTES`, `reports/`, or SFX/augment docs. `reports/video-backends.md` LTXV leg is empty "pending Stream A" though ltxv is default since 2026-09-29 (longlive2 leg solid — keep, mark historical after 079). Backend presentation (INSTALL/BACKENDS/MODELS/README order) lists LongLive first; default is ltxv. Unrecorded: CausVid 832×480@16 + fake testsrc lifted ~2.7× px + 2× fps by floors — no cost/quality note; BENCHMARKING e2e numbers predate floors. `ARCHITECTURE.md` layout lists `video.mp4/audio.wav/DONE` but not `audio/sfx/` stems, `sfx.jsonl`, `video_tail.mp4`, augment artifacts, 2-GPU pairing.

## Rationale

Stale geometry/download/flag lists cause real under-provisioning + expectation mismatch (slice-4 lesson: "default shift breaks foreign suites — own the fix"). Empty default-backend qual leg reads as missing evidence.

## Live evidence

- Sweep table 2026-09-30 (ARCHITECTURE partial, INSTALL stale list, MODELS fresh, BACKENDS gap, OPERATIONS partial, PROMPTING/AUDIO stale, TROUBLESHOOTING partial, BENCHMARKING minor gap, PATCHES one cell, LTXV/CausVid fresh/title-drift)
- `grep -n "models download" docs/INSTALL.md README.md` → 6 rows, zero film/realesrgan/sfx; `docs/MODELS.md` has all three

## Repro

```bash
rg -n "768.?512|1280|24 ?fps|32 ?fps|min-fps|min-resolution|no-augment|models download" README.md docs/ DESIGN.md | head -n 40
sed -n '85,117p' reports/video-backends.md
```

## Fix candidates

1. One-line batch commit (patches 8→16, README output/floors/downloads/flags, DESIGN §5.1 + §§56-57 rewrite + §87 coverage, INSTALL +3 stacks, ARCHITECTURE layout + pairing, OPERATIONS floors + captions, TROUBLESHOOTING stand-in entry, BENCHMARKING floors paragraph).
2. Fill or mark-superseded qual LTXV leg (record basis + E2E numbers or pointer to decision entry).
3. Reorder backend presentation ltxv-first (lands with 079 delete).
4. Gate: `rg` verifiers + `gates.sh` green.

## Refs

- Complementary with 091 (091 creates `SFX.md`/`AUGMENT.md`; this issue indexes them + polishes the rest). Benchmark-floors paragraph complements 060/154/163 (harness gaps) — docs here, code there.
- Issues 042 (README geometry), 065 (missing stacks); DESIGN §§5,56,57,87,140; `reports/video-backends.md:124`, `reports/longlive-audit.md:75`

## Progress log

- 2026-09-30: applied the in-scope one-line batch (`Voyage/docs/*,README.md` only; no Python changes):
  - `README.md`: ltxv native→floors note (768x512@24 native, ships ≥1280x720@≥32fps via floors, `--no-augment` escape) + 7 missing `generate` flags (`--min-fps/--min-resolution/--no-augment`, `--music-caption/--video-caption/--sfx-caption`, `--no-download`).
  - `docs/UPSTREAM_LONG_LIVE_PATCHES.md:50`: `local_attn_size` ours 8→16 (live `voyage/config.py:242`; `reports/video-backends.md:41` confirms 16/8).
  - `docs/BACKENDS.md`: ltxv-first video ordering (default first) + SFX rows (with 091).
  - `docs/ARCHITECTURE.md`: layout gains `video_tail.mp4`, `audio/sfx/` stems + `sfx.jsonl`, `final-sfx.mp4`, augment output + 2-GPU pairing line.
  - `docs/OPERATIONS.md`: Finalization gains floors + SFX pins with pointers.
  - `docs/TROUBLESHOOTING.md`: SFX ladder OOM note + augment FILM stand-in section.
  - `docs/BENCHMARKING.md`: floors paragraph (Causvid/fake lift math, compare-at-same-floors, `--no-augment` for native timings).
  - `INSTALL.md` +3 stacks (film/realesrgan/sfx-mmaudio): verified already present (`INSTALL.md:90-92`, `README.md:46-48`, `MODELS.md:45-65/116-131`) — no edit needed.
- Verification: `rg` verifiers non-empty (floors/flags/1280/32fps/downloads); in-container probe confirms `local_attn_size=16`, augment defaults 32/1280/720, `plan_augmentation` lifts; `test_run_sh.py` 9 passed; `bash -n` scripts OK. Pre-existing `ruff check` 11 errors + 1 format file are in other agents' in-flight CLI-split files (untouched, out of scope).

## Resolution

- Delivered (in-scope): the seven file edits above + 091's `SFX.md`/`AUGMENT.md` index.
- No code behavior changes; no `Voyage/issues/*.md` reformatting; no commits.
- Residuals (out of scope — `000_INDEX.md/DESIGN.md/AGENTS.md` + `reports/` untouched by contract):
  - DESIGN §5.1 as-built (Causvid "no worker/registry/preset" — false since 2026-09-24, VALID 144f). Proposed: replace with "CausVid worker (`voyage/workers/video_causvid.py`) registered in `VIDEO_WORKER_MODULES`/`STREAMING_VIDEO_BACKENDS`, preset `causvid` 832x480@16 in `BACKEND_REGISTRY`, VALID 144f 2026-09-24."
  - DESIGN §§56-step-10/57 normative 768x432/24fps (two generations behind). Proposed: one rewritten paragraph — "finalize keeps generation resolution, then `plan_augmentation` lifts to ≥1280x720@≥32fps (`AugmentConfig` defaults; `0` disables; 24fps `PRESENTATION_MIN_FPS` floor remains)."
  - DESIGN §87 coverage list (11 docs, missing 3 `UPSTREAM_*_NOTES` + `reports/` + SFX/AUGMENT). Proposed: append "`docs/SFX.md` (finalize effects), `docs/AUGMENT.md` (presentation floors), `docs/UPSTREAM_*_NOTES.md` (×3), `reports/` (qualification evidence)."
  - `reports/video-backends.md` LTXV leg (empty "pending Stream A" while ltxv is default). Proposed: record basis + E2E numbers or mark superseded with a pointer to the ltxv-default decision entry (needs an idle-GPU `qualify.sh --backend ltxv` run — card held during this pass).

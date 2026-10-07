# 305 — ARCHITECTURE.md commit pipeline + run-dir layout predate all-deferred audio

Severity: MEDIUM-HIGH (pass-2 DESIGN-docs drift sweep).

## Technical description

Commit pipeline still shows per-segment audio (`audio coverage (slow loop…)` →
`validate_audio` + A/V drift ±0.6s → metadata + `sha256.json`); layout lists
`audio-state/metrics.json`, `sha256.json`, `video_tail.mp4`, `final-sfx.mp4`.

## Rationale

Commit is video-only since 2026-10-04; there is no per-segment `audio.wav`, no
commit-time A/V gate, manifest is `manifest.json` (checksums video-only),
`sha256.json` is legacy-fallback only.

## Live evidence

- Docs: `docs/ARCHITECTURE.md:33-37` pipeline; `:64-71` layout (`world/transition/
  prompt-plan/audio-state/metrics.json, sha256.json`, `video_tail.mp4`, `audio/`
  takes+slices, `final-sfx.mp4`).
- Code: `voyage/supervisor.py:2360,2381-2382` ("No `audio.wav` is ever written at commit
  … video-only commit"); `voyage/segment_manifest.py:40-48`
  (`REQUIRED_CHECKSUM_ARTIFACTS = ("video.mp4",)`, audio no longer required);
  `voyage/media_audio.py:452-457` (validate is "video-only");
  `voyage/paths.py:32-33` (`SEGMENT_MANIFEST_FILENAME = "manifest.json"`).
- Command: `grep -n "validate_audio\|_cover_audio" voyage/supervisor.py` → only
  deferred-cover `_cover_audio` + finalize paths;
  `grep -n "REQUIRED_CHECKSUM" voyage/segment_manifest.py`.

Repro: commit a segment, list `segments/NNNNNN/` — no `audio.wav`/`audio-state.json`/
`sha256.json` on fresh runs.

## Source refs

As above.

## Online sources

- None (in-tree all-deferred change is the anchor).

## Fix candidates

- Pipeline → `director decide → video generate_blocks → video-only cover →
  validate_video → manifest.json → DONE`; layout → `manifest.json` + `video.mp4` +
  `recovery.pt` (GPU) + `DONE`; move takes-ledger/SFX wording to finalize-only.

## Evaluation (2026-10-07)

Claim CONFIRMED live. `voyage/supervisor.py:2359-2397` (`_cover_audio`:
"No `audio.wav` is ever written at commit … video-only commit");
`voyage/segment_manifest.py:41-48`
(`REQUIRED_CHECKSUM_ARTIFACTS = ("video.mp4",)`, audio no longer
required); `voyage/media_audio.py:451-461` (`_verify_segment`:
"video-only … no A/V gate"); `voyage/paths.py:32-35` (manifest
filenames — filed refs `:32-33` confirmed, doc's old `:29/:31/:32`
had drifted). The §50 "GPU time-sharing" paragraph (`ARCHITECTURE.md:52-58`)
already stated video-only commit and needed no change.

## Progress log

- 2026-10-07: rewrote pipeline → director decide → video
  `generate_blocks` → video-only cover → `validate_video` →
  `manifest.json` → `DONE` (with live code refs); rewrote layout →
  `manifest.json` + `video.mp4` + `recovery.pt` (GPU) + `DONE`, takes
  ledger/SFX as finalize-only, `sha256.json`/`final-sfx.mp4` retired to
  legacy-fallback/absent; fixed `paths.py` line refs (`:32/:34/:35`).
  No code touched.

## Resolution (2026-10-07)

RESOLVED docs-only. File: `Voyage/docs/ARCHITECTURE.md`.
Verify: `grep -n "audio.wav\|validate_audio\|final-sfx.mp4" Voyage/docs/ARCHITECTURE.md`
→ only the "no `audio.wav` ever written" + legacy-fallback mentions;
no per-segment audio, no commit-time A/V gate remains.

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

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.

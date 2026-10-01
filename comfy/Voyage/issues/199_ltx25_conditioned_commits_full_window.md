# 199 — ltx25 conditioned segments commit 121f, not 96 novel (timeline overlap vs ltx23)

## Technical description
For Mode A (121f window / 25f frozen prefix / 96 novel), `video_ltx23`
commits 121 + 96 + 96 = 313f, but `video_ltx25` commits 121 + 121 + 121 =
363f: conditioned ltx25 blocks return the FULL 121f window including the
25f prefix instead of the 96 novel frames. Finalize concatenates all
committed windows, so the ltx25 timeline (15.03 s) is ~2 s longer than the
same 3-segment plan on ltx23 (12.95 s) and the joints are hard cuts, not
continuations.

## Rationale
Planning (`cli_planning._frames_per_segment = 96`) assumes 96 novel/seg.
The 121f commits break duration math (`--duration 10s` → 15.0 s, not
~13.0 s), and the seam quality fails the native-continuation requirement
(see evidence).

## Live evidence (post-hoc, PIL grayscale 304x176, same protocol both runs)
- ltx25-compare: media frames [121, 121, 121]; seam diffs 46.98 / 46.24 vs
  within-motion baseline 29.21 → joints are ~1.6x the normal frame-to-frame
  motion (visible cuts). Duplicate-prefix scan: best tail match (frames
  93/91) still diffs ~44 → the prefix is re-generated content, not a copy.
- ltx23-compare: media frames [121, 96, 96]; seam diffs 17.74 / 15.43 vs
  baseline 33.70 → joints well UNDER normal motion (smooth, no visible
  seam; best match is the adjacent frame itself).
- Finals: `ltx25-compare/final.mp4` 1280x720@32 480f 15.0 s 35 MB;
  `ltx23-compare/final.mp4` 1280x720@32 414f 12.94 s 11 MB. Both VALID,
  AAC 48 kHz joint audio throughout (ltx25 −13…−16 dB, ltx23 −17…−19 dB).

## Repro
Same A/B pair as 198 (seed 1846428821, boba style). Compare
`segments/000001/manifest.json` frames + `video.mp4` frame counts.

## Fix candidates
- Slice the 25f prefix off conditioned ltx25 commits (return 96 novel +
  tail, mirroring ltx23), or mark prefix frames so the supervisor timeline
  advances by novel count only.
- If the full window is intentional (overlap for blending), finalize must
  dedupe/overlap-blend the joints instead of concatenating.
- Add a commit-time `boundary_diff` metric (tail-vs-head mean diff +
  baseline ratio) so regressions like this trip a gate instead of needing
  ad-hoc PIL forensics.

## Source refs
- `Voyage/voyage/workers/video_ltx25.py` `generate_blocks` return path
  (~844–874) vs `video_ltx23.py` (~868–898).
- `Voyage/voyage/supervisor.py` `_render_video` / `_cover_audio` timeline
  advance (trusts worker `timeline_frames`).
- Analysis driver: `/tmp/opencode/compare_metrics.py` (ephemeral; rerun
  against the two `*-compare` run dirs).

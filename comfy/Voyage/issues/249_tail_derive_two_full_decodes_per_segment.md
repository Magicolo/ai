# 249 — Frame-count full decodes: tail derive per segment + `presented_frames`/`validate` on multi-GB finals

Severity: MEDIUM (track D-05).

## Technical description

`tail_start_frame` → `count_video_frames` (`-count_frames` full decode) then
`derive_tail_from_segment_video` runs ffmpeg trim (second decode).

## Rationale

256 segments → 512 full decodes of 121–257f 1216×704 files just to learn `total - 25`.
Manifest `metrics.frames` already holds the exact count (supervisor truth everywhere
else); the derive path ignores it.

## Live evidence

```
$ grep -n "def count_video_frames\|def tail_start_frame\|def derive_tail_from" voyage/workers/video_common.py
476, 517, 575
```

`count_video_frames` argv uses `-count_frames … nb_read_frames` (decode-bound, not
header).

Repro: instrument/stub `subprocess.run` counting spawns during 10 sequential tail derives
— 20 ffmpeg/ffprobe spawns where 10 trims suffice.

## Source refs

`voyage/workers/video_common.py:476-531` (`count_video_frames`), `575-610`
(`derive_tail_from_segment_video`), `517-531` (`tail_start_frame`).

## Online sources

- ffmpeg `-count_frames` semantics (decode-bound vs `nb_frames` header).
- In-tree `_segment_timeline` precedent (frames from manifest, zero probes).

## Fix candidates

- Pass manifest frames into the derive; or `ffprobe nb_frames` fast path with
  `-count_frames` fallback; or single-pass trim-from-end (`-sseof`) avoiding the count
  entirely.

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Consolidated from 258_frame_count_full_decode_instead_of_header (2026-10-07)

Severity: LOW-MEDIUM (track D-13).

### Technical description

`presented_frames` (`media.py:771-800`) and `count_video_frames`
(`video_common.py:476-514`) use `-count_frames` full decode; `cli_finalize` + generate
freshness gate call it on the final (e.g. 7504f 2432×1408).

### Rationale

Tens of seconds per call on large finals for a number the pipeline already knows
(`(n-1)*m+1` math + drain counts + `output_frames` metric). Fast `nb_frames` header read
suffices with decode fallback; the current order pays the slowest method first, every
time.

### Live evidence

```
$ sed -n '771,800p' voyage/media.py   # -count_frames ... nb_read_frames ... int(stdout)
```

Repro: time `presented_frames` on any 1k+ frame mp4 vs `ffprobe -show_entries
stream=nb_frames` — order-of-magnitude delta.

### Source refs

`voyage/media.py:771-800`; `voyage/workers/video_common.py:476-514`.

### Online sources

- ffmpeg `-count_frames` (decode-bound) vs container `nb_frames` (header).
- In-tree `DURATION_FRAME_ESTIMATE_SLACK_FRAMES` already accepts estimates elsewhere.

### Fix candidates

- Header-first with decode fallback; or trust + verify (ledger counts, spot-check tail);
  cache per (path, mtime, size).

### Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.
- 2026-10-07: consolidated into 249 (same full-decode frame-count family).

## Evaluation (2026-10-07, group N)

Re-verified live against current code before fixing — claim CONFIRMED,
numbers qualified:

- `count_video_frames` (`voyage/workers/video_common.py:476`) still uses
  `-count_frames` full decode; `tail_start_frame` (`:517`) takes no frames
  param; `derive_tail_from_segment_video` (`:575`) costs 2 spawns
  (probe + trim). Only callers are `audio_finalize.py:142` and
  `ensure_conditioning_tail` (`video_common.py:632`) — neither threads
  manifest frames, even though `SegmentSource.total_frames` (manifest
  `metrics.frames`) is already trusted at the pollers.
- `presented_frames` (`voyage/media.py:772`) uses the same `-count_frames`
  shape on finals (`cli_finalize.py:154`, `cli_generate.py:105`);
  `augment_morph._probe_frames` (`voyage/augment_morph.py:200`) same shape
  for trims.
- The "512 decodes" / "tens of seconds per call" figures are
  computed-not-profiled projections (reviewer flag stands): per-segment
  cost is real (2 spawns, decode-bound on 121-257f files) but no wall
  timing was measured; the 30-80 ms/spawn storm math understates
  decode-bound probes. Recorded as estimates, not measurements.
- Fix adopted: optional known-frames params (default None = current
  behavior, out-of-scope callers untouched) + header-first `nb_frames`
  with decode fallback + stat-keyed cache. The `-sseof` single-pass
  alternative was rejected (keeps the trim path byte-identical).

## Progress log (2026-10-07, group N)

- Evaluated live (see Evaluation above); implemented in
  `voyage/workers/video_common.py` (`count_video_frames` gains
  `known_frames`, header-first + fallback + `_FRAME_COUNT_CACHE`;
  `tail_start_frame`/`derive_tail_from_segment_video` gain
  `total_frames`), `voyage/media.py` (`presented_frames` gains
  `known_frames`, header-first + fallback + `_PRESENTED_FRAMES_CACHE`),
  `voyage/augment_morph.py` (`_probe_frames` header-first + fallback +
  `_MORPH_FRAME_COUNT_CACHE`).
- New tests in `Voyage/tests/test_group_n_perf.py` (known-frames
  skip, header-first argv shape, decode fallback, cache hits,
  validation errors).
- Live proof in-container (real ffmpeg, no GPU): 120f testsrc mp4 —
  header path, decode path, and morph probe all agree (120 == 120);
  `tail_start_frame` = 95. Full scoped suite green (see Resolution).

## Resolution (2026-10-07)

- Verdict: RESOLVED. Tail derives cost one trim when the caller knows
  the count (new optional params; existing callers unchanged) and one
  header probe + one trim otherwise — the per-segment full-decode
  probe is gone except for containers that omit `nb_frames` (fallback
  preserved). Finals/trim probes share the same header-first shape
  with per-identity caches.
- Files changed: `voyage/workers/video_common.py`,
  `voyage/media.py`, `voyage/augment_morph.py`,
  `Voyage/tests/test_group_n_perf.py` (new).
- Verification: ruff check + format clean on touched files; mypy
  strict clean on touched modules; scoped pytest 318 passed /
  5 skipped (group-N + upscale/tail/drain/joint/morph/interp/
  finalize-wire/overlap/sfx-finalize neighbors) plus 153 passed
  (finalize/sfx/boundary/mix-cache neighbors); live header==decode
  proof above. No GPU workloads. No commits.
- Left open: threading manifest frames at the two derive call sites
  (`audio_finalize.py:142`, `ensure_conditioning_tail`) — those
  modules are outside this track's scope; the params are in place for
  the owning pass.

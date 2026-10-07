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

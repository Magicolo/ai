# 050 — Finalize double-encodes video + hardcodes `veryfast` with no CRF, no per-stage timing

**Severity:** MEDIUM

**File:line:** `voyage/media.py:1046-1110` (parts + final vf encode), `1062-1063`/`1101-1102` (`-preset veryfast`), `1073-1076` (vf only on final)

**Description:**
Non-native finalize encodes twice: per-segment `libx264/veryfast/-an` parts (1049-1067, no vf — pure transcode to parts), then concat + full `scale/pad/fps/minterpolate` re-encode (1077-1110). The first pass is wasted work — the vf could run per segment once, or the concat could feed a single encode. `preset veryfast` + default CRF (23) is hardcoded for both passes; no CRF knob, no `tune`, no `slow`/`medium` quality ladder, and the 2× upscale + 4× FILM design target (1536×1024 crf 15 in the Comfy `video_export.json` precedent) is nowhere plumbed. No per-stage wall/VRAM timings flow into `bench.py` — `stage_ms` exists only in the longlive worker, not in finalize.

**Rationale:**
Double-encode doubles the longest CPU stage on exactly the long runs that need chunking; silent quality default (CRF 23 veryfast) ships softer finals than the validated recipe the project already uses in Comfy.

**Live evidence (current tree):**
```
media.py:1050-1066: for segment in usable: ffmpeg -i segment/video.mp4 -c:v libx264 -preset veryfast -an part
media.py:1073-1076: vf = f"{lift}scale=...pad=...,setsar=1,fps={out_fps}"  # only on final
media.py:1101-1102: "-preset", "veryfast",  # final encode, no -crf
python3 -c "print(open('voyage/media.py').read().count('crf'))" → 0
```
`rg -n "preset|crf|libx264" voyage/media.py` → presets without CRF; ffmpeg spawns for N segments = N (parts) + blends + 1 (final). `augment.py:183-204` does expose `crf` (default 15) for chunk encodes — finalize does not.

**Repro:** `rg -n "preset|crf|libx264" voyage/media.py` → presets without CRF; count ffmpeg spawns for N segments = N + blends + 1.

**Fix candidates:**
- Single-pass: per-segment vf + concat of finished parts, or concat-demuxer + single vf encode (no intermediate libx264 pass).
- Expose `crf/preset` in `FinalizeOptions` (defaults matching the validated `video_export` recipe), log effective values in the finalize metric.
- Emit `stage_ms`-style timings (`parts_encode_ms`, `audio_blend_ms`, `final_encode_ms`) into `metrics.jsonl` so soak can trend them.

**Refs:** ffmpeg concat guidance (use concat demuxer when possible, avoid concat filter for large numbers); Comfy `video_export.json` §6.8 precedent (crf 15, RealESRGAN 2×, 4× FILM).

**Overlaps with:** 043 (finalize publish RAM — same finalize path, complementary: this file is encode-count/quality, 043 is publish-RAM; not duplicates).

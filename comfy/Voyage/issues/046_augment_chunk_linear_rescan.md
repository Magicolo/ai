# 046 — `ffmpeg_decode_chunk` re-decodes from file start per chunk (quadratic augment cost)

**Severity:** MEDIUM

**File:line:** `voyage/augment.py:136-175`, esp. `146-147` (docstring admission), `153-165` (argv), `169` (glob)

**Description:**
Chunk decode uses `-i source -vf select='between(n,start,end)'` with no input `-ss` fast-seek, so chunk K decodes (and discards) all K×32 preceding frames. A 1000-frame sequence at `chunk=32` (≈32 chunks) decodes ~16k frames to keep 1k — linear-from-start per chunk, quadratic total. The docstring flags a "`-ss` fast-seek follow-up" but ships the slow path. `dest_dir.glob("frame_*.png")` (line 169) also picks up stale files unless the caller guarantees a fresh dir (contract in docstring only, not enforced — no unlink or tempdir enforcement).

**Rationale:**
Augment is the long-sequence path (the VHS_BatchManager-32 precedent); quadratic decode dominates wall time on exactly the runs the chunking was built for. ffmpeg best practice is input-seek (`-ss` before `-i`) + frame-accurate `select` after, or segment-demuxed chunk inputs.

**Live evidence (current tree):**
```
augment.py:153-165: argv = ["ffmpeg", ..., "-i", str(source_video), "-vf",
    f"select='between(n\\,{start}\\,{end})',setpts=...", "-vsync 0", ...]
augment.py:146-147: "Seeking is exact but linear from the file start — a `-ss`
    fast-seek follow-up can skip the already-decoded prefix..."
augment.py:169: frames = sorted(dest_dir.glob("frame_*.png"))
```
`rg -ss voyage/augment.py` → zero hits outside docstring. `-nostdin` present (line 156), so only seek + freshness are missing.

**Repro:**
```bash
ffmpeg -hide_banner -nostdin -y -f lavfi -i testsrc=size=768x512:rate=32:duration=10 \
  -c:v libx264 -pix_fmt yuv420p /tmp/a.mp4
time ffmpeg -hide_banner -nostdin -y -i /tmp/a.mp4 \
  -vf "select='between(n\,288\,319)',setpts=N/FRAME_RATE/TB" -vsync 0 /tmp/c_%06d.png
# vs: time ffmpeg -hide_banner -nostdin -y -ss 9 -i /tmp/a.mp4 -vf "select=..." ...
```

**Fix candidates:**
- Input seek: `["-ss", str(start/fps), "-i", src, "-vf", f"select='between(n\\,0\\,{count-1})'..."]` with fallback to exact mode when keyframe granularity matters.
- Or split the source once with the segment demuxer and decode chunk files directly.
- Enforce fresh `dest_dir` (fail if `frame_*.png` present, or use `TemporaryDirectory` internally).

**Refs:** ffmpeg seek docs (`-ss` as input vs output option); `augment.py:34-35` chunk-size provenance comment.

**Overlaps with:** 047 (augment-worker reload/stack — same augment path, complementary halves; not a duplicate).

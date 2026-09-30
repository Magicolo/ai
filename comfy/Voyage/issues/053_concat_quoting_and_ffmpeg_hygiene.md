# 053 — `concat.txt` shell-unsafe quoting + inconsistent `-nostdin`/`-hide_banner` across ffmpeg call sites

**Severity:** MEDIUM

**File:line:** `voyage/media.py:1007-1010`, `1071-1072` (concat lists); `voyage/workers/audio_acestep.py:114-131` (`_convert`); contrast `voyage/media.py:76+`, `voyage/fake_backends.py:64+`

**Description:**
Concat demuxer lists are built as `f"file '{path}'\n"` with `-safe 0`. Any segment path containing a single quote (legal on POSIX; run-ids/segments derive from user `--run-id`/style strings upstream) breaks the list or escapes it — ffmpeg concat demuxer requires `'\''` escaping, which is never applied. Separately, `audio_acestep._convert` (the one ffmpeg call on the music-take hot path) omits `-nostdin` and `-hide_banner`/`-v error` consistency while every sibling uses them; an ffmpeg reading stdin in a pipelined supervisor can steal RPC bytes or block, and full-banner stderr pollutes worker logs. Arg-lists are used everywhere (good — no shell injection), so this is strictly the quoting + stdin-consistency gap.

**Rationale:**
Concat-demuxer quoting is a classic silent-corruption vector (one bad run-id breaks all later finalizes); `-nostdin` is the ffmpeg-daemon hygiene flag the rest of the tree already standardized on.

**Live evidence (current tree):**
```
media.py:1008-1010: "".join(f"file '{segment / 'video.mp4'}'\n" for segment in usable)
media.py:1072: concat_list.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
audio_acestep.py:114-129: ["ffmpeg","-y","-v","error",...]  # no -nostdin/-hide_banner
fake_backends.py:68: ["ffmpeg","-hide_banner","-nostdin","-y",...]  # the standard
media.py:1015-1016,1054,1081: ["ffmpeg","-hide_banner","-nostdin","-y",...]  # standard elsewhere
sfx_mmaudio.py:131,280: "-nostdin" present
```
Probe: `rg -c nostdin` → `media.py` 12, `sfx_mmaudio.py` 2, `fake_backends.py` 3, `audio_acestep.py` 0. `sfx_mmaudio._convert` (line 172-187) also omits `-nostdin`/`-hide_banner` (has `-y -v error` only) — same gap, second site.

**Repro:**
```bash
mkdir -p "/tmp/voyage o'brien/segments/000000"
# finalize concat_list line becomes: file '/tmp/voyage o'brien/.../video.mp4' → ffmpeg parse error
python3 -c "print(\"file '/tmp/a'b/c.mp4'\")"
```

**Fix candidates:**
- Escape `'` as `'\''` when writing concat lists, or use `-safe 0` + `file <escaped>` via a shared ffmpeg-escaping helper with `augment.py`.
- Add `-hide_banner -nostdin` to `audio_acestep._convert` (+ `sfx_mmaudio._convert`; standardize all three).
- Test with adversarial `run_id` (`o'brien`, spaces, `$()`) through `finalize_run` fast path.

**Refs:** ffmpeg concat-demuxer docs (`file` quoting + `-safe`); in-tree standard (`media.run_capture` arg-list discipline).

**Overlaps with:** 102 (finalize concat escape/RAM/tmp — the quoting half is the same defect; recommend merging the quoting half, keeping the RAM/tmp half in 102).

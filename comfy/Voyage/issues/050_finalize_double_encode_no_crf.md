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

## Progress log (2026-09-30, Rank 2 track)

- Re-read full issue + `voyage/media.py` as-read 2026-09-30 (tree drifts — cite as-read).
- VERDICT: still-relevant. Live verification in-container (`voyage:latest`, CPU-only):
  - `python3 -c "print(open('voyage/media.py').read().count('crf'))"` → `0` (no CRF knob).
  - Non-native path was per-segment `libx264/veryfast/-an` parts (N spawns) + concat + full vf re-encode with `-preset veryfast`, no `-crf` (ffmpeg default 23).
  - `augment.py:ffmpeg_encode_chunk` already defaults `crf=15` — finalize did not.
  - No `stage_ms`-style finalize timings in `logs/metrics.jsonl`.
- TDD: wrote `tests/test_finalize_encode_rank2.py` (8 tests) — all 8 failed pre-fix via `./scripts/test.sh -q tests/test_finalize_encode_rank2.py` (missing `crf`/`preset` attrs, missing `escape_concat_path`/`write_concat_list`, hygiene flags absent, adversarial `o'brien` concat failed with `Impossible to open '.../obrien'` truncation).
- Fix (single-pass option, second candidate): concat-demuxer + single vf encode over the originals — deleted the per-segment parts loop entirely (N+1 encodes → 1). `FinalizeOptions` gains `crf=15` / `preset="veryfast"` (validated recipe) with `validate_crf` (int, 0..51, bool-rejected → `TypeError` for type, `ValueError` for range) + `validate_preset` (full x264 vocabulary). `finalize_run` gains `crf`/`preset` scalar overrides (`None` = use `options`, mirroring `min_*`) and emits `finalize_completed` (`segments`, `out_w/h/fps`, `crf`, `preset`, `parts_encode_ms=0.0`, `audio_blend_ms`, `final_encode_ms`, `fast_path`) via `logrotate.append_line` for soak trending.
- Gates: `ruff check` + `ruff format --check` + `mypy` clean on `voyage/media.py`, `voyage/workers/audio_acestep.py`, `tests/test_finalize_encode_rank2.py` (one TRY004 fix: `TypeError` for non-int CRF; one format reflow). Never ran `ruff format` on `issues/*.md` (pyproject `exclude=["issues/"]`).
- Related suites in-container: `test_finalize_encode_rank2 + test_finalize_fastpath + test_state_integrity + test_media_memory + test_integration + test_augment_plan` → `67 passed, 1 skipped`; audio leg `test_audio_planner + test_audio_accounting + test_acestep_contract + test_sfx_finalize + test_sfx_contract + test_final_blend_scale + test_media_robustness_rank2` → `75 passed, 1 skipped`.

## Resolution (2026-09-30)

RESOLVED. Non-native finalize is now a single libx264 pass (`-preset <effective> -crf <effective>`, defaults 15/veryfast) over a quoting-safe concat list; `finalize_completed` carries the three stage timings + effective knobs. Post-fix probe: `crf` count in `media.py` = 25, `FinalizeOptions().crf/preset` = 15/veryfast, `libx264` appears once (final encode) + the stream-copy path, `segment mux failed` gone. New tests: `test_reencode_is_single_pass_with_crf` (exactly 1 libx264 argv with `-crf 18 -preset fast -vf`), `test_finalize_emits_stage_timings` (`finalize_completed` with non-negative timings + `crf 15/veryfast`).

Files changed: `Voyage/voyage/media.py`, `Voyage/tests/test_finalize_encode_rank2.py` (new).

Residuals: CLI has no `--crf/--preset` flags yet (owned scope was `media.py` only — `cli.py` untouched; a later slice can thread `FinalizeOptions.crf/preset` through `cmd_finalize`/TOML). `parts_encode_ms` is schema-stable `0.0` (no parts stage remains); soak should trend `final_encode_ms` + `audio_blend_ms`.

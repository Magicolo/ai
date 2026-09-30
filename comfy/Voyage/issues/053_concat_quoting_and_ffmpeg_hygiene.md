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

## Progress log (2026-09-30, Rank 2 track)

- Re-read full issue + `voyage/media.py` + `voyage/workers/audio_acestep.py` as-read 2026-09-30 (tree drifts — cite as-read).
- VERDICT: still-relevant. Live verification in-container (`voyage:latest`, CPU-only):
  - `media.py` both concat sites used raw `f"file '{path}'"` with `-safe 0`, no `'\''` escaping.
  - `audio_acestep._convert` argv was `["ffmpeg","-y","-v","error",...]` — `nostdin` count 0, `hide_banner` count 0 vs standard `["ffmpeg","-hide_banner","-nostdin","-y",...]` in `media.py` (11/12), `fake_backends.py` (3/3), `sfx_mmaudio` single-pass/benchmark (2/2).
  - `sfx_mmaudio._convert` has the same gap (has `-y -v error` only) — noted but OUT OF OWNED SCOPE (owned files: `media.py`, `audio_acestep.py` only), left as residual.
  - 102's quoting legs were explicitly deferred to 053 — 053 owns the quoting fix (no 102 edit made).
- TDD: same `tests/test_finalize_encode_rank2.py` (8 tests) — pre-fix the adversarial `o'brien run` finalize failed with `Impossible to open '.../obrien'` (quote truncation) and `_convert` hygiene assertions failed.
- Fix: shared `escape_concat_path` (`'` → `'\''`) + `write_concat_list` in `media.py`, used at both concat sites (native copy + single-pass encode, so both paths are safe); `audio_acestep._convert` now `["ffmpeg","-hide_banner","-nostdin","-y","-v","error",...]` with docstring noting the daemon hygiene. Adversarial test uses run dir `tmp_path / "o'brien run" / "run"` (quote + space in the concat-listed path) through the real fastpath finalize.
- Gates: same as 050 (shared change) — `ruff check` + `format --check` + `mypy` clean on touched files; never formatted `issues/*.md`.
- Related suites: same runs as 050 (fastpath/integrity/media-memory/audio legs all green — see 050 log).

## Resolution (2026-09-30)

RESOLVED (owned scope). Concat lists are quoting-safe via the shared helper (`escape_concat_path("/tmp/voyage o'brien/a.mp4")` → `"/tmp/voyage o'\\''brien/a.mp4"` verified live); `audio_acestep._convert` carries `-hide_banner -nostdin`. New tests: `test_escape_concat_path_handles_single_quote`, `test_write_concat_list_escapes_quote`, `test_audio_acestep_convert_uses_hygiene_flags` (mocked `subprocess.run` asserts both flags), `test_finalize_adversarial_path_with_quote` (real finalize under `o'brien run` succeeds, non-empty MP4).

Files changed: `Voyage/voyage/media.py` (`escape_concat_path` + `write_concat_list` + both call sites), `Voyage/voyage/workers/audio_acestep.py` (`_convert` flags), `Voyage/tests/test_finalize_encode_rank2.py` (new).

Residuals: `sfx_mmaudio._convert` (line ~312-331) still omits `-hide_banner`/`-nostdin` — same one-line fix, left for the owning track since `voyage/workers/sfx_mmaudio.py` was outside this change's owned files. Recommend the same `["ffmpeg","-hide_banner","-nostdin","-y","-v","error",...]` order + a hygiene test mirroring `test_audio_acestep_convert_uses_hygiene_flags`.

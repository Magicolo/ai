# 138 — `finalize --skip-bad` covers only one of three failure legs

- Severity: MEDIUM
- Area: media finalizer — error-policy inconsistency
- Overlaps with 188 (numbering-gap half — 188 owns the silent-disable; this file owns legs 1/3)
- Files (as-read 2026-09-30):
  - `voyage/media.py:928-933` (leg 1 — `video.mp4` / `audio.wav` existence, unconditional raise)
  - `voyage/media.py:324-360` (`_verify_segment` — checksums / frame ranges / A/V alignment)
  - `voyage/media.py:347-352` (metrics.json parse inside `_verify_segment`)
  - `voyage/media.py:935-949` (`skip_bad` gate — only wraps the `_verify_segment` loop)
  - `voyage/media.py:977` (`TemporaryDirectory` staging for the final mix/encode)
  - `voyage/media.py:1043-1044` (fast-path `validate_video` after stream-copy concat)
  - `voyage/media.py:1113-1115` (re-encode-path `validate_video` + atomic publish)
  - `voyage/media.py:802` (`FinalizeOptions.skip_bad: bool = False`)

## Technical description

`finalize_run` fails a run in three distinct places ("three legs"), but `skip_bad` only opens
the middle one:

- Leg 1 — existence probe (`voyage/media.py:928-933`, as-read):
  ```python
  for segment in committed:
      if not (segment / "video.mp4").exists():
          raise MediaError(f"segment {segment.name} missing video.mp4")
      if not (segment / "audio.wav").exists():
          raise MediaError(f"segment {segment.name} missing audio.wav")
  ```
  Unconditional — a segment with `DONE` but a missing artifact aborts even with `skip_bad=True`.
- Leg 2 — `_verify_segment` (`:324-360`: `sha256.json` checksums, `metrics.json` frame count
  (`:347-352`), probed durations, `check_av_alignment`). This is the *only* leg `skip_bad`
  covers (`:942-949`: `try: _verify_segment(segment) except MediaError: if not skip_bad: raise;
  print(skip); continue`).
- Leg 3 — post-assembly `validate_video` (`:1042-1044` fast path, `:1111-1115` re-encode path).
  Unconditional `MediaError` on geometry/fps mismatch — a corrupt input that passes leg 2 but
  fails the final presentation check still aborts a `--skip-bad` finalize.

So `--skip-bad` promises "corrupt segments are skipped with a warning instead of aborting the
whole finalize" (`voyage/media.py:890-891`) but delivers it only for checksum/metrics/alignment
failures. A run with one `DONE`-but-empty segment (leg 1) or one segment that breaks the final
`validate_video` (leg 3) still fails end-to-end.

## Why this is an issue

- Operator-facing flag with a narrower contract than documented; recovery runs (`--skip-bad` is
  the documented recovery path) still abort on the most common partial-failure shapes.
- Leg 1 and leg 2 overlap (both check artifact presence, differently): leg 1 raises before the
  numbering-gap check (`:935-940`) and before any skip logic. Note (per 188): the numbering-gap
  check itself is wrapped in `if not settings.skip_bad`, so under `skip_bad=True` numbering gaps
  are silently skipped too — triage order is existence (strict) → numbering (skipped) →
  verification (lenient).
- Not a duplicate of 109 (`stop --finalize` missing `--skip-bad` wiring): 109 is CLI plumbing;
  this is the finalizer's semantic gap once the flag arrives.

## Live evidence

Live re-verification 2026-09-30 (read-only; probes per task brief ran in `voyage:latest` CPU-only).
Track A draft command+output bundle (`ses_f0fbea412ffeh3V7K1SlQpXjsv`) was not recoverable from
this writer's context, so evidence below is the as-read code, not invented command output:

```
$ sed -n '928,949p' voyage/media.py
  928:    for segment in committed:
  929:        if not (segment / "video.mp4").exists():
  930:            raise MediaError(f"segment {segment.name} missing video.mp4")
  931:        if not (segment / "audio.wav").exists():
  932:            raise MediaError(f"segment {segment.name} missing audio.wav")
   935:    if not settings.skip_bad:          # numbering gap check skipped under skip_bad (see 188)
  942:    for segment in committed:
  943:        try:
  944:            _verify_segment(segment)   # 324-360 — the ONLY skip-gated leg
  945:        except MediaError as exc:
  946:            if not settings.skip_bad:
  947:                raise

$ sed -n '324,360p;1042,1044p;1113,1115p' voyage/media.py
  324: def _verify_segment(...)  # checksums + metrics.json frames + probe + check_av_alignment
  1042:            validate_video(staged, out_w, out_h, out_fps)   # fast path — unconditional
  1113:        validate_video(staged, out_w, out_h, out_fps)       # re-encode path — unconditional
```

## Minimal repro

1. Build a run with two committed segments; delete `segments/000001/audio.wav` (keep `DONE`).
2. `finalize_run(run_dir, out, skip_bad=True)` → `MediaError: segment 000001 missing audio.wav`
   from leg 1 (no `finalize: skipping …` line; abort, no output).
3. Contrast: corrupt `segments/000001/sha256.json` instead → `finalize: skipping 000001 (…)` and
   the finalize succeeds on the remaining segment.
4. Leg-3 variant: craft segments that pass `_verify_segment` but violate the presentation box so
   the staged `validate_video` raises — same abort under `skip_bad=True`.

## Fix candidates

1. (Preferred) Fold leg 1 into the skippable triage: move the existence probe inside the
   per-segment `try` (or route it through `_verify_segment`) so `skip_bad` skips missing-artifact
   segments the same way it skips checksum failures; keep the strict raise when `skip_bad=False`.
2. Decide the leg-3 contract explicitly: either (a) keep final `validate_video` strict (document
   that `skip_bad` is input-triage only, not output-validation), or (b) on `validate_video`
   failure under `skip_bad`, drop the offending segment(s) and re-render once. Option (a) + docs
   is the minimal honest fix.
3. Log skipped segments as structured metric events (not just `print`), so `status`/`scoreboard`
   can show which segments shipped.
4. Regression tests: DONE-with-missing-audio + `skip_bad=True` → finalizes rest; same with
   `skip_bad=False` → raises; staged `validate_video` failure under both settings pins the chosen
   leg-3 contract.

## References

- In-tree: `voyage/media.py:315-360,802,860-949,977-1045,1071-1116`; `voyage/cli.py:962-963,1383,1397`
  (`skip_bad` plumbing); `DESIGN §56` (finalize steps 4-6).
- Neighbor issues: 003 (A/V alignment), 095/096 (metrics/checksum gaps), 098 (orphan scan),
  102 (concat RAM/tmp), 109 (stop path missing `--skip-bad`), 110 (generate validates after init).
- External:
  - https://ffmpeg.org/ffmpeg.html (concat demuxer + `-shortest` semantics used by both paths)

## Investigation log

- 2026-09-30: filed by Track A sweep; live re-verified via Read (concurrent uncommitted edits
  noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py` — citations are
  as-read values above).

## Progress log

- 2026-09-30 (Group E1, joint reader of 138 + resolver of 188): live
  re-verified all three legs — leg 1 existence probe unconditional
  (`voyage/media.py:1207-1211` as-read), leg 2 `_verify_segment` the only
  skip-gated leg, leg 3 post-assembly `validate_video` now a single call
  (`:1380`, fast/re-encode paths unified since the issue was filed).
  Correction to this file's §"Why": the numbering-gap check is NOT strict
  under skip_bad — 188 is right and the "stays strict" line here is
  inverted (the guard is `if not settings.skip_bad`). Verdict: CONFIRMED;
  both regions ownable in `voyage/media.py`, landed once (cross-ref 188).
- TDD: `tests/test_e1_media_augment.py` 138/188 sections written first —
  lenient leg-1 (missing audio/video) + lenient gap tests failed pre-fix
  (aborted instead of skipping), strict tests passed pre-fix
  (characterization pins); all 5 green post-fix.

## Resolution

- Single chosen contract (resolving 188's strict-vs-lenient question):
  **lenient-with-record**: under `skip_bad=True`, missing artifacts fold
  into the skippable triage (`finalize: skipping <seg> (...)`, same line
  shape as leg 2) and numbering gaps print
  `finalize: skipping segment numbering gap: expected X, found Y` and
  continue — never silent. Under `skip_bad=False`, every leg raises as
  before. Leg 3 (post-assembly `validate_video`) stays STRICT under both
  settings — `skip_bad` is input triage, not output validation — now stated
  in the `finalize_run` docstring. Leg-1 probe extracted to
  `_check_segment_committed` (`voyage/media.py:478-488`) so the triage loop
  never raises inside its own `try` (TRY301). Strict-mode order note: the
  numbering check now precedes per-segment verification (both raise
  `MediaError` either way; no test pinned the old order).
- Files changed: `voyage/media.py` (`:478-488` helper, `:1313-1342`
  triage, docstring contract) + new tests in
  `tests/test_e1_media_augment.py`. Per-file gates green in-container
  (`voyage:latest`, CPU-only): ruff check + format-check + PLR2004 +
  mypy strict; 20/20 E1 + neighbors green (see 096 log for the full
  neighbor list; pre-existing `test_finalize_skip_bad_finalizes_rest`
  still passes unchanged).
- DESIGN proposal (quoted text only, not applied — DESIGN.md untouched):
  "> `finalize --skip-bad` is input triage with a record: missing
  > artifacts, checksum/metrics/alignment failures, and numbering gaps are
  > each skipped with a `finalize: skipping ...` warning naming the
  > segment; the post-assembly presentation check stays strict, so a
  > corrupt stage still aborts the finalize."
- Residuals: none in this file's scope. 188's file carries the same
  contract by cross-reference; 109 (CLI plumbing) stays with its owner.

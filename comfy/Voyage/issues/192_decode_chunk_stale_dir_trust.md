# 192 — `ffmpeg_decode_chunk` trusts a caller-fresh `dest_dir` on a docstring sentence alone

- Severity: LOW
- Area: augment chunk runner — stale-input hygiene
- Files (as-read 2026-09-30):
  - `voyage/augment.py:136-175` (`ffmpeg_decode_chunk`, esp. `:144-145` docstring, `:151` mkdir, `:169` glob)
  - `voyage/augment.py:34-50` (chunk-size/device constants — no staging-dir contract)
  - `tests/test_augment_runner.py:217-263` (decode tests use fresh `tmp_path` subdirs)

## Description

`ffmpeg_decode_chunk` decodes a frame window into `dest_dir` and collects its
output with `sorted(dest_dir.glob("frame_*.png"))` (`:169`). The only protection
against stale frames is a docstring sentence (`:144-145`: "`dest_dir` must be a
fresh per-chunk directory (stale `frame_*.png` files would be picked up by the
glob below)"). The function itself does `mkdir(parents=True, exist_ok=True)`
(`:151`) — which *succeeds* on a reused dir — then:

- surplus stale frames are silently included (a rerun with a shorter window, or
  two chunks sharing a dir, yields more PNGs than `frame_count` with no check);
- the empty-output check (`:170-171`) cannot catch surplus, only absence;
- per-frame `st_size == 0` checks (`:172-174`) validate presence, not provenance.

No production caller exists yet (Track D spike — only tests call it, always with
fresh `tmp_path` subdirs), so nothing is corrupt today; but the first wired
caller (finalize chunk loop, retry-after-partial-decode) inherits a silent-mix
trap where the failure (wrong frame count downstream, off-by-`k` interpolation
windows) surfaces far from the reused dir. The module already validates
everything else at the boundary (`_require_count` on start/count/fps/crf) —
directory freshness is the one precondition left to prose.

## Rationale

Boundary validation is the file's own idiom (`_require_count` × every numeric);
a directory precondition that controls output correctness belongs in code, not in
a docstring the future caller must remember. Retry paths reuse dirs by nature —
the exact case where stale frames appear is the case the docstring forbids.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '136,175p' voyage/augment.py` — `:151` `exist_ok=True` + `:169` bare
  glob; no pre-clean, no pre-existing-file check, no post-count assertion
  (`len(frames) == count` is never compared — over-delivery is accepted).
- `rg -n "augment_plan\(|ffmpeg_decode_chunk\(" voyage/ --glob '!tests'` →
  zero production callers: the trap is latent, filed now so the wiring change
  cannot inherit it silently.

## Repro

Static (deterministic, no ffmpeg needed for the logic): pre-place
`dest/frame_000001.png` (stale), call `ffmpeg_decode_chunk(src, dest, 0, 1)` →
on success the glob returns 2 PNGs (stale + fresh); no error, and the caller
upscales/interpolates a 2-frame chunk it planned as 1 frame.

## Fix candidates

1. (Preferred) Assert output count: after the glob, require
   `len(frames) == frame_count` (or at minimum reject a non-empty dir on entry
   — fail loud before decoding, so retries must pass a fresh dir or an explicit
   `allow_reuse` flag).
2. Alternatively scope the glob to the decoded range (`frame_<start>..frame_<end>`
   names) so strays outside the window are ignored rather than mixed in.
3. Test: pre-seeded dir → `MediaError`; post-decode count mismatch → `MediaError`.

## Refs

- In-tree: `voyage/augment.py:136-175,266-281` (`run_augment_chunks` — the future
  caller); `tests/test_augment_runner.py:217-263`.
- Not-a-duplicate: 046 is the linear-rescan *cost* (seek shape, same function,
  different facet — cost vs correctness); 047 is worker-side reload/stack;
  157 is preset/fanout/batch-peak. None names the stale-dir glob trust.

# 018 — `finalize` loads the whole film into RAM; artifacts triple-hashed; orphan scan walks the tree

- Status: open
- Severity: major (I/O + RAM wall growing with run length)
- Area: performance/durability — `media.py`, `supervisor.py`, `cli.py`
- Rank rationale: finalize OOM risk on long voyages + every segment's bytes hashed
  3× across its lifetime for no reason.

## Technical description

```python
# voyage/media.py:86-88 — unknown frame count bypasses the check entirely
frames = int(video.get("nb_frames", 0) or 0)
if frames < min_frames and frames != 0: raise MediaError(...)

# voyage/media.py:440-444 — unconditional, even with skip_bad=True
for segment in committed:
    if not (segment / "video.mp4").exists(): raise MediaError(...)
    if not (segment / "audio.wav").exists(): raise MediaError(...)

# voyage/media.py:549 — whole final in memory (multi-GB voyages)
atomic_write_bytes(output_path, staged.read_bytes())
```

(a) `staged.read_bytes()` materializes the entire final MP4 (GBs) plus a second
copy inside `atomic_write_bytes` — OOM/swap on the finalize host.
(b) `--skip-bad` skips checksum/drift failures but not missing artifacts — one
deleted file aborts an otherwise salvageable finalize; numbering-gap check is
skipped under `skip_bad`, so non-`^\d{6}$` dirs with DONE can enter `usable` out
of order (concat order = lexicographic, not timeline).
(c) `nb_frames` is container-provided and often `0`/absent (needs `-count_frames`);
the `!= 0` carve-out means "unknown" always passes `min_frames`.
(d) Every segment hashed 3×: commit (`supervisor.py:1215-1218`) → validate
(`cli.py:500-523`) → finalize-verify (`media.py:220-261`); `validate_run` also
`rglob`s the whole tree for `*.partial` — O(total files) per invocation, no
mtime/size short-circuit. (Three duplicate chunked-hash helpers — see 021.)

## Why this is an issue

Materializing the entire final MP4 in RAM twice over turns the system's core
deliverable — the finished film, gigabytes on a long voyage — into an OOM risk
on the finalize host. Hashing every segment's bytes three times across commit,
validate, and finalize with no size-plus-mtime short-circuit, plus an O(total
files) partial-file scan per validate invocation, makes export cost grow with
run length just when runs are longest. The `--skip-bad` path additionally
cannot skip missing artifacts while relaxing numbering checks, so salvageable
runs abort and unorderable ones can concat. The finalize step of every long
voyage pays in RAM pressure, I/O, and operator rescue time.

## Evidence

Source quotes above. Re-verified 2026-09-25:

```
$ rg -n "read_bytes|nb_frames|missing video.mp4|missing audio.wav" voyage/media.py
86:    frames = int(video.get("nb_frames", 0) or 0)
442:            raise MediaError(f"segment {segment.name} missing video.mp4")
444:            raise MediaError(f"segment {segment.name} missing audio.wav")
549:        atomic_write_bytes(output_path, staged.read_bytes())
```

Whole-final-in-RAM copy (549), `nb_frames==0` carve-out (86-87), and
unconditional missing-artifact abort (442-444) all still present.

## Reproduction

Finalize a long run while watching RSS (spike ≈ 2× final size); run
`validate_run` on a 500-segment run and time the `rglob` + rehash.

## Source references

- `voyage/media.py:86-88,211-217,220-261,406-549`; `voyage/supervisor.py:77-83,
  1215-1218`; `voyage/cli.py:500-523,555-604`.

## Resolution candidates

1. Replace `read_bytes` with streaming atomic copy (`shutil.copyfile` + fsync, or
   chunked write).
2. Store `(sha256, size, mtime_ns)` at commit; validate/finalize skip rehash when
   size+mtime match; replace `rglob("*.partial")` with a commit-time marker +
   `validate --quick`.
3. Make missing-artifact per-segment skippable under `skip_bad` (collect + report
   like `_verify_segment`); keep strict numbering even under `skip_bad`.
4. Derive frame truth from `metrics.frames` + duration cross-check
   (`|frames/fps - duration|`) instead of trusting `nb_frames==0` as pass.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness + perf sweeps (convergent).
- Open: implement streaming copy + hash-short-circuit + tests.
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  with live `media.py` output; refs verified current.

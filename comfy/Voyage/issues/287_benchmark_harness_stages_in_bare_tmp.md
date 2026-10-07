# 287 — `run_benchmark_harness` stages full-segment renders in bare `/tmp` — the boba quota fix doesn't cover benchmarks

Severity: MEDIUM (pass-2 worker-tails sweep).

## Technical description

`voyage/workers/video_common.py::run_benchmark_harness` creates its staging dir as
`tempfile.TemporaryDirectory(prefix=...)` with no `dir=`, i.e. host `/tmp` (31 G tmpfs
with usrquota — the boba incident). All four video workers route benchmarks through it.
Production session scratch was carefully moved under `run/tmp/` via
`session_scratch_parent`, but the benchmark path was left behind.

## Rationale

A benchmark probe renders a full fresh window (ltx25: 257 frames of PNGs + mp4 mux) into
`/tmp`; concurrent or repeated benches accumulate exactly the class of pressure that
killed the ltx25 verify run (`Errno 122 Disk quota exceeded`). Audio/SFX benches already
do it right (`paths.staging_parent(_scratch_dir)` in `audio_acestep.py:300,352` and
`sfx_mmaudio.py:391,436`).

## Live evidence

```
$ sed -n '661,686p' voyage/workers/video_common.py
def run_benchmark_harness(...):
    ...
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as tmp:
$ grep -n "run_benchmark_harness" voyage/workers/*.py
video_causvid.py:1126 / video_ltx23.py:1132 / video_ltx25.py:1224 / video_ltxv.py:1031
$ grep -n "session_scratch_parent\|mkdtemp" voyage/workers/video_ltxv.py voyage/workers/video_causvid.py
(no output — neither threads scratch into the harness)
```

Repro: any `benchmark` op on any video worker with `/tmp` under pressure; or code-read:
no caller passes a directory, the harness signature has no directory parameter.

## Source refs

`voyage/workers/video_common.py:661-686`; callers listed above.

## Online sources

- None (in-tree boba `/tmp` usrquota incident, AGENTS §11 2026-10-02, is the anchor).

## Fix candidates

- Add optional `dir=`/`scratch_parent` parameter to `run_benchmark_harness`, threaded
  from each worker's `_INIT_PARAMS["scratch_dir"]` (ltx25/23 have it; ltxv/causvid would
  need to record it first).

## Log

- 2026-10-07: filed from read-only pass-2 worker-tails sweep; no code touched.

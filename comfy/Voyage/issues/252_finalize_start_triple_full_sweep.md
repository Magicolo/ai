# 252 — Finalize-start does 3–4 full sweeps (hash + PIL-open + rglob-stat every file)

Severity: MEDIUM (track D-08).

## Technical description

`run_durable_model_pass` → `sweep_chunk_mp4s` (all ledgers + PNG completeness) →
`prune_orphan_plan_dirs` (re-derives live dirs + `sha256_file` every joint video +
`_newest_modification_time` rglob-stats every file) → `_poll_to_completion` per-pass
scans → `build_parallel_tasks` PIL-opens every chunk PNG header.

## Rationale

Each sweep is O(segments × chunks × files). `prune_orphan_plan_dirs` alone hashes every
joint video (GBs) and stats every file under every non-live plan dir.
`build_parallel_tasks` calls `chunk_frames_match_size` (PIL open per PNG) per ledgered
chunk just to build the queue — 256×8×32 ≈ 65k header opens before any GPU work. These
run serially at every finalize start, and `_poll_to_completion` re-enumerates per pass
(up to 10).

## Live evidence

```
$ grep -n "sweep_chunk_mp4s\|prune_orphan_plan_dirs\|_poll_to_completion" voyage/augment_finalize.py | head
685-723
$ grep -n "rglob\|chunk_frames_match_size" voyage/augment_drain.py voyage/augment_parallel.py | head
drain:359 (_newest_modification_time rglob); parallel:184,186
```

Repro: `strace -c`-style counter (or `LD_PRELOAD`-free Python stat counter via `pathlib`
shim) during finalize-start on a 10-seg fixture — stat/open counts scale with plans ×
chunks, not with missing work.

## Source refs

`voyage/augment_finalize.py:685-701`; `voyage/augment_drain.py:372-479`
(`_newest_modification_time` rglob); `voyage/augment_parallel.py:177-219`;
`voyage/augment_sidecar.py:632-760`.

## Online sources

- In-tree budget comments (`UPSCALE_TILE_BUDGET_PIXELS`, chunk-32 design notes show the
  project measures these floors — sweeps have no equivalent budget).

## Fix candidates

- Single enumeration shared by sweep/GC/queue-build; cache PIL sizes per dir+mtime; hash
  joint videos once per process; skip GC when `augment/` mtime unchanged.

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Evaluation (2026-10-07, group N)

Re-verified live against current code before fixing — claim CONFIRMED
with one correction:

- Finalize start runs `sweep_chunk_mp4s` (`voyage/augment_finalize.py:739`)
  then `prune_orphan_plan_dirs` (`:744`), plus `heal_augment_ledgers`
  (`voyage/media.py:1042`, `cli_generate.py:244`), then per-pass
  poller scans (`_poll_to_completion`, up to 10 passes re-enumerating
  segments + ledgers each pass).
- CORRECTION: the drain docstring ("NOT wired into finalize", `drain.py`
  GC docstring) is STALE — GC has been wired since the interleave work
  (`augment_finalize.py:744`); verified live, docstring left for the
  owning pass (out of scope here).
- `prune_orphan_plan_dirs` re-hashes every joint video (`:443-447`) and
  rglob-stats every file under every non-live dir
  (`_newest_modification_time`, `:349-369`); `build_parallel_tasks`
  PIL-opens every chunk PNG (`:177-189`) per queue-build.
- The "65k header opens" figure is a computed-not-profiled projection
  (reviewer flag stands): the per-PNG-open loop shape is confirmed in
  code, the segment count is hypothetical. Recorded as an estimate.
- Fix adopted: per-file PNG-size cache (stat-keyed), process-wide
  joint-sha cache (shared with 250), GC skip on unchanged fingerprint
  (deferred GC only, never wrong deletion), shared `list_plan_dirs`
  helper for sweep/heal/GC enumeration. Per-pass poller re-scans stay
  (ledger truth must be re-read as chunks land) — recorded as open.

## Progress log (2026-10-07, group N)

- Implemented: `voyage/augment.py` (`_PNG_SIZE_CACHE` +
  `_cached_png_size` consulted by `chunk_frames_match_size`);
  `voyage/augment_joints.py` (`joint_video_sha`, shared with 250);
  `voyage/augment_sidecar.py` (new `list_plan_dirs`, adopted by
  `heal_augment_ledgers`); `voyage/augment_drain.py`
  (`sweep_chunk_mp4s` + GC loop use `list_plan_dirs`;
  `prune_orphan_plan_dirs` hashes joints via `joint_video_sha` and
  skips on an unchanged `_gc_fingerprint` — params + segment
  manifests + augment top-level + joint identities, wall time
  deliberately excluded so skips defer collection only).
- New tests in `Voyage/tests/test_group_n_perf.py`: PNG opens
  happen once per identity, repeat GC returns 0 with zero recursive
  stats, a new orphan re-arms the GC, shared enumeration shape.

## Resolution (2026-10-07)

- Verdict: RESOLVED. Repeat finalize starts in one process skip the
  GC entirely when nothing changed; PNG headers open once per file
  identity across queue-build/heal passes; joint videos hash once
  per process; sweep/heal/GC share one enumeration helper.
- Files changed: `voyage/augment.py`, `voyage/augment_joints.py`,
  `voyage/augment_sidecar.py`, `voyage/augment_drain.py`,
  `Voyage/tests/test_group_n_perf.py` (new).
- Verification: ruff check + format clean on touched files; mypy
  strict clean on touched modules; scoped pytest 318 passed /
  5 skipped plus 153 passed neighbors. No GPU workloads. No commits.
- Left open: per-pass poller re-scans (up to 10 passes re-enumerate
  segments + ledgers) — ledger truth must be re-read as chunks land,
  so this stays; `heal_augment_ledgers` still walks every ledger at
  every start (now PIL-cheap via the size cache).

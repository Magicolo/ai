# 033 — Director prefetch serialized by the single JSONL lock ("parallel" rarely overlaps)

- Status: open
- Severity: medium (lost overlap; hit rate unreported)
- Area: performance — prefetch vs commit contention
- Rank rationale: design claims "CPU director vs GPU video — no contention", but
  the director worker is a single process with one stdin/stdout stream and a
  client-side mutex.

## Technical description

- `voyage/supervisor.py:452-523` (`_prefetch_decide_for_next` / `_take_prefetch`),
  `:183-191` (single `ThreadPoolExecutor(1)`).
- `voyage/rpc.py:90-94,156` (`_call_lock` serializes all ops on one worker;
  prefetch `director.call("decide")` vs commit `director.call("decide"/"embed")`
  share it).
- `voyage/workers/loop.py:16-53` (single-threaded dispatch loop).
- `voyage/workers/director.py:338-364` (`handle_benchmark` only times
  `deterministic`, never Qwen — prefetch benefit unmeasured).

When the commit path calls `embed` (line 671, per novelty attempt) or `decide`, a
concurrent prefetch `decide` either waits on the lock or is skipped
(`_take_prefetch` logs `director_prefetch_miss` when `not future.done()`). With
Qwen CPU decisions at ~minutes (Phase-3 notes: `~5min/segment on CPU`), the
prefetch window (video render) is the only overlap — but the next commit's
`embed`+`decide` contend with the still-running prefetch. Hit rate is unreported
(no metric beyond hit/miss events, never aggregated).

## Why this is an issue

The design promises free overlap between CPU director and GPU video, but a single JSONL lock means the "parallel" prefetch rarely overlaps in practice — so multi-minute Qwen decisions stay on the critical path and segment throughput never sees the designed speedup. With hit rate unreported, nobody can tell whether the machinery earns its complexity or just adds contention failure modes. Any fix (second worker process doubles ~16 GiB RAM) is expensive enough that it must be driven by measured data, which currently does not exist.

## Evidence

`rg -n "prefetch|_call_lock" Voyage/voyage/supervisor.py Voyage/voyage/rpc.py`;
fake-backend multi-segment run: count `director_prefetch_hit vs miss` in
`metrics.jsonl`.

Verified live 2026-09-25: supervisor `prefetch` hits at 183,189-191,226-227,
234-236,452,464-475,487,496,498-506; `ThreadPoolExecutor(max_workers=1)` at
`:227`; `handle_benchmark` still at `workers/director.py:338` (times
deterministic only, never Qwen).

## Reproduction

Run fake-backend multi-segment; aggregate hit/miss events (currently manual).

## Source references

- Files/lines above.

## Resolution candidates

1. Separate prefetch worker process (cheap when the backend is stateless
   `deterministic`; Qwen weights are mmap'd ~16 GiB — second process doubles RAM,
   so gate by backend).
2. Or make `embed` lock-free (MiniLM encode is fast; move to a supervisor CPU
   thread, drop the RPC).
3. Measure hit rate in the `soak` report before optimizing further.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; prefetch /
  executor / handle_benchmark refs re-verified live, current; pasted hit lines
  into Evidence.
- Open: measure hit rate; then pick (1) or (2).

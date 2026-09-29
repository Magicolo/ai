# 029 — CausVid pre-encodes with N separate 11 GiB T5 shuttles per segment

- Status: resolved (fixed 2026-09-25: single T5 shuttle + tests)
- Severity: medium-high (allocator churn → fragmentation OOMs)
- Area: performance — `CausvidSession._encode_conditional` + `generate_blocks`
- Rank rationale: the code's own comment warns against exactly what the code does.

## Technical description

`voyage/workers/video_causvid.py:717`:

```python
conditionals = [_encode_conditional(p) for p in prompts]
```

`_encode_conditional` (`:534-549`) does `text_encoder.to(cuda) … to(cpu) +
empty_cache()` **per call**; `:512-516` comment warns about fragmentation but the
loop still performs N shuttles — one `to(cuda)/to(cpu)/empty_cache()` cycle per
prompt, each allocating/freeing ~11 GiB. Repeated alloc/free is what the comment
blames for fragmentation OOMs.

## Why this is an issue

Each shuttle allocates and frees ~11 GiB on a 16 GB card, and repeated alloc/free cycling is precisely what fragments the CUDA allocator into OOMs — the scarcest resource on the target hardware, paid once per prompt instead of once per segment. Multi-block segments multiply the churn exactly when VRAM pressure is highest. The code's own comment warns against this pattern while the loop below it performs it, leaving future readers an unresolved contradiction to trip over.

## Evidence

Read `CausvidSession._encode_conditional` + `generate_blocks:678-728`.

Verified live 2026-09-25:

```
534:  def _encode_conditional(self, prompt: str) -> dict[str, Any]:
543:    pipeline.text_encoder.to("cuda")  ... .to("cpu") + empty_cache() per call
678:  def generate_blocks(
717:    conditionals = [self._encode_conditional(prompt) for prompt in prompts]
```
Note: the `:534-549` docstring now claims "callers pre-encode the whole segment
up front so N rollouts cost one 11 GB roundtrip, not N" — but `:717` still calls
`_encode_conditional` once per prompt, and each call does its own
`to(cuda)/to(cpu)/empty_cache()` cycle. The docstring describes the fix, not the
code; issue stands.

## Reproduction

Multi-block CausVid segment under memory pressure; observe per-prompt shuttle
cycles in logs vs a single-shuttle baseline.

## Source references

- `voyage/workers/video_causvid.py:512-549,678-728`.

## Resolution candidates

Move T5 to CUDA once, encode all prompts in one `inference_mode` block, move back
once. Trivial; saves (N-1) × ~11 GiB roundtrips + allocator churn; reduces
multi-block OOM flakes.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; refs re-verified
  live, current; noted in Evidence that the `:534-549` docstring now describes
  the fix ("one 11 GB roundtrip, not N") while `:717` still loops per-prompt
  shuttles — docstring-vs-code contradiction, issue stands.
- Open: implement + multi-block soak.
- 2026-09-26 (resolution, FIXED): new
  `CausvidSession._encode_conditionals` (`voyage/workers/video_causvid.py:581-605`)
  moves T5 to CUDA once, encodes every prompt with the same per-prompt
  call as before, and parks it back once (`finally`, so a failed
  encode still restores the CPU park); empty lists rejected.
  `_encode_conditional` (`:607-613`) delegates to it (single-prompt
  form kept for benchmark probes — identical shuttle semantics).
  `generate_blocks` (`:783`) calls the batch form, resolving the
  docstring-vs-code contradiction. Multi-block soak under memory
  pressure is the orchestrator's job (no GPU workloads here).
  Tests: `tests/test_issue_029_causvid_shuttle.py` (6 tests: one
  `cuda/cpu` pair + 2 `empty_cache` calls for 3 prompts, single
  delegation, `finally` restore on failure, empty rejection, class
  cache identity, precomputed passthrough).
  Gates: ruff + format + mypy strict clean on `video_causvid.py`;
  new tests pass in-container (see 014 log for full-tree gate state).
  Regression slice (causvid/ltxv/longlive/concepts-adjacent/metrics/
  recovery/inspector suites): 118 passed; the only 4 failures are
  `tests/test_ltxv_failure_hygiene.py`, caused by the 030 LRU type
  change (see 030 log) — `test_causvid_worker.py` itself is fully
  green with the new single-shuttle path.

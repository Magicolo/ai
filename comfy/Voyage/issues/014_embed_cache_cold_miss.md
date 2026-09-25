# 014 — Embed cache is session-resident: every evict/rebuild cold-misses CPU T5 (minutes)

- Status: open
- Severity: major (performance — minutes per post-audio segment)
- Area: workers — text-embedding cache lifecycle
- Rank rationale: turns every audio take into a repeated minutes-long T5 encode;
  interacts multiplicatively with 013.

## Technical description

- `voyage/workers/video_longlive.py:353` `_embed_cache`, `:479-487` `_encode`,
  `:701-709` `evict` (`del _stream` + `del _pipeline`).
- `voyage/workers/video_ltxv.py:295` `_embed_cache`, `:301-321` `_encode (~25s)`,
  `:602-612` `evict` (`_embed_cache.clear()`).
- `voyage/workers/video_longlive.py:408-471` `resume_from_tape` restores RNG +
  tail but **not** the embed cache.

T5-XXL CPU/bf16 encode costs "minutes per segment" (`video_longlive.py:94-95`;
LTXV `~25s` per `_encode` docstring). The cache lives in the stream/session object
that `evict()` destroys, so after every audio take the fresh session cold-misses;
the audit's `tape_encode_ms 0.01` is a **cache-hit number only**. Post-rebuild
first block with a repeated prompt re-encodes from scratch.

## Why this is an issue

T5-XXL CPU/bf16 encoding costs minutes per segment, and because the cache
lives in the session object that `evict()` destroys, every audio take forces a
cold re-encode on the next block — even when the prompt is identical to the one
encoded before the take. This multiplies with the swap cost in 013: each take
pays teardown plus minutes of repeated encoding of already-seen prompts, while
the audit's `tape_encode_ms 0.01` figure only ever describes the cache-hit path
that takes destroy. Over a long music-backed voyage with periodic takes, the
repeated work sums to hours. Post-audio segments — most of such a run — pay it
on every take boundary.

## Evidence

`rg -n "_embed_cache" Voyage/voyage/workers/video_*.py` — no disk persistence of
embeds (only `prompt_embeds` for the tail in the tape). Audit first-call excess
`72s vs 16.9s steady` attributed to `CPU T5 + autotune`
(`reports/longlive-audit.md:40-47`) — paid once per worker lifetime without swap,
once per **take** with swap.

Re-verified 2026-09-25:

```
$ rg -n "_embed_cache" voyage/workers/video_longlive.py voyage/workers/video_ltxv.py
voyage/workers/video_longlive.py:353:        self._embed_cache: dict[str, tuple[Any, Any]] = {}
voyage/workers/video_longlive.py:480:        cached = self._embed_cache.get(prompt)
voyage/workers/video_longlive.py:486:        self._embed_cache[prompt] = (cond, cond_list)
voyage/workers/video_ltxv.py:295:        self._embed_cache: dict[str, tuple[Any, Any]] = {}
voyage/workers/video_ltxv.py:303:        cached = self._embed_cache.get(text)
voyage/workers/video_ltxv.py:320:        self._embed_cache[text] = result
voyage/workers/video_ltxv.py:605:        self._embed_cache.clear()
```

Cache still session-resident; the `evict` paths (`video_longlive.py:701-716`,
`video_ltxv.py:602-612`) still destroy it.

## Reproduction

`acestep+longlive2` with a repeated prompt across a take boundary; compare
first-block latency pre- vs post-rebuild (cold T5 vs cached).

## Source references

- Files/lines above; `Voyage/reports/longlive-audit.md:40-47`.

## Resolution candidates

1. Persist embeds keyed by `prompt_plan_hash` to `<segment>/embeds.pt` (CPU
   tensors, ~4 MB) or a small file-backed LRU in `/models/.embed_cache`.
2. Restore the cache entry in `resume_from_tape` alongside RNG/tail.
3. Bound the cache (see 030) so persistence doesn't grow without limit.

Payoff: post-audio segments return to steady-state latency; saves minutes per take.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- Open: implement file-backed cache + measurement.
- 2026-09-25 (repair pass): added `## Why this is an issue`; Evidence enriched
  with live `_embed_cache` output; refs verified current.

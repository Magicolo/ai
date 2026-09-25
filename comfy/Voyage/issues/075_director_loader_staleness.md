# 075 — Director loaders ignore `model_id` after first load; `embed`/`decide`/`inspect` params unvalidated

- Status: open
- Severity: low (stale-model confusion in long-lived worker; `0`/huge token
  budgets reach the model)
- Area: director worker — `voyage/workers/director.py:49-50,89-96,114-115,
  159-178,168,237-238,316-323`
- Rank rationale: pass-2 worker finding; staleness (wrong model kept) is distinct
  from 030's unbounded growth and 014's cold-miss latency.

## Technical description

```python
def _load_embedder(model_id: str) -> Any:
    if "model" not in _EMBEDDER:
        _EMBEDDER["model"] = SentenceTransformer(model_id, device="cpu")
    return _EMBEDDER["model"]
```

All three loaders guard on key presence only — `model_id` never compared on later
calls; `handle_init:327-335` updates `_CONFIG` but loaders keep the first
weights. Plus: `handle_embed` (`:316-323`) `checked_request(payload, texts=list)`
then coerces — `[]` passes and reaches `model.encode([], ...)`; non-strings
coerced silently. `max_new_tokens=int(payload.get(...))` (`:168,237-238`) and
`temperature=float(...)` — `0`/negative/huge/`NaN` flow into `model.generate`
(`_qwen_generate:219-226`, `_inspector_generate:144-145`).

## Why this is an issue

In a long-lived worker, re-`init` with a new `model_id` silently keeps the old
weights — every later decision is attributed to a model that never loaded.
Alongside that, empty text lists and unbounded generation budgets (`0`/huge
`max_new_tokens`, `NaN` temperature) reach the model unchecked, turning typos
into confusing backend errors or runaway generations instead of boundary
rejections.

## Evidence

Loader guards (re-run 2026-09-25):

```
$ rg -n "if \"model\" not in _(QWEN|EMBEDDER|INSPECTOR)" Voyage/voyage/workers/director.py
50:    if "model" not in _QWEN:
96:    if "model" not in _INSPECTOR:
115:    if "model" not in _EMBEDDER:
```

All three guard on key presence only — `model_id` is never compared on later
calls. `handle_embed` (`:320-328`) has no `if not texts` check; no range checks
on `max_new_tokens` (`:168`, `:237-238`) or `temperature`.

## Reproduction

`init {"embedding_model_id":"A"}` → `embed` (loads A) → `init
{"embedding_model_id":"B"}` → `embed` still encodes with A; `embed {"texts":[]}`
→ `{"vectors":[]}`; `decide {"temperature":1e9,"max_new_tokens":1000000}` /
`inspect {"max_new_tokens":0}` reach `generate`.

## Source references

- Files/lines above.

## Resolution candidates

Key caches by `model_id` (`_QWEN[model_id]`) or invalidate on `handle_init`
change; require `texts` non-empty; clamp `max_new_tokens>=1` (upper-bound, e.g.
4096) and `temperature` finite `>=0`.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`director.py:49-50`, `:89-96`, `:114-115` loaders,
  `:159-178` inspect, `:168`/`:237-238` token params, `:320-328` embed —
  all match: note `handle_embed` def sits at `:320`, inside the cited range);
  re-ran loader-guard `rg` (past above).
- Open: implement + tests.

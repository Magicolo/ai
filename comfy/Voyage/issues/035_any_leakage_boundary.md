# 035 — `Any` leakage: 364 hits, `ANN401` 255 under ALL, boundary alias admitted incomplete

- Severity: LOW (typing hygiene — `Any` propagates silently)
- Group: standards/typing — Rank: 4/5
- File:line: `voyage/rpc.py:18,25-34,257-272`; `voyage/model_registry.py` (25 lines); `voyage/scoreboard.py:20,39,77,81`

## Description

`Any` is the default seam type: manifest/scoreboard/RPC payloads are
`dict[str, Any]`, test doubles are `(*args: Any, **kwargs: Any)`.
`rpc.py:25-34` introduces `RpcPayload = dict[str, JsonValue]` ("never bare
Any") but `call()` at `rpc.py:257` keeps `dict[str, Any]` with a comment
deferring per-op `TypedDict`s out of scope — the ratchet point exists but
nothing uses it. The `numpy/torch/transformers/…` mypy override
(`follow_imports="skip"`, `pyproject.toml:144-166`) additionally renders all
numeric/tensor code `Any`-typed, invisible to strict.

## Rationale

Mypy strict includes `warn_return_any`/`disallow_any_*` precisely because
`Any` propagates silently. Unselected `ANN401` (255 hits) means new `Any`
args land ungated.

## Live evidence (re-verified 2026-09-30)

Host `rg` today:

```
total Any in voyage/: 364
voyage/workers/video_causvid.py:37
voyage/workers/video_longlive.py:32
voyage/workers/video_ltxv.py:31
voyage/workers/augment_worker.py:26
voyage/model_registry.py:25
voyage/workers/director.py:20
voyage/supervisor.py:19
voyage/cli.py:18
```

`voyage/rpc.py:18,25-34,210,213,257-258,270-272` read live:

```python
from typing import Any, TextIO                      # :18
RpcPayload = dict[str, JsonValue]                   # :25
"""One worker request payload: JSON-shaped, never bare `Any` (issue 036)."""
raw_stdout: Any = stdout                            # :210
readable: Any = raw_stdout                          # :213
def call(self, op: str, payload: dict[str, Any], …  # :257
    ) -> dict[str, Any]:                            # :258
    `payload` stays `dict[str, Any]` (not `RpcPayload`) for now …  # :270-272
```

`voyage/scoreboard.py:20,39,77,81` (`dict[str, Any]` helpers/rows);
`voyage/model_registry.py:362,372,375,445` and 8 `_record_*` builders
returning `dict[str, Any]`; `voyage/models_ensure.py:150,152,203`
(`present`/`raw`/`future_to_entry`).

Sweep `ALL` stats (preserved): `ANN401 255, ANN001 10, ANN202 3`.

## Repro

```bash
rg -n "Any" voyage/*.py voyage/workers/*.py | head
ruff check --select ANN .
```

## Fix candidates

(a) Enable `ANN401` (and `ANN001/ANN201`) first — cheapest tripwire.
(b) Migrate `scoreboard/model_registry/models_ensure` dicts to
`JsonValue`-valued aliases.
(c) Adopt per-op `TypedDict`s at the RPC boundary per the `rpc.py` comment's
own plan.

## Refs

- Ruff `ANN401 any-type` docs; mypy strict flag set
  (`disallow_any_generics`, `warn_return_any`).
- `Voyage/pyproject.toml:144-166`
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §5.

## Progress log (2026-09-30, toolchain track)

- Re-verified premises live (host `rg` for counts, in-container ruff
  for families): total `Any` in voyage/ is now **387** (was 364;
  top files video_causvid 37, augment_worker 36, video_longlive 32,
  video_ltxv 31, model_registry 27, workers/director 20, supervisor 18,
  cli 18 — same shape, drifted up). ANN-family probe: **ANN 339
  (ANN401 326 + ANN001 10 + ANN202 3)** — ANN401 alone grew 255→326.
- ANN401 tripwire: canNOT be enabled — 326 hits would redden the gate
  on landing, and fixing them means annotating worker/GPU seams plus
  the `follow_imports="skip"` heavy-deps override (which renders all
  numeric/tensor code `Any`-typed by design until upstream ships
  3.10-clean stubs). Documented as staying out with the count above.
- Boundary aliases: `rpc.py` still defers — `call()` at `:327-329`
  keeps `payload: dict[str, Any] -> dict[str, Any]` with the
  per-op-TypedDict deferral comment at `:348` (unchanged since issue
  time; `RpcPayload`/`RpcResult` JsonValue aliases exist but only the
  alias definitions use them). scoreboard.py (`:40,56,118,122`),
  model_registry.py (25 hits incl. 8 `_record_*` builders),
  models_ensure.py (`:154,164,263`) still `dict[str, Any]`-valued.
  Migrating these is annotation surgery across supervisor + workers
  (incl. dirty supervisor.py) — outside this track's scope
  (docstrings + constants only). No alias migration this pass.
- No pyproject/gate change for 035 (nothing green to ratchet).

## Resolution

- Document-only: tripwire stays out (326 ANN401 hits as-read;
  enabling = instant red), alias migration proposed for the owning
  pass — adopt `RpcPayload` in `call()` first (single signature,
  callers already pass JSON-shaped dicts), then scoreboard →
  models_ensure → model_registry `dict[str, Any]` returns to
  `JsonValue`-valued aliases, with the heavy-deps `follow_imports`
  note (issue 035's override section) as the long pole.
- Files changed: none for 035. Gate evidence: n/a (no change; ANN
  probe output recorded above). DESIGN proposals: none. Residuals:
  full 035 scope (ANN401 326, boundary `dict[str, Any]` at
  rpc.py:327-329, scoreboard/model_registry/models_ensure sites).

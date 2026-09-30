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

## Progress log (2026-09-30, Group D pass)

- `call()` migration blocked (verified live, no edit): `supervisor.py:648`
  (`payload: dict[str, object]`), `:1383` (`audio_payload`), `:1599`
  (`payload`) all hold non-JSON-shaped dicts, and `supervisor.py` carries
  concurrent uncommitted edits — dict invariance would redden the gate at
  those call sites. `rpc.py:281-287` (`raw_stdout`/`readable: Any` fd
  juggling) cannot narrow by construction. Scoreboard leg is residual:
  `voyage/scoreboard.py` has concurrent edits and is explicitly out of
  scope for this pass.
- Landed the first greenable step in `voyage/models_ensure.py`:
  `_read_manifest_keys` now returns `dict[str, JsonValue] | None`
  (import from `voyage.atomic`, explicit `str(key)` narrowing instead of
  aliasing the raw dict). Callers (`_repair_manifest`, `atomic_write_json`)
  accept it without complaint — verified, not assumed.
- Registry builders (`_record_*` → `dict[str, Any]`, 27 hits) and the
  `snapshot_kwargs`/`file_kwargs` `dict[str, Any]` sites stay: consumed
  across modules incl. dirty ones, not greenable in isolation.

## Resolution (2026-09-30, Group D pass)

- Partially resolved: one alias hunk landed (`models_ensure`
  `_read_manifest_keys` → `JsonValue`-valued). `call()` → scoreboard →
  registry order stands behind it: `call()` needs the supervisor
  `dict[str, object]` call sites converted first (dirty file, other pass);
  scoreboard needs its concurrent edits to land.
- Files changed: `voyage/models_ensure.py` (JsonValue import + return-type
  narrowing + docstring). Gate evidence: `ruff check` + `ruff format
  --check` clean; `mypy voyage` clean (63 files); `test_generate_ensure.py`
  22 passed in-container. DESIGN proposals: none. Residuals: `call()`
  signature, scoreboard dicts, registry builders/kwargs.

## Progress log (2026-09-30, CLI track — re-verification only, this pass)

- Re-verified every leg live (host reads, 2026-09-30): `call()` still
  `payload: dict[str, Any] -> dict[str, Any]` (`voyage/rpc.py:327-329`,
  deferral comment intact); `scoreboard.py` legs still present (`:21`
  `Any` import, `:41` `_finite_float`, `:111/:115` `scoreboard_rows`
  dicts); `model_registry.py` legs still present (8+ `download_*` returns
  `:455/:980/:1080/:1095/:1109/:1125/:1144/:1184/:1200/:1217/:1232`,
  `_merge_manifest_record` `:465/:468`, `record_builder` `:546`,
  `snapshot_kwargs` `:959`, `file_kwargs` `:969`); `models_ensure.py`
  batch-8 hunk present (`:40` `JsonValue` import, `:154`
  `_read_manifest_keys -> dict[str, JsonValue] | None`) with one remaining
  `dict[str, Any]` at `:268` (`future_to_entry` — typeshed-driven, worker
  futures return untyped dicts).
- Cleanliness at runtime (`git diff --name-only`): `scoreboard.py`,
  `model_registry.py`, `rpc.py`, `models_ensure.py` all clean — but the
  pass brief bans editing `scoreboard.py`/`model_registry.py` ("record,
  don't touch") and `rpc.py` ("concurrent-adjacent, do NOT touch"), so no
  leg was attempted. No test (record-only, no behavior change).

## Resolution (2026-09-30, CLI track — this pass)

- Verdict: DEFERRED (record-only) — premises all hold, all legs residual.
- Files changed: none. Gate evidence: n/a (no change). DESIGN proposals: none.
- Residuals (exact handoffs for the owning pass): `voyage/rpc.py:327-329`
  (`call()` → `RpcPayload` first, then supervisor `dict[str, object]` call
  sites `supervisor.py:648/:1383/:1599` per the Group-D finding);
  `voyage/scoreboard.py:21,41,111,115` (JsonValue-valued aliases);
  `voyage/model_registry.py:455,465,468,546,959,969,980,1080,1095,1109,1125,
  1144,1184,1200,1217,1232` (builders/kwargs/returns);
  `voyage/models_ensure.py:268` (`future_to_entry` — narrow only with the
  worker-result type). Order stands: `call()` → scoreboard → models_ensure →
  registry.

## Progress log (2026-09-30, this pass — migration in recorded order)

- `git diff --name-only` at pass start: clean tree. `voyage/supervisor.py`
  was NOT touched (another group owns it — its sites are remainder only).
- Leg 1, `rpc.py call()` → `RpcPayload`: BLOCKED, re-verified live
  against the batch-9 tree — `supervisor.py` still holds
  `payload: dict[str, object]` (`:611`, `:1488`) plus `dict[str, Any]`
  `video_init`/`prefetched_raw`/`raw`, so narrowing `call()` reddens
  those foreign sites (dict invariance; same trap `voyage/atomic.py`
  documents for the write/read sides). `raw_stdout`/`readable: Any`
  (`rpc.py:281,284`) stay by construction (fd juggling). No edit.
- Leg 2, scoreboard dicts: MIGRATED `_finite_float(value: Any)` →
  `(value: JsonValue)` (`scoreboard.py:21,41` + docstring) — callers pass
  JSON-parsed `Any`, so the boundary documents without breaking.
  `scoreboard_rows -> list[dict[str, Any]]` (`:114,118`) STAYS: a live
  in-container mypy probe proved `dict[str, float]` variables
  (`stages.get(...)`, `current`, `deltas`) are invariant-blocked from
  `JsonValue` nesting (`Dict entry has incompatible type ... [dict-item]`),
  while all-literal nesting passes — casts would be needed, out of scope.
- Leg 3, models_ensure: MIGRATED `future_to_entry` →
  `dict[Future[dict[str, JsonValue]], RequiredModel]` (`:268`) once
  `download_model` migrated below. `raw: Any` (`:167`) stays (the
  `json.loads` narrow-with-`isinstance` idiom, same as `atomic`).
- Leg 4, registry builders: MIGRATED all 10 `_record_*` in
  `registry_records.py` → `dict[str, JsonValue]` (import swap; zero `Any`
  left in the file — probe shape `h()` verified comprehensions,
  `list(...)` calls, `str | None` revisions and nested `str→str` dicts
  all context-infer cleanly) + `model_registry.py` `record_builder`
  field, `_merge_manifest_record` param/return (internal `record` stays
  `dict[str, Any]` — the `json.loads` idiom), `download_model` + all 10
  thin `download_*` wrappers → `dict[str, JsonValue]`.
  `snapshot_kwargs`/`file_kwargs` (`:953,:963`) STAY: `**dict[str,
  JsonValue]` is not assignable to the hub signatures (invariance —
  same class as `future_to_entry`'s old typeshed block).
- Foreign coexistence note: `model_registry.py` carries a concurrent
  uncommitted hunk in the same file (`__all__` `_sha256` removal, issue
  021, another group) — different region, no overlap; left intact.

## Resolution (2026-09-30, this pass)

- Verdict: PARTIALLY RESOLVED — scoreboard `_finite_float`, all 10
  registry builders, `record_builder` field, `_merge_manifest_record`,
  `download_model` + 10 wrappers, `future_to_entry` now `JsonValue`-valued.
- Files changed: `voyage/registry_records.py` (import + 10 returns),
  `voyage/model_registry.py` (import + field + merge + 11 returns),
  `voyage/models_ensure.py` (1 annotation), `voyage/scoreboard.py`
  (import + param + docstring). Gate evidence: in-container `mypy`
  strict clean on all 4 files (plus full `mypy voyage`, 65 files,
  clean) + `ruff check` + `ruff format --check` clean; narrow tests
  green — `test_scoreboard` + `test_registry_split` +
  `test_generate_ensure` + `test_cli_split` (46 passed),
  `test_director_models_dir` + `test_single_source` (26 passed),
  `test_registry_pins` + `test_models_ranges_119` +
  `test_backend_registry` + `test_augment_models` (51 passed).
  DESIGN proposals: none (annotation-only, no behavior change).
  Residuals: `call()` signature + supervisor `dict[str, object]` sites
  (`:611,:1488` + `video_init`/`prefetched_raw`/`raw`); `scoreboard_rows`
  return + `rows` local (invariance probe on file); `snapshot_kwargs` /
  `file_kwargs`; `raw: Any` + merge-internal `record` (`json.loads`
  idiom); `raw_stdout`/`readable` (fd juggling).

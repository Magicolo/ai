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

## Progress log (2026-09-30, this pass — remainder in recorded order)

- Pre-flight: `git status`/`git diff` (repo root `ai/`, paths
  `comfy/Voyage/...`) showed concurrent-agent uncommitted hunks in
  `voyage/supervisor.py` (issue 081 prefetch move:
  `summarize_prefetch_outcome` → `voyage/supervisor_prefetch.py` +
  re-export, plus `supervisor_proposal`/`cli_observe`/`audio/*` tracks)
  and new untracked `voyage/supervisor_prefetch.py` +
  `voyage/cli_inspect_metrics.py`. No foreign hunks in the
  `model_registry.py`/`models_ensure.py`/`scoreboard.py` target regions;
  `supervisor.py` foreign hunk is at a different region from the
  `JsonValue` sites below — left intact, re-read before every edit,
  hunks minimal, never undid foreign work. `voyage/rpc.py` clean.
- Leg 1, `rpc.py:327-329 call()` → `RpcPayload`: BLOCKED, probed live
  then reverted (no edit landed). Changing payload to `RpcPayload`
  reddens `voyage/supervisor.py:631`
  (`dict(payload)` with `dict[str, object]` → `SupportsKeysAndGetItem`
  mismatch) and `voyage/supervisor.py:1182`
  (`{"texts": texts}` with `list[str]` vs `list[JsonValue]` invariance).
  `list[str]`/`dict[str,str]` literals are never assignable to
  `dict[str, JsonValue]` (list/dict invariance) — every call site would
  need an explicit `RpcPayload` annotation or cast, incl. foreign lines.
  Return stays `dict[str, Any]` (narrowing `response.result` would need
  a cast; `WorkerRequest`/`WorkerResponse` in `voyage/models.py:315-330`
  stay `dict[str, Any]` — out of scope). `raw_stdout`/`readable: Any`
  (`rpc.py:281,284`) stay by construction (fd juggling).
- Leg 2, supervisor `dict[str, object]` → `JsonValue`: SKIPPED/BLOCKED,
  no edit landed for payload/return types. `_call_with_restart`
  (`supervisor.py:590-598`, payload `:596`, return `:598`),
  `_with_audio_gpu` (`:1470-1475`, payload `:1473`, return `:1475`),
  `payload` (`:1694`), `gauges` (`:850`), `_log_metric` (`:574`),
  `extra` (`:1238`) form one invariant chain: inline
  `dict[str,str]` (`{"recovery_path": ...}`) and `list[str]` payloads
  fail against `dict[str, JsonValue]`, and `RpcPayload` fails against
  `dict[str, object]` consumers (`_log_metric`) — coordinated
  re-annotation of every producer + consumer plus `list[JsonValue]`
  reshaping, touching hot/foreign regions. Recorded as still-blocked.
- Leg 3, `scoreboard.py:114,118 scoreboard_rows` return: BLOCKED,
  probed live then reverted (no edit landed). Changing return + `rows`
  to `list[dict[str, JsonValue]]` reddens `:187`
  (`dict[str, list[str] | int | dict[str, float] | ...]` vs
  `dict[str, JsonValue]`): `stages.get` (`dict[str, float]`), `current`/
  `deltas` (`dict[str, float]`), `errors` (`list[str]`) are
  invariant-blocked from `JsonValue` nesting; fixing needs
  `JsonValue`-valued locals + `isinstance` re-narrowing for the
  arithmetic (code change, not annotation-only) in a hot file.
- Leg 4, hub `**kwargs` (`model_registry.py:953,963`): BLOCKED, probed
  live then reverted (no edit landed). `snapshot_kwargs` →
  `dict[str, JsonValue]` reddens `:960` with 11 errors
  (`**dict[str, JsonValue]` vs `str`/`str | None`/`str | Path | None`/
  `list[str] | str | None`/etc.): the `JsonValue` union is too wide for
  the hub signatures, and `allow_patterns list[str]` already fails the
  `list[JsonValue]` invariance. Stays `dict[str, Any]` (Any defeats
  both checks by design).
- Leg 5, `json.loads` idiom sites: MIGRATED where local + mypy-clean
  (annotation-only, no behavior change — `Any` → `JsonValue` on the
  `json.loads` temporary, `isinstance(..., dict)` narrows the union to
  `dict[str, JsonValue]`; downstream `.get`/`isinstance` guards
  unchanged):
  `model_registry.py:423` (`verify_checkpoint_against_manifest loaded`),
  `:462,:464` (`_merge_manifest_record record` +
  `loaded` — `record` was the last `dict[str, Any]` in the merge path,
  return was already `JsonValue`), `:1035`
  (`_manifest_hash_mismatches loaded`); `scoreboard.py:78` (`event`);
  `models_ensure.py:167` (`raw` + dropped now-unused `Any` import);
  `supervisor.py:795` (`raw`), `:1861` (`existing`), `:1868`
  (`recorded`) + `JsonValue` import (`:32`). All verified per-file
  then full-tree (below). Stay by design: `atomic.py:97,110,118`
  (`atomic_write_json`/`read_json` — docstring records the tried +
  reverted widening: manifests hold `dict[str, object]`);
  `rpc.py:281,284` (fd juggling, above).

## Resolution (2026-09-30, this pass)

- Verdict: PARTIALLY RESOLVED — `json.loads` narrowings landed (7
  sites across 4 files); `call()` + supervisor `dict[str, object]` +
  `scoreboard_rows` return + hub kwargs recorded still-blocked with
  live probe evidence (above).
- Files changed: `voyage/model_registry.py` (3 `loaded: JsonValue` +
  `record: dict[str, JsonValue]`), `voyage/models_ensure.py` (`raw:
  JsonValue` + `Any` import removal), `voyage/scoreboard.py` (`event:
  JsonValue`), `voyage/supervisor.py` (`JsonValue` import + `raw` /
  `existing` / `recorded: JsonValue`). `voyage/rpc.py` probed but
  reverted — no change. No test files changed (annotation-only; TDD
  n/a — gate evidence below instead, per contract). Foreign hunks
  (`supervisor.py` 081 prefetch move + `audio/*`/`cli_observe`/issues
  tracks, untracked `supervisor_prefetch.py`/`cli_inspect_metrics.py`)
  left intact; own hunks are disjoint regions.
- Gate evidence (in-container `voyage:latest`, CPU-only, no host pip):
  scoped `ruff check` + `ruff format --check` + `mypy` clean on all 5
  touched/probed files (`model_registry`/`models_ensure`/`scoreboard`/
  `supervisor`/`rpc`); `mypy voyage` clean (67 source files, incl.
  concurrent `supervisor_prefetch.py`); `ruff format --check .` clean
  (257 files); `pytest -m 'not gpu' tests/test_scoreboard.py
  tests/test_generate_ensure.py tests/test_registry_pins.py
  tests/test_backend_registry.py` — 53 passed. Full-tree `ruff check .`
  is RED from pre-existing committed `voyage/workers/augment_worker.py`
  E501/B905 (29 errors, batch-10 SRVGG loader content — out of scope,
  never touched, not caused by this pass); own scope is green.
  DESIGN proposals: none (annotation-only, no behavior change).
- Residuals (exact, post-edit line numbers): `voyage/rpc.py:327-329`
  (`call()` payload/return) + `:281,:284` (`raw_stdout`/`readable`);
  `voyage/supervisor.py:590-598` (`_call_with_restart`), `:1470-1475`
  (`_with_audio_gpu`), `:1694` (`payload`), `:850` (`gauges`),
  `:574` (`_log_metric`), `:1238` (`extra`), plus `dict[str, Any]`
  `:271` (`video_init`), `:293` (`audio_init`), `:330`, `:1045`,
  `:1071`, `:1189`, `:1273`, `:1325`, `:1933`, `:1937`, `:2172`,
  `:2493`, `:2163`; `voyage/scoreboard.py:114,118`
  (`scoreboard_rows` return + `rows`); `voyage/model_registry.py:953,963`
  (`snapshot_kwargs`/`file_kwargs`); `voyage/atomic.py:97,110,118`
  (write/read sides, by design).

## Progress log (2026-09-30, batch 12)

- Pre-flight: `git diff --name-only` (from repo root `ai/`, paths
  `Voyage/...`) showed a clean Voyage tree at pass start — no
  concurrent hunks in `rpc.py`/`scoreboard.py`/`model_registry.py`.
  `supervisor.py` was never touched (forbidden this batch — its sites
  are remainder only, read for evidence, never edited). Mid-pass a
  concurrent agent landed an uncommitted 3-line hunk in
  `voyage/workers/director.py:594-596` (`handle_init` unknown-field
  guard) — unrelated region, left intact; all gate evidence below was
  re-run with it present (tree still green).
- Leg 1, `rpc.py:327-329 call()` → `RpcPayload`/`RpcResult`: BLOCKED,
  probed live then reverted (no edit landed). The probe (signature
  only, docstring untouched) reddens foreign sites — `mypy voyage`
  (67 files) reported 11 errors in 2 files:
  `supervisor.py:610` (`dict(payload)` with `dict[str, object]` →
  `SupportsKeysAndGetItem` mismatch AND `dict[str, JsonValue]`
  return vs `dict[str, object]`), `supervisor.py:1161`
  (`{"texts": texts}` — `list[str]` vs `list[JsonValue]` invariance),
  `supervisor.py:1172` (`float(value)` — return-narrowing propagates
  the `JsonValue` union into an arithmetic site that previously took
  `Any`), plus `cli_observe.py:245,247,572,574,619`
  (`dict[str, JsonValue]` vs `dict[str, object]` invariance at the
  `format_report`/`_persist_benchmark_report` seams) and
  `cli_observe.py:613,614` (`list[str]`/`dict[str, float]` literals
  vs `JsonValue`). Every call site would need an explicit
  `RpcPayload` annotation or cast, incl. forbidden/foreign lines —
  same invariance class as batch 11. Return stays `dict[str, Any]`
  (`response.result` is `dict[str, Any]` via `WorkerRequest`/
  `WorkerResponse` in `voyage/models.py`, out of scope).
  `raw_stdout`/`readable: Any` (`rpc.py:281,284`) stay by
  construction (fd juggling) — untouched.
- Leg 2, supervisor `dict[str, object]` chain: SKIPPED entirely
  (forbidden this batch — read-only evidence: `_call_with_restart`
  `:596/:598`, call site `:610`, `_log_metric` `:574`, `gauges`
  `:850`, `extra` `:1238`, `_with_audio_gpu` `:1473/:1475`,
  `payload` `:1694`, `dict[str, Any]` `:271`/`video_init`,
  `:293`/`audio_init`, `:330`, `:1045`, `:1071`, `:1189`, `:1273`,
  `:1325`, `:1933`, `:1937`, `:2163`, `:2172`, `:2493` — same
  invariant chain as batch 11, still standing).
- Leg 3, `scoreboard.py:114,118 scoreboard_rows` return: LANDED with
  coordinated re-annotation inside the owned file (annotation-only,
  no behavior change). The bare return+`rows` narrowing reddened only
  the owned file — `scoreboard.py:187: error: Argument 1 to "append"
  of "list" has incompatible type "dict[str, list[str] | int |
  dict[str, float] | str | Any | None]"; expected "dict[str,
  JsonValue]" [arg-type]` (no foreign errors — callers in
  `cli_scoreboard.py`/tests accept the narrower return). Fix:
  `from typing import Any` → `from typing import cast` (`:21`),
  `row: dict[str, JsonValue]` + `cast(JsonValue, …)` on the four
  invariant-blocked members (`stages.get(...)`/`current`/`deltas`/
  `errors` — `dict[str, float]`/`list[str]` are never assignable to
  `JsonValue` nesting by invariance). Locals stay `dict[str, float]`
  so the delta arithmetic is untouched; `cast` erases at runtime.
  `metrics`/`transition`/`audio_state` loaders stay `dict[str, Any]`
  (`segment_manifest.py`, out of scope — their `.get()` is `Any`,
  compatible both ways).
- Leg 4, hub `**kwargs` (`model_registry.py:953,963`): BLOCKED,
  probed live then reverted (no edit landed). `dict[str, JsonValue]`
  reddens `:960` (11 errors) + `:971` (9 errors) — the `JsonValue`
  union is too wide for the hub signatures, e.g.
  `Argument 1 to "snapshot_download" has incompatible type
  "**dict[str, JsonValue]"; expected "str" [arg-type]` (× `str`/
  `str | None`/`str | Path | None`/`dict[Any, Any] | str | None`/
  `float`/`bool`/`bool | str | None`/`list[str] | str | None`/`int`/
  `type[Any] | None`/`dict[str, str] | None`), same class at `:971`
  for `hf_hub_download`. `allow_patterns list[str]` already fails
  `list[JsonValue]` invariance on its own. This dict is hub-typed,
  not JSON-shaped — `JsonValue` would be less precise, not more.
  Stays `dict[str, Any]` (Any defeats both checks by design).
- `atomic.py:97,110,118` + `rpc.py:281,284` fd idioms: stay by
  design, untouched (no probe — documented in prior passes).

## Resolution (2026-09-30, batch 12)

- Verdict: PARTIALLY RESOLVED — leg 3 landed (`scoreboard_rows` →
  `JsonValue`-valued); legs 1/4 probed-blocked with live mypy
  evidence above (reverted, no edit); leg 2 remainder (forbidden).
- Files changed: `voyage/scoreboard.py` only (`cast` import,
  return + `rows` + `row` annotations, 4 `cast(JsonValue, …)`).
  `voyage/rpc.py` + `voyage/model_registry.py` probed but reverted
  — no change. No test files changed (annotation-only; TDD n/a —
  gate evidence below instead, per contract). Concurrent
  `workers/director.py` hunk left intact (disjoint region).
- Gate evidence (in-container `voyage:latest`, CPU-only, no host
  pip): `ruff check voyage/cli_observe.py voyage/scoreboard.py
  voyage/model_registry.py voyage/rpc.py` — All checks passed;
  `ruff format --check voyage/cli_observe.py voyage/scoreboard.py`
  — 2 files already formatted; `mypy voyage/scoreboard.py` +
  `mypy voyage` — clean (67 source files, incl. the concurrent
  director hunk); `pytest -p no:cacheprovider -q
  tests/test_scoreboard.py tests/test_cli_split.py
  tests/test_cli_inspect_metrics.py` — 22 passed;
  `tests/test_observability_rank2.py tests/test_cli_scoreboard.py
  tests/test_registry_pins.py tests/test_generate_ensure.py
  tests/test_backend_registry.py` — 68 passed.
  DESIGN proposals: none (annotation-only, no behavior change).
- Residuals (exact, post-edit line numbers): `voyage/rpc.py:327-329`
  (`call()` payload/return) + `:281,:284` (`raw_stdout`/`readable`,
  by design); `voyage/supervisor.py:596/:598` (`_call_with_restart`),
  `:610` (call site), `:1161` (embed dict), `:1172` (float
  knock-on), `:1473/:1475` (`_with_audio_gpu`), `:1694`
  (`payload`), `:574` (`_log_metric`), `:850` (`gauges`), `:1238`
  (`extra`), plus `dict[str, Any]` `:271`, `:293`, `:330`, `:1045`,
  `:1071`, `:1189`, `:1273`, `:1325`, `:1933`, `:1937`, `:2163`,
  `:2172`, `:2493`; `voyage/cli_observe.py:245,247,572,574,613,614,
  619` (call()-narrowing knock-ons — they unblock only with leg 1);
  `voyage/model_registry.py:953,963` (`snapshot_kwargs`/
  `file_kwargs`, probe errors at `:960`/`:971`); `voyage/atomic.py:
  97,110,118` (write/read sides, by design). Scoreboard leg: none
  (landed at `scoreboard.py:114,118,172-176`).

## Progress log (2026-09-30, batch 13 — re-probe blocked legs only)

- Re-probed live (host grep + prior-pass mypy probe evidence — no
  source edits outside owned files; every leg below lives in
  `voyage/*`, forbidden this pass, so record-only):
  leg 1 `voyage/rpc.py:327-329 call()` still `payload:
  dict[str, Any] -> dict[str, Any]` (deferral comment intact;
  `raw_stdout`/`readable: Any` at `:281,:284` stay by construction);
  leg 2 supervisor chain intact (`_log_metric` `:540`,
  `_call_with_restart` `:562/:564`, `gauges` `:816`, `extra`
  `:1204`, `_with_audio_gpu` `:1439/:1441`, `payload` `:1660`,
  plus `dict[str, Any]` `:237`/`video_init`, `:259`/`audio_init`,
  `:296`, `:1011`, `:1037`, `:1155`, `:1239`, `:1291`, `:1899`,
  `:1903`, `:2129`, `:2138`, `:2459` — same invariant chain);
  leg 3 scoreboard already landed (batch 12: `:114,118,168`
  `JsonValue`-valued, `cast` import — no remaining leg);
  leg 4 hub kwargs still `dict[str, Any]`
  (`model_registry.py:953,963`, probe errors at `:960`/`:971` —
  `JsonValue` union too wide for the hub signatures, same class as
  batch 12).
- No leg is unblocked AND in a `tests/` or owned file: the only
  landed leg (scoreboard) is already done, and every still-blocked
  site lives in `voyage/*`. No fix attempted, none owed.

## Resolution (2026-09-30, batch 13)

- Verdict: DEFERRED (record-only) — all legs residual with live
  evidence above. Files changed: none for 035 (this issue file
  only). Gate evidence: n/a (no change). DESIGN proposals: none.
- Residuals (exact, post-probe line numbers): `voyage/rpc.py:327-329`
  + `:281,:284`; `voyage/supervisor.py:540,562/564,816,1204,1439/
  1441,1660` + `dict[str, Any]` `:237,:259,:296,:1011,:1037,:1155,:
  1239,:1291,:1899,:1903,:2129,:2138,:2459`; `voyage/model_registry.py:
  953,963`; `voyage/atomic.py:97,110,118` (by design).

## Progress log (2026-10-01, verification-only pass — blocked-leg re-probe)

- Read-only probes (host grep + live file reads, no source edits for
  035): comparable `Any` total in `voyage/*.py voyage/workers/*.py` is
  now **492** (was 387 — growth is new files since: `augment_worker.py`
  60, `video_causvid.py` 39, `video_ltxv.py` 33, `segment_manifest.py`
  21, `supervisor.py` 20, `workers/director.py` 20; the 079 delete took
  out `video_longlive.py`'s 32). ANN probe in-container: **360**
  (ANN401 345 + ANN001 11 + ANN202 4; was 339).
- Landed narrowings all HOLD (grep-verified): `scoreboard.py` zero code
  `Any` (docstring mention only), `models_ensure.py` zero code `Any`,
  `registry_records.py` zero `Any`, `model_registry.py` down to the
  import + `snapshot_kwargs`/`file_kwargs` (`:879,:889`).
- Blocked legs re-verified still-blocked (sites unchanged since the last
  live mypy probes — no edit attempted): leg 1 `rpc.py:328 call()`
  still `payload: dict[str, Any]` (deferral comment intact); the
  supervisor traps stand (`supervisor.py:570` `dict(payload)`, `:1121`
  `{"texts": texts}`, `:1132` `float(value)`); leg 2 supervisor
  `dict[str, object]` chain intact (`:534,556/558,810,1198,1233,1285,
  1433/1435,1466,1654` + `dict[str, Any]` `:239,:253,:290,:1005,:1031,
  :1149,:1893,:2123,:2132,:2453`); leg 4 hub kwargs still
  `dict[str, Any]` (`model_registry.py:879,889` — `JsonValue` union too
  wide for the hub signatures, batch-12 probe evidence stands). The 079
  delete touched `supervisor.py`/`model_registry.py` only at the
  longlive surface (module map, init branch, spec row, WAN pins) —
  disjoint from every chain site above.
- No annotation-only narrowing is greenable: the only remaining
  `dict[str, Any]` in quiet files are the three proven-blocked legs.

## Resolution (2026-10-01, verification-only pass)

- Verdict: DEFERRED (record-only) — all legs residual with live evidence
  above. Files changed: none for 035 (this issue file only). Gate
  evidence: n/a (no change). DESIGN proposals: none.
- Residuals (unchanged): `voyage/rpc.py:328` (`call()` payload/return)
  + `:281,:284` (fd juggling, by design); supervisor chain (lines
  above); `voyage/model_registry.py:879,889` (hub kwargs);
  `voyage/atomic.py:97,110,118` (by design).

## Progress log (2026-10-01, re-probe pass — legs re-verified, one new site probed then reverted)

- Pre-flight: `git diff --name-only HEAD -- comfy/Voyage/voyage/
  comfy/Voyage/tests/` EMPTY (only pre-existing `comfy/Voyage/LTX2.md`
  modified at repo root, out of scope). `voyage/supervisor.py` never
  touched (forbidden — its sites are remainder only, read for
  evidence, never edited).
- Baseline live in-container (`voyage:latest`, CPU-only, no host pip):
  `mypy voyage` → clean (73 source files). `rg -n "Any" voyage/
  --glob '*.py'` → 386 lines (top: `augment_worker` 60,
  `video_causvid` 39, `video_ltxv` 33, `segment_manifest` 21,
  `supervisor` 20, `workers/director` 20); landed narrowings all HOLD
  (grep-verified): `scoreboard.py` zero code `Any`, `models_ensure.py`
  zero code `Any`, `registry_records.py` zero `Any`,
  `model_registry.py` down to the `Any` import + `snapshot_kwargs`/
  `file_kwargs` (`:879,:889`).
- Blocked legs re-verified still-blocked (sites read live, no edits):
  leg 1 `rpc.py:328-329 call()` still `payload: dict[str, Any] ->
  dict[str, Any]` (deferral comment `:348-349` intact;
  `raw_stdout`/`readable: Any` at `:281,:284` stay by construction);
  leg 2 supervisor `dict[str, object]` chain intact (`:537`
  `_log_metric`, `:559/:561` `_call_with_restart`, `:784` gauges,
  `:1172` `extra`, `:1407/:1409` `_with_audio_gpu`, `:1628`
  `payload`, plus `dict[str, Any]` `:242`/`video_init`,
  `:256`/`audio_init`, `:293`, `:979`, `:1005`, `:1123`, `:1207`,
  `:1259`, `:1867/:1871`, `:2097`, and `state: Any` `:1117`,`:1201`);
  leg 4 hub kwargs still `dict[str, Any]` (`model_registry.py:879,889`
  — batch-12 probe evidence stands: the `JsonValue` union is too
  wide for the hub signatures); `atomic.py:97,110,118` stay by design
  (docstring records the tried + reverted widening).
- Newly-unblocked candidate probed live, then REVERTED (no edit
  landed): `voyage/bench.py:16 _finite_float(value: Any)` → `(value:
  JsonValue)` (same idiom as the landed `scoreboard.py` precedent;
  callers in `summarize_gauges` pass `Any` from `dict[str, Any]`
  events). In-container `mypy voyage/bench.py` on the probe →
  **2 errors** (`:172,:177: Argument 1 to "_finite_float" has
  incompatible type "object"; expected "JsonValue"`): the
  `summarize_sfx_windows(windows: list[dict[str, object]])` callers
  pass `record.get(...)` (`object | None`), the same `dict[str,
  object]`-chain invariance class as batch-11/12. Narrowing the
  callers cascades into the CLI collectors (foreign `dict[str,
  object]` sites) — coordinated surgery, out of scope. Probe
  reverted; post-revert `mypy` + `ruff check` + `ruff format --check`
  on `voyage/bench.py` all green.
- Verdict: no annotation-only narrowing is greenable — the only
  remaining `dict[str, Any]` in quiet files are the three
  proven-blocked legs (plus the `dict[str, object]` chain that just
  blocked the bench candidate from the other side).

## Resolution (2026-10-01, re-probe pass)

- Verdict: DEFERRED (record-only) — all legs residual with live
  evidence above; bench candidate probed-blocked and reverted.
  Files changed: none for 035 (this issue file only; `voyage/bench.py`
  probed then restored byte-identical — `git diff` clean).
  Gate evidence: baseline `mypy voyage` clean (73 files);
  post-revert per-file gates on `voyage/bench.py` green.
  DESIGN proposals: none (annotation-only scope, no behavior change).
- Residuals (exact, as-read line numbers): `voyage/rpc.py:328-329`
  (`call()` payload/return) + `:281,:284` (fd juggling, by design);
  supervisor chain (`:537,:559/:561,:784,:1172,:1407/:1409,:1628` +
  `dict[str, Any]` `:242,:256,:293,:979,:1005,:1123,:1207,:1259,:
  1867/:1871,:2097` + `state: Any` `:1117,:1201`);
  `voyage/model_registry.py:879,889` (`snapshot_kwargs`/`file_kwargs`);
  `voyage/atomic.py:97,110,118` (by design); `voyage/bench.py:16`
  (`_finite_float` — newly probed, blocked by the
  `summarize_sfx_windows list[dict[str, object]]` callers).

## Progress log (2026-10-01, record-only maintenance pass — blocked legs re-probed, no narrowing)

- Pre-flight: `git diff --name-only` shows only banned/foreign dirty
  (`Voyage/DESIGN.md`, `Voyage/LTX2.md`, `Voyage/tests/test_stage_a_telemetry.py`,
  `Voyage/voyage/supervisor.py`) plus untracked
  `Voyage/tests/test_commit_slice_compensation.py` (other group's).
  `voyage/supervisor.py` never touched (forbidden — its sites are remainder
  only, read for evidence, never edited). `voyage/rpc.py`,
  `voyage/model_registry.py`, `voyage/scoreboard.py`, `voyage/bench.py`
  all clean.
- Baseline live in-container (`voyage:latest`, CPU-only, no host pip):
  `mypy voyage` → clean (75 source files). Host `grep -rn Any`:
  377 lines (top: `augment_worker` 59, `video_causvid` 39, `video_ltxv` 33,
  `segment_manifest` 21, `workers/director` 20, `supervisor` 17, `cli` 16).
  `ruff check --select ANN --statistics` → **360**
  (ANN401 345 + ANN001 11 + ANN202 4, unchanged).
- Landed narrowings all HOLD (grep-verified): `scoreboard.py` zero code
  `Any` (docstring `:45` only), `models_ensure.py` zero code `Any`
  (docstring `:160` only), `registry_records.py` zero `Any`,
  `model_registry.py` down to the `Any` import (`:15`) +
  `snapshot_kwargs`/`file_kwargs` (`:879,:889`).
- Blocked legs re-verified still-blocked (sites read live, no edits):
  leg 1 `rpc.py:327-329 call()` still `payload: dict[str, Any] ->
  dict[str, Any]` (deferral comment intact; `raw_stdout`/`readable: Any`
  at `:281,:284` stay by construction); leg 2 supervisor
  `dict[str, object]` chain intact (`:539` `_log_metric`, `:561/:563`
  `_call_with_restart`, `:786` gauges, `:1174` `extra`,
  `:1409/:1411` `_with_audio_gpu`, `:1630` `payload`, plus
  `dict[str, Any]` `:244`/`video_init`, `:258`/`audio_init`, `:295`,
  `:981`, `:1007`, `:1119`, `:1125`, `:1209`, `:1261`, `:1909/:1913`,
  `:2139`, `:2148`, `:2469`, and `state: Any` `:1119`,`:1203`);
  leg 4 hub kwargs still `dict[str, Any]` (`model_registry.py:879,889`
  — batch-12 probe evidence stands: the `JsonValue` union is too wide
  for the hub signatures); `atomic.py:97,110,118` stay by design
  (docstring records the tried + reverted widening); `bench.py:16`
  `_finite_float(value: Any)` stays (prior-pass probe proved the
  `summarize_sfx_windows list[dict[str, object]]` callers block it —
  no new probe needed, no edit).
- Verdict: no annotation-only narrowing is greenable — the only
  remaining `dict[str, Any]` in quiet files are the three proven-blocked
  legs (plus the `dict[str, object]` chain that blocks the bench
  candidate from the other side).

## Resolution (2026-10-01, record-only maintenance pass)

- Verdict: DEFERRED (record-only) — all legs residual with live evidence
  above. Files changed: none for 035 (this issue file only).
  Gate evidence: baseline `mypy voyage` clean (75 files).
  DESIGN proposals: none (annotation-only scope, no behavior change).
- Residuals (exact, as-read line numbers): `voyage/rpc.py:327-329`
  (`call()` payload/return) + `:281,:284` (fd juggling, by design);
  supervisor chain (`:539,:561/:563,:786,:1174,:1409/:1411,:1630` +
  `dict[str, Any]` `:244,:258,:295,:981,:1007,:1119,:1125,:1209,:1261,:
  1909/:1913,:2139,:2148,:2469` + `state: Any` `:1119,:1203`);
  `voyage/model_registry.py:879,889` (`snapshot_kwargs`/`file_kwargs`);
  `voyage/atomic.py:97,110,118` (by design); `voyage/bench.py:16`
  (`_finite_float` — blocked by the `summarize_sfx_windows`
  `list[dict[str, object]]` callers).

## Progress log (2026-10-01, joint call()+supervisor pass — LANDED)

- Pre-flight: `git status` (Voyage dir) shows extensive foreign/concurrent
  dirty (DESIGN/LTX2/README/docs, issues/081/166, scripts/gates+run,
  tests/*, voyage/augment+cli+cli_models+media+model_registry+
  models_ensure+registry_records+sfx_finalize) plus untracked
  build-ltx/registry_ltx23+25/test_issue_152+166. Own targets
  (`voyage/rpc.py`, `voyage/supervisor.py`, `voyage/cli_observe.py`,
  `voyage/bench.py`) were clean at pass start — own hunks below are
  disjoint from every foreign hunk; foreign regions never touched,
  never undid foreign work. `model_registry.py`/`models_ensure.py`
  LTX hunks left intact (hub-kwargs leg stays blocked, not attempted).
- Baseline live in-container (`voyage:latest`, CPU-only, no host pip):
  `mypy voyage` clean (81 files); per-file `mypy`+`ruff check` clean
  on all 4 targets.
- Step 1, `rpc.py call()` → `RpcPayload`/`RpcResult` (signature only):
  `mypy voyage` → 11 errors in 2 files (same invariance class as
  batch-12, line numbers shifted): `supervisor.py:533` (`dict(payload)`
  `dict[str,object]` → `SupportsKeysAndGetItem` + `dict[str,JsonValue]`
  return vs `dict[str,object]`), `:1055` (`{"texts": texts}`
  `list[str]` vs `list[JsonValue]`), `:1066` (`float(value)` JsonValue
  union into arithmetic), `cli_observe.py:243,245,570,572,617`
  (`dict[str,JsonValue]` vs `dict[str,object]` at
  `format_report`/`_persist_benchmark_report`), `:611,612`
  (`list[str]`/`dict[str,float]` literals vs `JsonValue`).
- Step 2, coordinated bridge (ONE pass, `mypy voyage` after each edit):
  `supervisor.py:29` `cast` import; `:533-536` `_call_with_restart`
  cast-bridge (`cast(dict[str,JsonValue], dict(payload))` in,
  `cast(dict[str,object], ...)` out — producers/consumers stay
  `dict[str,object]`, only the RPC seam casts); `:1058-1061` embed
  payload `cast(dict[str,JsonValue], {"texts": texts})`; `:1073-1078`
  `float(value)` → explicit `bool`/`(int,float)` guard (same `None`
  outcome for hostile vectors, no contract change). Probe: 11→9→7
  (supervisor errors gone, 7 `cli_observe` left).
- Step 3, `cli_observe` knock-ons: `bench.py:10` `Mapping` import,
  `:75-76`/`91-92` `format_report`/`report_document` params
  `dict[str,object]` → `Mapping[str,object]` (covariant — both
  `dict[str,object]` and `dict[str,JsonValue]` pass; bodies only
  `.items()`/`dict()`, no behavior change);
  `cli_observe.py:15` `Mapping` import, `:150-151`
  `_persist_benchmark_report` → `Mapping`; `:602-618` end-to-end
  `setup`/`metrics` renamed `e2e_setup`/`e2e_metrics` with explicit
  `dict[str,object]` (fixes `no-redef` vs the benchmark-branch
  `metrics: RpcResult` + forces object-join instead of JsonValue
  inference). Probe: 7→2 (`:612,613` literal pair) → 1 (`no-redef`)
  → 0. Full `mypy voyage` clean (81 files). One transient
  `sfx_finalize.py:519,520` `_blend_*` red mid-pass was a concurrent
  half-edit (import already moved to `_join_audio_single_graph`);
  re-probe after their hunk landed: clean — not caused here (stash
  probe: without own 4 files `mypy voyage` clean; with them clean).
- `rpc.py:346-350` docstring updated (was stale `stays dict[str,Any]`);
  `ruff format` reflowed `call()` to one line (`:327`).
- No TDD (annotation + `Mapping`-covariance + explicit-guard refactor,
  no behavior change — gate evidence below per contract).

## Resolution (2026-10-01, joint call()+supervisor pass)

- Verdict: RESOLVED (joint leg landed) — `call()` is now
  `RpcPayload → RpcResult`; the supervisor `dict[str,object]` chain
  bridges at the seam via `cast` (no producer/consumer re-annotation);
  `bench`/`cli_observe` reporting takes `Mapping[str,object]`
  (covariant, accepts both sides); embed `float()` is explicitly
  narrowed; end-to-end metrics are `dict[str,object]`-annotated.
- Files changed: `voyage/rpc.py` (signature + docstring + format),
  `voyage/supervisor.py` (`cast` import + `_call_with_restart` bridge +
  embed cast + value guard), `voyage/bench.py` (`Mapping` import +
  2 signatures), `voyage/cli_observe.py` (`Mapping` import +
  `_persist` signature + e2e rename/annotate). No test files changed.
  Foreign hunks left intact (disjoint regions).
- Gate evidence (in-container `voyage:latest`, CPU-only, no host pip):
  per-file `ruff check` + `ruff format --check` + `mypy` clean on all
  4 files; full `mypy voyage` clean (81 files); full `ruff check .`
  RED from 2 pre-existing foreign errors only
  (`tests/test_ltxv_tensor_handoff.py:11` F401 unused `Any`,
  `voyage/media.py:740` E501 101>100 — neither touched nor caused
  here); `pytest -p no:cacheprovider -q` 70 passed
  (`test_scoreboard`+`test_generate_ensure`+`test_registry_pins`+
  `test_backend_registry`+`test_cli_split`+
  `test_cli_inspect_metrics`) + 46 passed (`test_benchmark`+
  `test_benchmark_counts`+`test_cli_benchmark_sfx_augment`+
  `test_e2_bench_sfx_augment_154_163`+`test_observability`+
  `test_cli_scoreboard`). DESIGN proposals: none (annotation-only).
- Residuals (exact, post-edit): `voyage/rpc.py:281,284`
  (`raw_stdout`/`readable`, by construction); supervisor
  `dict[str,object]`/`dict[str,Any]` chain stays object-typed by
  design (bridge via `cast` — full `JsonValue` re-annotation of every
  producer/consumer remains open but unneeded while green);
  `voyage/model_registry.py:1149,1159` (`snapshot_kwargs`/
  `file_kwargs`, batch-12 probe stands); `voyage/atomic.py:97,110,118`
  (by design); `voyage/bench.py:16` (`_finite_float`, still blocked by
  the `summarize_sfx_windows list[dict[str,object]]` callers).

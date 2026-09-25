# 022 — Stringly-typed backends: `backend: str` + scattered `==` + four disagreeing registries

- Status: resolved (Literals + single registry landed 2026-09-25; cross-file annotation adoption + supervisor worker-module merge are hook-noted below for the owning tracks)
- Severity: major (type safety — typo fails at runtime; lists already disagree)
- Area: structure — config/supervisor/CLI/TUI backend naming
- Rank rationale: only `quantization` got a `Literal`; the wire-critical field
  didn't. TUI's GPU warning already omits `causvid` while the CLI treats it as CUDA.

## Technical description

```python
# config.py:25, supervisor.py:94, backends.py:180
backend: str = "fake"
STREAMING_VIDEO_BACKENDS = ("longlive2", "ltxv", "causvid")
VIDEO_WORKER_MODULES = {"fake": ..., "longlive2": ..., "ltxv": ..., "causvid": ...}
# supervisor.py:146,733,1206; cli.py:719,721,724; tui_state.py:242,272
if config.video.backend == "longlive2": ...
```

Four parallel registries (`VIDEO_WORKER_MODULES`, `AUDIO_WORKER_MODULES`,
`_VIDEO_BACKEND_PRESETS`, `_AUDIO_BACKEND_PRESETS`, `BACKEND_STATE_MODES`) keyed
by bare strings, each with its own `try/except KeyError → "unknown backend"`.
`rg -n 'backend ==|backend in' Voyage/voyage` → ~20 scattered sites; compare
`tui_state.py:391-398` (omits causvid) vs `cli.py:734`
(`_CUDA_BACKENDS = frozenset({"ltxv","longlive2","causvid","acestep"})`).

## Why this is an issue

A bare `backend: str` with some twenty scattered equality checks and four
parallel registries means a typo fails at runtime — after GPU init — instead
of at typecheck, and the registries already disagree in shipped code: the TUI
shows no CUDA warning for `causvid` while the CLI treats it as a CUDA backend.
Backend selection drives geometry, device, audio pairing, and resume profile,
so a late failure wastes the most expensive minutes on the card. Centralizing
on one literal type plus one registry table turns the whole class into a
static error. Operators with a typo and every reader tracing backend dispatch
pay for the scattered strings today.

## Evidence

Sweep `rg` output (~20 sites); live `gpu_warning('causvid') == ''` probe
(orchestrator, 2026-09-25 — see 024).

## Reproduction

`rg -n 'backend ==|backend in' Voyage/voyage -g '*.py'`; pass
`backend="longlvie2"` (typo) and watch it fail at runtime, not typecheck.

## Source references

- `voyage/config.py:25,69,144,483-507`; `voyage/supervisor.py:86-94,109-124,141,
  146,163,733,1206`; `voyage/backends.py:68,185`;
  `voyage/cli.py:719-726,734`; `voyage/tui_state.py:242,272,391-393`;
  `voyage/workers/director.py:294`.

## Resolution candidates

`VideoBackendName = Literal["fake","longlive2","ltxv","causvid"]` (+
`AudioBackendName`) in one module; type all `backend:` fields, preset dicts, and
`STREAMING_VIDEO_BACKENDS` with it; single `BACKEND_REGISTRY` table replacing the
4 dicts + 4 unknown-backend errors.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- Open: introduce Literals + registry; mypy then enforces.
- 2026-09-25 (repair pass): added `## Why this is an issue`;
  `gpu_warning('causvid')==''` re-probed live (still omitted); refs verified
  current.
- 2026-09-25 (resolution, backend-typing track): FIXED in own scope.
  `voyage/config.py`: `VideoBackendName = Literal["fake","longlive2","ltxv",
  "causvid"]` + `AudioBackendName = Literal["fake","acestep"]` + `StateMode`
  (moved from backends.py; backends re-exports it) in one module;
  `BackendRecord` frozen dataclass + `BACKEND_REGISTRY`
  (`dict[VideoBackendName, BackendRecord]`: profile/geometry/latent/device
  + audio pairing + state_mode + streaming per row) replacing
  `_VIDEO_BACKEND_PRESETS` / `_AUDIO_BACKEND_PRESETS` / `BACKEND_STATE_MODES`
  as sources (kept as derived views for existing importers); typed
  `VideoConfig.backend: VideoBackendName`, `AudioConfig.backend:
  AudioBackendName`, `resolve_config/with_video_backend/default_config_toml`
  backend params, `BACKEND_STATE_MODES: dict[VideoBackendName, StateMode]`
  and `_STREAMING_BACKENDS: frozenset[VideoBackendName]` (both derived from
  the registry) in `voyage/backends.py`, plus `VideoBackendAdapter.backend`
  / `BackendCapabilities.backend` / `VideoSegmentResult.backend`. Boundary
  functions taking unchecked external strings (`_video_preset`,
  `_audio_preset`) stay `str` with runtime ValueError on purpose (their
  callers live in other tracks' files). Tests:
  `tests/test_backend_registry.py` (13 tests: vocabularies, registry
  coverage, defaults==fake row, state/streaming derivation, per-row
  dispatch, TOML geometry per backend, rejection messages, purity).
  Gates: ruff check + format clean on all touched files; `mypy
  voyage/config.py voyage/backends.py` clean; 238 targeted tests green
  (new 3 files + backends_adapter/draft/generate/generation_stack/
  tui_state/unit/cli_hardening); full suite 674 passed / 4 failed, all 4
  in other tracks' files (no config/backends frame in their tracebacks).
  Full `gates.sh` is red from concurrent in-flight work (supervisor.py
  ruff F401 + ~80 mypy name-defined from the supervisor rewire; 4 failing
  tests in new worker/tui files) — none in this track's scope.
- Hook notes for owning tracks (do NOT belong to this change):
  supervisor.py — type `VIDEO_WORKER_MODULES: dict[VideoBackendName, str]`
  (:89), `STREAMING_VIDEO_BACKENDS: tuple[VideoBackendName, ...]` (:97),
  `AUDIO_WORKER_MODULES: dict[AudioBackendName, str]` (:105),
  `audio_worker_module/video_worker_module(backend: …)` (:136/:145), and
  consider deriving all three from `config.BACKEND_REGISTRY` (worker-module
  strings were deliberately left out of the registry — supervisor-owned
  data); the `== "longlive2"` (:173, :1491) / `== "acestep"` (:190, :885)
  / `in STREAMING_VIDEO_BACKENDS` (:168, :886, :1337) sites keep working
  (Literal == str is runtime-true) and need no change for correctness.
  cli.py — `--backend choices=(…)` (:1321-1322, :1411-1412) can derive from
  `typing.get_args(VideoBackendName)`; `_CUDA_BACKENDS` (:888) stays the
  CUDA set (note: `gpu_warning('causvid')` is covered — tui_state delegates
  to `_CUDA_BACKENDS`, which already contains causvid); `_frames_per_segment`
  `== "ltxv"/"longlive2"/"causvid"` (:873-878) need no change; annotate
  `cmd_init`'s `backend` (`:160`, currently `Any | str`) as
  `VideoBackendName` to clear the one mypy arg-type error this change
  surfaces there. tui_state.py — annotate `_planning_frames_and_fps(backend:
  …)` (:248) as `VideoBackendName` to clear the one mypy arg-type error at
  `VideoConfig(backend=backend)` (:271); `gpu_warning` (:453) needs nothing
  (already delegates to cli._CUDA_BACKENDS). workers/director.py — `if
  backend == "qwen"` (:321) is director vocabulary, out of scope.
  model_registry.py — comment at :170 references the supervisor dicts;
  update when the supervisor merges them into the registry.

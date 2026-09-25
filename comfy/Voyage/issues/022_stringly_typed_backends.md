# 022 — Stringly-typed backends: `backend: str` + scattered `==` + four disagreeing registries

- Status: open
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

# 021 — `_CUDA_BACKENDS` is polluted and incomplete: impossible `acestep`, missing `mmaudio` → CUDA fast-fail blind to SFX, TUI warning dead/noisy

- Severity: MEDIUM (preflight correctness: false positive + false negative in the same set)
- Group: config/preflight — Rank: 3/5
- File:line: `voyage/cli.py:1174` (`_CUDA_BACKENDS`), `voyage/cli.py:1193-1205` (`_cuda_offenders`), `voyage/cli.py:1207-1220` (`_require_cuda_stack`)

## Description

`acestep` is an *audio* backend (`AudioBackendName`), never a valid `VideoBackendName`. It occupies a slot in a set used for video-backend checks (`cmd_generate:1206` tests `args.backend in _CUDA_BACKENDS`, where `args.backend` is always a video backend) and TUI warnings. Conversely `mmaudio` (the SFX CUDA backend, `SfxBackendName`) is absent, and `_cuda_offenders`/`_require_cuda_stack` only inspect `video.backend` + `audio.backend` — never `sfx.backend`. A run with `video=fake, audio=fake, sfx=mmaudio` on a torch-less image passes the fast-fail and dies late inside the SFX worker.

## Rationale

- A registry set that mixes vocabularies cannot be the single source of truth; every consumer inherits both a false positive (`acestep` warning for a backend the TUI never offers — `tui_state.BACKENDS = ("ltxv","longlive2","causvid","fake")`) and a false negative (`mmaudio` silence on a real CUDA need).
- The docstring on `gpu_warning` claims single-sourcing as the design; the set itself violates it.
- Late failure inside the SFX worker wastes the whole video+audio render before surfacing a condition knowable at preflight.

## Live evidence (read + grep, 2026-09-30)

`voyage/cli.py:1135`:

```python
_CUDA_BACKENDS = frozenset({"ltxv", "longlive2", "causvid", "acestep"})
```

`voyage/cli.py:1154-1163`:

```python
def _cuda_offenders(config: ProjectConfig) -> list[str]:
    offenders: list[str] = []
    if config.video.backend in _CUDA_BACKENDS:
        offenders.append(f"video {config.video.backend!r}")
    if config.audio.backend in _CUDA_BACKENDS:
        offenders.append(f"audio {config.audio.backend!r}")
    return offenders
```

No `sfx` branch exists (verified: `grep -n "sfx" voyage/cli.py` in the 1154-1190 window returns nothing).

Sweep probe (preserved Track B result, in-container):

```
_CUDA_BACKENDS: ['acestep', 'causvid', 'longlive2', 'ltxv']
video backends: ['causvid', 'fake', 'longlive2', 'ltxv']
mmaudio cuda-checked? False
gpu_warning('acestep'): 'acestep needs the CUDA worker image …'   # dead: TUI BACKENDS never contain acestep
gpu_warning('mmaudio'): ''                                        # silent: real CUDA need, no warning
```

Live grep (2026-09-30):

```
$ grep -n "_CUDA_BACKENDS" voyage/cli.py voyage/tui_state.py
voyage/cli.py:1135:_CUDA_BACKENDS = frozenset(...)
voyage/cli.py:1161:    if config.video.backend in _CUDA_BACKENDS:
voyage/cli.py:1163:    if config.audio.backend in _CUDA_BACKENDS:
voyage/cli.py:1178:    needs_cuda = config.video.backend in _CUDA_BACKENDS or config.audio.backend in _CUDA_BACKENDS
voyage/cli.py:1206:    if args.backend in _CUDA_BACKENDS and not _torch_available():
voyage/tui_state.py:545:    from voyage.cli import _CUDA_BACKENDS
voyage/tui_state.py:547:    if backend in _CUDA_BACKENDS:
```

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.cli import _CUDA_BACKENDS
from voyage import tui_state
print(sorted(_CUDA_BACKENDS))
print('mmaudio warned?', bool(tui_state.gpu_warning('mmaudio')))
print('acestep warned?', bool(tui_state.gpu_warning('acestep')))"
# Buggy: acestep True (dead), mmaudio False (silent)
grep -n "_CUDA_BACKENDS" voyage/cli.py voyage/tui_state.py
```

## Fix candidates

1. Derive the CUDA set from `BACKEND_REGISTRY` (e.g. `device.startswith("cuda")` per row for video/audio/sfx devices) instead of hand-maintaining it.
2. Add an explicit `SFX_CUDA = {"mmaudio"}` (or extend the derived set to cover the sfx device column) and extend `_cuda_offenders` with a `sfx.backend` branch.
3. Change `gpu_warning` to take the resolved triple (video/audio/sfx), not a bare video string — a bare string cannot express the `fake/fake/mmaudio` case.
4. Add a test: `fake/fake/mmaudio` without torch fast-fails naming `sfx 'mmaudio'`; `acestep` is not treated as a video backend anywhere.

## Refs (with links/quotes)

- "Encode that order in one place … with no `if ENV == …` branching anywhere." Same applies to backend sets — one registry, derived views. — https://python-config-secrets-hub.com/core-configuration-patterns-file-formats/configuration-precedence-rules/
- "A single source of truth for configuration values … A clear and logical precedence." — https://cobra.dev/docs/tutorials/12-factor-app/

# 081 — Split `supervisor.py` god module (1894L, commit + lifecycle + audio-cover)

- Severity: HIGH (structure)
- File: `voyage/supervisor.py:1` (1902 lines, 45 fns — sweep said 1894)
- Area: structure — supervisor decomposition

## Description

`supervisor.py` carries lifecycle state machine + transactional segment commit + audio coverage + video render + director accept in one file. Load-bearing giants: `_ensure_audio_coverage` 150L, `_commit_segment` 137L, `_accept_director_decision` 135L, `_render_video` 117L, `_propose_segment` + `run_segments` ownership. Side seams: `sha256_file as sha256_file` re-export shim (`:50`, import `hashing` directly), `STREAMING_VIDEO_BACKENDS` hand literal (`:96`) vs registry-derived `backends._STREAMING_BACKENDS` (`backends.py:82-88`, ownership open, issue 023), `VIDEO/AUDIO_WORKER_MODULES` maps (`:88-108`) with 3 near-identical unknown-backend errors, flat deterministic-backend payload compat block (`:859`), `LEGACY_MIGRATION_REMOVE_AFTER` threading (`:1460`), `run_id` legacy (`:469`).

## Rationale

Same §12 split-signal case as cli (4x over). Highest concurrent-edit collision file (§9 warning). Commit-pipeline reviewers must load lifecycle + audio + render context to review one path.

## Live evidence

- `wc -l voyage/supervisor.py` → 1902; `grep -n "^def \|^class " voyage/supervisor.py` → 45 fns
- `:88-108` worker-module maps + streaming tuple; `:242,595,605,666` bare `except Exception` samples (sweep cited `:373,570,599,1331` — drifted)
- `:859` flat deterministic payload comment; `:1460` legacy_path threading
- `backends.py:82-88` docstring: "supervisor track owns unifying this with supervisor.STREAMING_VIDEO_BACKENDS"

## Repro

```bash
wc -l voyage/supervisor.py; grep -n "def _ensure_audio_coverage\|def _commit_segment\|def _accept\|def _render_video\|STREAMING_VIDEO_BACKENDS\|VIDEO_WORKER_MODULES" voyage/supervisor.py
```

## Fix candidates

1. Extract commit pipeline (`commit.py`: propose/render/cover/commit/accept) vs lifecycle (`run_segments`/state/locks) vs audio-coverage; keep `supervisor.py` as facade re-exporting for one release, then cut over.
2. Delete `sha256_file` re-export; derive streaming set from `BACKEND_REGISTRY` (see 083); merge 3 worker-module maps into one table or registry derivation.
3. Gate: `gates.sh` green + crash-matrix/commit/integration suites (`test_crash_matrix`, `test_commit_hardening`, `test_commit_split`, `test_integration`) green.

## Refs

- Issues 023 (streaming unification owner), 025, 036; `voyage/backends.py:74-89`, `voyage/config.py:88-184`

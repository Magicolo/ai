# 090 — Hardcoded `/tmp` placeholder paths that never touch the filesystem (missing-file path untested)

- Status: open
- Severity: low (string-only paths; real loader's missing-file error has zero
  coverage)
- Area: tests + director loader — `test_rhythm.py:87,101`
  (`path="/tmp/x.wav"`), `test_inspector.py:35,43,47` (`"/tmp/frame.png"`),
  `test_longlive_stages.py:323` (`"/tmp/x/recovery.pt"`)
- Rank rationale: pass-2 finding; previously unmentioned.

## Technical description

All three are string-only: the files never exist and no test asserts behavior
when they don't. For the inspector this matters — `handle_inspect`
(`voyage/workers/director.py:159-178`) `checked_request`s the payload then
forwards `frame_path` straight to `_inspector_generate` with no existence check,
so the real loader's missing-file error path has zero coverage (the mocks bypass
it).

## Why this is an issue

String-only placeholder paths mean the real loader's missing-file error path
has zero coverage: the inspector mocks bypass exactly the boundary that needs
testing, so a regression in missing-file handling (wrong exception, hang,
silent skip) ships with green gates. The three tests assert behavior with
files that never exist, pinning nothing about the filesystem contract.

## Evidence

Placeholder paths verified live (re-run 2026-09-25 — none of the files exist):

```
$ rg -n "/tmp" Voyage/tests/test_rhythm.py Voyage/tests/test_inspector.py Voyage/tests/test_longlive_stages.py
Voyage/tests/test_rhythm.py:87:        path="/tmp/x.wav",
Voyage/tests/test_rhythm.py:101:        path="/tmp/x.wav",
Voyage/tests/test_longlive_stages.py:323:        video: dict[str, Any] = {"frames": 29, "recovery_path": "/tmp/x/recovery.pt"}
Voyage/tests/test_inspector.py:35:    result = director_worker.handle_inspect({"frame_path": "/tmp/frame.png"})
Voyage/tests/test_inspector.py:43:        assert frame_path == "/tmp/frame.png"
Voyage/tests/test_inspector.py:47:    result = director_worker.handle_inspect({"frame_path": "/tmp/frame.png"})
```

`handle_inspect` (`director.py:159-178`) `checked_request`s the payload then
forwards `frame_path` straight to `_inspector_generate` with no existence
check — the mocks bypass the real loader's missing-file path entirely.

## Reproduction

`handle_inspect({"frame_path": "/nonexistent/frame.png"})` against the real
(unmocked) loader path.

## Source references

- Files/lines above.

## Resolution candidates

Use `tmp_path`-backed files in the three tests and add one missing-file test for
`handle_inspect` (plus an existence check at the worker boundary).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (all `/tmp` hits match); re-ran `rg` (pasted above).
- Open: fix tests + boundary check.

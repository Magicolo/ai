# 247 — `VideoBackendAdapter.generate_segment` silently substitutes request values for bad worker reports

Severity: MEDIUM (track B-11).

## Technical description

Non-`int`/non-positive `frames`, `novel_frames`, `fps` fall back to
`requested`/`returned`/`request.fps` with no error or metric. Only
`conditioning_frames >= 0` passes through. A worker returning `frames: 0`, `frames:
"many"`, or a missing `video` block is indistinguishable from a healthy render in the
normalized result.

## Rationale

The adapter is the trust boundary (issue 023: "worker-reported frames win"). Silent
substitution converts worker bugs into timeline drift the supervisor then commits as
truth.

## Live evidence

```
reported_frames = video_block.get("frames")                              # voyage/backends.py:394
if isinstance(reported_frames, int) and reported_frames > 0: returned = reported_frames
reported_novel ... if isinstance(...) and ... > 0: novel = ...           # :400-402
reported_fps ... if isinstance(...) and ... > 0: native_fps = ...        # :403-405
# else: request values stand in, no raise/log
```

Repro: transport returning `{"video": {"frames": 0}}` →
`VideoSegmentResult(returned_frames==requested_frames)`; `{"video": {}}` → same; neither
raises `ConfigurationError`.

## Source refs

`voyage/backends.py:371-415`.

## Online sources

- None (in-tree issue-023 adapter contract is the anchor).

## Fix candidates

- Strict-validate worker frame/fps reports (`>0 int` else
  `ConfigurationError`/`WorkerError`); keep a lenient preview path only behind an
  explicit flag; test the four malformed shapes.

## Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.

## Evaluation (2026-10-07)
- Re-read `voyage/backends.py:371-415` live: the three silent-substitution
  branches still stand — NOT stale. Live repro in-container confirms all
  four shapes (`frames: 0`, `{}`, missing block, `frames: "many"`)
  substitute request values with no error.
- Integrity note: the supervisor already re-gates present `frames` with a
  `MediaError` 1..ceiling check (`supervisor.py:2397-2408`, issue 006), so
  `frames: 0` / `frames: "many"` fail at commit — but `fps`/`novel_frames`/
  `conditioning_frames` have NO supervisor gate, and the adapter-level
  contract (`test_missing_report_falls_back_to_requested`) still pins the
  lenient fallback, so the trust-boundary gap is live. The fix makes the
  adapter strict (fail loud at the boundary) while keeping the supervisor
  ceiling as the second layer for implausibly large counts.
- Error choice: `FatalWorkerError` (a `WorkerError`, per the issue's
  candidates) — worker output is not config (`ConfigurationError` would
  misdirect), and fatal (not recoverable) so a deterministic shape bug
  never burns restart budget. The supervisor's `MediaError` ceiling gate
  is untouched and still catches `frames: 10**9` (adapter accepts any
  positive int; only the supervisor knows the ceiling).

## Progress log (2026-10-07)
- `VideoBackendAdapter.generate_segment` gained keyword-only
  `strict: bool = True`: strict raises `FatalWorkerError` for a
  missing/non-dict `video` block, missing `frames`, or any
  present-but-malformed `frames`/`fps`/`novel_frames`/`conditioning_frames`
  (bool-safe `int` checks, `frames`/`fps`/`novel` > 0, `conditioning` >= 0);
  absent optional `fps`/`novel`/`conditioning` still fall back. The
  `elif` branch preserves the exact legacy lenient fallback under
  `strict=False` (including its new bool guards). `FatalWorkerError`
  imported in `voyage/backends.py`.
- `tests/test_adapter_contract.py`: `test_missing_report_falls_back_to_requested`
  now pins the lenient path via `strict=False`; four new strict tests —
  non-positive frames, non-int/bool frames, missing block, bad
  fps/novel/conditioning (+ absent-optionals fallback pin).

## Resolution (2026-10-07)
- RESOLVED. Files: `voyage/backends.py`, `tests/test_adapter_contract.py`.
  Scoped + neighbor pytest green in-container; ruff + format + mypy strict
  clean on touched modules.
- Left open (follow-up, supervisor track owns it): `supervisor.py:2282-2293`
  docstring still says "the adapter normalizes leniently by design" — now
  stale for malformed reports (adapter raises before the supervisor
  re-gate). The re-gate itself stays correct (ceiling + tape confinement).

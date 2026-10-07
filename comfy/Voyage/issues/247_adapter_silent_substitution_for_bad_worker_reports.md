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

# 286 — SFX single-pass extract streams `stdout` while `stderr=PIPE` sits undrained — textbook pipe deadlock, no timeout

Severity: MEDIUM (pass-2 worker-tails sweep).

## Technical description

`voyage/workers/sfx_mmaudio.py::_extract_frames` spawns ffmpeg with `stdout=PIPE,
stderr=PIPE`, then reads stdout frame-by-frame in a Python loop and only calls
`proc.communicate()` at the end to collect stderr. If ffmpeg writes more than the OS pipe
buffer (~64 KiB) to stderr (corrupt segment video spewing per-frame errors is the live
case), the child blocks on stderr while the parent blocks on stdout — permanent hang
with no timeout (`communicate()` called without `timeout=`).

## Rationale

The CPython docs warn exactly against this shape: "Use `communicate()` rather than
`.stdout.read` or `.stderr.read` to avoid deadlocks due to any of the other OS pipe
buffers filling up". `-v error` keeps the quiet path quiet, so this only bites on the
failure path that most needs a loud error.

## Live evidence

```
$ grep -n "Popen\|communicate()\|stderr=subprocess.PIPE" voyage/workers/sfx_mmaudio.py
293:    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
313:        _, stderr = proc.communicate()
```

Lines 299-310 read `proc.stdout` in a `while len(raw_frames) < wanted_sync` loop;
`communicate()` (313) is reached only after the loop. No `timeout=` anywhere in the
function. Sibling `_convert` (same file, line 346) correctly uses one-shot
`subprocess.run(..., capture_output=True)`.

Repro: feed a corrupt/truncated `video_path` that makes ffmpeg emit >64 KiB of stderr
while producing stdout frames: parent sits in `_read_frame_bytes` → `stdout.read`,
child sits in `write(stderr)` — both block; supervisor RPC deadline (600 s) is the only
bound.

## Source refs

`voyage/workers/sfx_mmaudio.py:268-326`.

## Online sources

- `https://docs.python.org/3/library/subprocess.html` ("Use `communicate()` rather than
  `.stdout.read`… to avoid deadlocks…", and `Popen.wait` "will deadlock when using
  stdout=PIPE…").

## Fix candidates

- (a) `stderr=subprocess.DEVNULL` — stderr text is only used for the already-failed case
  and the returncode check survives; (b) spawn a stderr-drain thread before the read
  loop; (c) pass `timeout=` to the trailing `communicate()` and kill on expiry.

## Log

- 2026-10-07: filed from read-only pass-2 worker-tails sweep; no code touched.

## Evaluation (2026-10-07)
- Claim CURRENT on re-read: `_extract_frames` still read stdout in a loop with
  `stderr=PIPE` undrained and a timeout-less trailing `communicate()`. Fix
  candidates (b)+(c) adopted in combination.

## Progress log
- Batch-6 Group Q fixed `voyage/workers/sfx_mmaudio.py::_extract_frames`: a
  stderr-drain thread runs ahead of the stdout read loop (pipe can no longer fill
  and block the child), and the trailing `communicate()` carries a timeout with
  kill-on-expiry. New `tests/test_issue_286_sfx_stderr_drain.py` covers the drain
  and timeout behavior. Scoped gates green (ruff + format + mypy strict + pytest).

## Resolution (2026-10-07)
- RESOLVED. The textbook pipe deadlock is closed on both ends: stderr is drained
  concurrently, and no path can hang past the communicate timeout.

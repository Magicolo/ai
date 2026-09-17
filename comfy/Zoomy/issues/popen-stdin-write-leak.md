# ffmpeg child process and pipe leaked on stdin write failure

- Severity: medium (resource leak + zombie process on the GPU host).
- Status: verified open. `zoomy/local_engine.py:1316-1332`.

## Evidence

Current code improved since the first audit (now catches `OSError` on
spawn, checks `returncode`, verifies non-empty output), but the write loop
still leaks:

```python
process = subprocess.Popen(command, stdin=subprocess.PIPE)
...
for frame in frames:
    process.stdin.write(np.asarray(frame).tobytes())
process.stdin.close()
process.wait()
```

If `write` raises (`OSError` on a dead child, `ValueError` on a closed
pipe), the handler at `:1330` raises `AssemblyError` while the child keeps
running with an open pipe: no `terminate()`/`kill()`, no `close()` on the
pipe object, no `wait()` to reap it. A 189-frame encode dying mid-stream
leaves a zombie ffmpeg plus one leaked FD per retry (and
`run_stage_with_retries` can run this window up to 3 times).

## Fix

Wrap the feed in `try/finally`: on any write failure, `close()` the pipe,
`terminate()`, `wait(timeout=…)`, `kill()` on timeout — then raise
`AssemblyError`. Or use `process.communicate(input=…)` with the whole
payload, which reaps unconditionally. Pin with a test that feeds a
failing stdin (stub `Popen`) and asserts the child was terminated.

## Verification

- New test with a stubbed `Popen` whose stdin raises on write: child
  terminated, pipe closed, `AssemblyError` raised.
- Gates: `Zoomy/scripts/quality-gates.sh` green.

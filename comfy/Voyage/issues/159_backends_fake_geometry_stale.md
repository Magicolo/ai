# 159 — BACKENDS.md fake-video geometry is stale (320×180-class vs 768×432 preset)

- **Severity:** LOW (docs-only — code is correct, the table lies about what `fake` renders)
- **File:line:** `Voyage/docs/BACKENDS.md:28` vs `Voyage/voyage/config.py:135-142` + `Voyage/voyage/fake_backends.py:73`
- **Area:** docs/backend-geometry drift (fake preset moved to 432p; the doc row never followed)

## Description

`docs/BACKENDS.md:28` still describes the fake video backend as:

```
| video | `fake` | deterministic `testsrc` 320×180-class H.264 |
```

The live fake preset (`config.py:135-142`, `BackendRecord` profile
`fake-432p`) is `768×432 @ 24 fps`, `segment_frames=48`:

```python
# config.py:135-142
"fake": BackendRecord(
    profile="fake-432p",
    width=768,
    height=432,
    fps=24,
    segment_frames=48,
    ...
```

And the fake video backend is fully parameterized — it renders
whatever geometry the request carries, not a hardcoded 320×180 frame:

```python
# fake_backends.py:73
f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
```

So on the current preset a fake segment is 768×432, 48 frames — the
"320×180-class" label is wrong by more than 2× per axis.

## Rationale

Geometry labels are load-bearing for readers: fake is the CPU smoke-run
backend (`generate`/`benchmark end-to-end` pin it), and anyone sizing a
test run, a VLM view, or a finalize expectation off "320×180-class"
will mispredict frame bytes, VAE-unrelated decode cost, and output
resolution. The fix is a one-cell doc edit; leaving it stale trains
every new reader to distrust the backend table.

## Live evidence

- `sed -n '26,29p' docs/BACKENDS.md` — row reads `deterministic
  \`testsrc\` 320×180-class H.264`.
- `sed -n '135,142p' voyage/config.py` — fake row is
  `profile="fake-432p", width=768, height=432, fps=24,
  segment_frames=48`.
- `sed -n '60,85p' voyage/fake_backends.py` — `testsrc=size={width}x{height}`
  at `:73`; no 320/180 literal anywhere in the file (`grep -n "320\|180"
  voyage/fake_backends.py` returns nothing).
- Overlap check: 042 is the README-geometry drift file (README, not
  BACKENDS.md); 135 is the §53/ltxv-576p addendum staleness (LTXV
  geometry, not fake). Neither names the BACKENDS.md fake-video cell.

## Repro

Static: open `docs/BACKENDS.md:28` next to `voyage/config.py:135-142`
and `voyage/fake_backends.py:73`. Or run the fake preset end-to-end
(`run.sh benchmark end-to-end`) and probe the segment mp4 — it is
768×432, not 320×180-class.

## Fix candidates

1. Update the cell to `deterministic \`testsrc\` 768×432 H.264
   (fake-432p preset, \`config.py:135-142\`)` — one line, no code change.
2. Optionally cite the parameterization (`fake_backends.py:73`
   `size={width}x{height}`) so the next preset move does not re-stale
   the doc.
3. Test: none needed beyond the doc edit (optional: a docs-vs-preset
   assertion pinning the fake row's geometry string — likely overkill
   for a LOW).

## Refs

- `Voyage/docs/BACKENDS.md:28`; `Voyage/voyage/config.py:135-142`;
  `Voyage/voyage/fake_backends.py:60-85`.
- Adjacent, not overlapping: 042 (README geometry drift — different
  file); 135 (LTXV 576p addendum — different backend).

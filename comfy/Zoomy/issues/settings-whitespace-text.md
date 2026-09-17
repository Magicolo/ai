# `_read_text` treats whitespace-only env vars as configured

- Severity: medium (config correctness — asymmetric with `_read_integer`).
- Status: verified open. `zoomy/settings.py:47-49`.

## Evidence

```python
def _read_text(name: str, default: str) -> str:
    return os.environ.get(name, default) or default
```

`"   "` is truthy, so `ZOOMY_OUTPUT_DIRECTORY="   "` (a common
copy-paste/CI artifact) is accepted as a directory named of spaces and
fails late in `LocalEngine`/Gradio. The sibling `_read_integer`
(`settings.py:58-60`) explicitly strips and falls back on blank. The
existing test (`tests/test_settings.py`) covers `""` only, never `"   "`
— which is exactly why the asymmetry survived.

## Fix

```python
raw = os.environ.get(name)
if raw is None or not raw.strip():
    return default
return raw
```

Decide and document whether the returned value is the raw or the
stripped text (stripped is friendlier; raw preserves intentional
padding — directories never want padding, so strip). Add the
whitespace case to the existing blank-value test.

## Verification

- Failing test first: env `"   "` → default (currently returns `"   "`).
- Gates: `Zoomy/scripts/quality-gates.sh` green.

# Garbage UI payloads coerce to silent wrong values

- Severity: low (misconfiguration renders with wrong LoRAs/strengths,
  no error).
- Status: FIXED. Chose the issue's 'documented-default' branch: the pinned
  property `test_ui_payload_coercions_never_crash` requires both coercions
  stay total, so semantics were tightened instead of raising —
  `_selected_lora_names` filters `isinstance(item, str)` (no more
  `str(None)`/`str(3)` phantom names), `_coerce_to_float` returns the
  documented 0.0 for bool/non-numeric/non-finite input. Tests:
  `test_selected_lora_names_keeps_only_strings`,
  `test_coerce_to_float_defaults_on_garbage`,
  `test_coerce_to_float_keeps_finite_numbers` (red first on str-ification,
  NaN/True/inf). `zoomy/interface.py:969-990`.

## Evidence

```python
def _selected_lora_names(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()
...
def _coerce_to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
```

A garbage `CheckboxGroup` payload (non-list) yields `set()` — renders with
*no* LoRAs instead of erroring — and a garbage slider payload yields
`0.0` strength — silently disabling the style. Note `_selected_lora_names`
also `str()`-ifies every item, so `[None, 3]` becomes `{"None", "3"}`,
which then matches nothing and deselects. Contrast `_as_durations` and
`_parse_target_seconds`, which fall back to principled defaults (`(0,
0.0, 0.0)`, unlimited) — those are *documented* blank-means-default
semantics, while these two are silent type-error swallowing.

## Fix

Keep the blank/default path (Gradio genuinely submits `0`/empty during
init), but distinguish "empty" from "wrong type": wrong-type payloads
should raise `ZoomyError` (surfaced by the handler error path) or at
minimum log. `str(item)` should become an `isinstance(item, str)` filter
so non-strings never masquerade as display names. Tests for each: `None`,
dict, and mixed-type list payloads.

## Verification

- New tests: wrong-type slider/checkbox payloads raise (or are
  documented-default); `["a", None, 3]` selects only `"a"`.
- Gates: `Zoomy/scripts/quality-gates.sh` green.

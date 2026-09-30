# 183 — `test_tui_app.py:717` carries an ignore neither 034 nor 150 names: the stale-ignore pattern keeps spreading outside the gate

- **Severity:** LOW (typing hygiene — one more invisible suppression in a file the mypy gate never looks at; systemic because it proves 150's pattern is still spreading)
- **Track:** D (tests/standards tails — below 150/034/033)
- **Verified live:** 2026-09-30 host read (tree as-read; no edits in these ranges)

## File:line (live-verified)

- `Voyage/tests/test_tui_app.py:717` (`app.save_screenshot(str(path))  # type: ignore[attr-defined]` inside `_form_svg_rows`)
- `Voyage/tests/test_tui_app.py:532` (`rows = app.query(".field-row")  # type: ignore[union-attr]` — the ONE ignore in this file that 034 names)
- `Voyage/tests/test_tui_app.py:1-40` (module docstring + `_require_app` + `app: object` convention — every Pilot helper types the app as `object`, so every attribute access needs a suppression by construction)
- `Voyage/scripts/gates.sh:22-28` (mypy scope: `voyage` + `tests/conftest.py` + `tests/test_seeds_properties.py` + `tests/test_beat_properties.py` + `tests/test_similarity_properties.py` — `test_tui_app.py` not listed)
- `Voyage/pyproject.toml:119` (`warn_unused_ignores = true` — active but blind where the gate never looks, per 033)

## Description

034 names exactly one ignore in `test_tui_app.py` (`:532`, `union-attr`). 150 names five other files (014/029/030/director_models_dir/032) and never mentions this one. Live the file carries TWO ignores — `:717` (`attr-defined` on `app.save_screenshot`, where `app: object`) is unfiled. The code is plausibly correct (`object` has no `save_screenshot`, so the code is right), but because the file sits outside the `gates.sh:28` mypy list, `warn_unused_ignores` can never confirm it: if a future Textual stub starts typing the Pilot app, this suppression goes stale silently, and if the call is ever removed the dead comment stays. It is 150's exact failure mode (new suppression, zero gate coverage) in a sixth file.

## Rationale

- 150's systemic point ("the gate stays green while the pattern spreads") is only as strong as its file list is complete — an unlisted sixth file with the same shape weakens the ratchet prescription (append listed files one at a time) by leaving a hole the next sweep re-files.
- The `app: object` convention makes this file an ignore factory: every new Pilot helper that touches the app needs another suppression, each individually reasonable, collectively invisible. Naming `:717` documents the factory, not just the instance.
- Fix cost is the same one-liner class as 150's scaffold fix (type the helper param as the real app/pilot type instead of `object`).

## Live evidence (host read, 2026-09-30)

```
$ rg -n "type: ignore" Voyage/tests/test_tui_app.py
Voyage/tests/test_tui_app.py:532:        rows = app.query(".field-row")  # type: ignore[union-attr]
Voyage/tests/test_tui_app.py:717:    app.save_screenshot(str(path))  # type: ignore[attr-defined]

$ sed -n '710,718p' Voyage/tests/test_tui_app.py
def _form_svg_rows(app: object, tmp_path: Path) -> list[str]:
    """Render the running app to SVG and return its text rows (stripped)."""
    import re

    path = tmp_path / "form.svg"
    app.save_screenshot(str(path))  # type: ignore[attr-defined]
```

034 cites `:532` only; 150's five files exclude `test_tui_app.py` entirely; `gates.sh:28` excludes it from mypy.

## Repro

```bash
rg -n "type: ignore" Voyage/tests/test_tui_app.py
sed -n '18,28p' Voyage/scripts/gates.sh  # test_tui_app.py absent from the mypy list
```

## Fix candidates

1. Type the helper params as the real Textual types (`VoyageApp` / pilot app) instead of `object` — kills `:532` + `:717` at the source (same prescription as 150's scaffold fix).
2. Add `test_tui_app.py` to 150's gate-conversion queue (one file at a time per 033's prescription) so `warn_unused_ignores` can fire on both lines.
3. Test/ratchet: extend the `rg "type: ignore" --count` ratchet (040's pattern) to fail on NEW suppressions in files outside the mypy list.

## Refs

- `Voyage/tests/test_tui_app.py:532,717`; `Voyage/scripts/gates.sh:28`; `Voyage/pyproject.toml:119`.
- Not-a-duplicate: 034 (names `:532` only — `:717` never appears); 150 (five files, this file not among them; the `:118`-clean-control note shows the filer checked per-file — this file was never checked); 033 (gate 4/88 scope — the scope this instance falls outside).

## Progress log (2026-09-30, Group D pass)

- Re-verified live plus one new finding: ad-hoc in-container `mypy` on
  the file reports `:532` as **wrong-code** (`[union-attr]` covering a
  genuine `[attr-defined]` — the ignore suppresses nothing AND the error
  stands), while `:717` is used-correct. So the file carried one stale
  and one live suppression, not two live ones.
- Fixed both at the source per candidate 1, with `Any` instead of the
  real `VoyageApp` type: a module-level app import would break the file's
  `importorskip("textual")` collection guard, and `Any` matches the
  existing `_click_generate_when_ready(pilot: Any, app: Any)` precedent.
  Both ignores removed; zero suppressions remain in the file.

## Resolution (2026-09-30, Group D pass)

- Resolved: the ignore factory instance is gone (`object` → `Any` on both
  helpers).
- Files changed: `tests/test_tui_app.py` (2 annotations + 2 deletions).
  Gate evidence: full file 33/33 in-container; ad-hoc `mypy` 0 errors
  (was 2); `ruff check` + `ruff format --check` clean; `grep "type:
  ignore"` empty. DESIGN proposals: none. Residuals: none (the factory
  pattern is documented by the precedent, not by new policy).

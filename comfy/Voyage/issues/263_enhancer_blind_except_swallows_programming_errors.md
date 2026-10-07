# 263 — Fail-soft `except Exception` in `prompt_enhancer.enhance` swallows programming errors indistinguishably from sidecar-down

Severity: MEDIUM (track C-05).

## Technical description

`voyage/prompt_enhancer.py:170` catches `Exception` over `_post_chat` + `_parse_expansion`
together and returns input-with-zero-counts. A bug in `_parse_expansion` itself (wrong
key, `TypeError`) degrades to the same zero-count "non-engagement" the metrics use to
mean "sidecar unreachable" — the 2026-10-05 "0-token ON failure" was in fact
environmental, showing this signal is already ambiguous in practice.

## Rationale

Best practice (and the codebase's own taxonomy — explicit exception types, "branch on
class never message", AGENTS.md §12) is to catch the transport/parse errors you expect
(`httpx` errors, `RuntimeError`, `ValueError`, `TypeError` from the sidecar contract) and
let programming errors fail loud. `motion_sense.sense_motion`
(`voyage/motion_sense.py:46`) shares the shape but is the weaker case — an explicitly
best-effort sensor whose `None` means "unknown", so it stays LOW (filed jointly here, fix
optional).

## Live evidence

```python
# voyage/prompt_enhancer.py:166-171
try:
    url = endpoint.rstrip("/") + COMPLETIONS_PATH
    parsed = _post_chat(url, body, timeout)
    return _parse_expansion(parsed, url)
except Exception:  # noqa: BLE001 — fail-soft prototype stage, never fails a commit
    return text, dict(zero_counts)
```

`--select BLE` probe: the only two `except Exception` in scope are these two (both carry
`noqa: BLE001`).

Repro: stub `_parse_expansion` to raise `TypeError` in a scratch test → `enhance`
returns `(input, zeros)`; metrics record non-engagement. Indistinguishable from a dead
sidecar.

## Source refs

`voyage/prompt_enhancer.py:148-171`; `voyage/motion_sense.py:38-48` (joint).

## Online sources

- ruff BLE001 ("blind except").
- AGENTS.md §12 errors clause.
- DESIGN note on the 0-token incident (AGENTS.md §11 "enhancer observable + clean ON
  rerun").

## Fix candidates

- (a) narrow to `(httpx.HTTPError, RuntimeError, ValueError, TypeError)` — still
  fail-soft on every sidecar-shaped failure, loud on internal bugs; (b) add a distinct
  metric/counter for parse-vs-transport failure so engagement accounting separates "down"
  from "malformed".

## Log

- 2026-10-07: filed from read-only Track C sweep; no code touched.

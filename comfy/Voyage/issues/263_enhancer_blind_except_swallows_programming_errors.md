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

## Evaluation (2026-10-07)
Live probe in-container: `enhance` catches bare `Exception` over
`_post_chat` + `_parse_expansion` together, returning input-with-zeros —
a `_parse_expansion` bug (`TypeError`) degrades to the same zero-count
non-engagement as sidecar-down, matching the ambiguous 0-token incident.
Confirmed as filed. Fix narrows to sidecar-shaped failures with a
transport/parse split: `_post_chat` normalizes missing httpx +
`httpx.HTTPError` to `RuntimeError` (no module-scope httpx import, slim
safe), so `enhance` needs only `(RuntimeError, OSError)` for transport
(`ConnectionError` included) and `(ValueError, TypeError)` for parse;
programming errors propagate. `OSError` is required beyond the issue's
literal list or the existing `ConnectionError` fail-soft test breaks.

## Progress log
- `prompt_enhancer.py`: `_post_chat` converts missing httpx
  (`ImportError`) + `httpx.HTTPError` (dynamic type, narrow-by-re-raise)
  to `RuntimeError`; added `enhance_with_failure_kind` returning
  `(text, counts, kind)` with `transport`/`parse`/`None`; rewrote
  `enhance` as a narrow delegating wrapper (no blind except);
  `enhance_many` summary gains `transport_failures`/`parse_failures`.
- Updated the one exact-equality pin in `tests/test_prompt_enhancer.py`
  (off-path summary now carries the two zeroed counters).
- New tests in `tests/test_issue_263_enhancer_kinds.py` (4 tests):
  transport vs parse kind split, programming-error loudness,
  `enhance_many` counter split, expected-failure softness.
- Verified: `ruff check` + `format --check` clean (one `TRY004` noqa
  for the transport convert, `TRY300` else-block); `mypy` strict clean;
  scoped pytest 113 passed.

## Resolution (2026-10-07)
Fixed as proposed (with `OSError` added for the `ConnectionError`
contract and httpx normalized in `_post_chat` so no module-scope import
is needed). No open items; supervisor's budget loop keeps its own broad
catch (out of scope) while the enhancer boundary is now narrow.

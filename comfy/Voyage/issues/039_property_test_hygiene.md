# 039 — Property-test hygiene gaps: surrogate-alphabet contradiction, no health-check/deadline policy, replay-disabled-by-default

- Severity: MEDIUM
- Files: `tests/test_seeds_properties.py:33-41`, `tests/conftest.py:42-70`, `tests/conftest.py:46-47`
- Area: tests / property-based testing
- Overlaps with: 089 (test-hygiene lock/markers — adjacent, not a duplicate)

## Description

Three linked issues: (a) `test_seeds_properties.py:35-37` claims
"`st.text()` already excludes surrogate code points by default" and draws
bare `strategies.text(max_size=24)`, while `conftest.py:58-63` documents the
opposite from a live probe and maintains a `short_texts` strategy the
property modules do not use — the two files disagree about Hypothesis's
surrogate behavior. (b) No property module sets `@settings(deadline=…)` or
documents `suppress_health_check` rationale; conftest registers only
`database=None` + `load_profile("container")`, so (c) the example database
is globally disabled — failing examples are "reported verbosely but never
replayed" (`conftest.py:14-16`), trading away shrinking-persistence for
bind-mount cleanliness with no per-test opt-out path documented.

## Rationale

Hypothesis docs warn to suppress health checks narrowly as encountered and
treat the database as the replay mechanism. A suite-wide `database=None`
plus contradictory alphabet assumptions risks both flaky-health-check
blindness and missed lone-surrogate crashes in tokenizer/JSON paths.

## Live evidence (re-verified 2026-09-30)

`tests/test_seeds_properties.py:33-37` read live:

```python
run_seeds = strategies.integers(min_value=-(2**63), max_value=2**63)
small_counts = strategies.integers(min_value=0, max_value=999999)
# `st.text()` already excludes surrogate code points by default, so labels
# are always encodable without an explicit alphabet restriction.
short_labels = strategies.text(max_size=24)
```

`tests/conftest.py:42-70` read live:

```python
settings.register_profile("container", database=None)   # :46
settings.load_profile("container")                      # :47
# Verified live: `characters()` with only `blacklist_characters` still
# draws lone surrogates (probed `\ud800`), so the surrogate category
# stays blacklisted explicitly alongside NUL.                    # :58-60
surrogate_category: tuple[Literal["Cs"], ...] = ("Cs",)  # :63
short_texts = hypothesis_strategies.text( … )            # :64-69
```

Host `rg` today:

```
tests/test_seeds_properties.py:35:# `st.text()` already excludes surrogate code points by default, so labels
tests/conftest.py:59:    # draws lone surrogates (probed `\ud800`), so the surrogate category
```

`rg -n "suppress_health_check|HealthCheck" tests/*.py` → zero hits (only
unrelated `deadline` hits in commit/recovery tests about RPC/event timing,
not Hypothesis `@settings`).

## Repro

```bash
rg -n "surrogate" tests/conftest.py tests/test_seeds_properties.py
rg -n "suppress_health_check|deadline" tests/*.py
pytest tests/test_seeds_properties.py -q   # observe no deadline/health-check config
```

## Fix candidates

(a) Unify on conftest's `short_texts`/`bounded_counts` everywhere and
correct the seeds-file comment after a live probe.
(b) Document the `database=None` trade-off per-module with an env-gated
opt-in for local replay.
(c) Add explicit `@settings(deadline=None)`-with-reason or tuned deadlines
where ffmpeg/subprocess-adjacent properties are added.

## Refs

- Hypothesis "Suppress a health check everywhere"; `@settings` /
  `register_profile` docs; pytest `tmp_path` docs.
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §9.

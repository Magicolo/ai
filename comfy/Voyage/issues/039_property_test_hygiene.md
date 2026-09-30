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

## Progress log (2026-09-30, resolution pass)

- As-read counts drifted from the issue: property modules are now 3
  (`test_seeds_properties`, `test_beat_properties`,
  `test_similarity_properties`) plus hypothesis use in
  `test_cli_validate_handoff` / `test_containers_rank2`; beat + similarity
  already import `bounded_counts` / `short_texts` from conftest, and
  `test_cli_validate_handoff.py:85` already blacklists the surrogate
  category explicitly — the seeds file was the sole outlier.
- Live probe in-container (`voyage:latest`, 500 examples,
  `database=None`): bare `st.text(max_size=24)` drew zero lone
  surrogates and 200/200 examples survived `json.dumps`. The seeds-file
  comment was therefore not crash-risky (upstream `text()` already
  excludes `Cs` by default); conftest's probe concerned bare
  `characters()`, a different default. Recorded as nuance, not
  contradiction — unification is still worthwhile (explicit over
  implicit).
- Fix applied (tests/ only): `test_seeds_properties.py` now aliases
  `small_counts = bounded_counts` / `short_labels = short_texts` from
  `tests.conftest` with a corrected comment citing the live probe;
  `conftest.py` gained the env-gated opt-in (`VOYAGE_HYPOTHESIS_DATABASE=1`
  restores the default database for local replay, otherwise `None`) plus
  a health-check/deadline policy comment (zero suppressions today —
  suppress narrowly as encountered; no custom deadlines — properties are
  CPU-only pure cores; any future ffmpeg-adjacent property needs an
  explicit `@settings(deadline=...)` with reason).
- Evidence: `pytest tests/test_init_run_ratchet.py
  tests/test_seeds_properties.py tests/test_beat_properties.py
  tests/test_similarity_properties.py` → 29 passed in-container;
  `ruff check` + `ruff format --check` + `mypy` (strict, gate scope +
  ratchet module) green.

## Resolution

- Resolved (tests/ slice). Residual: none for this issue — (a)(b)(c)
  all landed. The `VOYAGE_HYPOTHESIS_DATABASE` opt-in writes the default
  `.hypothesis/` location on replay runs (gitignored, never baked —
  `.dockerignore` already lists it).

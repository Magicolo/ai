# 259 — `docs/PROMPTING.md` novelty section contradicts the steer-and-accept code

Severity: MEDIUM (track C-01).

## Technical description

PROMPTING.md says a proposal "exceeding `novelty_threshold = 0.85` is rejected with
feedback and the director retries (bounded attempts, then the best candidate wins)". The
code has done steer-and-accept since 2026-10-01: novelty never rejects; the prompt steers
and every generation is accepted with truthful `novelty_accepted`.

## Rationale

An operator tuning `novelty_threshold` or debugging "holds" from this doc will expect
rejection/retry dynamics that no longer exist (e.g. expecting retry-burn latency that is
now single-serve). Docs-behavior mismatch is the highest-impact class in this scope.

## Live evidence

```
$ grep -rn "never rejects" voyage/supervisor.py | head -3
voyage/supervisor.py:7:  2. style check (code-level, §18.1) → novelty score (recorded, never rejects)
voyage/supervisor.py:1846:        director. Novelty never rejects (item 1): the prompt steers
docs/PROMPTING.md:31: history exceeds `novelty_threshold = 0.85` is rejected with feedback and
```

`voyage/supervisor.py:1954-1968` — "Item 1: novelty never rejects … simply carries
novelty_accepted=False. Single-serve accepts"; `tests/test_novelty_steer_accept.py` pins
the new behavior.

Repro: read `docs/PROMPTING.md:27-33` then `voyage/supervisor.py:1846-1868` — the two
cannot both be true.

## Source refs

`docs/PROMPTING.md:29-33` vs `voyage/supervisor.py:7,1846-1968`,
`voyage/director.py:119-130` (NOVELTY STEERING section).

## Online sources

- Project history (AGENTS.md §11 "Voyage novelty steer-and-accept done 2026-10-01").
- `tests/test_novelty_steer_accept.py` (7 tests).

## Fix candidates

- (a) rewrite the PROMPTING.md novelty paragraph to steer-and-accept (steering section,
  `novelty_accepted`/`hold` semantics, `allow_concept_revisit`); (b) move
  threshold/retry text to a "pre-2026-10-01" note or delete.

## Log

- 2026-10-07: filed from read-only Track C sweep (scope gates green: ruff + mypy clean);
  no code touched.

## Evaluation (2026-10-07)

Claim re-verified live: `docs/PROMPTING.md:27-33` described reject-with-feedback +
bounded retry with best-candidate-wins, while `voyage/supervisor.py:2748-2752,2864-2883`
(Item 1: novelty never rejects; first schema/style-valid generation always renders;
revisit carries `novelty_accepted=False`), `voyage/director.py:119-130` (NOVELTY
STEERING section, sentinel-gated), and `tests/test_novelty_steer_accept.py` (7 tests)
all implement steer-and-accept since 2026-10-01. File:line refs in the issue were
fresh (PROMPTING novelty paragraph, supervisor docstring + accept loop, director
steering builder). No downgrade: full fix scope applied.

## Progress log

- 2026-10-07: rewrote the `## Novelty system` paragraph in `docs/PROMPTING.md` to
  steer-and-accept (steering section + sentinel omission, `novelty_scored` score vs
  threshold, `novelty_accepted`/`hold` semantics, bounded retries now schema/style
  only with deterministic fallback, `allow_concept_revisit = false` kept,
  pre-2026-10-01 regime marked retired with test pin). Grep-verified no
  "rejected with feedback" remnant; PROMPTING claims cross-checked against
  `voyage/supervisor.py:7,2748,2864` + `voyage/director.py:36,126`.

## Resolution (2026-10-07)

RESOLVED. Docs now match the steer-and-accept code; nothing left open.

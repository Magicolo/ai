# 077 — DESIGN §74 vs code: accept-loop check order inverted (style before novelty)

- Status: open
- Severity: low (spec drift — both gates enforced, but violation attribution and
  retry sequencing differ from spec)
- Area: spec/code drift — DESIGN §74 (`DESIGN.md:3500`, was `:3487` before
  concurrent edits shifted lines — section ref stable),
  `voyage/supervisor.py:646-687`
- Rank rationale: pass-2 DESIGN finding; §74 ordering not previously covered.

## Technical description

Spec order: `Pydantic validation → novelty check → style policy check → accept`.
Code order: schema validate (647) → empty-stages (651) → amendments (654) →
**style check (660-670)** → embed + **novelty check (673)**.

## Why this is an issue

Both gates are enforced, but in the opposite order from the spec — so a
proposal violating both gets a style rejection (and a style-rejection concept
record) where the spec predicts novelty-first. That changes violation
attribution, retry sequencing, and the audit trail in the concept store, and
any operator reading §74 will mispredict what the system does.

## Evidence

Order diagram is at `DESIGN.md:3500` (`novelty check` → `style policy check`,
`rg -n "novelty check"` confirms); code order verified live
(`supervisor.py:646-687`: `check_prompt_against_style` + style-rejection
record precede `_embed_texts`/`check_novel`):

```
$ sed -n '646,690p' Voyage/voyage/supervisor.py | rg -n "style|novel|embed"
16:                    check_prompt_against_style(stage_text, style_spec)
21:                    summary=f"style-policy rejection: {exc}",
24:                feedback = f"style-policy rejection: {exc}"
26:            vectors = self._embed_texts([decision.destination_concept])
28:            accepted, last_score = store.check_novel(decision.destination_concept, vector)
```

## Reproduction

Proposal violating both gates → operator sees a style rejection (and a
style-rejection concept record) where the spec predicts novelty-first.

## Source references

- Files/lines above.

## Resolution candidates

Either reorder to spec (novelty before style) or annotate §74 as-built with the
actual order + rationale.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI/DESIGN sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; fixed stale
  DESIGN line (`:3487` → `:3500`, section ref stable); re-verified code order
  live (style gate precedes novelty gate — pasted above).
- Open: reorder or annotate.

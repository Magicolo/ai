# 076 — DESIGN §19 vs code: director prompt demands `novelty.distinguishes_from`, schema drops it

- Status: open
- Severity: low-medium (spec'd contract unfulfillable; distinction rationale
  never persisted)
- Area: spec/code drift — DESIGN §19 (`DESIGN.md:1530`),
  `voyage/director.py:80-86`, `voyage/models.py:113-114`
- Rank rationale: pass-2 DESIGN finding; no prior schema-vs-prompt item.

## Technical description

```python
# director.py user message:
"novelty MUST be an object with why_new and distinguishes_from strings."
```

```python
class DirectorNovelty(BaseModel):
    why_new: str = ""   # no distinguishes_from field
```

## Why this is an issue

The prompt contract demands a field the schema silently drops (pydantic ignores
the extra key by default), so distinction rationale the director was explicitly
asked to produce is never persisted anywhere. The spec'd contract is
unfulfillable: compliant model output loses evidence at validation, and nobody
reading the stored decision can tell what distinguished the new concept from
history.

## Evidence (live probes by pass-2 sweep)

```
DirectorNovelty fields: ['why_new']
model_validate({'why_new':'x','distinguishes_from':'y'}) → ACCEPTED {'why_new': 'x'} (extra dropped)
EvolutionDecision with distinguishes_from → novelty={'why_new': 'w'} (evidence lost)
```

## Reproduction

Snippet above.

## Source references

- Files/lines above.

## Resolution candidates

Add `distinguishes_from: str = ""` to `DirectorNovelty`, or remove it from the
demanded prompt keys. Persist it in the concept record if kept.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 CLI/TUI/DESIGN sweep with live probes.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`director.py:80-86` prompt demand, `models.py:113-114`
  `DirectorNovelty`, `DESIGN.md:1530` §19 header — all match).
- Open: align schema with prompt (or prompt with schema).

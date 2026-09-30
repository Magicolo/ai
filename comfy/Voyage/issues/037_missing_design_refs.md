# 037 — Seven modules lack DESIGN § refs in their docstrings (§12 violation)

- Severity: LOW
- Files: `voyage/augment.py:1`, `voyage/bench.py:1`, `voyage/hashing.py:1`, `voyage/__init__.py:1`, `voyage/models_ensure.py:1`, `voyage/sfx_finalize.py:1`, `voyage/tui.py:1`
- Area: standards / docs
- Overlaps with: 091 (docs SFX/augment follow-ups may add the missing refs as a side effect — not a duplicate)

## Description

AGENTS.md §12 requires "every module docstring states purpose + DESIGN §
ref". All 42 modules have a docstring, but the first-15-lines scan shows
these 7 carry no `DESIGN` reference (contrast `cli.py:1` "DESIGN §§58, J",
`rpc.py:1` "DESIGN §§45-46, D", `seeds.py:1` "DESIGN §62",
`supervisor.py:1` "DESIGN §73"). `augment.py` additionally carries a
`TODO (unify)` in shipped code.

## Rationale

DESIGN refs are the project's traceability mechanism (gates comments cite
§83/§87/§92). Unref'd modules rot first — reviewers cannot tell which spec
section governs them.

## Live evidence (re-verified 2026-09-30)

Host scan today (`head -n 15` per file):

- `voyage/augment.py:1` — "GPU augment runner orchestration … (Track D
  spike). Pure orchestration — stdlib only, never torch (supervisor section
  12 GPU ban) … TODO (unify): if media.py ever gains chunk/frame-count plan
  math, move `interpolated_frame_count` there…" — no `DESIGN` token.
- `voyage/bench.py:1` — "Benchmark + soak helpers: timing stats, §104
  reports…" — bare `§104`, no `DESIGN` token.
- `voyage/hashing.py:1` — "Shared SHA-256 helpers (issue 021)…" — no
  `DESIGN` token.
- `voyage/__init__.py:1` — "Voyage — autonomous infinite audiovisual voyage
  (Phase 0 skeleton)…" — no `DESIGN` token.
- `voyage/models_ensure.py:1` — "Inline model ensure for `voyage generate`
  (selective + parallel)…" — no `DESIGN` token.
- `voyage/sfx_finalize.py:1` — "Finalize-time SFX pass (slice 3,
  three-caption doctrine)…" — no `DESIGN` token.
- `voyage/tui.py:1` — "Interactive launcher TUI…" — no `DESIGN` token in the
  first 15 lines.

Track-C sweep loop (preserved): `for f in voyage/*.py; do head -n 15 "$f" |
grep -q DESIGN || echo "NO-DESIGN-REF: $f"; done` → the same 7.

## Repro

```bash
for f in voyage/*.py; do head -n 15 "$f" | grep -q DESIGN || echo "NO-DESIGN-REF: $f"; done
rg -n "TODO|FIXME" voyage/
```

## Fix candidates

One-line docstring amendments pinning each module to its DESIGN section +
converting the `augment.py` TODO into a tracked follow-up or deleting it.

## Refs

- AGENTS.md §12 ("every module docstring states purpose + DESIGN § ref").
- Preserved track result: `ses_f10013fc5ffeLLDtqZEwFbf3JR`, §7.

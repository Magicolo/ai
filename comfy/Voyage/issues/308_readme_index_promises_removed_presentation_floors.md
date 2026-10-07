# 308 — README docs-index still promises "presentation floors (≥24 fps, ≥1216×704)"

Severity: MEDIUM (pass-2 DESIGN-docs drift sweep).

## Technical description

`README.md:110` indexes AUGMENT.md as "finalize presentation floors (≥24 fps,
≥1216×704)". Floors were removed (2026-10-05 explicit-quality change): defaults are
`upscale=1/interpolate=1`, old `min_fps`/`min_*`/`use_model_pass` fail loud.

## Rationale

Entry-point doc misroutes every new reader to a deleted knob model.

## Live evidence

- Docs: `README.md:110` vs `docs/AUGMENT.md:3-8,12-25,47-49` (no floors; 1/1 ships
  source; old floor keys fail loud).
- Code: `voyage/config.py:534+` (`AugmentConfig`: `upscale = 1`, `interpolate = 1`,
  `interp_backend = "rife"`, `presentation_fps = None`).
- Command: `sed -n '100,112p' README.md` vs `sed -n '1,50p' docs/AUGMENT.md`.

Repro: read README index, then `configure` help — no floor flags exist.

## Source refs

`README.md:110`; `docs/AUGMENT.md:3-8,12-25,47-49`; `voyage/config.py:534+`.

## Online sources

- None (in-tree explicit-quality change is the anchor).

## Fix candidates

- Re-index as "explicit finalize quality (`--upscale`/`--interpolate`/
  `--presentation-fps`/`--interp-backend`, 1/1 ships source)".

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.

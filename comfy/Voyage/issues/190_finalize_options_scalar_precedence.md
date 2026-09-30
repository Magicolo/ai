# 190 — `finalize_run` legacy scalars override `options` for floors but are ignored for audio joints

- Severity: LOW
- Area: media finalizer — caller-contract asymmetry
- Files (as-read 2026-09-30):
  - `voyage/media.py:857-872` (`finalize_run` signature: legacy scalars + `options`)
  - `voyage/media.py:901-918` (settings build + `effective_min_*` overrides)
  - `voyage/media.py:828-830` (`effective_overlap_fraction`: hard-splice forces zero)

## Description

When both `options=` and legacy scalars are passed, the three `min_*` scalars
override `options` (`:916-918`: "Explicit `min_*` scalars override `options` when
both are given"), but the audio-joint scalars (`overlap_fraction`,
`overlap_cap_seconds`, `sample_rate`, `channels`) are silently ignored — the
`options` values win with no warning:

```python
# media.py:901-918 (as-read)
settings = options if options is not None else FinalizeOptions(
    skip_bad=skip_bad, sample_rate=sample_rate, channels=channels,
    overlap_fraction=overlap_fraction, overlap_cap_seconds=overlap_cap_seconds,
    joint_style="hard-splice" if overlap_fraction <= 0 else "blend", ...)
effective_min_fps = min_fps if min_fps is not None else settings.min_fps
# … no effective_overlap_fraction / effective_sample_rate / effective_channels
```

So `finalize_run(…, options=blend_opts, overlap_fraction=0)` renders a *blend*
despite the explicit `0` that means hard-splice on the legacy path (the
`overlap_fraction <= 0 → hard-splice` mapping at `:910` never runs when `options`
is given). A caller migrating incrementally — pinning floors via scalars while
passing an `options` built earlier — gets the legacy values honored for geometry
and silently dropped for audio. The docstring (`:880-883`) documents the `min_*`
override rule, which makes the silence on the other four scalars read as
"likewise honored" when they are not.

## Rationale

Mixed override rules in one signature are a caller trap: auditing one call site
requires knowing per-parameter precedence, and the two plausible readings
("scalars always win" / "`options` always wins") are each wrong for half the
parameters. The failure mode is silent behavior change (blend vs hard-splice is
audible in the final mix), not a crash.

## Live evidence (verified live 2026-09-30, host reads)

- `sed -n '901,918p' voyage/media.py` — only `min_*` get `effective_` resolution;
  `overlap_fraction`/`overlap_cap_seconds`/`sample_rate`/`channels` scalars have
  no use-site when `options is not None` (`rg -n "overlap_fraction" voyage/media.py`
  shows uses only inside the `options is None` constructor and
  `settings.effective_overlap_fraction()`).
- Docstring `:880-883` states the `min_*` override rule and nothing else,
  confirming the asymmetry is unspecified, not designed.

## Repro

Static: `opts = FinalizeOptions(joint_style="blend", overlap_fraction=0.10, …)`;
`finalize_run(d, out, options=opts, overlap_fraction=0)` → `settings` is `opts`
verbatim (`overlap_fraction=0.10`, blend) — the explicit `0` vanishes. Contrast
`finalize_run(d, out, options=opts, min_fps=0)` → floors disabled (scalar wins).

## Fix candidates

1. (Preferred) Decide one rule and document it: either all-or-nothing (passing
   `options` ignores every scalar — fail loud via `ValueError` when both are
   given, forcing call sites to pick), or uniform scalar-wins (add
   `effective_overlap_fraction`/`effective_overlap_cap`/`effective_sample_rate`/
   `effective_channels` mirroring `:916-918`). All-or-nothing is the smaller,
   safer change.
2. At minimum, document the actual precedence in the `finalize_run` docstring
   (`min_*` scalars win; joint/audio scalars are ignored when `options` given).
3. Tests: options+conflicting-scalars pins the chosen rule for each parameter.

## Refs

- In-tree: `voyage/media.py:785-830` (`FinalizeOptions`, `effective_overlap_
  fraction`); `voyage/media.py:857-918`; callers via `rg -n "finalize_run\("
  voyage/ tests/`.
- Not-a-duplicate: 022 is dropped *caption* pins in generate→worker handoff;
  147 is dropped *verbose/no_color* in generate→finalize handoff (different
  parameters, different handoff; extend 147's contract test if the all-or-nothing
  rule lands). 045 is the dataclass introduction (structure, not precedence).

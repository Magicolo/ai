# 025 — Config geometry has 5 sources of truth (defaults / TOML / presets / draft overlay / CLI)

- Status: open
- Severity: major (stale-value hazards; the file's own comment admits the
  causvid preset "lie")
- Area: structure — `voyage/config.py`
- Rank rationale: width/height/fps/latent/device can disagree silently; draft is
  a second model duplicating 4/6 fields.

## Technical description

`voyage/config.py:24-56` (`VideoConfig`), `:208-231` (`DraftConfig`), `:271-309`
(`default_config_toml`), `:372-421` (`apply_draft_overrides`, 111 lines, 7
`Model(**{**old.model_dump(), ...})` rebuilds), `:424-480`
(`_VIDEO_BACKEND_PRESETS` repeating `latent_shape [1,8,48,44,80]` in
fake+longlive2+ltxv), `:510-522` (`with_video_backend`). `DraftConfig` duplicates
4/6 `VideoConfig` fields with different defaults (640×352 vs 768×432).

## Why this is an issue

Geometry that can disagree silently will disagree eventually: a preset row, a worker native size, and a draft overlay each claim authority over width/height/fps/latent, and nothing reconciles them at startup — so a wrong-size latent or a surprise rescale surfaces hours into a run, not at config time. Every geometry change must now be made in five places and reviewed in five places, and the file's own comment admits the causvid preset already "lies", proving the hazard is live rather than theoretical. Whoever adds the next backend pays a full five-place audit just to stand still.

## Evidence

`sed -n '424,480p' Voyage/voyage/config.py`; `rg -n "latent_shape.*1, 8, 48"
Voyage/voyage/config.py` → 4 hits; `rg -n "class VideoConfig|class DraftConfig|
_VIDEO_BACKEND_PRESETS|def apply_draft|def with_video_backend|
def default_config_toml"`.

Verified live 2026-09-25 (`Voyage/`):

```
$ rg -n "class VideoConfig|class DraftConfig|_VIDEO_BACKEND_PRESETS|def apply_draft|def with_video_backend|def default_config_toml" voyage/config.py
24:class VideoConfig  208:class DraftConfig  271:def default_config_toml
372:def apply_draft_overrides  424:_VIDEO_BACKEND_PRESETS  510:def with_video_backend
$ rg -n "latent_shape.*1, 8, 48" voyage/config.py → 7 hits; the preset table
  repeats [1,8,48,44,80] 3× (lines 447,456,465)
```

## Reproduction

Compare preset geometry vs worker native geometry per backend; hardcode fps/latent
in one place and watch another disagree (the causvid case per the in-file comment).

## Source references

- `voyage/config.py` lines above.

## Resolution candidates

One `BACKEND_GEOMETRY` table (backend → width/height/fps/latent/device/
audio-pairing); `VideoConfig` defaults = `fake` row; delete `DraftConfig` as a
separate model (draft = named overlay `{width:640,height:352,latent:…,
take_seconds:45}`); single `resolve_config(base, backend, draft, overrides)`
replacing `apply_draft_overrides` + `with_video_backend`.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; all def-line refs
  re-verified live, current; pasted rg output into Evidence.
- Open: unify geometry; keep `with_video_backend` behavior via tests.

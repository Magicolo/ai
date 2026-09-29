# 045 — Boolean traps + overloaded signatures (`finalize_run`, `generate_blocks`, `apply_draft_overrides`)

- Status: open
- Severity: medium (silent semantic changes at a distance; two "absent" encodings)
- Area: structure — API design
- Rank rationale: `skip_bad=False` + `overlap_fraction=0.10` change finalize
  semantics from afar; `scene_cut: bool` threads through 4 layers; draft has two
  absent-encodings (`None` vs `""`).

## Technical description

```python
def finalize_run(run_dir, output_path, width=768, height=432, fps=24,
    skip_bad: bool = False, min_free_space_gib=0.0, sample_rate=48000,
    channels=2, overlap_fraction=0.10, overlap_cap_seconds=0.5) -> Path:  # media.py:406-418, 12 params
def generate_blocks(self, prompts, seeds, scene_cuts, output_path,
    width=NATIVE_WIDTH, height=NATIVE_HEIGHT, fps=NATIVE_FPS,
    segment_id="000000", prompt_plan_digest=None, requested_frames=None):  # causvid.py:678-690, 9 params
def apply_draft_overrides(config, *, draft=False, director=None, blocks=None,
    take_seconds=None, quantization=None, beats_per_segment=None, drift_every_n_segments=None):  # config.py:372-382
def apply_scene_cut_prefix(prompt: str, scene_cut: bool) -> str:
```

`apply_draft_overrides`' seven `None`-means-default params duplicate
`GenerateFormState`'s empty-string-means-default. LongLive `__init__(…,
quantization="fp8", local_attn_size=8, sink_size=8)` mixes precision + cache
geometry positionally.

## Why this is an issue

Twelve-parameter `finalize_run` and nine-parameter `generate_blocks` move semantic decisions — skip-bad, overlap, scene cuts — to call sites where positional-arg mistakes compile cleanly and change behavior silently; correctness currently depends on hand-auditing every call. The `scene_cut: bool` threaded through four layers cannot grow a third state without touching all four, and draft's two absent-encodings (`None` vs `""`) force every consumer to agree on which emptiness means "default". API-shape debt taxes every future caller.

## Evidence

`rg -n "def finalize_run" -A12 Voyage/voyage/media.py`; `rg -n "scene_cut: bool"
Voyage/voyage/workers/`; `rg -n "skip_bad|draft.*bool|force.*bool"
Voyage/voyage/cli.py Voyage/voyage/tui_state.py`.

Verified live 2026-09-25:

```
media.py:406:def finalize_run( ... 12 params ending overlap_cap_seconds=0.5
video_causvid.py:678: def generate_blocks( ... video_longlive.py:718: def generate_blocks(
config.py:372:def apply_draft_overrides( ... 7 None-means-default kwargs
video_longlive.py:42:def apply_scene_cut_prefix(prompt: str, scene_cut: bool) -> str
video_longlive.py:521 / video_causvid.py:667: further scene_cut: bool params
```
All def-line refs current.

## Reproduction

Call-site review: find every `finalize_run(` / `generate_blocks(` call and check
positional-arg correctness by hand (that's the bug class).

## Source references

- `voyage/media.py:406-418`; `voyage/workers/video_causvid.py:678-690`;
  `voyage/workers/video_ltxv.py:463`; `voyage/workers/video_longlive.py:608-616`;
  `voyage/config.py:372-382`; `voyage/workers/video_longlive.py:42,521`.

## Resolution candidates

`FinalizeOptions(skip_bad, overlap_fraction, overlap_cap, sample_rate, channels)`
+ `GenerateBlocksRequest(prompts, seeds, scene_cuts, geometry…)` dataclasses;
replace `scene_cut: bool` with `BoundaryKind.FRESH|CONTINUE`; collapse TUI
empty-string + config `None` into one `Unset` sentinel.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; all def-line refs
  re-verified live, current; pasted rg output into Evidence.
- 2026-09-25: partial resolution (this track) — `FinalizeOptions` dataclass
  landed in `voyage/media.py` (`skip_bad`, `sample_rate`, `channels`,
  `overlap_fraction`, `overlap_cap_seconds`, `joint_style: blend|hard-splice`
  with `__post_init__` validation + `effective_overlap_fraction()`), and
  `finalize_run` accepts `options=` (legacy scalars build an equivalent
  instance; `overlap_fraction=0` maps to hard-splice, so behavior is
  identical for existing callers; pinned by new integration tests). The
  explicit `joint_style` also answers 046's blend-vs-splice collapse for
  the finalize path. HOOK NOTES for the concurrent pass (out of my scope):
  `GenerateBlocksRequest` for the three `generate_blocks` (causvid/ltxv/
  longlive — triplicated, coordinate with 019), `BoundaryKind.FRESH|CONTINUE`
  for the `scene_cut: bool` thread (longlive:42,521 / causvid:667+), one
  `Unset` sentinel collapsing TUI empty-string + config `None`
  (config.py:372 `apply_draft_overrides` + `GenerateFormState`), keyword-only
  args + LongLive `__init__` precision/geometry split.
- Open: option dataclasses for video workers + Unset (concurrent pass owns).
- 2026-09-29: verification (this track) — re-read `voyage/media.py:505-613`
  live: `JointStyle`, `FinalizeOptions` (`skip_bad`, `sample_rate`,
  `channels`, `overlap_fraction`, `overlap_cap_seconds`, `joint_style` with
  `__post_init__` validation + `effective_overlap_fraction()`), and
  `finalize_run(..., options=...)` with legacy-scalar equivalence
  (`overlap_fraction=0` maps to hard-splice) all present. Pinned by
  `tests/test_integration.py:100-112` (blend + hard-splice finalize
  identically, unknown style rejects). No edit needed (already fixed);
  `GenerateBlocksRequest` / `BoundaryKind` / `Unset` remain concurrent-owned
  (loop.py/rpc.py/cli.py/supervisor.py untouched).
- 2026-09-29 (orchestrator): FinalizeOptions/JointStyle landed
  (verified); GenerateBlocksRequest + BoundaryKind + Unset sentinel
  remain — a worker-payload refactor needing GPU-adjacent validation.
  Kept OPEN narrowed to exactly that remainder.

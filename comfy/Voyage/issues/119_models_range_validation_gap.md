# 119 — `models.py` calibration/plan knobs have no range validation: inverted intent passes silently

- **Severity:** Low-Medium (config-adjacent — out-of-range style/audio/transition values persist, then mis-flag every segment)
- **File:line:** `Voyage/voyage/models.py:67-103` (`StyleSpec`), `:147-162` (`DirectorAudioPlan`), `:118-127` (`TransitionPlan`), `:36-39` (`ArtifactRef`), `:42-59` (`PromptStage`)
- **Area:** workers-internals tail — `voyage/models.py` validators (pass 1 covered `PromptStage` ordering (091) and `StyleSpec` band ordering; the *ranges themselves* were never checked)

## Description

`bands_ordered` rejects `min > max`, but nothing checks the values are meaningful in the first place. Verified live (all accepted without error):

- `StyleSpec(surrealism=5.0, transition_smoothness=-2.0, style_similarity_min=99.0)` — the inspector compares 0..1 metrics against these; `style_similarity_min=99` flags *every* segment as style collapse, `surrealism=-2` is nonsense input to prompt staging.
- `StyleSpec(motion_energy_min=5.0, motion_energy_max=6.0)` — passes `bands_ordered` (ordered!) while sitting entirely outside the 0..1 metric domain, so every segment reads BELOW band forever.
- `DirectorAudioPlan(energy=999.0, tempo_bpm=-120)` — energy feeds `bpm_for_energy` (clamped downstream by luck) and the fake worker's 0..1 validator; negative tempo flows toward ACE payloads.
- `TransitionPlan(transition_strength=42.0, estimated_duration_seconds=-1.0)` — negative duration into timeline math.
- `ArtifactRef(path='', bytes=-10)` — empty path + negative size persist into segment metadata.
- `PromptStage(block_start=-3, ...)` — negative block index passes the ordering check.

The tree validates the same quantities elsewhere (`AudioConfig.energy` 0..1 at `config.py:333-338`, audio-worker energy 0..1), so the *model layer* — the single source everything else trusts — is the one layer that doesn't.

## Rationale

These models are the supervisor↔worker contract (DESIGN §82: "no tensors, only artifact references"). A validator at this layer fails one bad director decision loudly at parse time; without it, a corrupted value (torn JSON, bad Qwen output that still parses, operator TOML typo surfacing via director defaults) flows into per-segment comparisons and silently biases the whole voyage. The `5.0..6.0` ordered-but-out-of-domain band is the worst shape: it passes the one check that exists.

## Evidence (verified live 2026-09-30, `voyage:latest`, `PYTHONPATH=/app/Voyage`)

```
stylespec junk accepted: 5.0 -2.0 99.0
audioplan junk accepted: 999.0 -120
transition junk accepted: 42.0 -1.0
artifact junk accepted: '' -10
promptstage neg start accepted: -3 5
```

Each line is a successfully constructed model that should have raised.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app -e PYTHONPATH=/app/Voyage voyage:latest python3 -c "
from voyage.models import StyleSpec
print(StyleSpec(prompt='x', style_similarity_min=99.0).style_similarity_min)  # 99.0, no error
print(StyleSpec(prompt='x', motion_energy_min=5.0, motion_energy_max=6.0).motion_energy_min)  # 5.0
"
```

## Fix candidates

1. `StyleSpec`: `0 <= min <= max <= 1` for the three metric bands + `style_similarity_min`; `0 <= surrealism/transition_smoothness <= 1` (match the documented 0..1 semantics; the inspector's `summarize_segment` guarantees 0..1 inputs).
2. `DirectorAudioPlan`: `0 <= energy <= 1`, `tempo_bpm > 0` (mirror `acestep.validate_bpm`/config `unit_range`).
3. `TransitionPlan`: `0 <= transition_strength <= 1`, `estimated_duration_seconds >= 0`.
4. `ArtifactRef`: non-empty `path`, `bytes >= 0` when present; `PromptStage`: `block_start >= 0` (keep the 091 ordering check).
5. Deliberately excluded: `RunState.fps`/`audio_buffer_seconds` (documented legacy tolerance at `models.py:209-214` — do not "fix").

## Refs

- `Voyage/voyage/models.py:36-103,118-162`; `Voyage/voyage/config.py:333-338` (energy precedent); `Voyage/voyage/audio/acestep.py:35-43` (BPM bounds precedent); DESIGN §82.
- Adjacent, not overlapping: 091 (`PromptStage` range *ordering* — this file is the *value domains*); 026 (prompt staging truncation — consumer side); 084 (validator consolidation — structural home for these checks).

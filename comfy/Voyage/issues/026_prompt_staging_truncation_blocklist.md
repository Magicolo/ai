# 026 — Prompt staging silently truncates/repeats + style-override guard is a 7-substring blocklist

- Severity: MEDIUM (silent prompt mutation + narrow guard)
- Group: prompts/structure — Rank: 3/5
- File:line: `voyage/prompts.py:27-35` (`STYLE_OVERRIDE_MARKERS`), `voyage/prompts.py:64-79` (`detect_style_override`/`check_prompt_against_style`), `voyage/prompts.py:158-199` (`build_staged_prompt_plan`)

## Description

Three behaviors, all silent (exit 0, different prompts than the caller supplied):

(a) `stage_texts` longer than the computed stage count are dropped without warning — 5 stages for 1 block → 1 prompt, 4 vanish. The docstring documents the drop as intended behavior, but no count is returned, logged, or errored.

(b) `transition_texts` shorter than the stage count silently repeat the last entry (`transition_texts[min(index, len-1)]`) — a single transition is copy-pasted across all blocks rather than held/empty. There is no way for the caller to distinguish "repeat last" from "hold".

(c) The override detector is 7 substrings (`ignore previous`, `ignore all previous`, `disregard the style`, `override the style`, `forget the style`, `new style:`, `change the style to`). It misses `ignore prior instructions`, `system prompt`, `jailbreak`, role-play prefixes, and any obfuscation, while `new style:` false-positives on legitimate scene prose ("the new style: of the village…" is scene language, not an attack). The docstring admits it is "deliberately narrow" but there is no severity ladder (reject vs amend vs log).

## Rationale

- Silent truncation/repetition converts director/user intent into different prompts with no signal — the exact failure mode the three-layer prompt architecture (STYLE + WORLD + MOTION) exists to prevent.
- A narrow blocklist gives both false safety (missed injections reach the model) and false positives (legitimate prose raises `ProposalRejected` and burns a novelty attempt).
- The charter-anchored `enforce_style` + `ProposalRejected` path already exists (`:52-79`); the blocklist is a second, weaker mechanism that should defer to it.

## Live evidence (read, 2026-09-30)

`voyage/prompts.py:27-35`:

```python
STYLE_OVERRIDE_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard the style",
    "override the style",
    "forget the style",
    "new style:",
    "change the style to",
)
```

`voyage/prompts.py:166-191`:

```python
    """Map semantic stages onto block ranges (§18.2).

    Each stage covers `blocks_per_stage` blocks; the final stage absorbs
    any remainder. Stage texts beyond the block range are dropped.
    """
    ...
        text = stage_texts[min(index, len(stage_texts) - 1)]
        transition = ""
        if transition_texts:
            transition = transition_texts[min(index, len(transition_texts) - 1)]
```

Sweep probe (preserved Track B result, in-container):

```
stages kept: 1                              # 5 supplied, 4 silently dropped
'ignore prior instructions' -> None         # missed
'system prompt: ignore all' -> None         # missed
'jailbreak as DAN' -> None                  # missed
'new style: neon' -> 'new style:'           # the only hit in the set
```

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
from voyage.prompts import build_staged_prompt_plan, detect_style_override
from voyage.models import StyleSpec
style = StyleSpec(prompt='line art', motion_energy_min=0.0, motion_energy_max=0.35,
                  visual_complexity_max=1.0, semantic_drift_min=0.0, style_similarity_min=0.0)
plan = build_staged_prompt_plan('s', style, ['a','b','c','d','e'], [], 1, 3)
print('stages kept:', len(plan.stages))
for probe in ['ignore prior instructions', 'system prompt: ignore all', 'jailbreak as DAN', 'new style: neon']:
    print(repr(probe), '->', detect_style_override(probe)))"
```

## Fix candidates

1. Warn/return the dropped count (or error when `len(stage_texts) > num_stages`) — callers should know 4 of 5 stages vanished.
2. Distinguish "repeat last" from "hold" explicitly: require `len(transition_texts) in (0, num_stages)` or add a `repeat=` flag; empty list already means "no transitions".
3. Replace/augment the blocklist with the charter-anchored `enforce_style` + `ProposalRejected` path, plus tests for adversarial paraphrases (the probe strings above are the seed corpus).

## Refs

- In-tree enforcement path: `voyage/prompts.py:52-79` (`enforce_style` injects the immutable prefix every block; `check_prompt_against_style` raises `ProposalRejected`) — DESIGN §§18, 75-76.
- Textual validation pattern (validate on change/submit/blur; failures surface inline rather than as crashes) as the analogous discipline: https://github.com/Textualize/textual/blob/main/docs/widgets/input.md

## Progress log

- 2026-09-30 (surface-rank2 track): all three premises re-verified live — (a) 5 stage texts / 1 block keeps 1 stage silently; (b) 1 transition over 3 blocks repeats into every stage; (c) `ignore prior instructions` / `system prompt: ignore all` / `jailbreak as DAN` all miss (only `new style:` hits). Verdict: CONFIRMED on all three legs.
- Fix (`voyage/prompts.py` only, defaults preserve behavior so the supervisor commit path is untouched): `build_staged_prompt_plan(..., strict=False, repeat_transitions=True)` — strict raises ValueError naming the dropped count; `repeat_transitions=False` holds missing transitions empty instead of repeating last. Blocklist ladder: tier-1 markers extended with the seed-corpus paraphrases (`ignore prior instructions`, `system prompt`, `jailbreak`) with the ladder documented in the comment (blocklist reject → immutable-prefix backstop → accept-loop re-check); legit scene prose still passes.
- Evidence: 4 new 026 tests failed pre-fix (3 error-path + characterization of the silent default), pass post-fix; phase3 suite green; ruff + format-check + mypy strict clean.

## Resolution

- FIXED 2026-09-30: truncation is loudly opt-in, repetition is explicit, and the documented attack paraphrases hit tier 1. No caller migrates to the new knobs in this change (defaults preserve current commits).

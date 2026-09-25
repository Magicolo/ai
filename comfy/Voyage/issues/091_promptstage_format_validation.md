# 091 — `PromptStage` accepts inverted ranges; `format_segment_id` accepts negatives/huge

- Status: fixed
- Severity: low (constructible invalid values break the `%06d`
  lexicographic/contiguous invariant `validate_run` relies on)
- Area: models/paths validation — `Voyage/voyage/models.py:42-47`,
  `Voyage/voyage/paths.py:31-32`
- Rank rationale: pass-2 finding; 038 asked for the *tests* — the missing
  *validation itself* is new.

## Technical description

`PromptStage(stage=0, block_start=5, block_end=2, prompt="x")` constructs fine;
`format_segment_id(-1)` → `'-00001'` and `format_segment_id(12345678)` →
`'12345678'`. No test touches either function directly (`rg format_segment_id
tests/` → only indirect use via run flows with 0,1,2…).

## Why this is an issue

Constructible invalid values break the `%06d` lexicographic/contiguous
invariant that `validate_run` relies on: an inverted `PromptStage` range or a
`'-00001'` segment id flows into directory names and ordering checks that
assume well-formed inputs. No direct unit test pins either function, so the
validation gap and the coverage gap protect each other.

## Evidence (code experiments, verified by orchestrator 2026-09-25)

```
$ python3 -c "...PromptStage(stage=0, block_start=5, block_end=2, prompt='x')..."
stage=0 block_start=5 block_end=2 prompt='x'
$ python3 -c "...format_segment_id(-1), format_segment_id(12345678)..."
'-00001' '12345678'
```

## Reproduction

One-liners above.

## Source references

- Files/lines above.

## Resolution candidates

`field_validator` on `PromptStage` (`block_end >= block_start`) and a range check
in `format_segment_id`; add the direct unit tests 038 already asked for.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep; both probes reproduced live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`models.py:42-47`, `paths.py:31-32` — both match); re-ran
  both probes (inverted range constructs, `-1→'-00001'` — confirmed).
- 2026-09-25 (fix): FIXED — `PromptStage` gained a
  `@model_validator(mode="after")` in `voyage/models.py:48-58` rejecting
  `block_end < block_start` (`ValueError`); `format_segment_id` in
  `voyage/paths.py:31-41` rejects out-of-range numbers (`ValueError` unless
  `0 <= number <= 999999`). Verified live: inverted ranges and `-1`/`1000000`
  raise; `0`/`42`/`999999` format exactly; both `prompts.py` construction
  sites use ordered ranges (unaffected). Tests:
  `test_prompt_stage_rejects_inverted_range` (plus valid/equal-bound pins)
  + `test_format_segment_id_range` in `tests/test_unit.py`. Gates: ruff +
  format + mypy strict clean on all scope files, 424 pytest passed
  in-container.
- Resolution: fixed as above; invalid ranges/ids are unconstructible and
  pinned by direct unit tests.

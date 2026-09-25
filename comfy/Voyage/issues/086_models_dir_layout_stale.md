# 086 — `models_dir_layout` frozen stale; test pins the stale key set

- Status: open
- Severity: low (helper lies about the models tree; exact-set test locks the lie)
- Area: registry — `Voyage/voyage/model_registry.py:654-663`,
  `Voyage/tests/test_longlive.py:22-33`
- Rank rationale: pass-2 finding; the helper is *used* (2 call sites) and stale,
  not dead (distinct from 046's unused helpers and 026's duplication).

## Technical description

`models_dir_layout` returns 7 keys (`wan_dir generator_ckpt ltxv_dir qwen_dir
minilm_dir acestep_dir manifest`) — no `causvid_dir`, `wan21_dir`,
`inspector_dir`, or LTXV text-encoder dir, although those stacks shipped later
(Stream D, Phase 5, Phase 7). `test_models_dir_layout_keys` asserts exact set
equality, so the staleness is now locked in: adding a key breaks the test.

## Why this is an issue

The helper lies about the models tree — tooling built on it misses real
directories (causvid, inspector, LTXV text encoder) that shipped later — and
the exact-set test punishes anyone who fixes the helper by failing on the new
key. Staleness plus a lock-in test means the lie is now self-preserving: each
new stack widens the gap while the test forbids closing it.

## Evidence

Live key set (re-run 2026-09-25, `PYTHONPATH=Voyage`):

```
$ python3 -c "from voyage.model_registry import models_dir_layout; ..."
['acestep_dir', 'generator_ckpt', 'ltxv_dir', 'manifest', 'minilm_dir', 'qwen_dir', 'wan_dir']
```

Seven keys — no `causvid_dir`, `inspector_dir`, or LTXV text-encoder dir —
while `download_causvid_models` et al. write those trees; `test_longlive.py:25`
asserts exact set equality (verified live).

## Reproduction

Read them; `models_dir_layout(tmp)` has no `causvid` key while
`download_causvid_models` writes one.

## Source references

- Files/lines above; call sites `test_ltxv.py:97`, `test_longlive.py:23`.

## Resolution candidates

Add the missing keys (or document the helper as longlive-era legacy) and update
the exact-set assertion.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`model_registry.py:654-663`, `test_longlive.py:22-33` —
  both match); re-ran layout probe (7 keys, pasted above).
- Open: update helper + test.
- 2026-09-25 (resolution): FIXED via the add-keys candidate (not the
  legacy-documentation one — the helper is used and should tell the
  truth). `models_dir_layout` (`voyage/model_registry.py`) gains
  `wan21_dir`, `causvid_dir`, `ltxv_text_encoder_dir`, and
  `inspector_dir` (full-word names, existing `*_dir` style) with a
  docstring listing every covered stack. Exact-set test
  `test_models_dir_layout_keys` (`tests/test_longlive.py`) updated to
  the 11-key set plus per-key suffix assertions; new
  `test_models_dir_layout_covers_shipped_stacks` in
  `tests/test_checkpoint_safety.py` pins each downloader subdir
  constant. `test_ltxv.py::test_layout_includes_ltxv_dir` unaffected.
  Scoped gates green (ruff + format + mypy strict + 73 tests); full
  `gates.sh` stays red only on another agent's in-flight `voyage/rpc.py`
  F401. Status: fixed.

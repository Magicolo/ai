# 064 — LTXV multi-block temp tails leak on failure; short-novel tail silently short; `fps` never validated

- Status: open
- Severity: medium (orphan `_chain*.mp4` in segment dirs; short anchor committed
  as 25-frame; bad fps reaches `mimsave`)
- Area: LTXV worker — `voyage/workers/video_ltxv.py:488-493,498,529-545,490-493`
- Rank rationale: pass-2 worker finding; distinct from 028 (which is the
  chaining-strategy choice — this is failure hygiene + tail-length + fps checks).

## Technical description

```python
novel_clips: list[Any] = []
chain_tails: list[Path] = []
...
tail_clip = novel[:, :, -conditioning_tail_frames:, :, :]
chain_tail = output_path.parent / f"{output_path.stem}_chain{index:02d}.mp4"
_save_mp4(tail_clip, chain_tail, fps)
chain_tails.append(chain_tail)
...
tape_path ... tape_tmp.replace(tape_path)
```

No `try/finally`: if block 1 raises (OOM — the documented retry path in
`_generate_block:390-408`), `chain00.mp4` remains beside the segment.
Separately, `novel[:, :, -25:]` on a short `novel` (e.g. 5 frames when the model
under-returns) silently yields 5 frames (negative-slice semantics) — the 25-frame
invariant is guarded in `split_prefix_novel:132-133` for the discard math only,
not the tail write. `generate_blocks:490-493` validates spatial size + frame
counts but never `fps` (0/negative flows into `_save_mp4`/`mimsave` and the tape).

## Why this is an issue

Orphan `_chain*.mp4` files pollute segment directories and confuse recovery and
tape accounting after the documented OOM-retry path fires. Worse, a short
`novel` tail is silently committed as a full 25-frame anchor, degrading the
next block's conditioning without any error — and an unvalidated `fps` reaches
both the encoder and the persisted tape, spreading one bad input across two
artifacts.

## Evidence

`rg` output (re-run 2026-09-25, chain-tail lifecycle + fps checks):

```
$ rg -n "chain_tails|try:|finally" Voyage/voyage/workers/video_ltxv.py
498:        chain_tails: list[Path] = []
511:                if not chain_tails:
516:                conditioning = str(chain_tails[-1])
534:            chain_tails.append(chain_tail)
540:        if len(chain_tails) == 1:
541:            chain_tails[0].replace(tail_path)
543:            chain_tails[-1].replace(tail_path)
544:            for stale in chain_tails[:-1]:
```

Tails are created/appended/replaced/unlinked only on the success path (the
`try/finally` hits at 741/768/788 belong to other functions); `generate_blocks`
validates spatial size + frame counts but has no `fps` validator
(`checked_request` only asserts `fps=int`).

## Reproduction

Force `_generate_block` to raise on block index 1 in a 2-block call →
`<stem>_chain00.mp4` orphan in `output_path.parent`; slice `-25:` on a 5-frame
`novel` → 5-frame tail committed as 25-frame anchor.

## Source references

- Files/lines above.

## Resolution candidates

Wrap the block loop in `try/except` that unlinks `chain_tails` on failure;
assert `tail_clip.shape[2]==conditioning_tail_frames` (fail loudly, don't commit
a short anchor); validate `fps>0` in `generate_blocks` (and
`handle_generate_blocks`).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`video_ltxv.py:488-493` validators, `:498` chain init,
  `:529-545` tail write/commit — all match); re-ran `rg` lifecycle probe
  (success-path only, confirmed).
- Open: implement + failure-path test.

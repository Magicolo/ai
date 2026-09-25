# 074 — Hardcoded device/shape constants: LTXV mask `.to("cuda")`; LongLive `stream_start_frame - 8*len`

- Status: open
- Severity: low (breaks `cuda:1`; misreports on config change)
- Area: worker constants — `voyage/workers/video_ltxv.py:319`,
  `voyage/workers/video_longlive.py:842`
- Rank rationale: pass-2 worker finding; device correctness + accounting
  correctness (distinct from 015's traffic volume / 029's shuttle count).

## Technical description

```python
result = (embeds, inputs.attention_mask.to("cuda"))   # ltxv.py:319
"stream_start_frame": self._stream.next_start_frame - 8 * len(prompts),  # longlive.py:842
```

Session device is `self._device` (LTXV `__init:234`, `_generate_block:375` uses
it for the generator); block size is `int(pipe.num_frame_per_block)`
(`video_longlive.py:506,536`; config sets `num_frame_per_block:8` at line 189).

## Why this is an issue

The hardcoded `"cuda"` breaks any multi-GPU run (`cuda:1` puts the mask on
`cuda:0` while the DiT and generator sit on `cuda:1` — a device-mismatch at
cross-attention), and the hardcoded `8` silently misreports stream start
frames the moment block size is configured differently. Both are config-drift
time bombs: correct today only because the defaults happen to match.

## Evidence

`rg` output (re-run 2026-09-25):

```
$ rg -n 'to\("cuda"\)' Voyage/voyage/workers/video_ltxv.py
319:        result = (embeds, inputs.attention_mask.to("cuda"))
```

Single hit while the rest of the file threads `self._device`; every other
`stream_start_frame`/`num_frame_per_block` site in `video_longlive.py` reads
`pipe.num_frame_per_block` (`:252-253`, `:506`, `:536`) or the config default
(`:189`) — only `:842` hardcodes `8`.

## Reproduction

Init LTXV with `"device":"cuda:1"` → mask on `cuda:0`, DiT/generator on `cuda:1`
→ device-mismatch at cross-attention; change `num_frame_per_block` → start-frame
misreport by `(8-actual)*len`.

## Source references

- Files/lines above.

## Resolution candidates

`.to(self._device)`; `self._stream.next_start_frame -
int(pipe.num_frame_per_block)*len(prompts)`.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 worker sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`video_ltxv.py:319`, `video_longlive.py:842` — both match);
  re-ran `rg` probes (outputs pasted above).
- Open: implement + multi-device/config test.

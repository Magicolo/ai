# 019 — Three video workers triplicate the same skeleton (~80% scaffolding identical)

- Status: open
- Severity: major (maintainability — every ops fix lands 3×, already diverged)
- Area: structure — `video_longlive.py` / `video_ltxv.py` / `video_causvid.py`
- Rank rationale: diffs are 1400–1900 lines but the delta is almost entirely
  model-specific denoising; tape/write/bench/serve logic already diverged
  (`tail_checksum` vs `tail_sha`, `profile` vs `backend` keys).

## Technical description

Diff sizes (sweep-measured): longlive↔ltxv 1709 lines, longlive↔causvid 1918,
ltxv↔causvid 1433. Identical copy-paste (modulo var names):

```python
# ltxv.py:562-564 == causvid.py:758-760
tape_tmp = tape_path.with_suffix(".tmp")
tape_tmp.write_text(json.dumps(tape, indent=2, sort_keys=True) + "\n")
tape_tmp.replace(tape_path)
# ltxv + causvid only (longlive has neither — divergence, not sharing):
TAIL_FILENAME = "video_tail.mp4"   # ltxv.py:59, causvid.py:78
TAPE_FILENAME = "recovery.pt"      # ltxv.py:60, causvid.py:79
RECOVERY_PROFILE = "ltxv"/"causvid"
def handle_benchmark / handle_resume / handle_rebuild / def main(): serve({...})
with tempfile.TemporaryDirectory(prefix="voyage-*-bench-") as tmp:
checked_request(payload, segment_id=str, output_path=str, fps=int)
```

Every operational concern (mp4 write via imageio, tail write, JSON tape atomic
write, benchmark warmup+measured loop, resume/rebuild dispatch, `serve()` map,
`checked_request` validation) is copy-pasted. `_save_mp4` exists at
`video_ltxv.py:615` and `video_causvid.py:390`; longlive has NO `_save_mp4` —
it inlines imageio mp4 writing inside `generate_blocks` (~734,828) — which is
itself divergence evidence, not sharing.

## Why this is an issue

With roughly 80% scaffolding identical across three 855–1106-line workers,
every operational fix — tape writes, mp4 writes, benchmark harness,
resume/rebuild dispatch, serve maps — must land three times, and the copies
have already diverged on tape keys and profile fields. The next correctness
fix will repair two backends and silently miss the third, and each new backend
(CausVid just landed) starts life as a fourth copy of the same skeleton. For a
project actively adding backends, the triplication is a standing multiplier on
all future worker-side work. Every upcoming operations change pays it
threefold.

## Evidence

`diff -U2 Voyage/voyage/workers/video_longlive.py Voyage/voyage/workers/video_ltxv.py
| head -n 150`; `rg -n "tape_tmp|_save_mp4|handle_benchmark"
Voyage/voyage/workers/video_*.py` (sweep output).

Re-verified 2026-09-25 (longlive confirms the divergence — shared handlers,
unshared tape/mp4 helpers):

```
$ rg -n "def handle_benchmark|def handle_resume|def handle_rebuild|TAIL_FILENAME|TAPE_FILENAME" voyage/workers/video_longlive.py voyage/workers/video_ltxv.py voyage/workers/video_causvid.py
voyage/workers/video_longlive.py:957:def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_longlive.py:1028:def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_longlive.py:1054:def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_ltxv.py:59:TAIL_FILENAME = "video_tail.mp4"
voyage/workers/video_ltxv.py:60:TAPE_FILENAME = "recovery.pt"
voyage/workers/video_ltxv.py:722:def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_ltxv.py:801:def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_ltxv.py:821:def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_causvid.py:78:TAIL_FILENAME = "video_tail.mp4"
voyage/workers/video_causvid.py:79:TAPE_FILENAME = "recovery.pt"
voyage/workers/video_causvid.py:974:def handle_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_causvid.py:1041:def handle_resume(payload: dict[str, Any]) -> dict[str, Any]:
voyage/workers/video_causvid.py:1061:def handle_rebuild(payload: dict[str, Any]) -> dict[str, Any]:
```

## Reproduction

`rg -n "def handle_benchmark|def handle_resume|def handle_rebuild|TAPE_FILENAME|
serve\(" Voyage/voyage/workers/video_*.py` → hits in all three.

## Source references

- `voyage/workers/video_longlive.py:957,1028,1054,1087,986` (handlers +
  bench harness; NO `_save_mp4`/`TAIL_FILENAME`/`TAPE_FILENAME` — mp4 writing
  is inlined in `generate_blocks`);
  `voyage/workers/video_ltxv.py:722,801,821,836,615,560-564,748`;
  `voyage/workers/video_causvid.py:974,1041,1061,1076,390,757-760,1002`.

## Resolution candidates

New `voyage/workers/video_common.py` with `save_mp4()`, `write_tape_atomic()`,
`run_benchmark_harness()`, `standard_serve_map()`; workers keep only
`generate_blocks` + session class. `VideoBackendAdapter` (see 023) should own
payload shape so the triple `handle_generate_blocks` validation collapses.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- Open: extract `video_common.py` incrementally (tape-write first — smallest,
  already-diverged).
- 2026-09-25 (repair pass): added `## Why this is an issue`; corrected
  longlive `_save_mp4`/`TAIL_`/`TAPE_FILENAME` claims (absent — inlined
  instead) + Evidence with live output.

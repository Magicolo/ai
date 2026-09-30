# 118 — Worker/RPC boundary silently coerces wire types: `None→"None"`, `"false"→True`, `1.9→1`, `True→1px`

- **Severity:** Medium-Low (boundary trust — wrong-type payloads execute with invented values instead of failing as INVALID_PAYLOAD)
- **File:line:** `Voyage/voyage/workers/video_common.py:103-108` (element coercion), `:116-124` (geometry/frames coercion), `:137` (`profile_stages=bool(...)`)
- **Area:** workers-internals tail — `GenerateBlocksRequest.from_payload` (issue 045 built the single validated struct, but the per-element conversions coerce instead of validating)

## Description

`from_payload` enforces only the *outer* container types (`checked_request(payload, prompts=list, ...)`), then converts every element with bare `str()` / `int()` / `bool()`, which never fail and frequently invent values. Verified live (all in-container, evidence below):

| wire value | expression | result | correct |
|---|---|---|---|
| `prompts: [None, 123]` | `str(item)` | `('None', '123')` | TypeError |
| `seeds: ['5', 1.9]` | `int(item)` | `(5, 1)` | TypeError (string seed; truncated float seed) |
| `scene_cuts: ['false', 0]` | `bool(item)` | `(True, False)` | the string `"false"` arms a scene cut |
| `profile_stages: 'false'` | `bool(payload.get(...))` | `True` | any non-empty string enables CUDA timing |
| `width: True` | `int(raw_width)` | `1` | 1px render (bool is int, `checked_request` never sees it) |
| `width: '512'` / `10.9` | `int(...)` | `512` / `10` | silent string-accept / truncation |

The `bool("false") == True` rows are the sharpest: a `"false"` scene-cut flag or profile flag does the opposite of what it says, with no error. The `None → "None"` prompt renders the literal word "None" as video content. Same coercion family at the scalar benchmark knobs (`int(payload.get("warmup", 1))` in `video.py:113-114`, `audio.py:122-123`, `sfx.py:87-88`, `director.py:493-494`, `audio_acestep.py:200-201`, `sfx_mmaudio.py:262-263`, `video_ltxv.py:844-845`, `video_causvid.py:1042-1043`, `video_longlive.py:1109-1110`): `warmup="2"` renders twice, `warmup=1.9` runs once — never INVALID_PAYLOAD.

`checked_request` (loop.py:144-167) already implements the right discipline for scalars (presence + exact type, bools rejected for int); the list-element layer simply never got it.

## Rationale

The worker boundary is the one place a malformed supervisor payload can still be caught before GPU work burns (the module's own §12-adjacent contract: "Raises KeyError/TypeError/ValueError — never asserts"). Coercion converts deterministic caller bugs into silent wrong-behavior runs: a mistyped seed truncates instead of erroring, breaking the determinism story (§62); a `"false"` scene cut restarts the stream the operator explicitly continued. Fail-loud is strictly cheaper than a 121-frame render of `"None"`.

## Evidence (verified live 2026-09-30, `voyage:latest`, `PYTHONPATH=/app/Voyage`)

```
prompts: ('None', '123')
seeds: (5, 1)
cuts: (True, False)
profile_stages string-false -> True
bool/str geometry -> 1 512
float geometry 10.9 -> 10
```

from `GenerateBlocksRequest.from_payload` with the payloads in the Repro table.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app -e PYTHONPATH=/app/Voyage voyage:latest python3 -c "
from voyage.workers.video_common import GenerateBlocksRequest as R
base = {'segment_id':'s','output_path':'/tmp/x.mp4','fps':24}
print(R.from_payload({**base,'prompts':[None],'seeds':[1]}, width_default=1, height_default=1).prompts)
print(R.from_payload({**base,'prompt':'hi','seed':1,'profile_stages':'false'}, width_default=1, height_default=1).profile_stages)
"
# → ('None',) and True — both should raise TypeError
```

## Fix candidates

1. Per-element `isinstance` checks in `from_payload` (mirror `checked_request` semantics: `str` elements must be `str` — and non-empty for prompts; `int` elements must be `int` excluding `bool`; `scene_cuts` elements must be `bool`), raising `TypeError` before any coercion.
2. Geometry: reject `bool` explicitly (`isinstance(raw, bool)` → TypeError) and require `int` (no `int(str)`/`int(float)` coercion); same for `fps` (currently only `checked_request` int-checked at the top, but `width`/`height` skip it entirely).
3. `profile_stages`: accept only real bools (or parse `"true"/"false"` case-insensitively and reject everything else).
4. Benchmark knobs: route `warmup`/`measured` through strict int validation (reject `bool`, `str`, non-integral `float`) before `validate_benchmark_counts` — one shared helper (084's `_validators.py` is the natural home).
5. Tests: wrong-element-type payloads → `TypeError` for every video worker (slim-safe, no GPU).

## Refs

- `Voyage/voyage/workers/video_common.py:81-138`; `Voyage/voyage/workers/loop.py:144-183` (the strict precedent + `validate_benchmark_counts`).
- Adjacent, not overlapping: 007 (top-level error taxonomy — this is element-level, inside the "validated" struct); 084 (structural collapse proposal — this file is the behavior it would fix); 060 (benchmark counts — validated *values*, not *types*).

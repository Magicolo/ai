# 023 — `VideoBackendAdapter` exists but the supervisor bypasses it (dead abstraction)

- Status: open
- Severity: major (two frame-accounting implementations = drift risk)
- Area: structure — `voyage/backends.py` vs `voyage/supervisor.py`
- Rank rationale: `backends.py:15` claims "contract is `VideoBackendAdapter`", but
  production never calls it; e.g. causvid 72-novel handling can drift between the
  two copies.

## Technical description

```python
# backends.py:230-254 — adapter builds the payload
def build_payload(self, request, output_path):
    ...
    payload["prompts"] = [request.prompt] * count
    payload["seeds"] = [request.seed + index for index in range(count)]
    payload["scene_cuts"] = [request.scene_cut] + [False] * (count - 1)
# supervisor.py:1092-1112 — same logic re-implemented inline, adapter never called
video_payload: dict[str, Any] = {"segment_id": ..., ...}
if streaming:
    video_payload["prompts"] = list(block_prompts)
    video_payload["seeds"] = [video_seed(config.seed, number, block) for block in range(num_blocks)]
    video_payload["scene_cuts"] = [state.destination_concept != state.current_concept] + [False]*(num_blocks-1)
```

`rg VideoBackendAdapter Voyage/voyage` hits only its own definition — production
(`supervisor._call_with_restart(... "generate_blocks", video_payload)`) duplicates
`build_payload` + `generate_segment` frame-accounting (`backends.py:270-300` vs
`supervisor.py:1145-1156`). Only `tests/test_backends_adapter.py` uses it.

## Why this is an issue

The module docstring declares `VideoBackendAdapter` the contract, but
production never calls it — so two frame-accounting implementations exist
(adapter versus supervisor-inline) with only the dead one covered by contract
tests. The live copy and the tested copy can drift silently, and the exact
boundary at risk (per-backend novel-frame math such as causvid's 72) is the
one Stream A just re-architected. The next accounting fix will land in one
copy while the other rots, and the test suite will keep passing because it
exercises the unused path. Frame-count correctness for every future segment
pays for the phantom abstraction.

## Evidence

```
$ rg -n "VideoBackendAdapter" Voyage/voyage Voyage/tests   # verified live 2026-09-25
Voyage/tests/test_backends_adapter.py:19:    VideoBackendAdapter,
... (10 test hits, zero supervisor/cli hits)
```

## Reproduction

The `rg` above; diff `backends.py:246-254` vs `supervisor.py:1106-1112`.

## Source references

- `voyage/backends.py:167-300`; `voyage/supervisor.py:1092-1141`;
  `Voyage/tests/test_backends_adapter.py`.

## Resolution candidates

Make `commit_one_segment` construct a `VideoSegmentRequest` and call
`adapter.generate_segment()` (transport = `self._call_with_restart` partial);
delete the inline payload/frame-accounting branches; keep one streaming-vs-fake
fork inside the adapter.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep; non-use re-verified live.
- Open: rewire supervisor through the adapter; adapter tests become contract tests.
- 2026-09-25 (repair pass): added `## Why this is an issue`;
  `VideoBackendAdapter` non-use re-verified live (only test hits); refs
  verified current.

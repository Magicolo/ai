# 023 — `VideoBackendAdapter` exists but the supervisor bypasses it (dead abstraction)

- Status: resolved (2026-09-25, supervisor-half track)
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
- 2026-09-25 (supervisor-half track): WIRED — no hook note from the
  backends track ever landed in this log (the note materialized instead
  as code: `backends.transport_from_restarting_call` + the
  `request_from_config` factory with `block_prompts/block_seeds`, which
  the backends track built for exactly this rewire). `commit_one_segment`
  now constructs the request via `VideoBackendAdapter.request_from_config`
  (geometry + seconds bridge from config; staged §18.2 prompts/seeds ride
  along for streaming) and calls `adapter.generate_segment` with the hook
  transport over `Supervisor._call_with_restart` (+ resume hook for
  streaming); the inlined payload/frame-accounting branches are deleted
  (`supervisor.py` `_render_video`, no more `video_payload` literal or
  `video_result` fork). Two supervisor-side gates stay by design — the
  issue-006 implausible-frame ceiling and the recovery-tape confinement —
  because the adapter normalizes leniently and drops tapes (helper-level
  tests pin both: `tests/test_commit_split.py` implausible/foreign-tape
  cases). Existing `test_implausible_reported_frames_rejected` +
  `test_foreign_recovery_path_rejected` green unmodified; full suite 691
  passed (3 worker-test failures are the concurrent `video_common`
  serve-map refactor, out of scope). Status → resolved.
- 2026-09-25 (contract half, backend-typing track — supervisor wiring stays
  with the supervisor track, status unchanged): `VideoSegmentRequest` +
  `VideoBackendAdapter.generate_segment` are now the documented contract in
  `voyage/backends.py` (module + class docstrings say so). Added, all in
  `voyage/backends.py`: optional `block_prompts`/`block_seeds` on the
  request (streaming only; length must equal the commit block count or
  `build_payload` raises ConfigurationError; fake rejects them) so the
  request can express the staged §18.2 plan that previously only the
  supervisor's inline code could; `VideoBackendAdapter.request_from_config`
  (stored VideoConfig → request: geometry + segment_frames/fps seconds
  bridge + registry state_mode + optional staged sequences); and
  `transport_from_restarting_call(restarting_call, worker, worker_name,
  segment_id, restart_hook=None)` — the transport injection point mapping
  the bound `Supervisor._call_with_restart` onto `Transport` (retries stay
  supervisor-side; the adapter never retries). The streaming-vs-fake fork
  lives only in `build_payload`. Tests: `tests/test_adapter_contract.py`
  (10 tests: request_from_config per row, staged sequences verbatim +
  mismatch rejections, fake shape + block rejection, transport forwarding
  incl. restart_hook, end-to-end causvid 72-novel over the injected
  transport, no-retry passthrough). Gates: same green scope as 022.
- Precise hook note for the supervisor track (verified against your
  in-flight rewire in `git diff`, which already builds the request inline
  in `_render_video` and overlays staging in `_staged_video_payload`):
  (1) replace the inline `VideoSegmentRequest(…)` constructor with
  `VideoBackendAdapter.request_from_config(config.video,
  segment_id=segment_id, prompt=proposed.prompt_plan.stages[0].prompt,
  seed=video_seed(config.seed, number, 0),
  scene_cut=(state.destination_concept != state.current_concept),
  block_prompts=proposed.block_prompts,
  block_seeds=[video_seed(config.seed, number, block) for block in
  range(proposed.num_blocks)])`, then DELETE `_staged_video_payload` —
  the new request fields make the overlay redundant (adapter raises on
  length mismatch, which is the fail-loud you want if plan and block
  count ever disagree); `display_payload` for `_segment_plan_info` then
  comes straight from `adapter.build_payload(request, video_out)`.
  (2) replace the local `_transport` closure's forward with
  `transport_from_restarting_call(self._call_with_restart, self._video,
  "video", segment_id, restart_hook=self._resume_video_worker if streaming
  else None)`, keeping only the `raw_result` capture wrapper around it
  (the issue-006 clamp re-gating intentionally stays supervisor-side:
  the adapter normalizes leniently where the supervisor raises MediaError
  — documented divergence, do not "fix" by aligning silently).
  Signature check: `_call_with_restart(worker, worker_name, segment_id,
  op, payload, restart_hook=None)` matches the helper's positional
  forward exactly. `BACKEND_STATE_MODES[config.video.backend]` and
  `VideoBackendAdapter(_transport, config.video.backend, config.video)`
  already typecheck against the new Literals — no annotation changes
  needed on your side for these.

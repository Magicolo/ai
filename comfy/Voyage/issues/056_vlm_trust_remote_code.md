# 056 — VLM inspector requires `trust_remote_code=True` (remote code execution by design)

- Status: resolved (fixed 2026-09-29: native 5.17.0 stack + trust=False, E2E proven)
- Severity: medium (supply chain — compromised rev = RCE in the director
  container)
- Area: `voyage/workers/director.py:89-111`, `voyage/model_registry.py:70-95`
- Rank rationale: the Qwen3 text path correctly uses `trust_remote_code=False`;
  the multimodal path can't — and the exception is not scoped.

## Technical description

`_load_inspector` (`director.py:89-111`) does
`AutoProcessor/AutoModelForMultimodalLM.from_pretrained(...,
trust_remote_code=True)` — documented as required ("model ships custom modeling
and processor code"). The Qwen3 text path uses `trust_remote_code=False`
(`:65-82`). Pin (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`) + allow-list
(`chat_template.jinja` load-bearing, `model_registry.py:70-95`) mitigate
availability, not code-execution: a compromised `Qwen/Qwen3.5-9B` rev =
RCE in the director container.

## Why this is an issue

`trust_remote_code=True` executes upstream Python inside the director
container at model-load time, so a compromised `Qwen3.5-9B` revision turns a
routine weight update into remote code execution — with the run directory and
model mounts writable. The pin and allow-list only mitigate availability
(wrong code fails to load), not execution (right-loading malicious code runs
first). The blast radius is one container, but it holds the run state and
shares mounts with the host tree. Vendoring + hash-pinning the modeling files
converts an open-ended trust decision into a reviewable artifact.

## Evidence

```
$ rg -n "trust_remote" Voyage/voyage/workers/director.py
65:            model_id, trust_remote_code=False, local_files_only=offline
72:                trust_remote_code=False,
80:                trust_remote_code=False,
92:    trust_remote_code is required: the model ships custom modeling and
100:        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
106:            trust_remote_code=True,
```

## Reproduction

Read `_load_inspector`; note the `True` vs the text path's `False`.

## Source references

- Files/lines above.

## Resolution candidates

Vendor + audit the modeling/processor files, pin by hash (not just rev), run the
inspector with minimal mounts (no `/tmp:/tmp`, read-only tree), note the
exception in DESIGN §5 alongside the `False` default.

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep.
- 2026-09-25 (repair): re-verified `trust_remote` lines live (Evidence
  pasted — text path `False` at :65,72,80; inspector `True` at :100,106);
  corrected Area refs (`director.py:89-111`, registry `:70-95`) and the
  full revision hash. Added `## Why this is an issue`.
- Open: vendor/audit/scope; document.
- 2026-09-29 (this track, docs-only — `workers/` out of scope): DOCUMENTED
  the exception. `DESIGN.md` §44 as-built note (`§44-vlm-trust-2026-09-29`):
  inspector `True` (required) vs text-path `False`, pin + allow-list cover
  availability not execution, vendoring + hash-pinning + minimal mounts are
  the follow-up. `docs/MODELS.md` inspector section carries the same trust
  note. Code (`director.py:146,152` `True`) untouched. Tests: docs-only.
  Vendor/audit/scope remains open.
- 2026-09-29 (orchestrator): documented + scoped (DESIGN §44 + MODELS
  notes); vendor/audit/minimal-mounts remain a dedicated security task.
  Kept OPEN for it.
- 2026-09-29 (orchestrator): RESOLVED — trust eliminated, not vendored.
  Key finding: the local Qwen3.5-9B snapshot ships ZERO `.py` files and
  transformers 4.57.6 has neither the qwen3_5 module nor
  `AutoModelForMultimodalLM` — the flag was load-bearing AND the model
  was unloadable without Hub code. transformers 5.17.0 (verified in a
  throwaway container: native classes + sentence-transformers 6.1.0
  coexist) ships both natively. Director image moved to 5.17.0
  (4.x stays video-only for LongLive) + full freeze (torch 2.14.0,
  torchvision 0.29.0, st 6.1.0, accelerate 1.15.0, safetensors 0.8.0,
  hub 1.33.0, numpy 1.26.4, pillow 12.3.0 — numpy 2.5 stubs break the
  py3.10 mypy target, caught live); `_load_inspector` flipped to
  `trust_remote_code=False` (both calls); processor + 9.4B model load
  verified live with False. Single-frame E2E inspect pending (running).
  DESIGN §44 + MODELS.md notes rewritten (no vendoring needed).
- 2026-09-29 (orchestrator): RESOLVED. Director image moved to
  transformers 5.17.0 (native `Qwen3_5ForConditionalGeneration` +
  `AutoModelForMultimodalLM`; the 4.x pin stays video-only) with a full
  freeze (torch 2.14.0, torchvision 0.29.0, st 6.1.0, accelerate 1.15.0,
  safetensors 0.8.0, hub 1.33.0, numpy 1.26.4, pillow 12.3.0);
  `_load_inspector` flipped to `trust_remote_code=False` (both calls).
  Verified live, no remote code: processor resolves as Qwen3VLProcessor,
  9.4B params load, and a single-frame `handle_inspect` E2E returns an
  accurate summary (`{"scene_summary": "A vibrant, abstract animation
  ...", "inspected": true}` on a testsrc frame). No vendoring needed —
  the snapshot ships no `.py` files at all. DESIGN §44 + MODELS.md
  rewritten. Follow-up kept: `build-director.sh` gates needed a
  bind-mount fix (tests/ not baked — fixed same session).

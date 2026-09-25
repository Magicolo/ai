# 056 — VLM inspector requires `trust_remote_code=True` (remote code execution by design)

- Status: open
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

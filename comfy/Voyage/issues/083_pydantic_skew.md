# 083 — Pydantic version skew across the RPC boundary (slim unbounded vs workers `<2.11`)

- Status: open
- Severity: low-medium (wire-type behavior can diverge silently between images)
- Area: packaging contract — `Voyage/pyproject.toml:7`
  (`"pydantic>=2.7"`), `Voyage/worker/Dockerfile.video:22`
  (`"pydantic>=2.7,<2.11"`), `Voyage/worker/Dockerfile.director:21` (same `<2.11`)
- Rank rationale: pass-2 finding; distinct from 011's general pins and 042's dev
  pins — this is a cross-image contract skew on the shared wire types.

## Technical description

The supervisor (slim image, floats to latest 2.x) validates `WorkerRequest`/
`WorkerResponse` against worker processes pinned below 2.11. Any 2.11+
serialization/validation behavior change silently straddles the JSONL-RPC
boundary (supervisor ↔ video/director). `voyage/models.py:170-186`
(`WorkerRequest`/`WorkerResponse`) is the shared contract with no version-gate test.

## Why this is an issue

The supervisor and the workers can validate the shared JSONL-RPC wire types
under different pydantic minor versions, so any 2.11+ serialization or
validation behavior change straddles the process boundary silently — one side
accepts what the other rejects, with no version-gate test to catch the skew.
Pinning the same bound everywhere (or documenting the float) makes the contract
explicit instead of accidental.

## Evidence

Pin lines (re-run 2026-09-25):

```
$ rg -n "pydantic" Voyage/pyproject.toml Voyage/worker/Dockerfile.video Voyage/worker/Dockerfile.director
Voyage/pyproject.toml:7:  "pydantic>=2.7",
Voyage/worker/Dockerfile.video:22:RUN pip install torchao==0.13.0 "huggingface_hub[cli]" "pydantic>=2.7,<2.11"
Voyage/worker/Dockerfile.director:21:    "huggingface_hub[cli]" "numpy>=1.26" "pydantic>=2.7,<2.11" "ruff" "mypy" "pytest" \
```

## Reproduction

Build slim vs video images and compare `pip show pydantic` versions;
`rg -n "pydantic" Dockerfile worker/Dockerfile.* pyproject.toml`.

## Source references

- Files/lines above; `voyage/models.py:170-186`.

## Resolution candidates

Pin the same upper bound in `pyproject.toml` (or document the float as
deliberate) and add a test asserting the bound appears in all three places.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`pyproject.toml:7`, `Dockerfile.video:22`,
  `Dockerfile.director:21`, `models.py:170-186` — all match); re-ran pin
  `rg` (pasted above).
- Open: align pins + contract test.

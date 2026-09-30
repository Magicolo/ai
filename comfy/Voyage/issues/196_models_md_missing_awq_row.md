# 196 — `docs/MODELS.md` documents the CPU director stack but not the default-CUDA AWQ stack

- Severity: LOW
- Area: docs — model catalog drift
- Files (as-read 2026-09-30):
  - `docs/MODELS.md:67-80` (Director section: `director-qwen8b` only, ~16 GB)
  - `voyage/model_registry.py:805-838` (`director-qwen4b-awq` spec: repo, revision `74d4bd2b…`, license)
  - `voyage/models_ensure.py:113-124` (CUDA placements serve AWQ; CPU keeps bf16 8B)
  - `voyage/config.py:433` (`cuda:1 … via a 4-bit AWQ model; "cpu" keeps the legacy bf16`)
  - `docs/INSTALL.md:43-45` (the only doc naming `Qwen3-4B-AWQ`, ~2.6 GiB)

## Description

`MODELS.md`'s Director section documents one stack:

```markdown
# docs/MODELS.md:67 (as-read)
## Director — Qwen3-8B + MiniLM (`models download director-qwen8b`, ~16 GB)
```

with the Qwen3-8B pin row (`:71`) and no AWQ row anywhere in the file
(`rg -ni "awq|4b" docs/MODELS.md` → zero content hits — the only `4b…` hit is
the unrelated LTXV commit hash). But the default CUDA path downloads the other
stack: `required_specs` picks `director-qwen4b-awq` whenever
`config.director.device != "cpu"` (`models_ensure.py:116-118`), and the default
device is CUDA (`config.py:601`, `:433` comment). So an operator following the
model catalog provisions ~16 GB of Qwen3-8B (+ MiniLM) while `generate` on a GPU
box fetches and serves `Qwen3-4B-AWQ` (~2.6 GiB) — different weights, different
disk, different VRAM, different license row, none of it in the catalog. The
information exists exactly once, in passing, in `INSTALL.md:43-45`
(`Qwen3-4B-AWQ`, ~2.6 GiB, cuda:1) — the install narrative, not the catalog —
and in the registry pins, which operators are told not to read.

## Rationale

MODELS.md is the single weight catalog (repos, revisions, sizes, licenses,
`models download` targets). A default-path 2.6 GB stack with its own revision
pin and license missing from it means under-provisioning (disk), wrong
expectations (which model serves decisions), and an unaudited license surface.
The fix is one table section mirroring the qwen8b row.

## Live evidence (verified live 2026-09-30, host rg)

- `rg -n -i "qwen|director" docs/MODELS.md` → Director section `:67-80` names
  only `director-qwen8b`; `rg -ni "awq" docs/MODELS.md` → no hits.
- `sed -n '67,80p' docs/MODELS.md` vs `sed -n '805,838p' voyage/model_registry.py`
  (awq spec: `QWEN4B_AWQ_HF_REPO`, revision `74d4bd2bd4bff9cafc9345221320bffb08b406a3`,
  `record_builder`, `success_message`) — cataloged nowhere.
- `sed -n '113,124p' voyage/models_ensure.py` — the default-CUDA selection rule
  making AWQ (not 8B) the common-case stack.

## Repro

Docs-only: `rg -ni "awq" docs/MODELS.md README.md docs/INSTALL.md` → INSTALL
only; follow MODELS.md's Director section to provision a GPU box → first
`generate` still downloads `director-qwen4b-awq` (verify message names the
missing AWQ snapshot, not the cataloged 8B).

## Fix candidates

1. Add a `Director — Qwen3-4B-AWQ (models download director-qwen4b-awq, ~2.6 GB)`
   section to MODELS.md: repo + revision + files + license + "default on CUDA;
   `--director-device cpu` opts into the 8B bf16 stack" (mirror INSTALL:43-45
   wording, keep INSTALL as pointer).
2. Note the selection rule once in the Director intro (device-gated, same rule
   `models_ensure` implements) so the two rows read as alternatives, not rivals.
3. Gate: extend 065's `rg` mirror test with an awq row assertion.

## Refs

- In-tree: `docs/MODELS.md:67-80`; `voyage/model_registry.py:74-76,630-660,
  805-838,1216-1227`; `voyage/models_ensure.py:113-124`; `voyage/config.py:425-440,
  595-610`; `docs/INSTALL.md:40-50`.
- Not-a-duplicate: 065 names `film`, `realesrgan-anime`, `sfx-mmaudio` in
  INSTALL/README download lists — AWQ is a different stack in a different file
  (the MODELS catalog, which 065 never cites); 091 proposes `SFX.md`/`AUGMENT.md`
  operator docs (feature guides, not catalog rows); 146 is the `models list`
  printout (CLI surface, which already lists both director backends correctly).

## Progress log (Group C, 2026-09-30)

- Verdict: premise PARTIALLY fixed already — the AWQ section exists live
  (`docs/MODELS.md:81-89`, added by a concurrent agent) but carried no
  literal repo/revision/license (only "pinned snapshot in registry").
- Enriched the row with literals verified live in code:
  `registry_records.py:101-119` (`QWEN4B_AWQ_HF_REPO = "Qwen/Qwen3-4B-AWQ"`,
  `QWEN4B_AWQ_HF_REVISION = "74d4bd2bd4bff9cafc9345221320bffb08b406a3"`,
  `QWEN4B_AWQ_LICENSE = "Apache 2.0"` + license URL) and the selection rule
  (`models_ensure.py:124-129`: non-cpu device → `director-qwen4b-awq`).
  Also recorded that the AWQ spec bundles the shared MiniLM snapshot
  (`model_registry.py:628`) so operators do not double-provision embeddings.
- Files changed: `docs/MODELS.md` (AWQ section only).
- Gates: docs-only.

## Resolution

- Done. Residual: none (the 065-mirror-test extension from fix candidate 3
  is left for the test-owning track).

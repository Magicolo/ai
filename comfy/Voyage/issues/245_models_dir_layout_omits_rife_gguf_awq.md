# 245 — `models_dir_layout()` omits `rife/`, `director-gguf/`, AWQ dirs; docstring claims full coverage

Severity: MEDIUM (track B-09).

## Technical description

`MODEL_SPECS` has 13 rows (`director-qwen8b`, `director-qwen4b-awq`,
`director-qwen35-gguf`, `inspector-qwen35`, `audio-acestep`, `sfx-mmaudio`, `ltxv-2b`,
`causvid`, `ltx25`, `ltx23`, `film`, `rife`, `realesrgan-anime`). `models_dir_layout()`
returns 13 paths but no `rife_dir`, no `director_gguf_dir`, no AWQ dir;
`film_dir`/`realesrgan_dir` present. Both `film` and `rife` live under the same
`frame_interpolation/` subdir, so one key can't distinguish them anyway.

## Rationale

An inventory function that silently misses 2–3 shippable stacks misleads disk-preflight
and docs. The docstring ("Covers all shipped stacks… FILM …, Real-ESRGAN…") doesn't
mention RIFE or the GGUF at all.

## Live evidence

`models_dir_layout` return keys: `wan21, causvid, ltx25, ltx23, ltxv, ltxv_text_encoder,
qwen, inspector, minilm, acestep, sfx, film, realesrgan, manifest` — no `rife`, no
`QWEN35_GGUF_SUBDIR` (`Qwen3.5-4B-GGUF`), no `QWEN4B_AWQ_SUBDIR`.

Repro: `models_dir_layout(models)` → `KeyError`-free but `rife`/`gguf` weights have no
listed home; compare `MODEL_SPECS.keys()` vs layout values to see the gap.

## Source refs

`voyage/model_registry.py:1625-1649,699-1196`; `voyage/registry_rife.py`;
`voyage/model_registry.py:584-596` (GGUF pins).

## Online sources

- None (in-tree MODEL_SPECS table is the anchor).

## Fix candidates

- Add `rife_dir` (even if same dir as film, name it), `director_gguf_dir`,
  `director_awq_dir`; or document that film/rife share a dir and AWQ/GGUF are
  intentionally unlisted; test asserting every spec's `relative_dir` appears in the
  layout.

## Log

- 2026-10-07: filed from read-only Track B sweep; no code touched.

# 310 — README video/director bullets omit the defaults (ltx25, llama)

Severity: MEDIUM (pass-2 DESIGN-docs drift sweep).

## Technical description

Bullets list video as "fake, LTXV 2B, or CausVid — all on CUDA" and director as
"deterministic or Qwen3-8B + MiniLM". Defaults are `ltx25` (joint A/V) and `llama`
(Qwen3.5-4B GGUF sidecar); `ltx25`/`ltx23` video and `llama`/`qwen-AWQ` director paths are
absent.

## Rationale

Quick-start readers never learn the default path exists.

## Live evidence

- Docs: `README.md:8-14` (+ `:40-44` model list also centers ltxv).
- Code: `voyage/config.py:240,333` (video default `ltx25`),
  `voyage/config.py:595` (`DirectorConfig.backend = "llama"`), `voyage/config.py:20` (5
  video backends incl. ltx25/ltx23).
- Command: `grep -n "backend.*llama\|backend.*ltx25" voyage/config.py`.

Repro: read README bullets, then `configure vdemo` — manifest says `ltx25`/`llama`,
neither introduced on the front page.

## Source refs

`README.md:8-14,40-44`; `voyage/config.py:20,240,333,595`.

## Online sources

- None (code defaults are the authority).

## Fix candidates

- Video bullet → "fake | ltxv | causvid | ltx25 (default, joint A/V) | ltx23"; director
  bullet → "deterministic, qwen-AWQ (opt-in), or llama Qwen3.5 sidecar (default)".

## Log

- 2026-10-07: filed from read-only pass-2 docs-drift sweep; no code touched.

## Evaluation (2026-10-07)

Claim re-verified live: `README.md:8-14` front bullets listed video as
"fake, LTXV 2B, or CausVid" and director as "deterministic built-in, or Qwen3-8B
+ MiniLM", while `voyage/config.py:240,333` default video to `ltx25`
(5 backends at `:20`) and `DirectorConfig.backend = "llama"` at `:620`.
Deeper README sections (`:40-44` model list, `:63` native note, `:72` backend
flag incl. default) already documented ltx25 — the drift was confined to the
front bullets, as filed. No downgrade.

## Progress log

- 2026-10-07: `README.md:8-13` bullets rewritten per the fix candidate —
  director: "deterministic built-in, llama Qwen3.5 sidecar by default, or opt-in
  Qwen AWQ on CUDA, all with MiniLM novelty embeddings"; video: "fake testsrc
  built-in, LTXV 2B tail-chained extensions, CausVid DMD causal rollouts, or
  LTX-2.5 / LTX-2.3 joint A/V — ltx25 is the default — all but fake on CUDA".
  Grep-verified against `voyage/config.py:20,240,333,620`.

## Resolution (2026-10-07)

RESOLVED. Quick-start readers meet the default path on the front page; nothing left open.

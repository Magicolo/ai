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

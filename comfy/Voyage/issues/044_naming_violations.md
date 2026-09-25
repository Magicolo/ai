# 044 — Naming-standard violations are pervasive (single letters, abbrevs, `vae` in our identifiers)

- Status: open
- Severity: medium (standard is explicit; `s`/`p`/`tmp` mean different things per
  file)
- Area: standards — naming (`s/p/i`, `tmp`, `num/den`, `dim`, `vae`)
- Rank rationale: the standard exists so GPU/multimedia code stays greppable;
  ~25+ single-letter sites plus banned-domain acronyms in our own identifiers.

## Technical description

Standard: no acronyms/abbreviations/single-letters; domain terms
`lora/comfy/json/http` allowed; prefer `autoencoder/text_encoder/base_model`
over `vae/clip/unet` in OUR names; node `class_type` literals stay literal.

- Single-letter loop/comprehension vars: `media.py:72,73,97,98,186,191,194,199,
  294,353,367,370,435` (`[s for s in …]`, `for s in slices`), `supervisor.py:
  314-315` (`p for p in …`, `lambda p: p.name`), `cli.py:566,596,620,1069`,
  `model_registry.py:322,332,350,395,416` (`p.stat() for p in …`),
  `vision/metrics.py:44,45,82,86`, `concepts.py:57,58`,
  `video_longlive.py:882` (`int(v) for v in raw_shape`),
  `video_causvid.py:169,810,870` (`dim` — itself an abbrev).
- Abbreviations: `tmp` ×7+ (`workers/video.py:59,63`, `video_ltxv.py:748,756`,
  `video_longlive.py:986,993`, `workers/audio.py:54,58`, `cli.py:944,946`,
  `media.py:466,467`), `tape_tmp` (`video_ltxv.py:562`, `video_causvid.py:758`),
  `num, den` (`media.py:82-83`), `dim`, `cfg` in f-string
  (`video_causvid.py:238`), `tmp_npy` (`concepts.py:158`).
- Banned-domain acronyms in OUR identifiers (not upstream attribute accesses,
  which are correctly left literal): `video_causvid.py:414`
  `def _vae_encode_slice(vae: Any, …)` + `video_ltxv.py:264`
  `vae = CausalVideoAutoencoder.from_pretrained(…)` — standard demands
  `autoencoder`. (`pipe.vae`, `clip`, `unet` attribute accesses on upstream
  objects are excluded.)

## Why this is an issue

The naming standard exists so GPU and multimedia code stays greppable under pressure — when `s` means streams in one file, slices in the next, and shards in a third, every debugging session starts with re-deriving local dialects instead of reading code. `vae` in our own identifiers contradicts the documented `autoencoder` convention and teaches newcomers the standard is decorative. Renames are cheap mechanical work now and expensive archaeology later, after the names have spread into logs, docs, and muscle memory.

## Evidence

`rg -n "for (s|p|i|d|v|x|y) in" Voyage/voyage` → 25+ hits (sweep transcript);
`rg -n "\btmp\b|tape_tmp|\bnum\b|\bdim\b" Voyage/voyage`.

Verified live 2026-09-25: per-file single-letter-loop counts —
`media.py:15, model_registry.py:5, vision/metrics.py:4, cli.py:4, concepts.py:2`
(+ others); `tape_tmp` confirmed at `video_ltxv.py:562-564`,
`video_causvid.py:758`; `tmp_npy` at `concepts.py:158`; `num, den` at
`media.py:82-83`. Refs current.

## Reproduction

The two `rg` commands above; `rg -n "(def \w+\([^)]*\bvae\b|^\s*vae\s*=)"
Voyage/voyage`.

## Source references

- Files/lines above.

## Resolution candidates

Mechanical rename pass — `stream_info`, `segment_dir`, `shard_path`,
`frame_index`, `numerator/denominator`, `temp_dir`, `tape_scratch_path`,
`latent_dim`, `autoencoder`; add the three patterns to ruff (`PLR` + a local `rg`
gate in `gates.sh`) so they stay fixed.

## Investigation / progress / resolution log

- 2026-09-25: found by structure sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; single-letter /
  tmp / tape_tmp / num-den refs re-verified live, current; pasted counts into
  Evidence.
- Open: rename pass + lint gate.

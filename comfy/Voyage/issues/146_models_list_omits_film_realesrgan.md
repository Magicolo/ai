# 146 — `models list` omits `film` + `realesrgan-anime` although download/verify support them

- Severity: LOW (docs gap in `--help`-adjacent output; users can't discover two real targets)
- Area: config/CLI — models-verb consistency
- Files (as-read 2026-09-30; no edits in these ranges):

## File:line

- `voyage/cli.py:337-347` (`cmd_models list`: prints video/audio/sfx/director/inspector lines — no film/realesrgan line)
- `voyage/cli.py:307-319` (`_download_film`: real downloader, prints checkpoint bytes + manifest)
- `voyage/cli.py:322-334` (`_download_realesrgan`: real downloader, same contract)
- `voyage/cli.py:405-418` (`cmd_models download` dispatch: `film` → `_download_film`, `realesrgan-anime` → `_download_realesrgan`, plus `known_targets` list)
- `voyage/cli.py:372-376` (as-read: `download` branch header, `target` default `longlive2-bf16`, `ltxv-2b` case — cited per draft, content noted as-read)

## Description

`voyage models download film` and `voyage models download realesrgan-anime` work (dispatch at :408-411, downloaders at :307-334), and `voyage models verify` checks both (`verify_film_models`/:363, `verify_realesrgan_models`/:365, folded into `all_ok`/:370). But `voyage models list` (:339-347) prints only the video/audio/sfx/director/inspector rows — the two augment-floor stacks are undiscoverable from the list output. A user reading `list` to learn what to download never learns these two exist; a user reading `verify` output sees `film`/`realesrgan` lines that `list` never mentioned.

## Rationale

- `list` is the discovery verb; `download`/`verify` are the action verbs. Discovery must be a superset of the actionable targets, never a subset.
- The omission dates to the Track-A augment slice (downloaders + verify + registry specs landed; the `list` print block was not extended).
- Zero behavioral risk — pure output addition — so this is the cheapest file in the batch.

## Live evidence

```
cli.py:340-346  video: fake|longlive2-bf16|ltxv-2b|causvid
                 audio: fake|acestep
                 sfx: fake|mmaudio
                 director: deterministic|qwen3-8b
                 inspector: skipped|qwen3.5-9b
                 # (no film / realesrgan line)
cli.py:363-366  verify film + realesrgan, folded into all_ok at :370
cli.py:408-411  target film → _download_film; realesrgan-anime → _download_realesrgan
```

## Repro

```bash
docker run --rm -v "$PWD/Voyage:/app" -w /app voyage:latest \
  python3 -m voyage.cli models list 2>&1 | grep -ci 'film\|realesrgan'
# Expect: 0 (both supported by download/verify, neither listed).
```

## Fix candidates

- Add two `print` lines to the `list` branch (e.g. `augment: film (FILM interpolation weights)` + `augment: realesrgan-anime (Real-ESRGAN anime upscaler)`), matching the existing row style.
- Test: `models list` output names every `known_targets` entry (or every download-dispatch target).

## Refs

- `voyage/cli.py:337-347` vs `:348-371` (verify) vs `:372-418` (download dispatch).

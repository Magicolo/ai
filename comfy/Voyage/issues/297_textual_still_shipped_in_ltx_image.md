# 297 — `textual==8.2.8` still shipped in `voyage-ltx` image

Severity: MEDIUM (pass-2 TUI-remnant sweep).

## Technical description

`DESIGN.md:8527` claims `textual` removed (pyproject + lock + worker Dockerfiles). Slim
side is true; LTX side is not: `worker/requirements-ltx.txt:105` still pins
`textual==8.2.8` and `worker/Dockerfile.ltx:53-55` installs that file.

## Rationale

#264 covers only `pyproject.toml` per-file-ignores for deleted `voyage/tui*.py`. This is
the runtime dep itself, different file, different effect (bytes + attack surface in the
GPU image).

## Live evidence

```
$ rg -n "^textual|textual==" pyproject.toml requirements.lock Dockerfile
(empty=absent in slim)
$ rg -n "textual" worker/requirements-ltx.txt worker/Dockerfile.ltx worker/Dockerfile.video
worker/requirements-ltx.txt:11:# Dockerfile.ltx ... and textual (console/TUI parity
worker/requirements-ltx.txt:105:textual==8.2.8
worker/Dockerfile.ltx:53:# except the torch trio above, plus textual for console/TUI parity.
worker/Dockerfile.ltx:54:COPY worker/requirements-ltx.txt /tmp/requirements-ltx.txt
worker/Dockerfile.ltx:55:RUN python3.11 -m pip install -r /tmp/requirements-ltx.txt
worker/Dockerfile.video:177:# ... (textual launcher TUI removed ... — comment-only, no install)
```

`worker/Dockerfile.video:175-179` installs only `rich==15.0.0` (clean); `Dockerfile.ltx`
has no such exclusion.

Repro: `grep -n textual worker/requirements-ltx.txt` → line 105; `grep -n textual
pyproject.toml` → empty; `ls voyage/tui*.py` → No such file (no consumer).

## Source refs

`worker/requirements-ltx.txt:11,105`; `worker/Dockerfile.ltx:53-55`.

## Online sources

- None (in-tree two-verb deletion as-built is the anchor).

## Fix candidates

- Delete `textual==8.2.8` + header comment from `requirements-ltx.txt`, reword
  `Dockerfile.ltx:53` to the video-image pattern; re-freeze notes per
  `requirements-ltx.txt:3-8`.

## Log

- 2026-10-07: filed from read-only pass-2 TUI-remnant sweep; no code touched.

## Evaluation

- 2026-10-07 (Group L): re-read live — `worker/requirements-ltx.txt:105`
  pinned `textual==8.2.8`, `worker/Dockerfile.ltx:53` cited console/TUI
  parity; `ls voyage/tui*.py` → no such file, `rg textual voyage/` → only a
  comment containing the English word "textually" plus removed-TUI notes,
  zero imports. Slim side already clean. Issue is LIVE, not stale.

## Progress log

- 2026-10-07 (Group L): deleted the `textual==8.2.8` pin (freeze 106 → 105
  rows), dropped the stale parity clause from the requirements header, and
  reworded the `Dockerfile.ltx` comment to the video-image pattern
  (TUI-removed note, rich-only console). No other pins touched, so the
  `requirements-ltx.txt:3-8` re-freeze procedure does not trigger (pure row
  deletion cannot skew remaining pins; rebuild-time `pip freeze` will
  confirm). No image builds per mandate.

## Resolution (2026-10-07)

- RESOLVED. Files changed: `worker/requirements-ltx.txt`,
  `worker/Dockerfile.ltx` (comment only). Verification: `grep -n textual`
  shows no pin/install/import — only historical removed-TUI notes and one
  English-word comment. Left open: nothing.

# 055 — Unverified `git clone https://…` for LongLive/ACE-Step/LTXV/CausVid (+ floating reqs; license note)

- Status: resolved (verified 2026-09-25)
- Severity: medium (supply chain; plus a non-commercial/share-alike weight in a
  general image)
- Area: `Voyage/worker/Dockerfile.video:26-28,46-47,63,79-80`
- Rank rationale: commits pinned (good) but fetched over plain HTTPS with full
  history and no verification; upstream `requirements.txt` floats everything.

## Technical description

Clones pinned to commits (`6b36d20`, `ca1e85fe`, `4b2d053`, `adb6a5e` — good)
but: plain HTTPS git, no hash/signature check beyond the checkout hash, full
history, no `--depth 1`/`--branch` guard, and `pip install -r
/opt/longlive/requirements.txt` (`:28`) floats the entire upstream transitive
set. The CausVid weight (`CC BY-NC-SA 4.0`, `model_registry.py:193`) is
non-commercial/share-alike — pulling it into a general-purpose image has license
implications the Dockerfile doesn't surface.

## Why this is an issue

Pinned commits without transport verification or hash checks mean a
compromised mirror or MITM'd fetch can inject arbitrary code into the GPU
worker image at build time — the highest-privilege container in the fleet —
with no signal until (or unless) someone diffs the checkout. Floating
transitive requirements add non-reproducible builds on top: two builds weeks
apart can pull different code under identical pins. The CausVid NC/SA weight
adds legal exposure if the image is ever shared. Blast radius is the entire
video stack; hardening is cheap (`--depth 1`, `rev-parse --verify`, vendored
tarballs, license notice).

## Evidence

```
$ grep -n "git clone|requirements.txt|checkout" Voyage/worker/Dockerfile.video
26:RUN git clone https://github.com/NVlabs/LongLive /opt/longlive \
27:    && git -C /opt/longlive checkout 6b36d20ec6f7958d29d11a704dfa64611a9f2572 \
28:    && pip install -r /opt/longlive/requirements.txt
46:RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5 \
47:    && git -C /opt/ACE-Step-1.5 checkout ca1e85fe9430179831e6bc6be790c332190a3866 \
63:(ltx-video via pip git+https, not clone — same unverified-fetch class)
79:RUN git clone https://github.com/tianweiy/CausVid /opt/causvid \
80:    && git -C /opt/causvid checkout adb6a5ecd07666b4d0290042915c8406e6d5ce22 \
```

`sed -n '26,28;46,47;63,64;79,80p' Voyage/worker/Dockerfile.video` (output
above — LTXV at :63 is `pip install ... git+https://...@4b2d053...`, not a
clone; same unverified-fetch class).

## Reproduction

Inspect the Dockerfile clone lines; check for `--depth`, hash verification, and
license surfacing (all absent).

## Source references

- `Voyage/model_registry.py:194-195` (`CAUSVID_LICENSE` NC/SA declaration).

## Resolution candidates

`--depth 1` + `git rev-parse --verify <commit>`, vendor tarballs with sha256,
`pip install --require-hashes` for the LongLive requirements, surface the CausVid
NC/SA license in `models download` help.

## Investigation / progress / resolution log

- 2026-09-25: found by supply-chain sweep.
- 2026-09-25 (repair): re-verified clone lines live (Evidence pasted) —
  corrected refs (`:79-80` not `:79-81`; LTXV is pip-git+https at :63, not a
  clone — description now notes the distinction);
  `model_registry.py:193` → `:194-195`. Added `## Why this is an issue`.
- Open: harden clones + requirements; add license notice.
- 2026-09-29 (this track): re-read live — ADOPTED, no duplicate work.
  FIXED by another track: `--depth 1` + `fetch <commit>` + `checkout` +
  `rev-parse --verify HEAD` on all three clones
  (`Dockerfile.video:50-54,76-80,116-120`); LTXV pip-git noted as accepted
  same-class (`:96-99`); CausVid NC/SA surfaced in image comment +
  `model_registry.py:CAUSVID_LICENSE` + `docs/MODELS.md:40-41`; upstream
  `requirements.txt` float noted as remainder (needs GPU-box
  `--require-hashes` freeze). No edit in this track. Note: full
  video-image build skipped per task.

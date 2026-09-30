# 123 — Resume paths never re-verify the taped tail checksum, and longlive skips the shape check causvid has

- **Severity:** Low-Medium (crash recovery — bit-rot/truncated tails adopted silently; geometry-mismatched tapes resume the wrong stream)
- **File:line:**
  - Tape *carries* sha: `Voyage/voyage/workers/video_ltxv.py:263-266,668` (`conditioning_tail_sha256` built + stored), `Voyage/voyage/workers/video_causvid.py:303-306,812,816` (same).
  - Resume *ignores* it: `video_ltxv.py:283-302` (`parse_recovery_tape` checks backend/mode/path-existence only), `:712-719` (`resume_from_tape` adopts without hashing); `video_causvid.py:341-370` + `:870-897` (same — existence only; causvid *does* check `latent_shape` at `:878-887`, longlive/ltxv don't).
  - Longlive resume: `Voyage/voyage/workers/video_longlive.py:496-563` (`resume_from_tape` uses `tape["tail_latents"]`/`tape["prompt_embeds"]`/`tape["noise_rng_state"]` with zero shape/dtype checks; `handle_resume` at `:1195-1205` checks only the profile string).
- **Area:** workers-internals tail — recovery/resume paths (below pass-1 coverage)

## Description

Three instances of "trust the tape":

1. **Stored-but-never-checked checksums.** Both ltxv and causvid compute `sha256_file(tail)` at commit and persist it in the tape — then no resume path ever recomputes it. A truncated tail mp4 (disk-full mid-write, partial copy, bit-rot) passes `Path.exists()` and becomes the conditioning anchor: ltxv conditions the next 121-frame clip on garbage frames, causvid VAE-re-encodes garbage into `start_latents`. The checksum exists precisely for this; nothing reads it.
2. **Longlive resumes unshaped tensors.** `resume_from_tape` indexes `tape["tail_latents"].to(device)`, reads `tail.shape[1]`, and calls `torch.as_tensor(tape["noise_rng_state"])` — a tape from a different geometry run (or a future format addition) fails deep inside the generator forward or `set_state` instead of at the boundary. Causvid's `resume_from_tape` already does the right thing (`:878-887`: tape `latent_shape` vs session shape → loud `ValueError`); longlive never got the equivalent, and ltxv checks neither shape nor geometry (width/height/fps ride in the tape at `build_recovery_tape:274-278` but `parse_recovery_tape` ignores them).
3. Causvid's shape check is also *session-relative only*: it compares against the live session, not against the tape's own `profile_hash` — a same-shape/different-checkpoint tape still resumes (arguably intended; recorded here so the fix can decide deliberately).

## Rationale

Resume is the path that runs when things already went wrong (crash, OOM-evict, audio swap rebuild). Silent adoption of a corrupt anchor converts a recoverable restart into a subtly degraded voyage: conditioning garbage doesn't crash, it just makes the next segment incoherent — the worst failure mode for a continuity system (§§22, 27). The checks are nearly free (one streaming hash + a few int comparisons) against a multi-minute GPU rebuild.

## Evidence (verified live 2026-09-30, tree reads)

- `video_ltxv.py:297-302`: tail checks are `isinstance str` + non-empty + `Path.exists()` — no `sha256_file` recompute anywhere in `:283-302` or `:712-719`.
- `video_causvid.py:360-370`: same existence-only shape; `:878-887` has the `latent_shape` check (the good precedent cited above).
- `video_longlive.py:526-559`: direct `tape[...]` access, no `latent_shape`/dtype/profile-geometry comparison (only the `profile` string match in `handle_resume:1201-1203`).
- `video_common.write_tape_atomic` callers prove tapes are JSON for ltxv/causvid (hashable in-CPU) — re-verification needs no GPU.

## Repro

```bash
# Corrupt a committed tail by one byte, resume, observe silent adoption:
cp <seg>/video_tail.mp4 /tmp/t.mp4 && printf '\x00' | dd of=/tmp/t.mp4 bs=1 seek=100 conv=notrunc status=none
# point a copied recovery tape at /tmp/t.mp4 (edit conditioning_tail_path), call handle_resume →
# {"resumed": True, ...} with no checksum complaint (tail sha in tape no longer matches).
```

## Fix candidates

1. Re-hash on resume: `parse_recovery_tape` (ltxv/causvid) recomputes `sha256_file(tail)` and compares to the taped value → `ValueError` ("tail sha mismatch — re-render from seed") on drift. Streaming hash, CPU-only.
2. Port causvid's shape check to longlive (`tape["latent_shape"]` vs session) + dtype/profile-geometry compare; add width/height/fps compare to ltxv's parse (values already in the tape).
3. Long-term: single shared `verify_tape_for_resume` in `video_common` (019's home) so the three workers can't diverge again.

## Refs

- `Voyage/voyage/workers/video_ltxv.py:241-302,669-719`; `Voyage/voyage/workers/video_causvid.py:271-370,870-897`; `Voyage/voyage/workers/video_longlive.py:496-563,1195-1205`; DESIGN §§5.3, 5.4, 27.1.
- Adjacent, not overlapping: 006 (supervisor trusts worker *frame counts* — this file is worker-side trust of *tape contents*); 059 (concept-vector dangling detector — the analogous fail-loud precedent); 122 (tape *write* durability — this file is tape *read* verification); 134 (causvid *unreadable*-anchor fallback silence — this file is *readable*-but-corrupt adoption).

## Progress log

- 2026-09-30 (Group B): re-verified live first (tree reads + `voyage:latest`
  probes): ltxv still stores `conditioning_tail_sha256` (`video_ltxv.py`
  build path) while `parse_recovery_tape` checks existence only; causvid
  keeps its `latent_shape` session check (the good precedent); longlive
  `resume_from_tape` indexes taped tensors with zero shape/dtype checks;
  no size gate anywhere (171's probe: 3 GiB sparse `.pt` ACCEPTED).
  Premise confirmed on all three legs.
- TDD: wrote `tests/test_tape_trust_123_171.py` first — collection error
  (helpers absent) before the fix, 8 passed after.
- Fix (shared-contract delivery, candidate 3): new
  `verify_conditioning_tail_sha(segment_dir, tape)` in
  `voyage/workers/video_common.py` — recomputes the taped sha when both
  keys are present (streaming `voyage.hashing.sha256_file`, CPU-only),
  `ValueError` on mismatch, no-op when the tape carries no hash (pruning
  records the would-be path with nothing to hash) or the tail is absent
  (the derive path materializes it — absence is not corruption). Plus a
  supervisor discovery leg (owned hunk): `_latest_recovery_tape` now runs
  every candidate through `_tape_tail_sha_matches` (same semantics) and
  skips mismatches to the next-newest tape with a `recovery_tape_skipped`
  metric — torn/non-JSON tapes still return True (adopt), so 139/197's
  territory is never masked.
- Gates (in-container): new tests + `test_recovery` +
  `test_supervisor_hardening` + `test_video_common` + `test_tail_derive` +
  `test_state_integrity` = 81 passed; `ruff check` + `ruff format --check`
  + `mypy` on `voyage/supervisor.py` + `voyage/workers/video_common.py`
  clean. Supervisor diff re-checked (disjoint hunks only).

## Resolution

- Verdict: FIXED at the shared + supervisor-discovery layers. Files
  changed: `voyage/workers/video_common.py` (`verify_conditioning_tail_sha`
  + 171's `MAX_RECOVERY_TAPE_BYTES`/`check_recovery_tape_size`),
  `voyage/supervisor.py` (`_tape_tail_sha_matches` + discovery skip),
  `tests/test_tape_trust_123_171.py` (new: match/corrupt/noop pins).
- DESIGN proposal (quoted text only, for the DESIGN owner): in §27.1, after
  the resume description, add: "Resume re-verifies the taped conditioning
  tail: when the tape carries `conditioning_tail_sha256`, discovery
  recomputes it and skips to the next-newest tape on mismatch
  (metric-visible) — a truncated tail degrades to an older anchor instead
  of silently conditioning the next segment on garbage."
- Residuals (other files, not touched per scope): wire
  `verify_conditioning_tail_sha` into `video_ltxv.py:283-302`
  (`parse_recovery_tape`) + `:712-719` (`resume_from_tape`) and
  `video_causvid.py:341-370` + `:870-897`; port causvid's `:878-887`
  `latent_shape` session check to `video_longlive.py:496-563`
  (`resume_from_tape`) plus dtype/profile-geometry compare, and add
  width/height/fps compare to ltxv's parse (values already ride in the
  tape) — the shared helper makes each call site a 3-line change.

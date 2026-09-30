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

# Voyage pedantic investigation — ranked index (2026-09-30, five passes + review, pass 3 clean)

191 files: 190 issues + this index. Pass 1 (137-167 reactivation, 2026-09-30):
six tracks (correctness/RPC/persistence/state, structure/config/CLI/TUI,
standards/tests, perf/VRAM/media, docs/observability, supply-chain). Pass 2
(168-172, 178-185, 188-197 — targeted tails below all prior windows; number
gaps 173-177 and 186-187 left free by range reservations, not missing files).
Plus prior passes below. Review pass (2026-09-30, six range tracks 001-035 /
036-070 / 071-104 / 105-136 / 137-167 / 168-197): every file re-read,
structure standardized, severities recalibrated to the HIGH=crash/data-loss/
security/liveness rubric, citations re-anchored to as-read values, overlaps
mapped to keep/fold decisions (no files renamed/deleted/created). Every file
is self-contained: title, severity, file:line refs, live evidence (command +
output), repro, fix candidates, references.

Conventions: severities are post-review (HIGH > MEDIUM > LOW, with a few
MEDIUM-HIGH / LOW-MEDIUM / MEDIUM-LOW retained where the filer argued a
half-step). "Resolved in live tree" = the file documents a fix that already
landed; kept for the record, ranked last. Ownership: files 012-019, 030,
043-054, 095-104 are the earlier correctness/perf tracks; 137-167 are this
reactivation (137-142 correctness/RPC/state, 143-149 structure/config/CLI/TUI,
150-151 standards/tests, 152-158 perf/VRAM/media, 159-164 docs/observability,
165-167 supply-chain); 020-042, 055-078 mixed tracks; 001-011 + 079-094
include concurrent-agent re-verifications and structure/test batches — do not
renumber, cite live file:line (concurrent agents have uncommitted edits in
`voyage/cli.py`, `voyage/tui_state.py`, `Voyage/tests/test_generate.py`).

Citation drift warning: the tree is moving under concurrent edits (cli.py
~+40 lines, supervisor.py ~+8, model_registry 1267→1368 during the review
alone). All file:line cites are as-read 2026-09-30 — re-verify with grep
before fixing. Prefer symbol grooming over line pins.

Deleted upstream during this pass: `Voyage/worker/Dockerfile.director` and
`Voyage/scripts/build-director.sh` are gone (unified `voyage-video` image per
§140 tail; only `worker/Dockerfile.video` remains in `worker/`). Issues
076/078/090/106/165 cite them — read those notes as historical; the
AGENTS.md §11 Phase-3 bullet still references the director Dockerfile (see
162, orchestrator-owned fix).

Collision note: 086/087/088/089 were double-written by two agents in the
same window; the other agent's files kept the numbers,
this pass's four moved to 095-098 with headers fixed. No content lost.

## Rank 1 — open correctness / security / liveness (fix first)

- 004 — No inter-process mutual exclusion (CRITICAL, data corruption)
- 003 — A/V alignment never enforced at commit/validate (CRITICAL)
- 006 — Worker-reported frames/recovery_path trusted blindly (MAJOR)
- 007 — Worker error taxonomy erased over RPC; checked_request empty (MAJOR)
- 015 — resolve_stored_path traversal + absolute-path trust (HIGH)
- 016 — _checked_tape_path symlink bypass + TOCTOU (HIGH)
- 013 — DONE-before-state crash window + silent overwrite on retry (HIGH)
- 014 — worker.restart() bypasses budget + circuit breaker (HIGH)
- 052 — ACE-Step OOM wrapped non-retryable → instant Fatal (HIGH)
- 068 — requirements.lock ships httpx2/httpcore2 lookalike names (HIGH)
- 071 — checkpoint SHA self-attested post-download, pass-through verify (HIGH)
- 070 — Wan2.2 snapshot revision=None (floating heaviest base) (HIGH)
- 073 — LTXV TE bypasses pinned snapshot (per-session hub fetch) (HIGH)
- 067 — video image ~20 floating pip deps, no hashes (HIGH)
- 043 — final publish loads entire MP4 into RAM (HIGH)
- 044 — _decode_frames captures whole rawvideo + full copies (HIGH)
- 045 — SFX conditioning 3 spawns/window + triple-held tensors (HIGH)
- 020 — TOML escaper duality → unparseable voyage.toml (HIGH)
- 022 — generate caption pins silently dropped (HIGH, exit 0 wrong behavior)
- 056 — worker logs rotate only on start() (HIGH, unbounded growth)

## Rank 2 — open medium (correctness-adjacent, perf, preflight)

- 021 — _CUDA_BACKENDS polluted/incomplete (MEDIUM; review-trimmed from MEDIUM-HIGH)
- 023 — TUI always-override clobbers stored tuning (MEDIUM; review-trimmed)
- 059 — summarize_gauges drops worker VRAM fields (MEDIUM; owns the gauge half of the 051 overlap)
- 063 — console split/timing/tracker (MEDIUM; review-raised — keep, fold 028 on fix)
- 072 — MMAudio vocoder *.py allow-list into persistent volume (MEDIUM; review-trimmed)
- 095 — metrics.json etc. never hashed (MEDIUM; review-raised from LOW-MEDIUM)
- 055 — inspect metrics rotation-blind (MEDIUM; review-trimmed from HIGH — scoreboard workaround exists)
- 139 — recovery-tape discovery unvalidated (MEDIUM; review-raised — discovery-side sibling of 016)
- 144 — NaN metrics mislabel WITHIN + drop amendments (MEDIUM; review-raised — director steers on a lie)
- 153 — SFX stem cache truncate/unlink/orphan (MEDIUM; review-raised — single-worker logic vs 054's race)
- 156 — SFX/ACE bench VRAM device-default (MEDIUM; review-raised — sfx/audio instance of 124's class)
- 158 — SFX two-worker hardcodes cuda:1 (MEDIUM; review-raised — 021 covers the set, not the presence)
- 017 — validate_run tracebacks on hostile inputs (MEDIUM)
- 018 — probe() leaks raw JSONDecodeError (MEDIUM)
- 019 — run_capture has no timeout (MEDIUM)
- 100 — NaN/inf pass positive_seconds into select (MEDIUM)
- 101 — ledger/concept/metrics fsync gaps (MEDIUM)
- 102 — finalize /tmp staging vs preflight fs (MEDIUM; quoting/RAM → 053/043)
- 103 — _embed_texts garbage bypasses fallback; NaN novelty (MEDIUM)
- 104 — unbounded audio slice loop → ffmpeg-spawn storm (MEDIUM)
- 030 — prefetch + shutdown(wait=False) hang (MEDIUM)
- 024 — unbounded --output/--final-video paths (MEDIUM)
- 025 — god modules + quadruple registries (MEDIUM)
- 026 — prompt staging truncation + blocklist (MEDIUM)
- 029 — inspect rotation + fps=0 divergence (MEDIUM; 055/061 are siblings, not duplicates)
- 036 — god modules vs §12 split signal (MEDIUM)
- 039 — property-test hygiene gaps (MEDIUM)
- 040 — _init_run ×121 duplication (MEDIUM)
- 046 — augment chunk linear rescan O(N²) (MEDIUM)
- 047 — augment worker reload-per-call + stack-first (MEDIUM)
- 048 — LongLive full-segment copies, fake chunking (MEDIUM)
- 049 — LTXV OOM narrow catch, no gc (MEDIUM)
- 050 — finalize double-encode, no CRF (MEDIUM)
- 051 — bench drops VRAM, no percentiles (MEDIUM; keeps the timing half of the 059 overlap)
- 053 — concat quoting + ffmpeg hygiene (MEDIUM; owns the quoting half of the 102 overlap)
- 054 — SFX two-worker ledger race (MEDIUM)
- 057 — rotation mtime-only, no size/compress/fsync (MEDIUM)
- 058 — metrics schema misses baseline (MEDIUM)
- 060 — benchmark env thin, stdout-only artifacts (MEDIUM)
- 061 — status omits config/health (MEDIUM; 029/055 are siblings, not duplicates)
- 062 — scoreboard crash/paths/partials (MEDIUM; keep, fold 027 on fix)
- 064 — qualify.sh fragile gate/paths/artifacts (MEDIUM)
- 065 — INSTALL/README missing film/realesrgan/sfx (MEDIUM; keep, fold 146 on fix)
- 066 — doctor root-disk + all-or-nothing models (MEDIUM)
- 069 — .dockerignore gaps (MEDIUM)
- 074 — augment torch.load on .safetensors (MEDIUM)
- 075 — chmod 777 /opt on PYTHONPATH (MEDIUM)
- 076 — unversioned apt in video image (MEDIUM)
- 077 — manifest repair best-effort (MEDIUM)
- 079 — full-delete longlive2 backend (HIGH structure; listed here as scoped proposal)
- 080 — split cli.py god module (HIGH structure)
- 081 — split supervisor.py god module (HIGH structure)
- 082 — split model_registry.py (HIGH structure)
- 083 — unify media/augment finalize (HIGH structure)
- 084 — worker validators + resident fakes (HIGH structure)
- 085 — single-source collapse (MEDIUM structure)
- 086 — tier-1 dead code batch (MEDIUM)
- 088 — tests 86→65 fold (MEDIUM)
- 089 — test hygiene/lock/markers (MEDIUM; lock rows fold into 068 on fix)
- 090 — scripts + containers cleanup (MEDIUM)
- 091 — SFX/AUGMENT operator docs (MEDIUM)
- 093 — TASK.md prune-merge (MEDIUM docs)

## Rank 3 — lows (policy, precision, docs)

- 028 — console stream split (LOW; review-trimmed — fold into 063 on fix)
- 027 — scoreboard overlap (LOW-MEDIUM; fold into 062 on fix)
- 031 — ruff select gap vs ALL (LOW; review-trimmed from HIGH — toolchain ratchet, not liveness)
- 032 — per-file-ignores debt incl. RUF100 (LOW; review-trimmed from HIGH)
- 033 — mypy gate 4/88 files (LOW; review-trimmed from HIGH — 034 blocks the ratchet)
- 034 — stale/wrong-code type: ignores (LOW; review-trimmed — cluster with 031/032/033)
- 035 — Any leakage (LOW; review-trimmed — cluster with 031/032/033/034)
- 037 — modules lack DESIGN refs (LOW; review-trimmed from MEDIUM — pure hygiene)
- 038 — PLR2004 dark (LOW; review-trimmed from MEDIUM)
- 041 — coverage ratchet 65 vs 78 (LOW; review-trimmed from LOW-MEDIUM)
- 078 — no --require-hashes/SBOM (LOW; review-trimmed — defense-in-depth)
- 087 — timeboxed legacy shims (LOW; review-trimmed)
- 092 — docs one-line batch + qual leg (LOW; review-trimmed)
- 151 — DESIGN refs missing in workers/audio (LOW; review-trimmed — 037 was voyage/*.py-scoped)
- 042 — README geometry vs augment floors (LOW docs)
- 094 — toolchain ratchet policy (LOW)
- 096 — validate_video 0/0 + nb_frames carve-out (LOW)
- 097 — run-lock exotic-FS/racy-holder/stale-pid (LOW)
- 098 — orphan scan misses audio/ + run root (LOW)

## Pass 2 (24 new: 105-128, targeted sweeps below pass-1 windows)

- 109 — stop --finalize always crashes, missing --skip-bad (HIGH)
- 105 — qualify gate pipefail abort on idle GPU (MEDIUM; corrects 064 leg a)
- 110 — generate validates numerics after init, orphans run dir (MEDIUM)
- 111 — TUI misses take_seconds > ahead_seconds invariant (LOW-MEDIUM)
- 112 — parse_duration grammar surprises (LOW-MEDIUM)
- 113 — TuiProgress drops console lines (LOW-MEDIUM)
- 114 — 5 TUI checkboxes lack help (LOW)
- 115 — soak/benchmark skip CUDA preflight (LOW)
- 116 — run-name strip asymmetry TUI vs CLI (LOW)
- 117 — concept tokenizer ASCII-only, CJK false-duplicates (MEDIUM)
- 118 — wire-boundary silent coercion (MEDIUM-LOW)
- 119 — models range-validation gaps (LOW-MEDIUM)
- 120 — beats/BPM cap interaction fatal at render (MEDIUM)
- 121 — quantize banker's-rounding shortens takes (LOW-MEDIUM)
- 122 — tape write no fsync (MEDIUM-LOW; 101 never named the tape writer)
- 123 — resume trusts tape blindly (LOW-MEDIUM)
- 124 — causvid device param ignored, hardcoded cuda (MEDIUM-LOW)
- 125 — longlive uint8 wrap without clip (MEDIUM)
- 126 — vision metrics edge inputs crash (LOW)
- 127 — director init ignores unknown keys (LOW)
- 128 — causvid overlap=1 empty window (LOW)
- 106 — gpu marker vs build-gate gap (LOW)
- 107 — ARCHITECTURE omits inspector pipeline (LOW)
- 108 — stray root-owned Voyage/Voyage/ dir (LOW)

## Pass 3 (8 new: 129-136, remaining tails + TUI/DESIGN drift)

- 129 — longlive recovery.pt bare torch.save, not even atomic (MEDIUM)
- 136 — prefetch hit logged before amend-discard inflates hit-rate (LOW-MEDIUM)
- 130 — TUI Pilot fixed-sleep waits vs file's own poll precedent (LOW-MEDIUM)
- 131 — backend-select operability test never opens overlay (LOW)
- 132 — qualification math crashes on minimal inputs (LOW)
- 133 — §140 handoff staleness batch, 4/17 claims drifted (LOW docs)
- 134 — causvid resume fallback silent, write-only flags (LOW)
- 135 — DESIGN §5.3 1024x576 addendum stale vs live 768x512 (LOW docs)

## Pass 4 (31 new: 137-167, 2026-09-30 reactivation — six tracks below all prior windows)

- 137 — RESOLVED batch 1 (was: RPC timeout desync poisons next call)
- 143 — cmd_init mkdirs before validating (MEDIUM; same family as 110, init-verb site)
- 152 — SFX bed left-fold O(N²) re-encode (MEDIUM; audio analogue of 050, unfiled)
- 154 — benchmark has no SFX/augment targets (MEDIUM; 051/060 cover shape, not targets — owns the target half vs 163)
- 155 — ACE take duration unbounded (MEDIUM; SFX sibling caps at 60 s)
- 157 — augment chunk path preset/fanout gaps (MEDIUM; preset + thread-interaction 047 doesn't cover)
- 161 — sfx_caption invisible in console+TUI (MEDIUM; contradicts DESIGN :7112-7113; 113 explicitly not captions)
- 163 — benchmark/soak have no SFX axis (MEDIUM; soak-only half — dedupe with 154 on fix)
- 166 — registry provisions FILM+RealESRGAN weights augment_worker cannot load (MEDIUM; 074 covers loader mechanics, not the contract)
- 150 — new test files repeat stale-ignore pattern (MEDIUM; blocks the 033 ratchet; 034 never listed these files)
- 138 — finalize --skip-bad covers one of three legs (LOW-MEDIUM; finalize side of 017 + 095-adjacent; 102 legs not re-filed; read 188's silent-disable jointly)
- 139 — recovery-tape discovery unvalidated (LOW-MEDIUM; discovery-side sibling of 016, different function)
- 144 — NaN metrics mislabel WITHIN + drop amendments (MEDIUM; 103 is embed-side, 126 a producer)
- 153 — SFX stem cache truncate/unlink/orphan (LOW-MEDIUM; single-worker logic vs 054's race)
- 151 — DESIGN refs missing in workers/+audio/ (LOW; 037 was voyage/*.py-scoped)
- 156 — SFX/ACE bench VRAM device-default (MEDIUM-LOW; sfx/audio instance of 124's class)
- 158 — SFX two-worker hardcodes cuda:1 (MEDIUM-LOW; 021 covers set pollution, not presence)
- 140 — inspect_frame.png orphan in segment dirs (LOW; *.png gap 098 doesn't name)
- 141 — manifest still advertises 768x432 (LOW; machine-written side of 042)
- 142 — inspect reads live stores lock-free (LOW; read-only-tool principle behind 017)
- 145 — TUI drops --no-download (LOW; getattr-default class of 022; coordinate help with 114)
- 146 — models list omits film/realesrgan (LOW; fold into 065 on fix; extend 065's mirror test)
- 147 — generate->finalize drops verbose/no_color (LOW; same handoff as 022/109 — extend the contract test)
- 148 — TUI _run_dir_for ignores --name (LOW; latent until a divergent pair is constructed)
- 149 — generate --output help stale since --name (LOW; TUI help already states the new truth)
- 159 — BACKENDS fake geometry stale 320x180 (LOW docs; 091 repeats the number, never files the table)
- 160 — OPERATIONS console list omits sfx (LOW docs; 133's probe omitted sfx too)
- 162 — AGENTS.md §11 voyage bullet stale (LOW; orchestrator-owned fix, do not edit AGENTS here)
- 164 — STATE_AND_RECOVERY omits SFX ledger (LOW docs; contract half vs 092's layout half)
- 165 — stale (issue NNN) citations (LOW; only the 011-family resolves; add a resolve-gate)
- 167 — Voyage/ run-dir name in neither ignore file (LOW; narrow gap beyond 069/108)

## Pass 2 (23 new: 168-172, 178-185, 188-197 — tails below all prior windows)

- 168 — prefetch hit logged before drift-hold discard (LOW-MEDIUM; new discard site vs 136; drift_every_n>1 only)
- 170 — RESOLVED batch 1 (was: worker start keeps stale handle on failed init)
- 171 — recovery-tape size unbounded (LOW-MEDIUM; count+containment landed, size open — CWE-400 third leg)
- 188 — skip_bad silently disables numbering-gap check (LOW-MEDIUM; 138 misdescribes it as strict — read both)
- 189 — single-slice audio paths skip duration/output checks (LOW-MEDIUM)
- 191 — sfx bounds fallback reads wrong stream (LOW-MEDIUM; shifts downstream captions)
- 195 — doctor check_models stale spec set, 6 vs 10 (LOW-MEDIUM; models_ok:true while ~16 GB missing)
- 180 — feedback_amendments steers on 4/6 metrics (MEDIUM-LOW; two families invisible)
- 178 — TUI plan renders fallback for invalid backend (LOW-MEDIUM; two answers, one input)
- 181 — zero Pilot coverage of checkbox widget→state (LOW-MEDIUM)
- 182 — TUI has zero SFX fields, always default pass (LOW-MEDIUM; 145's sibling)
- 169 — LTXV block-0 silent fresh, no counter (LOW; ltxv has nothing where causvid has write-only 134)
- 172 — causvid tail mp4 non-atomic overwrite (LOW; producer side of 134's silent consumer)
- 179 — stop --finalize always muted (LOW; stop parser defines no console flags)
- 183 — test_tui_app.py:717 ignore unlisted by 034/150 (LOW; pattern still spreads)
- 184 — DESIGN frontpage omits augment floors (LOW docs)
- 185 — cmd_init reads style/seed directly (LOW; AttributeError trigger for 143's litter)
- 190 — finalize min_* scalars override options, joint/audio ignored (LOW)
- 192 — decode_chunk trusts stale dir (LOW; docstring-only guard)
- 193 — resolve_device silent CPU fallback (LOW; zero observability)
- 194 — benchmark_env omits 32/1280/720 floors (LOW; post-060 axis)
- 196 — MODELS.md missing AWQ row (LOW; ~2.6 GB default-CUDA stack)
- 197 — torn manifest crashes ensure (LOW; loud-vs-077-silent — read both)

## Pass 3 (micro-pass 2026-09-30: CLEAN — worker-tail sweep below every window returned zero genuine findings; 198-207 left free; investigation closed until the next code change)

## Resolved in live tree (record only, do not re-fix)

- 012 — start_workers() leak (was HIGH, fixed 2026-09-30 batch 1: exception-safe unwind + start inside try/finally)
- 099 — control-plane lost update (was MEDIUM, fixed 2026-09-30 batch 1: commit preserves STOP/PAUSE_REQUESTED via pre-write re-read)
- 137 — RPC timeout desync (was HIGH, fixed 2026-09-30 batch 1: generation fencing discards stale ids within the deadline)
- 170 — stale handle on failed init (was LOW-MEDIUM, fixed 2026-09-30 batch 1: start() fence reaps child + closes log)
- 001 — RPC readline deadline bypass (was CRITICAL, fixed: non-blocking reader + cap)
- 002 — non-VoyageError escapes commit (was CRITICAL, fixed: ingress wraps)
- 005 — torch.load RCE (was CRITICAL, fixed: weights_only + pre-verify)
- 008 — CLI path traversal (was MAJOR, fixed: flat-name + resolve_run_dir)
- 009 — TOML injection (was MAJOR, fixed: escaper; residual C0 gap → 020)
- 010 — audio-GPU finally masks error (was MAJOR, fixed)
- 011 — unpinned deps (was MAJOR, fixed: lockfile; residual → 067/068/078)

## Known overlap / dedupe on fix (review decisions — keep both files, fix jointly)

- 063 KEEPS the console stream-split defect; 028 FOLDS into 063 on fix (same file, 063 broader: +timing/spinner/KeyError).
- 062 KEEPS the scoreboard defect; 027 FOLDS into 062 on fix (same module, 062 broader: +paths/partials/baseline/headers).
- 065 KEEPS the stale download-list defect; 146 FOLDS into 065 on fix (CLI surface of the same omission; extend 065's mirror test to `models list`).
- 059 OWNS the gauge-VRAM half; 051 KEEPS the timing-percentile half (split one path, fix together).
- 053 OWNS the concat-quoting half; 102 KEEPS the RAM/tmp-staging half (three legs, one fix pass).
- 068 OWNS the lockfile corrupt-rows; 089 KEEPS the cache/marker halves (fold only the lock section).
- 154 OWNS the benchmark-SFX-target half; 163 KEEPS the soak-aggregation half (fold target half on fix, leave 163 soak-only).
- 029 ~ 055 + 061 NOT duplicates (inspect-metrics vs rotation-blind reader vs status omissions — fix readers together, keep all three).
- 031/032/033/034 NOT duplicates (select-gap vs ignores vs gate-scope vs wrong-codes — one toolchain ratchet, four files).
- 043 ~ 102-partial, 044/045/048 cluster, 049 ~ 052, 046 ~ 047: cross-ref only, different halves of the same paths.
- 100 ~ 104 (NaN/inf family), 097 ~ 099 (lock vs control-plane bypass), 095 ~ 098 (integrity vs residue), 071 ~ 077 (ingest vs repair), 122 ~ 129 (JSON fsync vs pt atomicity), 123 ~ 134 (readable-corrupt vs unreadable-anchor), 130 ~ 131 (same Pilot file), 168 ~ 136 (amend vs drift-hold discard — one shared `invalidated` outcome), 188 ~ 138 (read both before choosing the skip_bad numbering contract): keep both, fix jointly.
- 113 vs 161 NOT overlap (113 is TUI-vs-console line drops, captions never named — fix separately).
- Track-B issue 11 (workers/loop.py stdout quarantine/strict/taxonomy) never
  materialized as its own file — angles covered by 007 (taxonomy) + 056
  (stdout quarantine); file it separately if those fix without covering it.

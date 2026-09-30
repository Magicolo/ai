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

- 004 — RESOLVED batch 2 (was: no inter-process mutual exclusion)
- 003 — RESOLVED batch 3 (was: A/V alignment never enforced at commit/validate)
- 006 — RESOLVED batch 3 (was: worker-reported frames/recovery_path trusted blindly)
- 007 — RESOLVED batch 2 (was: worker error taxonomy erased over RPC)
- 015 — RESOLVED batch 2 (was: resolve_stored_path traversal + absolute-path trust)
- 016 — RESOLVED batch 2 (was: _checked_tape_path symlink bypass + TOCTOU)
- 013 — RESOLVED batch 2 (was: DONE-before-state crash window + silent overwrite)
- 014 — RESOLVED batch 2 (was: worker.restart() bypasses budget + circuit breaker)
- 052 — RESOLVED batch 2 (was: ACE-Step OOM wrapped non-retryable → instant Fatal)
- 068 — RESOLVED batch 2 (was: lockfile httpx2/httpcore2 rows — premise refuted, genuine)
- 071 — RESOLVED batch 3 (was: checkpoint SHA self-attested, pass-through verify)
- 070 — Wan2.2 revision=None (HIGH, procedure-only 2026-09-30 batch 3: wire + record + pin procedure landed, value still None — pin from a provisioned box)
- 073 — RESOLVED batch 3 (was: LTXV TE bypasses pinned snapshot)
- 067 — RESOLVED batch 3 (was: video image ~20 floating pip deps, no hashes)
- 043 — RESOLVED batch 3 (was: final publish loads entire MP4 into RAM)
- 044 — RESOLVED batch 3 (was: _decode_frames captures whole rawvideo + copies)
- 045 — RESOLVED batch 3 (was: SFX conditioning 3 spawns/window + triple-held tensors)
- 020 — RESOLVED batch 3 (was: TOML escaper duality → unparseable voyage.toml)
- 022 — RESOLVED batch 3 (was: generate caption pins silently dropped)
- 056 — RESOLVED batch 3 (was: worker logs rotate only on start)

## Rank 2 — open medium (correctness-adjacent, perf, preflight)

- 021 — RESOLVED batch 4 (was: _CUDA_BACKENDS polluted/incomplete)
- 023 — RESOLVED batch 4 (was: TUI always-override clobbers stored tuning)
- 059 — RESOLVED batch 5 (was: summarize_gauges drops VRAM; gauges now carry per-worker vram_* series + timing_stats_ex p50/p95/std)
- 063 — RESOLVED batch 5 (was: console stream-split/timing/tracker; error() via injected stream, elapsed parity, BaseException teardown, guarded tracker; 028 folded)
- 072 — RESOLVED batch 4 (was: MMAudio vocoder *.py allow-list into persistent volume)
- 095 — RESOLVED batch 4 (was: metrics.json etc. never hashed)
- 055 — SUPERSEDED by 029 batch 4 (was: inspect metrics rotation-blind; same branch/helper/repro/fix, no distinct leg)
- 139 — recovery-tape discovery unvalidated (MEDIUM; review-raised — discovery-side sibling of 016)
- 144 — NaN metrics mislabel WITHIN + drop amendments (MEDIUM; review-raised — director steers on a lie)
- 153 — SFX stem cache truncate/unlink/orphan (MEDIUM; review-raised — single-worker logic vs 054's race)
- 156 — SFX/ACE bench VRAM device-default (MEDIUM; review-raised — sfx/audio instance of 124's class)
- 158 — SFX two-worker hardcodes cuda:1 (MEDIUM; review-raised — 021 covers the set, not the presence)
- 017 — RESOLVED batch 5 (was: validate_run tracebacks on hostile inputs; every filesystem anomaly now maps to an INVALID line)
- 018 — RESOLVED batch 5 (was: probe() leaks raw JSONDecodeError; wrapped as MediaError, retryable/--skip-bad-able)
- 019 — RESOLVED batch 5 (was: run_capture has no timeout; 600 s RPC-mirroring budget, TimeoutExpired→MediaError)
- 100 — RESOLVED batch 4 (was: NaN/inf pass positive_seconds into select)
- 101 — RESOLVED batch 6 (was: ledger/concept/metrics fsync gaps; concepts jsonl/index + metrics append/prune + SFX-twin dir-sync closed; O_APPEND multi-writer deferred)
- 102 — finalize /tmp staging vs preflight fs (MEDIUM; quoting/RAM → 053/043)
- 103 — _embed_texts garbage bypasses fallback; NaN novelty (MEDIUM)
- 104 — RESOLVED batch 4 (was: unbounded audio slice loop → ffmpeg-spawn storm)
- 030 — RESOLVED batch 4 (was: prefetch + shutdown(wait=False) hang)
- 024 — unbounded --output/--final-video paths (MEDIUM)
- 025 — RESOLVED batch 5 (was: god modules + quadruple registries; unity now test-guarded)
- 026 — RESOLVED batch 5 (was: prompt staging truncation + blocklist; strict + repeat_transitions knobs + tier-1 markers)
- 029 — RESOLVED batch 4 (was: inspect rotation + fps=0 divergence; 055 superseded into it, 061 is a sibling, not a duplicate)
- 036 — god modules vs §12 split signal (MEDIUM; still OPEN as tracker — cli.py split landed batch 7, supervisor/registry/workers splits remain)
- 039 — RESOLVED batch 7 (was: property-test hygiene; conftest unification + replay opt-in + health-check policy)
- 040 — RESOLVED batch 7 (was: _init_run ×153 duplication; 153→144 + test_phase2 folded + cap-144 ratchet test)
- 046 — RESOLVED batch 5 (was: augment chunk linear rescan O(N²); input -ss fast-seek + exact-fallback + fresh dest_dir)
- 047 — RESOLVED batch 5 (was: augment worker reload-per-call + stack-first; resident cache + list-halving + gc-before-empty_cache + load/infer split)
- 048 — RESOLVED batch 5 (was: LongLive full-segment copies, fake chunking; true chunked decode with per-chunk write+del)
- 049 — RESOLVED batch 6 (was: LTXV OOM narrow catch, no gc; broadened except + is_oom() predicate + gc-before-empty_cache with VRAM logging)
- 050 — RESOLVED batch 6 (was: finalize double-encode, no CRF; single concat-demuxer + vf pass, crf/preset knobs defaulting 15/veryfast, finalize_completed metric)
- 051 — RESOLVED batch 5 (was: bench drops VRAM, no percentiles; timing_stats_ex is the percentile source; SFX/augment targets stay with 154)
- 053 — RESOLVED batch 6 (was: concat quoting + ffmpeg hygiene; write_concat_list with `'\''` escaping + -hide_banner -nostdin on audio_acestep._convert)
- 054 — RESOLVED batch 6 (was: SFX two-worker ledger race; serial plan-order append after pool joins, no lock by construction; validate already last-wins)
- 057 — RESOLVED batch 6 (was: rotation mtime-only, no size/compress/fsync; size-or-time rotation + fsync_dir on rename/append/prune)
- 058 — RESOLVED batch 6 (was: metrics schema misses baseline; METRICS_SCHEMA_VERSION=1 + format/parse_metric_lines with torn accounting + run_id filter)
- 060 — RESOLVED batch 6 (was: benchmark env thin, stdout-only artifacts; rich _benchmark_env + report_document() JSON artifacts in logs/)
- 061 — RESOLVED batch 6 (was: status omits config/health; Config section + recorded-vs-live hardware + gauges trend + reserve WARN; fps half already dead)
- 062 — RESOLVED batch 5 (was: scoreboard crash/paths/partials; per-row degrade with errors cell + baseline id + existence flags; 027 folded)
- 064 — RESOLVED batch 6 (was: qualify.sh fragile gate/paths/artifacts; fail-closed gates + absolute-path enforcement + df preflight + tee'd JSON artifact; leg-a mechanism corrected per 105)
- 065 — RESOLVED batch 5 (was: INSTALL/README missing film/realesrgan/sfx; mirror rows + mirror test covering docs + list output; 146 folded)
- 066 — RESOLVED batch 6 (was: doctor root-disk + all-or-nothing models; disk_by_mount + models_ok_required/all split + per-GPU facts + health_alerts())
- 069 — RESOLVED batch 6 (was: .dockerignore gaps; 9 patterns added + COPY-graph header corrected)
- 074 — RESOLVED batch 4 (was: augment torch.load on .safetensors)
- 075 — RESOLVED batch 6 (was: chmod 777 /opt on PYTHONPATH; 755 + rationale comments; shim fallback unverified until next video build)
- 076 — RESOLVED batch 6 (was: unversioned apt in video image; all 8 packages =-pinned to the jammy freeze, CPU-only dpkg-query; rebuild-proven pending)
- 077 — RESOLVED batch 6 (was: manifest repair best-effort; atomic merge under _MANIFEST_LOCK with retries + fail-loud unrepaired entries; fetch+merge in download_model stays unlocked)
- 079 — PARTIAL batch 7 (was: full-delete proposal; DEPRECATED_VIDEO_BACKENDS + init-time warn shipped; full delete gated on quiet tree)
- 080 — RESOLVED batch 7 (was: cli.py 2403L god module; 735L seam + 10 verb modules + seam-dispatch rule)
- 081 — split supervisor.py god module (HIGH structure)
- 082 — PARTIAL batch 7 (was: registry 1543L; records/builders extracted to registry_records.py 767L, registry 1270L; per-family split + manifest race remain)
- 083 — RESOLVED batch 7 (was: media/augment finalize split; single homes + resolve_finalize_settings(); full FILM port open)
- 084 — RESOLVED batch 7 (was: triplicated validators; _validators.py + _resident.py + fake video serve-map port)
- 085 — RESOLVED batch 7 (was: literal duplication; _DEFAULT_ROW/_RESERVED_FOLDER_NAMES/FLOAT_DUST_EPSILON + 8 agreement tests)
- 086 — tier-1 dead code batch (MEDIUM; still OPEN — needs voyage/ scope, returned batch 7)
- 088 — tests 86→65 fold (MEDIUM; still OPEN — proof fold landed batch 7, one cluster per pass)
- 089 — test hygiene/lock/markers (MEDIUM; still OPEN — needs lock/pyproject/scripts owners, returned batch 7; lock rows fold into 068 on fix)
- 090 — RESOLVED batch 7 (was: scripts cleanup; lib/common.sh + run.sh audio/sfx gap + qualify.sh backends)
- 091 — RESOLVED batch 7 (was: missing operator docs; docs/SFX.md + docs/AUGMENT.md per §87)
- 093 — PARTIAL batch 7 (was: TASK prune-merge; TASK-only slice done, DESIGN write + delete approval pending)

## Rank 3 — lows (policy, precision, docs)

- 028 — FOLDED into 063 batch 5 (was: console stream split; fixed once under 063, no distinct leg)
- 027 — FOLDED into 062 batch 5 (was: scoreboard overlap; fixed once under 062, no distinct leg)
- 031 — ruff select gap vs ALL (LOW; still OPEN — document-only, adoption retry order PERF→N→PT recorded batch 7)
- 032 — RESOLVED batch 7 (was: per-file-ignores incl. RUF100; console.py T201 removed as tripwire, rest logged)
- 033 — RESOLVED batch 7 (was: mypy gate 4 files; now voyage + 82 test modules)
- 034 — RESOLVED batch 7 (was: stale tomli-shim ignores + override; wrong-code test ignores stay with owners)
- 035 — Any leakage (LOW; still OPEN — document-only, alias-migration proposal recorded batch 7)
- 037 — RESOLVED batch 7 (was: 7 modules lack DESIGN refs; all carry refs + gates.sh check; cli_core.py catch closed same batch)
- 038 — RESOLVED batch 7 (was: PLR2004 dark; 10 constants + scoped check)
- 041 — RESOLVED batch 7 (was: coverage floor 65; now 73 per measured-76-minus-3)
- 078 — no --require-hashes/SBOM (LOW; review-trimmed — defense-in-depth)
- 087 — timeboxed legacy shims (LOW; review-trimmed)
- 092 — RESOLVED batch 7 (was: docs one-line batch; README/BACKENDS/ARCHITECTURE/OPERATIONS/TROUBLESHOOTING/BENCHMARKING rows)
- 151 — DESIGN refs missing in workers/audio (LOW; review-trimmed — 037 was voyage/*.py-scoped)
- 042 — README geometry vs augment floors (LOW docs)
- 094 — RECORDED batch 7 (was: toolchain ratchet policy; policy + 7 do-not-regress incidents in-file, no code change needed)
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
- 139 — RESOLVED batch 5 (was: recovery-tape discovery unvalidated; empty/un-stat-able tapes skipped with metric; containment already mirrored from 016)
- 144 — RESOLVED batch 5 (was: NaN metrics mislabel WITHIN + drop amendments; per-metric isfinite skip, never WITHIN)
- 153 — RESOLVED batch 5 (was: SFX stem cache truncate/unlink/orphan; fuzzy 0.6 s match + render-to-tmp+replace + prune-on-plan)
- 156 — RESOLVED batch 5 (was: SFX/ACE bench VRAM device-default; session-device-indexed peaks + honest-null report shape)
- 158 — RESOLVED batch 5 (was: SFX two-worker hardcodes cuda:1; augment_devices() visibility gate, fail fast at plan time)
- 140 — inspect_frame.png orphan in segment dirs (LOW; *.png gap 098 doesn't name)
- 141 — manifest still advertises 768x432 (LOW; machine-written side of 042)
- 142 — inspect reads live stores lock-free (LOW; read-only-tool principle behind 017)
- 145 — TUI drops --no-download (LOW; getattr-default class of 022; coordinate help with 114)
- 146 — FOLDED into 065 batch 5 (was: models list omits film/realesrgan; augment rows added to list verb; mirror test covers list output)
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
- 178 — RESOLVED batch 5 (was: TUI plan renders fallback for invalid backend; invalid backend blocks the plan with a message)
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

- 003 — A/V alignment never enforced (was CRITICAL, fixed 2026-09-30 batch 3: commit + adopt paths enforce the 0.6 s gate via check_av_alignment; validate uses the shared av_drift_seconds helper, read-only error strings)
- 006 — worker-reported frames/tape trusted blindly (was MAJOR, fixed 2026-09-30 batch 3: adoption path clamps to 1..REPORTED_FRAMES_SLACK × segment_frames with implausible MediaError; live-report/tape gates verified already correct)
- 020 — TOML escaper duality (was HIGH, fixed 2026-09-30 batch 3: one full-C0+DEL config._toml_basic_string, tui_state alias; Hypothesis round-trip pins it)
- 022 — caption pins silently dropped (was HIGH, fixed 2026-09-30 batch 3: inner cmd_run namespace forwards music_caption/video_caption + contract test)
- 043 — final publish whole-MP4 RAM (was HIGH, fixed 2026-09-30 batch 3: atomic_copy chunked publish at both media.py sites; sfx_finalize.py:563 left as noted follow-up)
- 044 — _decode_frames full-stdout + copies (was HIGH, fixed 2026-09-30 batch 3: streamed Popen reads, pick-count/FALLBACK_MAX_FRAMES caps, zero .copy() fan-out; pixel-identical)
- 045 — SFX conditioning triple-spawn (was HIGH, fixed 2026-09-30 batch 3: single-pass 25fps@384 streamed into one stack, 2 GiB pre-stack budget, pad-to-16 tails, copy-free evict; torch execution needs a GPU box)
- 056 — worker logs rotate only on start (was HIGH, fixed 2026-09-30 batch 3: copytruncate rotate_worker_logs tick once per committed segment, 10 MiB cap)
- 067 — video image floating deps (was HIGH, fixed 2026-09-30 batch 3: all rows frozen to the 2026-09-30 live-image freeze + re-freeze procedure; rebuild + --require-hashes still open)
- 071 — checkpoint SHA self-attested (was HIGH, fixed 2026-09-30 batch 3: EXPECTED_*_SHA256 ingest verify pre-merge, manifest-conditional ensure leg, fail-closed manifests with opt-in; CausVid DMD baseline still open)
- 073 — LTXV TE hub bypass (was HIGH, fixed 2026-09-30 batch 3: _resolve_te_source offline-first local snapshot + pinned revision; AST grep-gate; needs HF_HUB_OFFLINE=1 GPU-box probe)
- 013 — DONE-before-state crash window (was HIGH, fixed 2026-09-30 batch 2: adopt-or-refuse reconciler + DONE guard, no silent re-render)
- 014 — restart-budget bypass (was HIGH, fixed 2026-09-30 batch 2: restart()/hook Recoverable consumes an attempt + worker_restart_failed, breaker still trips)
- 004 — run-lock limitations (was LOW, fixed 2026-09-30 batch 2: EWOULDBLOCK/EAGAIN-only contention, liveness-checked holder, pid-guarded unlink; residual unlink/check TOCTOU)
- 016 — tape symlink/TOCTOU (was HIGH, fixed 2026-09-30 batch 2: resolve-containment + is_file() on report and discovery paths; residual check-then-use TOCTOU)
- 007 — supervisor-side error taxonomy (worker-side already landed; fixed 2026-09-30 batch 2: call() isinstance fail-fast as Fatal + wire mapping pins)
- 015 — stored-path traversal (was HIGH, fixed 2026-09-30 batch 2: resolve-then-relative_to confinement both branches, MediaError; 4 old tests re-pinned; validate sites convert to error strings)
- 052 — ACE OOM fatal-not-retryable (was HIGH, fixed 2026-09-30 batch 2: is_oom() re-raise unwrapped so WORKER_ERROR stays retryable)
- 068 — lock non-canonical names (was HIGH, premise refuted 2026-09-30 batch 2: httpx2/httpcore2 genuine via huggingface_hub==2.0.0; lock↔manifest agreement now gated by test)
- 012 — start_workers() leak (was HIGH, fixed 2026-09-30 batch 1: exception-safe unwind + start inside try/finally)
- 099 — control-plane lost update (was MEDIUM, fixed 2026-09-30 batch 1: commit preserves STOP/PAUSE_REQUESTED via pre-write re-read)
- 137 — RPC timeout desync (was HIGH, fixed 2026-09-30 batch 1: generation fencing discards stale ids within the deadline)
- 170 — stale handle on failed init (was LOW-MEDIUM, fixed 2026-09-30 batch 1: start() fence reaps child + closes log)
- 021 — CUDA preflight sets (was MEDIUM, fixed 2026-09-30 batch 4: sets derived from BACKEND_REGISTRY device columns + SFX branch; union kept for the TUI warning)
- 023 — TUI always-override (was MEDIUM, fixed 2026-09-30 batch 4: untouched-at-default quantization/min_fps/min_resolution emit Unset, stored TOML wins; test_augment_config re-pinned to the inherit contract)
- 029 — inspect metrics rotation + fps (was MEDIUM, fixed 2026-09-30 batch 4: rotation-aware reader with N-events-across-K-files header; validate reports state fps corrupt for fps<=0, 24-fallback SFX-only after reporting)
- 055 — SUPERSEDED by 029 (2026-09-30 batch 4: same branch/helper gap/repro/fix, no distinct leg; fixed once under 029)
- 072 — vocoder allow-list (was MEDIUM, fixed 2026-09-30 batch 4: data-only snapshot config.json + bigvgan_generator.pt; verify fails loud on any .py; per-file hashes await 071 follow-up)
- 074 — augment weight-load gates (was MEDIUM, fixed 2026-09-30 batch 4: suffix branch safetensors vs torch.load, torch-free size+manifest pre-checks, UnpicklingError→ModelCompatibilityError; pinned weights still unloadable until 166 port lands)
- 095 — segment-manifest checksums (was MEDIUM, fixed 2026-09-30 batch 4: sha256.json covers all 7 artifacts in both verifiers + adoption; legacy 2-entry manifests pass as not-covered; inspector metrics rewrite refreshes its checksum)
- 100 — non-finite config timeouts (was MEDIUM, fixed 2026-09-30 batch 4: positive_seconds + Draft take_seconds reject non-finite; call()/_read_response_line map non-finite to RecoverableWorkerError; safe siblings audited)
- 101 — ledger fsync (was MEDIUM, partial 2026-09-30 batch 4: planner append_take gains fsync_dir; concepts/metrics/SFX legs recorded as follow-ups in owners' scopes)
- 104 — audio slice bound (was MEDIUM, fixed 2026-09-30 batch 4: from_dict validates geometry→StateError; both walks bounded 128 slices/50 ms floor; slice_take rejects slivers)
- 030 — prefetch shutdown hang (was MEDIUM, fixed 2026-09-30 batch 4: explicit 60 s prefetch budget + ~2 s best-effort drain in stop_workers; 137 resync semantics intact)
- 001 — RPC readline deadline bypass (was CRITICAL, fixed: non-blocking reader + cap)
- 002 — non-VoyageError escapes commit (was CRITICAL, fixed: ingress wraps)
- 005 — torch.load RCE (was CRITICAL, fixed: weights_only + pre-verify)
- 008 — CLI path traversal (was MAJOR, fixed: flat-name + resolve_run_dir)
- 009 — TOML injection (was MAJOR, fixed: escaper; residual C0 gap → 020)
- 010 — audio-GPU finally masks error (was MAJOR, fixed)
- 011 — unpinned deps (was MAJOR, fixed: lockfile; residual → 067/068/078)
- 059 — bench VRAM/percentiles (was MEDIUM, fixed 2026-09-30 batch 5: per-worker vram_* gauge series + timing_stats_ex p50/p95/std)
- 051 — bench timing stats (was MEDIUM, fixed 2026-09-30 batch 5: same bench.py pass; SFX/augment targets stay with 154)
- 062 — scoreboard robustness (was MEDIUM, fixed 2026-09-30 batch 5: per-row degrade + baseline id + existence flags + partial list; 027 folded)
- 063 — console contracts (was MEDIUM, fixed 2026-09-30 batch 5: stream rule + elapsed parity + BaseException teardown + guarded tracker; 028 folded)
- 027 — FOLDED into 062 (2026-09-30 batch 5: same module/root cause, no distinct leg)
- 028 — FOLDED into 063 (2026-09-30 batch 5: same file/contract, no distinct leg)
- 017 — validate_run robustness (was MEDIUM, fixed 2026-09-30 batch 5: every filesystem anomaly maps to an INVALID line, never raises)
- 018 — probe taxonomy (was MEDIUM, fixed 2026-09-30 batch 5: malformed ffprobe JSON raises MediaError)
- 019 — ffmpeg timeout (was MEDIUM, fixed 2026-09-30 batch 5: 600 s budget, TimeoutExpired→MediaError)
- 102 — finalize staging (was MEDIUM, fixed 2026-09-30 batch 5: TemporaryDirectory(prefix="voyage-final-", dir=run_dir); quoting stays with 053)
- 103 — embed garbage (was MEDIUM, fixed 2026-09-30 batch 5: coercion inside guard + math.isfinite, degrades to token-set fallback)
- 046 — augment chunk rescan (was MEDIUM, fixed 2026-09-30 batch 5: input -ss fast-seek + exact-fallback + fresh dest_dir)
- 047 — augment worker reload (was MEDIUM, fixed 2026-09-30 batch 5: resident caches + list-halving + gc-before-empty_cache + load/infer split)
- 048 — LongLive segment copies (was MEDIUM, fixed 2026-09-30 batch 5: true chunked decode with per-chunk write+del; tiling stays measure-only)
- 153 — SFX stem cache (was LOW-MEDIUM, fixed 2026-09-30 batch 5: fuzzy match + tmp-atomic-replace + prune-on-plan)
- 156 — SFX/ACE bench VRAM (was MEDIUM-LOW, fixed 2026-09-30 batch 5: session-device-indexed peaks + device in report)
- 158 — SFX 2-worker gate (was MEDIUM-LOW, fixed 2026-09-30 batch 5: augment_devices() visibility gate, fail fast)
- 065 — docs mirror (was MEDIUM, fixed 2026-09-30 batch 5: film/realesrgan-anime/sfx-mmaudio (+AWQ) rows in INSTALL/README/MODELS + list verb + mirror test; 146 folded)
- 146 — FOLDED into 065 (2026-09-30 batch 5: same omission, CLI surface; fixed once under 065)
- 144 — NaN director metrics (was MEDIUM, fixed 2026-09-30 batch 5: per-metric isfinite skip in context + amendments, never WITHIN)
- 139 — tape discovery (was LOW-MEDIUM, fixed 2026-09-30 batch 5: empty/un-stat-able skip with metric; containment already mirrored from 016)
- 178 — TUI invalid-backend plan (was LOW-MEDIUM, fixed 2026-09-30 batch 5: _plan_details→None + cannot-plan message)
- 025 — registry unity (was MEDIUM, fixed 2026-09-30 batch 5: unity test-guards the sets; literal derivation still with supervisor/CLI owners)
- 026 — prompt staging (was MEDIUM, fixed 2026-09-30 batch 5: strict + repeat_transitions knobs + 3 tier-1 markers; real 024 untouched, concurrent owner)
- 049 — LTXV OOM fallback (was MEDIUM, fixed 2026-09-30 batch 6: broadened except + is_oom() predicate + gc-before-empty_cache with VRAM logging)
- 050 — finalize single-pass encode (was MEDIUM, fixed 2026-09-30 batch 6: concat-demuxer + one vf pass, crf/preset knobs defaulting 15/veryfast, finalize_completed metric)
- 053 — concat quoting + ffmpeg hygiene (was MEDIUM, fixed 2026-09-30 batch 6: write_concat_list `'\''` escaping + -hide_banner -nostdin on audio_acestep._convert)
- 054 — SFX ledger race (was MEDIUM, fixed 2026-09-30 batch 6: serial plan-order append after pool joins; validate already last-wins)
- 057 — rotation size/fsync (was MEDIUM, fixed 2026-09-30 batch 6: size-or-time rotation + fsync_dir on rename/append/prune)
- 058 — metrics schema (was MEDIUM, fixed 2026-09-30 batch 6: schema v1 + format/parse_metric_lines with torn accounting + run_id filter)
- 060 — benchmark env + artifacts (was MEDIUM, fixed 2026-09-30 batch 6: rich _benchmark_env + report_document() JSON in logs/)
- 061 — status visibility (was MEDIUM, fixed 2026-09-30 batch 6: Config section + recorded-vs-live hardware + gauges trend + reserve WARN)
- 064 — qualify.sh gates (was MEDIUM, fixed 2026-09-30 batch 6: fail-closed gates + absolute-path enforcement + df preflight + tee'd artifact)
- 066 — doctor depth (was MEDIUM, fixed 2026-09-30 batch 6: disk_by_mount + models split + per-GPU facts + health_alerts())
- 069 — dockerignore (was MEDIUM, fixed 2026-09-30 batch 6: 9 patterns + corrected COPY-graph header)
- 075 — /opt perms (was MEDIUM, fixed 2026-09-30 batch 6: 755 + rationale; shim fallback unverified until next video build)
- 076 — apt pins (was MEDIUM, fixed 2026-09-30 batch 6: all 8 packages =-pinned to jammy freeze)
- 077 — manifest repair (was MEDIUM, fixed 2026-09-30 batch 6: atomic merge under lock + retries + fail-loud)
- 101 — ledger fsync legs (was MEDIUM, fixed 2026-09-30 batch 6: concepts/metrics/SFX-twin dir-sync closed)
- 080 — cli.py god-module split (was HIGH, fixed 2026-09-30 batch 7: cli.py 2403→735L seam over 10 verb-group modules ≤365L + seam-dispatch rule + __all__ contracts; 15/15 --help goldens byte-identical)
- 082 — registry split, partial (was HIGH, 2026-09-30 batch 7: pins + 10 _record_*/_describe_* extracted to registry_records.py 767L, registry 1543→1270L; per-family table split + in-core manifest-race rewrite remain)
- 083 — media/augment unification (was HIGH, fixed 2026-09-30 batch 7 safe subset: interpolated_frame_count single home, FINALIZE_CRF_* alias, resolve_finalize_settings(); defaults unchanged — fast-path contract; full FILM port stays open)
- 084 — worker validators (was HIGH, fixed 2026-09-30 batch 7 safe subset: _validators.py + _resident.py, 5 workers rewired, fake video serve-map/harness port; audio/sfx generalization stays open)
- 085 — single-source collapse (was MEDIUM, fixed 2026-09-30 batch 7: _DEFAULT_ROW/_RESERVED_FOLDER_NAMES/FLOAT_DUST_EPSILON collapsed, 8 agreement tests; supervisor-side unification deferred)
- 079 — longlive2 removal, partial (was HIGH, 2026-09-30 batch 7: DEPRECATED_VIDEO_BACKENDS + init-time warn toward ltxv; full delete gated on a quiet tree + TOML migration decision)
- 032 — per-file-ignores (was LOW, ratcheted 2026-09-30 batch 7: console.py T201 removed as tripwire; remaining entries logged with fire counts)
- 033 — mypy gate scope (was LOW, ratcheted 2026-09-30 batch 7: gate covers voyage + 82 test modules, 38 error files + 9 in-flight excluded with reasons)
- 034 — stale ignores (was LOW, fixed 2026-09-30 batch 7: 2 tomli-shim comments + unused-ignore override deleted; wrong-code test ignores stay with owning passes; 150's four files untouched)
- 037 — DESIGN refs (was LOW, fixed 2026-09-30 batch 7: 7/7 top-level modules carry refs + gates.sh DESIGN check; tripwire caught cli_core.py same-day, closed by owning track in this batch)
- 038 — PLR2004 slice (was LOW, fixed 2026-09-30 batch 7: 10 named constants with why-comments in 4 modules + scoped gates.sh check; full select stays out)
- 039 — property-test hygiene (was MEDIUM, fixed 2026-09-30 batch 7 tests slice: conftest unification + VOYAGE_HYPOTHESIS_DATABASE=1 replay opt-in + health-check/deadline policy)
- 040 — _init_run duplication (was MEDIUM, converged 2026-09-30 batch 7: 153→144 refs, test_phase2 folded into test_recovery, ratchet test caps at 144; keep converging 2-3 files/pass)
- 041 — coverage floor (was LOW, ratcheted 2026-09-30 batch 7: 65→73 per measured-76-minus-3; per-dir floors deferred with measured numbers)
- 090 — scripts cleanup (was MEDIUM, fixed 2026-09-30 batch 7: lib/common.sh + run.sh audio/sfx gap + qualify.sh --backend/--segments; gate-matrix replacement stays open)
- 091 — SFX/AUGMENT docs (was MEDIUM, fixed 2026-09-30 batch 7: docs/SFX.md + docs/AUGMENT.md created per §87, indexed, claims verified live)
- 092 — docs one-liners (was LOW, fixed 2026-09-30 batch 7 in-scope edits: README floors + 7 flags, BACKENDS/ARCHITECTURE/OPERATIONS/TROUBLESHOOTING/BENCHMARKING rows; DESIGN §§ + reports GPU leg proposed only)
- 093 — TASK prune-merge, partial (was MEDIUM, 2026-09-30 batch 7: TASK-only slice — §30.4 corrected; DESIGN write + delete approval pending)
- 094 — ratchet policy (was LOW, recorded 2026-09-30 batch 7: policy + 7 do-not-regress incidents written into the file)

## Known overlap / dedupe on fix (review decisions — keep both files, fix jointly)

- 063 KEEPS the console stream-split defect; 028 FOLDS into 063 on fix (same file, 063 broader: +timing/spinner/KeyError) — done 2026-09-30 batch 5.
- 062 KEEPS the scoreboard defect; 027 FOLDS into 062 on fix (same module, 062 broader: +paths/partials/baseline/headers) — done 2026-09-30 batch 5.
- 065 KEEPS the stale download-list defect; 146 FOLDS into 065 on fix (CLI surface of the same omission; extend 065's mirror test to `models list`) — done 2026-09-30 batch 5.
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

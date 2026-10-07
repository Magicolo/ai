# Voyage issues — consolidated index

**2026-10-07 — 68 live findings.**

An aggressive bug hunt (six parallel read-only sweeps across correctness/RPC/state, structure/CLI/config, standards/tests, perf/media, docs/observability, and supply-chain, with orchestrator live-verification of load-bearing claims) produced 116 issue files (`198`–`313`). A consolidation pass kept only design-preserving, truly relevant findings, removed cosmetic/hygiene/unreachable/fixed-in-tree items, and merged duplicates into lower-numbered canonicals; the 48 removed/absorbed files are archived at `/tmp/opencode/voyage-issues-archived-2026-10-07/`. This index lists the surviving 68 findings — severity as stated in each file, with the three `MEDIUM-HIGH` documentation inversions grouped at the foot of HIGH; numbering stays append-only (AGENTS.md §9).

## HIGH

| File | Title | Why it matters |
|------|-------|----------------|
| `207_rife_sha_truncated.md` | `EXPECTED_RIFE_SHA256` is 63 hex chars (truncated, can never match) | The security pin can never pass an honest fetch, pushing operators to bypass the ingest gate entirely. |
| `208_ltx23_checkpoint_shas_key.md` | ltx23 DiT `checkpoint_shas` key omits `distilled/`; only 2 of 5 pinned files recorded | Verify always mismatches on good volumes (and 60% of files are presence-only) — trains the bypass behavior hashes exist to prevent. |
| `209_loadtime_sha_only_causvid.md` | Load-time sha verification only for CausVid; six weight families have zero ingest hash | Stale or tampered MMAudio/ACE/LTX/GGUF weights load unchecked (size floor only). |
| `202_models_manifest_nonatomic.md` | `_merge_manifest_record` writes `models/manifest.json` non-atomically | SIGKILL mid-write tears the tamper-evidence baseline; every later verify degrades or crashes. |
| `216_double_rotate_invisible.md` | Double-rotate archives invisible to readers AND never pruned | Silent metric-history loss plus unbounded disk growth despite the 30-day prune contract. |
| `201_worker_recoverable_demoted.md` | Worker-side `RecoverableWorkerError` demoted to `Fatal` over the wire | Workers cannot signal transience: a correct restart request becomes an immediate run FAILED. |
| `212_upscale_natives_accumulation.md` | Upscale direct path accumulates full-chunk 4x natives on GPU outside OOM-halving | Full-chunk device tensors pile up, bypassing the OOM safeguard the tiled path depends on. |
| `215_partial_leg_legacy_path.md` | Partial-leg finalize still takes the legacy all-at-once path | Half-provisioned stacks route to an unbounded ~400 GB PNG staging / TB-scale tensor path. |
| `213_sfx_conditioning_collapse.md` | SFX `existing` dict collapses conditioning_source (proxy vs shipped) | Dedupe by `window_id` alone can serve the wrong source and re-render every cached window. |
| `214_music_sfx_timeline_clocks.md` | Music timeline (frames/fps) vs SFX bounds (probed seconds) diverge per segment | Worst-case ~128 ms systematic skew misaligns SFX captions, the dub, and the music mix. |
| `203_sfx_flags_silently_dropped.md` | `configure --sfx-*` flags accepted but silently dropped; `generate` hardcodes them away | Exit 0, configured-looking manifest, zero behavior change — silent option loss. |
| `204_qualify_rundir_mismatch.md` | `qualify.sh` checks one run dir then generates another; `--segments N` silently ignored | `reports/qual-*.json` can attest the wrong run; qualification artifacts lose their meaning. |
| `205_runsh_ignores_director_backend.md` | `run.sh` image sniff ignores the director backend and swallows malformed manifests | The default `fake`+`llama` shape lands in a sidecar-less image, failing late with `WORKER_ERROR`. |
| `206_runref_validation_asymmetry.md` | `resolve_run_ref --run` unvalidated; `output_root()` cwd-coupled; containment half-warned | One run gets two identities across cwd and traversal-unguarded paths. |
| `218_benchmarking_docs_dead_verbs.md` | Deleted CLI verbs (`benchmark`/`soak`/`models`) still prescribed in docs and runtime strings | The primary bench/soak/model runbook leads to `exit 2` after a full image-selection + `docker run`. |
| `303_docs_gpu_pairing_inverted_vs_code.md` | SFX.md + ARCHITECTURE.md 2-GPU pairing is inverted vs code (MEDIUM-HIGH) | Operators co-locate the wrong stacks and send OOM triage to the wrong card. |
| `304_augment_doc_hitch_tradeoff_is_fixed_bug.md` | AUGMENT.md chunk-hitch tradeoff is the fixed bug, not current behavior (MEDIUM-HIGH) | Users tune chunk size and diagnose jumps against a retired tradeoff. |
| `305_architecture_commit_pipeline_predates_deferred_audio.md` | ARCHITECTURE.md commit pipeline + run-dir layout predate all-deferred audio (MEDIUM-HIGH) | Documents a per-segment `audio.wav`, commit-time A/V gate, and layout that no longer exist. |

## MEDIUM

| File | Title | Why it matters |
|------|-------|----------------|
| `222_atomic_mode_narrowing.md` | `atomic_write_*` silently narrows file mode 0644 → 0600 | Every manifest/state/metrics rewrite locks out other-uid readers on bind-mounted runs. |
| `223_control_plane_race.md` | Control-plane read-then-write race survives `_write_state_preserving_control_plane` | A `stop`/`pause` landing in the window is clobbered — the only operator brake on an infinite run. |
| `219_restart_oserror_escapes_budget.md` | `OSError` from `worker.restart()` escapes the restart budget | A respawn failure evades the counter and circuit breaker, losing worker/op/budget context. |
| `221_validators_nan_inf.md` | `_validators` accept `NaN`/`inf` (and bool-as-int slips past range checks) | Bad numbers fail deep inside ffmpeg/torch instead of at the op boundary as `INVALID_PAYLOAD`. |
| `225_embed_texts_narrow_catch.md` | `_embed_texts` catches only `VoyageError`, so transport bugs crash the commit | An advisory novelty probe can turn a transport hiccup into a run-killing `FAILED`. |
| `247_adapter_silent_substitution_for_bad_worker_reports.md` | `VideoBackendAdapter` silently substitutes request values for bad worker reports | Worker bugs become committed timeline drift the supervisor trusts as truth. |
| `285_ltx23_tape_vae_revision_aliases_dit.md` | ltx23 recovery tape `vae_revision` aliases the DiT revision — VAE swaps resume silently | The no-resume-across-numerics guarantee is void for the ltx23 VAE leg. |
| `288_causvid_overlap_block_size_and_fps_ordering_gaps.md` | CausVid validates overlap against the wrong block size and skips the fps-first boundary | Init passes bad configs and failures land after minutes of model load, off the documented fail-fast convention. |
| `286_sfx_extract_stdout_streams_stderr_pipe_deadlock.md` | SFX extract streams `stdout` while `stderr=PIPE` sits undrained — pipe deadlock, no timeout | A corrupt segment video hangs the worker permanently on the failure path that most needs a loud error. |
| `235_allow_missing_manifest_env_silently_skips_sha_gate.md` | `VOYAGE_ALLOW_MISSING_MANIFEST=1` silently skips the only load-time sha gate | One stale env var disables fail-closed verification process-wide with no log. |
| `234_tarfile_extractall_without_filter_llama_source.md` | `tarfile.extractall` without `filter=` for the llama.cpp source (path traversal) | A single sha-gate bypass means arbitrary file write as root at build. |
| `232_no_require_hashes_index_url_no_build_isolation_git_url.md` | No `--require-hashes` anywhere; extra index URLs, unisolated builds, unverified git URL, bare SFX-venv packages | PyPI/index compromise or a swapped git ref becomes arbitrary code at image build. |
| `237_llama_sidecar_fixed_port_endpoint_binary_override.md` | Llama sidecar: fixed port 8080, endpoint-controlled port, arbitrary-binary env override | Port collisions misroute traffic; the loopback claim is contradicted; the server binary is substitutable unverified. |
| `238_ltxv_revision_kwarg_dead_on_local_snapshot_path.md` | `LTXVSession` passes dead `revision=` alongside a local snapshot path | A stale or hand-rolled TE snapshot loads silently under a "pinned" log line. |
| `231_metric_readers_silent_vs_counted_torn_lines.md` | `read_all_metric_events` silently drops torn lines; `parse_metric_lines` loudly counts them | The most-used reader breaks the §60 counted-torn contract — crash-torn commits vanish without a signal. |
| `224_log_metric_no_schema.md` | `_log_metric` bypasses the metric schema (`schema`/`ts_iso`/`truncated`) | Two writers, two contracts in one file: readers cannot distinguish v0 from v1 lines, and oversize events skip the cap. |
| `226_manifest_no_version.md` | Run manifest has no schema/format version; `paths.SCHEMA_VERSION = 1` is dead | Every breaking change needs another content-sniff carve-out; downgrades misread silently. |
| `242_boundary_metrics_exit_code_conflates_fail_and_tool_error.md` | `boundary_metrics.main` conflates verdict-FAIL and tool-error on exit 1 | CI/qualify drivers cannot gate on continuity without tripping on infra errors. |
| `244_dead_preset_int_models_dir_hardcoded_no_cli_surface.md` | Dead `_preset_int` helper; `models_dir` hardcoded `/models`; no CLI surface for model roots | Silent clobber on a path every backend switch executes, with no way to customize roots. |
| `245_models_dir_layout_omits_rife_gguf_awq.md` | `models_dir_layout()` omits `rife/`, `director-gguf/`, AWQ dirs | Inventory under-reports shippable stacks while its docstring claims full coverage. |
| `243_hand_rolled_mutex_instead_of_argparse_groups_double_errors.md` | Mutual exclusion hand-rolled in 5 places; `store_true(default=None)` tri-state; double error prints | Help shows no mutual-exclusion groups and invalid input can print twice. |
| `248_duplicated_planning_constants_unguarded_division_import_time_cuda_sets.md` | Duplicated planning constants, unguarded division, import-time CUDA sets, dead fallback | Three drift/crash vectors in the preflight every `generate` runs. |
| `260_director_input_untyped_any_boundary.md` | `director_input_from_state(state: Any, …)` — untyped boundary under mypy strict | Strict mypy is disabled exactly at a boundary constructor; one typo becomes a runtime `AttributeError`. |
| `263_enhancer_blind_except_swallows_programming_errors.md` | Fail-soft `except Exception` in `prompt_enhancer.enhance` swallows programming errors | Enhancer bugs degrade to the same zero-count signal used for "sidecar unreachable". |
| `297_textual_still_shipped_in_ltx_image.md` | `textual==8.2.8` still shipped in the `voyage-ltx` image | The documented TUI removal is incomplete: bytes and attack surface remain in the GPU image. |
| `298_cmd_validate_orphan_deleted_verb.md` | `cmd_validate` orphan in `voyage/cli_validate.py:375` | Dead CLI surface accepting a `--run` flag no live parser provides. |
| `230_ffmpeg_major_drift_unpinned_ltx_apt.md` | Image reproducibility drift: ffmpeg 7.1.5 vs 4.4.2 + unpinned apt and deadsnakes PPA | Identical configs can ship different `final.mp4` bytes with no provenance trail. |
| `249_tail_derive_two_full_decodes_per_segment.md` | Frame-count full decodes: tail derive per segment + `presented_frames`/`validate` on multi-GB finals | 256 segments pay 512 full decodes to learn a count the manifest already holds. |
| `250_finalize_probe_spawn_storm.md` | Finalize/prewarm spawn storm: redundant ffprobe + weights/joint re-hash per pass, per commit | ~1500 process spawns (45–120 s) before any pixel/audio render. |
| `251_upscale_poller_always_decodes_from_frame_zero.md` | Upscale poller always decodes from frame 0 (O(N²) H.264 decode tax) | Decodes 218 frames to keep 32; an 8-chunk segment pays ~4× its frames. |
| `252_finalize_start_triple_full_sweep.md` | Finalize start does 3–4 full sweeps (hash + PIL-open + rglob-stat every file) | ~65k PNG header opens plus GB-scale hashing serially before any GPU work. |
| `255_dual_pan_sfx_double_model_load_teardown.md` | Dual-pan SFX loads and tears down the MMAudio stack twice | Two cold model loads and 2× failure window for one model rendering two seed streams. |
| `256_png_bridge_triple_copy_transient_ram.md` | PNG bridge copies every frame 3–4× through 8 threads (GBs transient RAM) | The pool buys wall time by spending ~2 GB transient per chunk leg per thread. |
| `257_rife_serial_per_pair_cpu_sync_no_batching.md` | RIFE interp strictly serial per pair with a `.cpu()` sync per mid | Kernel-launch and sync latency dominate; free VRAM cannot buy throughput. |
| `287_benchmark_harness_stages_in_bare_tmp.md` | `run_benchmark_harness` stages full-segment renders in bare `/tmp` | Benchmarks recreate the 31 G tmpfs quota pressure that killed the ltx25 verify run. |
| `239_doctor_gap_list_stale_compute_cap_cuda_runtime_shipped.md` | Doctor-gap docs stale (`compute_cap`/CUDA runtime shipped since issue 066) | On-call re-probes manually or concludes doctor is useless and skips it. |
| `259_prompting_doc_contradicts_steer_and_accept.md` | `docs/PROMPTING.md` novelty section contradicts the steer-and-accept code | Operators expect rejection/retry dynamics (and latency) that no longer exist. |
| `253_scoreboard_documents_deleted_cli_target.md` | `scoreboard.py` documents a deleted CLI target; `partial_segment_ids` has zero production callers | The documented per-segment inspection view lost its CLI with no pointer to a replacement. |
| `276_operations_toml_section_syntax_for_json_manifest.md` | Doc drift: TOML `[section]` syntax + deleted-verb pins across OPERATIONS.md and SFX.md | Operators hunt for an `[augment]` table that cannot exist in the JSON-only manifest. |
| `306_design_s56_finalizer_predates_compression_two_verb.md` | DESIGN §56 finalizer steps + command predate compression + two-verb CLI | Three conflicting finalize truths in one section (stream-copy vs crf 30, `voyage finalize` vs two verbs). |
| `307_operations_presets_omit_ltx25_ltx23.md` | OPERATIONS.md backend presets + image selection omit ltx25/ltx23 | The default backend's geometry, device, and image choice are undocumented on the runbook page. |
| `308_readme_index_promises_removed_presentation_floors.md` | README docs-index still promises "presentation floors (≥24 fps, ≥1216×704)" | Entry-point doc misroutes every new reader to a deleted knob model. |
| `309_readme_quotes_superseded_rrdb_weights.md` | Model docs quote stale pins: README RRDB size, MODELS.md license note + LTXV "(default)" | Users provision/verify the wrong upscaler file; the size mismatch looks like corruption. |
| `310_readme_bullets_omit_ltx25_llama_defaults.md` | README video/director bullets omit the defaults (ltx25, llama) | Quick-start readers never learn the default generation path exists. |
| `313_design_s118_s120_frozen_at_2026_09_24.md` | DESIGN §118 as-built (+ §120 profile) frozen at 2026-09-24 | Prose as-built directly contradicts the registry (5 wired backends, ltx25 default). |

## LOW

| File | Title | Why it matters |
|------|-------|----------------|
| `282_rpc_timeout_true_accepted_as_1s.md` | `timeout=True` accepted as 1 s; `timeout=1.0` vs `True` indistinguishable | A caller flag where a duration belongs silently yields a 1 s deadline and burns restart budget. |
| `293_ltxv_tensor_handoff_broad_except_swallows_oom.md` | LTXV tensor-handoff fallback catches everything including OOM — silent quality downgrade | Real resource failures degrade to a lossy mp4 tail with no metric recorded. |
| `271_scoreboard_new_metric_masked_as_zero_delta.md` | Scoreboard masks new-metric appearance as 0.0 delta | A backfilled metric looks unchanged and keeps a baseline pointer that never applied. |
| `275_gates_sh_manual_mypy_list_head_heuristic.md` | `gates.sh` rot: explicit ~150-file mypy list + `head -15 \| grep DESIGN` heuristic | New test files are silently un-typechecked and the DESIGN-ref ratchet drifts. |
| `277_video_backends_report_stale_snapshot_deleted_verbs.md` | `reports/video-backends.md` is a 2026-09-24 snapshot citing deleted verbs and a pending leg | Unmarked stale measurements invite citing fake-backend numbers as GPU evidence. |

## Consolidation log (2026-10-07)

**Merged (canonical <- absorbed):**

- `208 <- 210` — ltx23 checkpoint sha key and coverage
- `204 <- 217` — qualify.sh run binding and segments flag
- `230 <- 233` — image reproducibility drift and apt pinning
- `232 <- 236` — package pinning and SFX venv names
- `209 <- 211` — checkpoint hash verification coverage gaps
- `250 <- 254` — finalize probe and re-hash storm
- `249 <- 258` — frame-count full-decode cost
- `206 <- 274` — run-ref validation and output containment
- `309 <- 311` — model docs stale pins and headers
- `276 <- 300` — docs TOML syntax and deleted-verb pins
- `218 <- 296` — deleted CLI verbs in docs and errors
- `205 <- 240` — run.sh image sniff robustness
- `243 <- 227` — hand-rolled argparse mutual exclusion
- `244 <- 228` — dead preset helper and models_dir surface

**Removed** (archived at `/tmp/opencode/voyage-issues-archived-2026-10-07/`):

- Fixed in-tree: `198`, `199`
- Intentional documented policy: `200`, `220`, `241`, `246`, `261`, `262`
- Cosmetic/hygiene: `264`–`270`, `272`, `273`, `299`, `301`, `302`, `312`
- Unreachable/defensive-only: `281`, `283`, `289`–`292`, `294`, `295`
- Orphaned surface: `229`
- Misc drift: `278`–`280`, `284`

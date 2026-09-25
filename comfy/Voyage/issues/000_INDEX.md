# Issue index (ranked — most critical first)

Pass 1 of the aggressive pedantic investigation, 2026-09-25. Numeric filename
order = rank order. Each file is self-contained (technical description, rationale,
evidence with live command output, reproduction, source refs, fix candidates, log)
so an agent with no project knowledge can pick it up.

## Critical (liveness / state integrity / RCE)

- 001_rpc_readline_deadline_bypass — `select`+blocking-`readline` hangs past timeout
- 002_non_voyageerror_escapes_commit — torn ledger/concepts/`0/0` fps leave RUNNING corpse
- 003_av_alignment_not_enforced_commit_validate — commit/validate weaker than finalize
- 004_no_interprocess_lock — two supervisors corrupt segment + state
- 005_torch_load_rce — `torch.load` without `weights_only` (only RCE)
- 006_worker_reports_untrusted — unbounded `frames` / foreign `recovery_path`

## Major (security / reproducibility / dominant perf)

- 007_worker_error_taxonomy_erased — RPC erases error class; `checked_request` unchecked; malformed line = 600 s hang
- 008_cli_path_traversal — `--run-id`/`--output` escape the tree (verified `/home/.../ai/tmp/evil-run`)
- 009_toml_injection — crafted `--style` injects tables / self-DoS (verified `injection-parses: True`)
- 010_audio_gpu_finally_masks_error — teardown `finally` hides audio failure, strands video evicted
- 011_unpinned_deps_no_lockfile — `>=` floors everywhere; transformers-5.x precedent
- 012_unpinned_base_images — no digests; unversioned apt/ffmpeg/pip
- 013_video_audio_gpu_swap — full evict+rebuild per take (dominant wall)
- 014_embed_cache_cold_miss — T5 re-encode (minutes) after every evict
- 015_longlive_pcie_pingpong — ~11.5% measured offload wedge; tiling never attempted
- 016_absolute_paths_break_relocation — `mv` a run breaks next commit + validate
- 017_gauges_timeout_stall — optional gauges can add ~30 min per commit
- 018_finalize_ram_triple_hash — whole-film `read_bytes`; 3× hashing; `rglob` per validate

## Structure / maintainability

- 019_video_workers_triplicated — 1400–1900-line diffs, ops logic copy-pasted 3×
- 020_god_modules — 6 files ≈ 60% of lines; 261-line `commit_one_segment`
- 021_sha256_5x — five chunked-hash helpers, already drifted
- 022_stringly_typed_backends — `backend: str` + 4 disagreeing registries
- 023_dead_adapter — `VideoBackendAdapter` unused by production (verified: 0 supervisor hits)
- 024_tui_planning_drift — TUI predicts 25/48 where CLI does 96/29/72; causvid omitted + `gpu_warning('causvid')==''` (all probed live)
- 025_config_geometry_sources — 5 geometry truth sources; causvid "lie" comment
- 026_model_registry_12x — 12 download/verify copies of one function

## Performance (remainder)

- 027_conceptstore_quadratic — O(history²) re-read + full-matrix rewrite per commit
- 028_ltxv_lossy_chaining — lossy mp4 roundtrip per chained block
- 029_causvid_n_shuttles — N × 11 GiB T5 shuttles despite the comment warning
- 030_unbounded_embed_caches — CUDA tensors pinned forever; `"cuda"` hardcoded
- 031_finalize_double_encode — 2 full encodes; O(pieces) ffmpeg spawns
- 032_percommit_fanout_seg0 — 3 health RPCs + seg0 re-decode every segment
- 033_director_prefetch_serialized — single lock; hit rate unmeasured

## Standards / tests

- 034_ruff_lint_gap — 9 families vs ALL; live assert/print/raise sites; 4 dead `noqa: BLE001`
- 035_mypy_tests_untyped — 11 invisible ignores
- 036_any_223 — 223× `Any` (36/31/30/17 per-file verified); untyped JSON boundary
- 037_scripts_quoting_sniff — qualify literal `'$run_dir'`; run.sh grep-sniff; `/tmp:/tmp`
- 038_missing_tests — audio/GPU/fakes/doctor/paths untested; 3-test integration contract
- 039_zero_hypothesis — 0 properties; `nan→(4,nan)`, `inf→5.6e162` beats (probed live)
- 040_fake_backends_ignore_seed — seed never reaches bytes
- 041_markers_coverage_bench — `gpu` marker 0 uses; no coverage gate; claims lack artifacts
- 042_cache_hygiene_devpins — `__pycache__` on disk; unpinned dev gates
- 043_tui_swallows_exceptions — 13 bare excepts; failure paths untested
- 044_naming_violations — 25+ single-letter sites; `tmp`/`num`/`vae` renames
- 045_boolean_traps — 12-param finalize; `scene_cut: bool` through 4 layers; two absent-encodings
- 046_legacy_shims_unused — migration shims served; 7 unused helpers; double-decode

## Docs / observability / UX

- 047_docs_drift — README omits TUI + 5 verbs; BACKENDS/MODELS miss CausVid+LTXV; LTXV notes missing
- 048_doctor_gaps — 3 of 13 spec checks (probed live, `torch_cuda: None`); build-time false gate
- 049_status_scoreboard_rotation — §59 drift; readers blind after rotation
- 050_console_flags_ignored — `--verbose`/`--no-color` dropped on 3 verbs
- 051_cli_help_polish — anonymous requireds; bare `models_target`; duration help; stop-finalize path; CUDA blame; streams
- 052_min_free_space_defaults — implicit 20 GiB trap; reserve invisible

## Containers / supply chain (remainder)

- 053_container_workdir_user — `/opt/longlive` CWD trap; root-owned mode-600 artifacts (observed live)
- 054_image_layer_hygiene — COPY invalidates pip layer; tests in prod images; `:latest`; no dockerignore
- 055_unverified_git_clone — plain-HTTPS full clones; floating reqs; NC/SA license unsurfacing
- 056_vlm_trust_remote_code — RCE-by-design unscoped
- 057_cmd_init_rundir_normalization — known-open TASK §30.2; one-line fix
- 058_done_atomic_fd_zombie — two-step DONE; fd leaks; zombies; `*.tmp.npy` blind spot
- 059_novelty_silently_disabled — lost vectors ⇒ duplicates accepted as novel

## Method (pass 1)

Six parallel sub-agent sweeps (correctness / structure / standards-tests /
performance-VRAM / docs-observability / supply-chain), all read-only; orchestrator
re-verified load-bearing claims live on the host (rpc `rg`, `Any` counts,
`torch.load` sites, FROM lines, adapter non-use, sha256 sites, `nan`/`inf` beats,
`gpu_warning`, `_frames_per_segment` matrix, `checked_request`, TOML injection,
path traversal, `noqa`/markers/hypothesis, qualify/run snippets, `doctor`
probe, `causvid-e2e` ownership). GPU idle at probe time (4060 Ti 15.1 GiB free).

## Pass 2 (done 2026-09-25 — 34 new files, 2 candidates closed without filing)

Three targeted sweeps (worker internals / CLI-TUI-DESIGN edges / tests-docs-root),
all read-only; orchestrator re-verified load-bearing claims live (manifest `rg`,
seed collision `True 952123984`, director `ZeroDivisionError`, quantize `inf`,
`format_segment_id(-1)='-00001'`, `PromptStage` inverted constructs,
`.dockerignore` absent + `output/` 406M, `_init_run` ×13, ltxv/causvid clip
`sed`, `run.sh` loop `sed`). GPU stayed idle.

- HIGH, filed as `006a` (sorts with the criticals): manifest clobber
  (`download_longlive2_bf16` raw `write_text`, 5 others merge — verified).
- Mediums `060-073`: benchmark `measured=0` ZeroDivision ×5 workers (060,
  director reproduced live); CLI numeric-override `ValidationError` traceback
  (061); TUI `take_seconds` nan/inf + `AudioConfig` accepts (062); audio RPC
  unvalidated (`bpm=0`, negative duration clamped — 063); LTXV chain-tail leak +
  short tail + unvalidated `fps` (064); ltxv `_save_mp4` missing `np.clip`
  (065, both sites verified); longlive `handle_init` no latent validation (066);
  vision-metrics empty-input crashes + NaN (067); `quantize_take_seconds`
  nan/inf (068, `inf` verified); `run.sh` drops `--run=` form (069, loop
  verified); `derive_seed` join collision (070, verified); `bench` empty/unknown
  helpers (071); `tui-last.toml` control chars (072); `--director` free string
  (073).
- Lows `074-092`: hardcoded `to("cuda")` + `stream_start_frame - 8*len` (074);
  director loader staleness + unvalidated gen params (075); DESIGN §19
  `distinguishes_from` dropped (076); §74 check-order inverted (077); TUI Stop
  unguarded (078); `--segments 0` no-op exit 0 (079); TUI `name="."` (080); §§56-57
  finalize-resolution drift (081); vacuous assert (082); pydantic cross-image skew
  (083); no `.dockerignore` (084, verified); `.gitignore` holes (085);
  `models_dir_layout` stale + pinning test (086); py312-target vs py3.10-runtime
  skew (087); 13× `_init_run` scaffolds (088, count verified); time-sensitive
  asserts + always-on endurance (089); `/tmp` placeholder paths (090);
  `PromptStage`/`format_segment_id` validation holes (091, both verified);
  gate-scope divergence + zero video gates (092).
- Closed without filing (2): `tests/test_probe_scratch.py` ("deleted before
  final gates") — already deleted by a concurrent agent (`ls` confirms absent;
  no action). Pass-2 `tests/`-mypy note — duplicate of 035 (recorded here as
  convergent re-confirmation with extra evidence `capsys: object`).
- Pass-2 explicitly checked with nothing new: zero assert-less test functions
  (AST sweep); no shared-mutable/order-dependent fixtures (`monkeypatch`-only,
  empty `tests/__init__.py`); docs fences match the CLI parser verb-by-verb;
  intra-repo file references resolve; `reports/video-backends.md` numbers
  consistent with code (4.61 FAIL recorded honestly); TASK §30.2 opens already
  tracked.

## Pass 3 (next)

Target: worker-internals tails below the previous line windows, TUI Pilot
failure-path matrices, DESIGN §§ cross-check of whatever lands next. New files
as `093+`; if a HIGH appears, prefix `0xxa/b` to sort with its rank and note it
here. Stop only when a full 3-track sweep returns zero new findings.

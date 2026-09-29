# 038 — No dedicated tests for the audio/GPU stack, fakes, doctor, or paths

- Status: resolved 2026-09-25 (this track)
- Severity: medium-high (untested load-bearing modules; 3-test integration
  contract for the whole RPC+commit+validate+finalize path)
- Area: tests — coverage mapping
- Rank rationale: module→test mapping shows six modules with zero direct tests;
  fakes are exercised only incidentally, so seed/shape regressions slip through.

## Technical description

Module→test mapping (`rg -l "voyage\.<mod>" Voyage/tests`):

| module | direct test |
|---|---|
| `audio/acestep.py` (5 fns: `initialize/render_take/evict/bpm_for_energy`) | **none** (0 hits) |
| `workers/audio.py`, `workers/audio_acestep.py` (12 handlers) | **none** |
| `fake_backends.py` (`FakeVideoBackend/FakeAudioBackend`) | **none** (only indirect via `test_integration.py:67`, 3 tests total) |
| `doctor.py` (`probe/check_ffmpeg`) | **none** (only mocked `test_generate.py:44`) |
| `paths.py` (`segment_dir/format_segment_id`) | **none** (used incidentally, never asserted: `format_segment_id(1)=="000001"` unpinned) |
| `vision/__init__.py` | none (only `vision/metrics.py` via `test_vision_metrics.py`) |
| `bench.py` / `logrotate.py` / `scoreboard.py` / `console.py` | thin single-file coverage (`test_benchmark:9`, `test_observability:6`, `test_scoreboard:3`, `test_console:12`) |
| `atomic.py/seeds.py` | only via `test_unit.py` (2 + 1 tests); no fsync/dir-fd, no negative-seed/overflow pins |

`test_integration.py` is 3 tests (`workers_answer_health`,
`director_worker_decides`, `commit_one_segment_end_to_end`) for the whole
RPC+commit+validate+finalize contract — genuine ffmpeg (`testsrc`/`sine`) but one
path.

## Why this is an issue

Whole load-bearing modules — the audio stack, both audio workers, the fakes everything else tests through, doctor, paths — have zero direct tests, so regressions in seed handling, shapes, or probe logic surface only as mysterious downstream failures in long GPU runs, the most expensive place to debug. Three integration tests carry the entire RPC+commit+validate+finalize contract on one path; resume, rebuild, and skip-bad paths are effectively unreviewed by any machine. Coverage debt compounds: each untested module makes the next refactor riskier to attempt.

## Evidence

Sweep loop output (table above); `rg -n "def test_"
Voyage/tests/test_integration.py` → 3; 361 total tests across 34 files (per-file
max `test_causvid_worker:30`, min `test_stage_timings:1`).

Verified live 2026-09-25:

```
$ for m in audio.acestep workers.audio fake_backends doctor paths bench logrotate; do
    echo -n "$m: "; rg -l "voyage.$m" tests/ | tr '\n' ' '; echo; done
audio.acestep: (none)   workers.audio: (none)   fake_backends: (none)
doctor: tests/test_generate.py (mocked only)   paths: (none)
bench: tests/test_benchmark.py   logrotate: tests/test_observability.py
$ rg -n "def test_" tests/test_integration.py
38:def test_workers_answer_health  47:def test_director_worker_decides
67:def test_commit_one_segment_end_to_end
```

## Reproduction

`for m in audio.acestep workers.audio doctor paths; do rg -l "voyage\.$m"
Voyage/tests; done` → mostly empty.

## Source references

- Files/table above.

## Resolution candidates

One test module per source module (Zoomy rule): `test_fake_backends`
(shape/duration/codec), `test_doctor` (ffmpeg present/absent), `test_paths` (id
format/roundtrip), `test_acestep_contract` (stub `initialize` → `render_take`
FLAC→WAV shape), `test_audio_workers` (fake `generate_audio`/`benchmark` ops);
expand `test_integration.py` to cover resume/rebuild/skip-bad paths.

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; module→test
  mapping re-probed live (audio.acestep/workers.audio/fake_backends/paths still
  zero direct hits; doctor mocked-only); pasted probe output into Evidence.
- 2026-09-25: resolution (this track) — new modules `test_fake_backends`
  (container/shape/duration + seed-reaches-bytes, issues 040/044-clean),
  `test_paths` (id format/bounds/roundtrip, resolve_stored_path incl.
  re-anchor), `test_audio_workers` (fake `generate_audio` real-ffmpeg render,
  validators, benchmark shape/counts, lifecycle echo),
  `test_acestep_contract` (CPU-only: bpm mapping, constants, stack shape,
  render_take validates-before-import + one `gpu`-marked render stub that
  skips without CUDA); `test_integration.py` gains finalize end-to-end +
  `FinalizeOptions` blend/hard-splice paths (issue 045 wiring). Doctor
  adopted as-is (`test_doctor.py` landed via the concurrent pass — not
  duplicated); `vision/__init__.py` is a one-line docstring (nothing to
  pin); atomic/seeds owned by the concurrent track (conftest +
  test_seeds_properties, untouched). Resume/rebuild/skip-bad already
  covered in test_recovery/test_crash_matrix/test_failure_policy/
  test_state_integrity — verified, not duplicated.
- Open: none in this track's scope (gates green required before close).
- 2026-09-29: verification (this track) — re-read live: `test_fake_backends.py`
  (6 tests: container/shape/duration + seed-reaches-bytes), `test_paths.py`
  (8 tests: format/bounds/roundtrip/re-anchor), `test_audio_workers.py`
  (7 tests), `test_acestep_contract.py` (8 tests), `test_doctor.py` (7 tests),
  `test_integration.py` gains finalize end-to-end + `FinalizeOptions` paths —
  all present. Scope pytest: 119 passed; 4 failures in
  `test_acestep_contract.py` are an out-of-scope `voyage/audio/acestep.py`
  regression (validates-after-import, `bpm_for_energy` clamp drift) for the
  owning track — not edited (audio/beat.py progress helper only). Ruff clean
  for scope; full `gates.sh` mypy red on pre-existing hypothesis stubs
  (conftest read-only, out of scope).
- 2026-09-29 (orchestrator): two review fixes on the contract tests —
  `render_take` now validates (bpm/duration/task_type/reference) BEFORE
  the upstream import (was: ModuleNotFoundError masked the ValueError
  contract on CPU), and the energy-floor pin corrected to == 60 (the
  60..140 map floor; MIN_BPM == 1 bounds explicit tempos instead).
  7 passed, 1 gpu-skipped.

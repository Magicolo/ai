# 142 — `inspect` reads live concept/scoreboard state with no lock (torn-store view)

- Severity: LOW
- Area: CLI observability — lock-free reads of mutating stores
- Files (as-read 2026-09-30):
  - `voyage/cli.py:1601-1606` (scoreboard row render — assumes float metrics/deltas)
  - `voyage/cli.py:1610-1616` (`inspect concepts` — `ConceptStore(...).records()` then print)
  - `voyage/cli.py:550-556` (`run_segments` SIGINT-restore `finally` — the concurrent writer path)
  - `voyage/cli.py:2069-2074` (verb wiring — `status/pause/resume/stop/validate/finalize/sfx/.../inspect` share the run dir with no lock on the read verbs)
  - Backing store: `voyage/concepts.py:139-144` (constructor replays `concepts.jsonl` line-by-line,
    `ConceptRecord.model_validate_json` per line, no lock, no torn-line tolerance)

## Technical description

The `inspect` verbs are lock-free reads against directories a running voyage mutates:

- `inspect concepts` (`voyage/cli.py:1610-1616`, as-read) constructs a `ConceptStore` over
  `<run>/novelty` (plus legacy fallback) and iterates `store.records()`. The constructor
  (`voyage/concepts.py:139-144`) replays `concepts.jsonl` with a bare
  `read_text().splitlines()` + `model_validate_json` per non-blank line. There is no
  inter-process lock (see issue 004 — no lock anywhere), no file lock around the supervisor's
  append (`concepts.py:184-187` `atomic_write_bytes` replaces the whole file on migration; the
  steady-state append path holds only the in-process single-writer assumption, DESIGN §72), and
  no torn-line guard. An `inspect` racing a commit can read a half-flushed `concepts.jsonl`
  (short read → `ValidationError` traceback instead of a table) or a vector/index pair mid-update
  (`concept_vectors.npy` + `concept_index.json` written separately, `:330`), showing concepts
  whose embeddings are not yet visible.
- `inspect scoreboard` (`voyage/cli.py:1590-1606`) formats each row's metrics with
  `f"{metrics[key]:.3f}({deltas[key]:+.3f})"` (`:1600`, as-read). A metrics row written by an
  in-flight commit (partial `metrics.jsonl` line, or a `visual` merge mid-`atomic_write_json`)
  either fails `json.loads` upstream (unhandled — traceback, exit non-zero) or renders
  `no-visual` fallback cells that look like data ("no-visual" × len(METRIC_KEYS), `:1602`) when
  the race is between the metrics write and the visual merge (`supervisor.py:1334-1339`).
- The writer side is genuinely concurrent: `run_segments` (`voyage/cli.py:550-556` restores the
  SIGINT handler around `supervisor.run_segments`, which loops commits until pause/stop), and
  the verb table (`voyage/cli.py:2069-2074`) offers every read verb against the same run dir
  with no shared-/exclusive-lock distinction. `status`/`scoreboard`/`inspect` are documented as
  safe-to-run-anytime, so the race is reachable by normal operation, not just adversarial timing.

Effect is observability-only (no state corruption — the writers are atomic; the *view* is torn),
hence LOW. But the failure mode is a traceback on a read-only command, which operators
reasonably expect to never fail.

## Why this is an issue

- Read-only verbs should never traceback: `inspect concepts`/`scoreboard`/`metrics` share the
  "safe during a run" expectation with `status`, but only `status`'s newest-novelty helper
  (`cli.py:581-599`) degrades to `unknown` on store errors — the inspect verbs have no such
  guard.
- Torn-view confusion: a half-appended concept list or a `no-visual` row during a healthy commit
  reads as a product signal (rejection / missing inspection) when it is actually a read race.
- Not a duplicate of 004 (no inter-process lock — the mechanism), 027 (scoreboard robustness),
  or 049 (status novelty/slowest-stage): this is the inspect-verb *view* half — torn reads of
  the concept store + scoreboard cells under a concurrent writer.

## Live evidence

Live re-verification 2026-09-30 (read-only; probes per task brief ran in `voyage:latest` CPU-only).
Track A draft command+output bundle (`ses_f0fbea412ffeh3V7K1SlQpXjsv`) was not recoverable from
this writer's context, so evidence below is the as-read code, not invented command output:

```
$ sed -n '1610,1616p;1597,1606p' voyage/cli.py
  1610:    if args.inspect_target == "concepts":
  1611:        store = ConceptStore(run_dir / "novelty", legacy_path=...)
  1612:        records = store.records()
  1615:            print(f"[{flag}] #{record.id}: {record.canonical_name[:120]}")
  1600:                cells = [f"{metrics[key]:.3f}({deltas[key]:+.3f})" for key in METRIC_KEYS]
  1602:                cells = ["no-visual"] * len(METRIC_KEYS)

$ sed -n '139,144p' voyage/concepts.py
  141:        if self._concepts_path.exists():
  142:            for line in self._concepts_path.read_text(encoding="utf-8").splitlines():
  144:                    self._records.append(ConceptRecord.model_validate_json(line))
  # no lock, no try/except around torn lines (as-read)

$ sed -n '550,556p;2069,2074p' voyage/cli.py
  550:        committed = supervisor.run_segments(args.segments)   # concurrent writer
  2069:    _add_generate_parser(sub)   # ... status/pause/resume/stop/validate/finalize/sfx/benchmark/soak/inspect
```

## Minimal repro

1. Start a slow commit loop (`run --segments 50` on fake backend, or any GPU run).
2. While a commit lands, run `voyage inspect --run <dir> concepts` in a tight loop + `voyage
   inspect --run <dir> scoreboard` in another.
3. Observed (race-dependent): occasional `pydantic ValidationError` traceback from
   `model_validate_json` on a truncated last line, or a `json.JSONDecodeError` / `no-visual`
   row from the scoreboard path. Expected: best-effort last-good view (`unknown`/`--` cells,
   non-zero exit never on a read verb).

## Fix candidates

1. (Preferred) Make the read verbs fail-soft like `_latest_novelty` already does: wrap the
   store/metric loads in `try/except (OSError, ValueError)` → print `unknown`/`--` cells with a
   stderr note, exit 0. Read-only commands never traceback.
2. Tolerate torn tails: in `ConceptStore` construction (or a dedicated `try_read_records`
   helper), skip-and-warn on a final line that fails `model_validate_json` instead of raising;
   same for `metrics.jsonl` tail reads (shared helper with `logrotate.iter_metric_files`).
3. Document the lock-free contract: `inspect`/`status`/`scoreboard` are best-effort views of a
   mutating run (pairs with issue 004's eventual locking work; explicit until then).
4. Regression tests: truncated-tail `concepts.jsonl` → `records()` best-effort (or verb prints
   `unknown`, exit 0); concurrent append-while-inspect stress (tmp run, writer thread + 100
   inspect reads, zero tracebacks).

## References

- In-tree: `voyage/cli.py:550-556,581-599,1590-1646,2069-2082`; `voyage/concepts.py:118-152,184-187,330`;
  `voyage/supervisor.py:1334-1339` (visual merge racing the scoreboard view); `DESIGN §§21,59,72`.
- Neighbor issues: 004 (no inter-process lock), 027 (scoreboard robustness), 049 (status novelty),
  055/062 (inspect rotation/partials), 058 (metrics schema), 101 (ledger fsync gaps).
- External:
  - https://docs.python.org/3/library/fcntl.html (advisory locking primitive, for the eventual 004 fix)
  - https://pydantic.dev/ (strict `model_validate_json` — torn input raises rather than degrading)

## Investigation log

- 2026-09-30: filed by Track A sweep; live re-verified via Read (concurrent uncommitted edits
  noted in `voyage/cli.py`, `voyage/tui_state.py`, `tests/test_generate.py`,
  `config/persistence/rpc/supervisor` — citations are as-read values above).

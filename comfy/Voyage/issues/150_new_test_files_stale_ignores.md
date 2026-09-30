# 150 — New test files repeat the stale-ignore pattern outside the mypy gate

- Severity: MEDIUM
- Files: `Voyage/tests/test_issue_014_embed_restore.py:127-136`, `Voyage/tests/test_issue_029_causvid_shuttle.py:71-73,105`, `Voyage/tests/test_issue_030_embed_bounds.py:160-165,197`, `Voyage/tests/test_director_models_dir.py:229`, `Voyage/tests/test_issue_032_commit_fanout.py:118` (clean — see evidence)
- Area: typing / gates

## Description

Four test files added after issues 033/034 carry the same `type: ignore`
patterns those issues already flagged as rotting — `attr-defined` on
`__new__`-built session scaffolds and stub assignments, plus one
`dict-item` on a negative-path `handle_init` call — and all sit outside
the 4-file test scope in `Voyage/scripts/gates.sh:28`. `warn_unused_ignores`
(`Voyage/pyproject.toml:119`) therefore cannot fire on any of them: wrong
codes stay wrong, unused ignores stay unused, and the gate stays green
while the pattern spreads. `test_issue_032:118` is the exception that
proves the rule — live it carries no ignore at all (clean control, or a
draft line-shift; documented below rather than silently dropped).

## Rationale

033 established the gate covers 4/88 test files and full `mypy tests` is
~108 errors red; 034 established the three rot modes (unused ignores,
wrong-code ignores, config-level `disable_error_code` silencing the
detector). New files that copy the `session = Klass.__new__(Klass)` +
per-attribute `# type: ignore[attr-defined]` scaffold re-enter the exact
debt 034 describes, at 10 + 4 + 7 + 1 new suppressions, with zero gate
coverage to ever clean them. `vision/metrics.py:12` is the contrast: the
typed `voyage` side (`from voyage.media import probe`) is inside the gate,
while the tests calling it are not — so the boundary the gate is supposed
to guard is exactly where the new ignores pile up.

## Live evidence (re-verified 2026-09-30, host reads)

`rg -n "type: ignore" tests/test_issue_014_embed_restore.py` (10 hits):

```
tests/test_issue_014_embed_restore.py:
  Line 127:     session._pipeline = pipe  # type: ignore[attr-defined]
  Line 128:     session._latent_shape = [1, 8, 4, 4, 4]  # type: ignore[attr-defined]
  Line 129:     session._device = "cpu"  # type: ignore[attr-defined]
  Line 130:     session._next_start_frame = 0  # type: ignore[attr-defined]
  Line 131:     session._blocks_appended = 0  # type: ignore[attr-defined]
  Line 132:     session._embed_cache = EmbedCache()  # type: ignore[attr-defined]
  Line 133:     session._noise_rng = None  # type: ignore[attr-defined]
  Line 134:     session._seq_noise = None  # type: ignore[attr-defined]
  Line 135:     session._seq_output = None  # type: ignore[attr-defined]
  Line 136:     session._seq_offset = 0  # type: ignore[attr-defined]
```

Read live `test_issue_014_embed_restore.py:125-137` — `_stream_session`
builds via `LongLiveStreamSession.__new__` then 10 per-attribute ignores.

`rg -n "type: ignore" tests/test_issue_029_causvid_shuttle.py` (4 hits):

```
  Line 71:     session._torch = torch  # type: ignore[attr-defined]
  Line 72:     session._pipeline = pipeline  # type: ignore[attr-defined]
  Line 73:     session._device = "cuda:0"  # type: ignore[attr-defined]
  Line 105:     session._pipeline.text_encoder = failing  # type: ignore[attr-defined]
```

Read live `:67-74` (`_encode_session`, `CausvidSession.__new__`) and
`:96-105` (failing-encoder swap) — same `__new__` + `attr-defined`
scaffold, same class as 014.

`rg -n "type: ignore" tests/test_issue_030_embed_bounds.py` (9 hits, cited subset):

```
  Line 160:     session._torch = _FakeTorch()  # type: ignore[attr-defined]
  Line 161:     session._device = device  # type: ignore[attr-defined]
  Line 162:     session._tokenizer = tokenizer  # type: ignore[attr-defined]
  Line 163:     session._text_encoder = encoder  # type: ignore[attr-defined]
  Line 164:     session._embed_cache = EmbedCache()  # type: ignore[attr-defined]
  Line 165:     session._negative = None  # type: ignore[attr-defined]
  Line 197:     session._pipeline = types.SimpleNamespace(text_encoder=object())  # type: ignore[attr-defined]
```

Read live `:156-166` (`_ltxv_session`, `LTXVSession.__new__`) and
`:195-198` (`_longlive_session`, real constructor + one stub ignore) —
third copy of the pattern (further hits at `:212,214` on
`conditioning_module`/`package_module` stubs, outside the cited range).

`rg -n "type: ignore" tests/test_director_models_dir.py` (6 hits, cited one):

```
  Line 229:         director_worker.handle_init({"models_dir": 123})  # type: ignore[dict-item]
```

Read live `:220-230` — negative-path `handle_init` with `dict-item`;
companion stub ignores at `:260-261,295,332-333` (`attr-defined` on
`AutoTokenizer`/`AutoModelForCausalLM`/`SentenceTransformer`/
`AutoProcessor`/`AutoModelForMultimodalLM` stubs).

`test_issue_032:118` does NOT reproduce as an ignore. Live
`tests/test_issue_032_commit_fanout.py:110-122` read:

```python
def test_sample_frames_select_path_matches_full_decode(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "clip.mp4")
    ...
    # Full-decode fallback path over the same bytes serves identical picks.
    info = metrics.probe(clip)   # :118 — no ignore, plain call
```

and `rg "type: ignore" tests/test_issue_032_commit_fanout.py` returns zero
hits — the file is fully typed, zero suppressions. Kept in the file list
as the draft cited it: either a line-shift in the draft or the intended
clean control. Either way the systemic point holds: this file is also
outside `gates.sh:28`, so a future ignore here would be equally invisible.

Gate scope, live `Voyage/scripts/gates.sh:22-28`:

```bash
docker run ... bash -c \
  "ruff check . && ruff format --check . && mypy voyage tests/conftest.py tests/test_seeds_properties.py tests/test_beat_properties.py tests/test_similarity_properties.py && coverage run ..."
```

None of the five files above is in that list. Detector switch, live
`Voyage/pyproject.toml:119`: `warn_unused_ignores = true` — active but
blind where the gate never looks (033's exact failure mode). Typed-side
contrast, live `Voyage/voyage/vision/metrics.py:12`:
`from voyage.media import probe` — inside `mypy voyage`, fully checked,
while its test callers are not.

## Repro

```bash
rg -n "type: ignore" Voyage/tests/test_issue_014_embed_restore.py Voyage/tests/test_issue_029_causvid_shuttle.py Voyage/tests/test_issue_030_embed_bounds.py Voyage/tests/test_director_models_dir.py Voyage/tests/test_issue_032_commit_fanout.py
sed -n '125,137p' Voyage/tests/test_issue_014_embed_restore.py
sed -n '67,74p;96,105p' Voyage/tests/test_issue_029_causvid_shuttle.py
sed -n '156,166p;195,198p' Voyage/tests/test_issue_030_embed_bounds.py
sed -n '220,230p' Voyage/tests/test_director_models_dir.py
sed -n '110,122p' Voyage/tests/test_issue_032_commit_fanout.py
sed -n '18,28p' Voyage/scripts/gates.sh
sed -n '115,120p' Voyage/pyproject.toml
```

## Fix candidates

1. Gate first (zero-behavior): append the four ignore-carrying files to the
   `gates.sh:28` mypy list one at a time (033's own prescription), or add a
   non-blocking full-`tests/` `warn_unused_ignores` report so the count can
   only fall.
2. Scaffold fix: replace `Klass.__new__` + N×`attr-defined` with a typed
   test factory (real constructor with fakes, or a `SimpleNamespace`-typed
   helper) — one helper per session class kills 10 + 3 + 6 ignores at the
   source instead of suppressing each attribute.
3. `dict-item` at `test_director_models_dir:229`: keep (genuine negative
   test) but pin it — it is the one ignore in this batch that tests
   behavior rather than scaffolding, so it belongs in the gate's converted
   set first.

## Refs

- `Voyage/tests/test_issue_014_embed_restore.py:127-136`
- `Voyage/tests/test_issue_029_causvid_shuttle.py:71-73,105`
- `Voyage/tests/test_issue_030_embed_bounds.py:160-165,197`
- `Voyage/tests/test_director_models_dir.py:229`
- `Voyage/tests/test_issue_032_commit_fanout.py:118` (live: clean, no ignore)
- `Voyage/pyproject.toml:119` (`warn_unused_ignores = true`)
- `Voyage/scripts/gates.sh:28` (4-file test scope)
- `Voyage/voyage/vision/metrics.py:12` (typed-side contrast)
- Issues 033 (gate 4/88) / 034 (stale/wrong-code ignores)
- Preserved track result: `ses_f0fbea3bbffe0HlTFo2fwHJi5B`, TRACK C NEW FINDING 1.

## Progress log (2026-09-30, Group D pass)

- Classified every ignore with ad-hoc in-container `mypy` (the authority;
  none of these files is in the `gates.sh:28` list): 014 → 10 unused;
  029 → 4 unused; 030 → 7 unused (`_ltxv_session` 6 + `_longlive_session`
  1) + 2 used ModuleType stubs (`:212,214`); director → `:229` unused +
  5 used (`:260,261,295,332,333`). All 22 stale ones deleted.
- 014 scaffold replaced with the real `LongLiveStreamSession(...)`
  constructor (assignment-only, CPU-safe — same ten values, no
  private-member poking; precedent: the 030 cluster already constructs
  this way). 029/030 `__new__` scaffolds kept: both real constructors
  load CUDA/heavy models, so the scaffold is load-bearing and the bare
  assignments are authority-clean.
- The 2 remaining ModuleType ignores (merged `:683,685`) kept as
  documented dormant suppressions: ruff B010 forbids the `setattr`
  alternative, and they fire only once the file enters the mypy gate
  (033). Director's 5 used ignores left verbatim. `metrics.probe`
  (`:802`, ad-hoc attr-defined, same call the issue logged as clean at
  old 032:118) needs a voyage-side export — out of scope, noted.
- Divergence noted: host pyright resolves `torch` and flags the 029/030
  `_torch` assignments, while project-config mypy (`follow_imports=skip`)
  types them `Any` — in-container mypy is the authority per AGENTS.md §11.
- Applied in `tests/test_perf_regressions.py` (post-088-fold single site;
  original paths above) + `tests/test_director_models_dir.py`.

## Resolution (2026-09-30, Group D pass)

- Resolved in scope: 22 stale suppressions gone; 2 documented dormant +
  5 used-correct remain, each justified in place.
- Files changed: `tests/test_perf_regressions.py` (fold + deletions +
  014 constructor + 2 documented ignores), `tests/test_director_models_dir.py`
  (`:229` deletion). Gate evidence: 64/64 (45 merged + 19 director)
  in-container; `ruff check` + `ruff format --check` clean on both;
  ad-hoc `mypy` reports zero errors on any changed line. DESIGN proposals:
  none. Residuals: `gates.sh` conversion queue (scripts track, 033
  prescription); `:802` probe export (voyage-side pass).

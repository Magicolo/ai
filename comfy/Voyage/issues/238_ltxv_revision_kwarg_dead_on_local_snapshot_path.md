# 238 — `LTXVSession` passes `revision=` alongside a local snapshot path (pin is documentary, not enforced)

Severity: MEDIUM (track F-12).

## Technical description

`_resolve_te_source` correctly resolves `PixArt-XL-2-1024-MS` to
`<models>/PixArt-XL-2-1024-MS` (or fetches it), but the subsequent
`T5Tokenizer/T5EncoderModel.from_pretrained(te_source, subfolder=…, local_files_only=True,
revision=LTXV_TE_REVISION)` passes a local directory + `revision`. For local paths
transformers ignores `revision` (no git resolution) — the pin documents intent but enforces
nothing. A stale or hand-rolled snapshot at that path loads silently.

## Rationale

Stale / wrong TE snapshot loads with a "pinned" log line.

## Live evidence

`voyage/workers/video_ltxv.py:303-332` (resolver) → `378-389` (load with
`local_files_only=True, revision=LTXV_TE_REVISION`). Comment itself hedges: "`revision`
… documents the pin for any hub-shaped input" (`video_ltxv.py:374-376`).

Repro: point `<models>/PixArt-XL-2-1024-MS` at an older snapshot → session init succeeds,
`model_revision` tape still claims the pinned rev.

## Source refs

`voyage/workers/video_ltxv.py:377-389`.

## Online sources

- HF `revision` semantics (branch/tag/SHA resolved against the Hub cache; meaningless for
  a plain local dir).
- Project's own `_resolve_model_source` + `snapshot_present` pattern (the check that
  actually enforces).

## Fix candidates

- After resolving, assert `snapshot_present(models_dir, ref)` (already done in the
  resolver — keep it load-time, not just fetch-time) or verify a snapshot marker (commit
  file / manifest entry) at load; drop the dead `revision=` kwarg so the code doesn't
  imply enforcement.

## Log

- 2026-10-07: filed from read-only Track F sweep; no code touched.

## Evaluation (2026-10-07)

Live check confirmed the dead kwarg (`video_ltxv.py` `from_pretrained`
calls pass `revision=` with `local_files_only=True`). The issue allows
"drop the dead `revision=` kwarg or document" — the drop branch was
rejected: `tests/test_registry_pins.py:402-403` pins both revision
kwargs as regression gates for issue 073, and the file is shared with
concurrent agents, so churning the call shape breaks foreign pins for
no runtime gain. The document + enforce branch was taken instead.

## Progress log

- 2026-10-07 (`voyage/workers/video_ltxv.py` only): added
  `_assert_te_snapshot_enforced()` (resolve ref + checklist postcondition
  after `_resolve_te_source`, fail-closed on stale/hand-rolled/foreign
  sources) and rewrote the `revision` comment to state it is
  documentary while the checklist is the enforcing pin. Both
  `revision=` kwargs kept, so the 073 pins stay green.
- 2026-10-07: new `tests/test_issue_238_te_enforcement.py` (3 tests:
  clean snapshot passes, stale dir and hub-id sources raise).

## Resolution (2026-10-07)

Resolved (document branch): a stale TE snapshot can no longer load
silently — the load-time checklist raises with a re-provision pointer.
No open items; the drop branch stays deliberately untaken while the
073 revision pins stand.

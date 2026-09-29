# 034 — Ruff selects 9 families vs Zoomy `ALL`: most bug-catching families off (+ dead `noqa: BLE001`)

- Status: resolved (fixed 2026-09-25: BLE/TRY/EM/SIM/RUF100/S101/T201 enabled + per-rule ignores + ratchet; full ALL explicitly rejected, see log)
- Severity: medium-high (lint blind spots; reviewers think blind-except is
  "acknowledged" while the linter sees nothing)
- Area: standards — `Voyage/pyproject.toml:33-38`
- Rank rationale: one config change unlocks ~8 finding classes; includes live
  `assert`/`print`/f-string-raise sites and 4 dead suppressions.

## Technical description

```toml
# Voyage/pyproject.toml:33-38
select = ["E","F","I","UP","B","A","C4","DTZ","W"]
# Zoomy: select = ["ALL"] + 14-line ignore with per-rule rationale +
# per-file-ignores + google convention
```

Missing families that matter here: `D` docstyle, `S` bandit, `BLE` blind-except,
`TRY`/`EM` exception style, `SIM` simplify, `PL` pylint, `RUF` (incl. `RUF100`
unused-noqa), `N` naming, `ANN` annotations, `PTH` pathlib, `C90` complexity,
`T201` print, `S101` assert. Live catches:

- `BLE001` blind `except Exception` — ~30 sites, e.g. `voyage/tui.py`
  (13 sites: `:165,186,198,228,642,652,690,697,709,728,735`), `voyage/console.py:
  176,190`, `voyage/supervisor.py:373` — unchecked.
- `S101` `assert` in shipped code (`video_ltxv.py:636,686,690`,
  `video_longlive.py:857,881,928,932`, `video_causvid.py:844,847,849,899,938,942`,
  `cli.py:149,165,181,223,236,268` — stripped under `-O`).
- `T201` `print` as interface (`cli.py:104,118,125,126,129,131,142,146,150-152,
  158,162,166-168,174,178,182,183`).
- `EM101/102` + `TRY003` f-string/long raises (`persistence.py:60,63,74,78`,
  `rpc.py:154,163,178,182`, `supervisor.py:216,746,842,1180`,
  `vision/metrics.py:42,47,51,74,79`, `config.py:267,361,363`).
- Dead suppressions: `noqa: BLE001` at `supervisor.py:516,886`,
  `workers/director.py:180,269` never fires (`BLE` unselected) and `RUF100` is
  also unselected so nothing reports it:

```
$ rg -n "noqa" Voyage/voyage   # verified live 2026-09-25 (excerpt)
Voyage/voyage/supervisor.py:516:        except Exception:  # noqa: BLE001 — prefetch must never break a commit
Voyage/voyage/supervisor.py:886:        except Exception as exc:  # noqa: BLE001 — inspector never breaks a commit
Voyage/voyage/workers/director.py:180:  except Exception as exc:  # noqa: BLE001 — inspector must never raise
Voyage/voyage/workers/director.py:269:  except Exception as exc:  # noqa: BLE001 — chain must survive any bad output
```

(`UP017`/`F401`/`E402` noqas are live and documented — keep those.)

## Why this is an issue

A linter that cannot see blind-excepts, shipped asserts, or print-as-interface teaches reviewers that these patterns passed review — ~30 `except Exception` sites and asserts that vanish under `-O` accumulate silently while everyone assumes gates are watching. The four dead `noqa: BLE001` suppressions are worse than noise: they signal "acknowledged risk" for a rule that is not even enabled, so future readers trust a safety net that was never strung. One config change buys back eight finding classes at once.

## Evidence

`rg -n "except Exception" Voyage/voyage | wc -l` (~30+); `rg -n "^\s*assert "`,
`rg -n "^\s*print\("` outputs in the sweep report; `rg -n "noqa"` above.

## Reproduction

Compare `Voyage/pyproject.toml:33-38` vs `Zoomy/pyproject.toml:12-38`.

## Source references

- `Voyage/pyproject.toml:33-38`; site lists above.

## Resolution candidates

Adopt `select=["ALL"]` with Zoomy-style `ignore=` + `[per-file-ignores]` for
`tests/*` (`S101,PLR2004`) + `pydocstyle convention="google"`; run
`ruff check --fix` + `ruff format` in gates; convert swallows to narrow
`except` or `log.debug(..., exc_info=True)` (see 043).

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep; `noqa` list re-verified live.
- 2026-09-25: repair pass — added `## Why this is an issue`; `noqa` hits
  re-verified live (4 dead `BLE001` at `supervisor.py:516,886`,
  `workers/director.py:180,269`, current); note `rg -c "except Exception"
  voyage/tui.py` now reports 16 vs the 13 sites enumerated in Technical
  description — count drifted, enumeration needs refresh when the rule lands.
- 2026-09-26 (resolution, policy-sized slice — FIXED as configured, fallout
  as follow-ups): the seven families are now enabled. Full `ALL` was
  rejected under the concurrent no-mass-fix constraint (measured 2308 hits
  tree-wide: S101 ×1524, TRY003 ×270, EM102 ×182, T201 ×152, EM101 ×93,
  BLE001 ×36, SIM105 ×14, RUF100 ×14, TRY004 ×9, SIM300 ×4, TRY300 ×3,
  SIM108/SIM117 ×2, TRY301 ×2, SIM103 ×1).
  - `pyproject.toml` `select` += `BLE,TRY,EM,SIM,RUF100,S101,T201`.
  - `ignore` = `EM101,EM102,TRY003` (exact Zoomy parity: messages embed
    values inline / carry context in the exception string).
  - `per-file-ignores`: `tests/* → S101`, `scripts/* → T201` (Zoomy parity)
    plus one entry per out-of-scope violating file (rule-scoped, each a
    follow-up below) — `tui,supervisor,cli,doctor,config,logrotate,concepts,
    prompts,tui_state,model_registry,persistence,media,console,
    workers/{loop,director,video_causvid,video_longlive,video_ltxv,
    audio_acestep}` + 7 test modules. Delete the entry (not the rule) as
    the owning pass converts each file.
  - Fallout fixed in owned files only: `SIM105 → contextlib.suppress` ×5
    (`atomic.py:1`, `rpc.py:4`), `S101` assert → explicit `FatalWorkerError`
    (`rpc.py`, asserts vanish under `-O`), `RUF100` dead `E402` noqas ×3
    removed (`test_seeds_properties.py`), `UP007` Union → `X|Y`
    (`atomic.py`). `ruff check .` is green tree-wide with the exemptions.
- Follow-ups (un-ignore per file, owning passes): BLE narrowing in
  `tui(16)/supervisor(7)/cli(7)/loop(2)/doctor(2)/causvid(1)/director(1)`;
  TRY004→TypeError / TRY300-else / TRY301-inner-function in
  `config,logrotate,workers/{audio_acestep,director,causvid,longlive,ltxv}`;
  SIM105/108/117/103/300 conversions (list in pyproject comments);
  shipped `S101` asserts (`causvid 6/cli 6/longlive 4/ltxv 3/supervisor 1/
  model_registry 1` — triage intentional-invariants vs real asserts);
  `T201` print interface (`cli 131` needs a rich/console decision first);
  dead noqas (`supervisor` BLE001 ×2 now at :699,1178 after drift,
  `director` ×2 at :232,330, plus UP017/PLC0415/F401/E402 noqas for rules
  that stay unselected — kept deliberately, RUF100-exempted per file).
- Open: none in this slice — rule lands are done; remaining work is the
  per-file conversions above.

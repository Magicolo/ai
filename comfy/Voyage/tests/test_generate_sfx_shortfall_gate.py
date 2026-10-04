"""Generate's pre-finalize gate skips the healable SFX shortfall (CPU-only).

SFX stems render only at finalize (`render_sfx_bed` cache-hits old windows
and renders the new ones), so a resume-generate on a run whose ledger
predates the new segments ALWAYS trips `sfx coverage Xs short of timeline
Ys` — a shortfall finalize itself would heal. The gate must filter exactly
that line (hard gaps / missing stems / unreadable ledgers stay fatal),
and only when the SFX pass will actually run (sfx enabled); with
`no_sfx` (or a fake sfx backend) nothing heals it, so it stays hard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voyage.config import preset_config


def _write_sfx_window(run_dir: Path, start: float, duration: float) -> str:
    stems = run_dir / "audio" / "sfx"
    stems.mkdir(parents=True, exist_ok=True)
    stem = stems / "w0000.wav"
    stem.write_bytes(b"fake-stem")
    record = {
        "window_id": "w0000",
        "start": start,
        "duration": duration,
        "path": "audio/sfx/w0000.wav",
        "caption": "rumble",
        "seed": 7,
    }
    (stems / "sfx.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return "sfx coverage"


def _mmaudio_effective() -> Any:
    config = preset_config("shortfall", "pastel neon line-art, peaceful", 11)
    return config.model_copy(update={"sfx": config.sfx.model_copy(update={"backend": "mmaudio"})})


def test_shortfall_predicate_matches_ledger_format(tmp_path: Path) -> None:
    from voyage.sfx_finalize import is_healable_sfx_shortfall, validate_sfx_ledger

    _write_sfx_window(tmp_path, 0.0, 7.44)
    errors = validate_sfx_ledger(tmp_path, 10.08)
    assert len(errors) == 1
    assert errors[0].startswith("sfx coverage ")
    assert is_healable_sfx_shortfall(errors[0]) is True


def test_shortfall_predicate_rejects_hard_errors() -> None:
    from voyage.sfx_finalize import is_healable_sfx_shortfall

    assert (
        is_healable_sfx_shortfall("sfx coverage gap: window w0001 starts at 8.00s, expected ~7.44s")
        is False
    )
    assert is_healable_sfx_shortfall("sfx w0001 missing audio/sfx/w0001.wav") is False
    assert is_healable_sfx_shortfall("sfx ledger unreadable: boom") is False
    assert is_healable_sfx_shortfall("timeline frames 242 != sum of segment frames 121") is False
    assert is_healable_sfx_shortfall("") is False


def test_pre_finalize_errors_filters_only_shortfall(tmp_path: Path, monkeypatch: Any) -> None:
    import voyage.cli_generate as generate_module

    shortfall = "sfx coverage 7.44s short of timeline 10.08s"
    gap = "sfx coverage gap: window w0001 starts at 8.00s, expected ~7.44s"
    other = "timeline frames 242 != sum of segment frames 121"
    monkeypatch.setattr(generate_module, "validate_run", lambda _run_dir: [shortfall, gap, other])
    kept = generate_module._pre_finalize_errors(tmp_path, {}, _mmaudio_effective())
    assert kept == [gap, other]


def test_pre_finalize_errors_keeps_shortfall_when_sfx_disabled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import voyage.cli_generate as generate_module

    shortfall = "sfx coverage 7.44s short of timeline 10.08s"
    monkeypatch.setattr(generate_module, "validate_run", lambda _run_dir: [shortfall])
    assert generate_module._pre_finalize_errors(
        tmp_path, {"no_sfx": True}, _mmaudio_effective()
    ) == [shortfall]
    fake_effective = _mmaudio_effective().model_copy(
        update={"sfx": _mmaudio_effective().sfx.model_copy(update={"backend": "fake"})}
    )
    assert fake_effective.sfx.backend == "fake"
    assert generate_module._pre_finalize_errors(tmp_path, {}, fake_effective) == [shortfall]

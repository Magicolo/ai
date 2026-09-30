"""Issue 079 TDD: longlive2 deprecation (failing-first, partial — not deletion).

Full deletion is unsafe while `supervisor.py`/`director.py`/`video_ltxv.py`
carry concurrent Stage-A telemetry hunks and longlive2 still has live
users across the registry, CLI, TUI, model registry, Dockerfile, scripts,
and six test modules. This pins the safe subset instead:
- `DEPRECATED_VIDEO_BACKENDS` names the deprecated backend.
- `warn_if_deprecated_backend` warns once per selection.
- The registry row stays (documents intentional retention).
- TUI help marks the backend deprecated.
"""

from __future__ import annotations


def test_deprecated_backends_name_longlive2() -> None:
    from voyage import config

    assert "longlive2" in config.DEPRECATED_VIDEO_BACKENDS
    for live in ("ltxv", "causvid", "fake"):
        assert live not in config.DEPRECATED_VIDEO_BACKENDS


def test_warn_if_deprecated_backend_warns_only_for_longlive2() -> None:
    import warnings

    from voyage import config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.warn_if_deprecated_backend("longlive2")
    assert any("longlive2" in str(warning.message).lower() for warning in caught)
    with warnings.catch_warnings(record=True) as caught_live:
        warnings.simplefilter("always")
        for live in ("ltxv", "causvid", "fake"):
            config.warn_if_deprecated_backend(live)  # type: ignore[arg-type]
    assert caught_live == []


def test_longlive2_registry_row_retained() -> None:
    from voyage import config

    assert "longlive2" in config.BACKEND_REGISTRY


def test_tui_help_marks_longlive2_deprecated() -> None:
    from voyage import tui_state

    assert "deprecat" in tui_state.FIELD_HELP["backend"].lower()

"""Unset absent-encoding (issue 045).

The CLI spells absence as None, the TUI form as "": one sentinel (Unset)
plus one conversion point (provided_or_none) so consumers only ever
check `is not None`. CPU-only.
"""

from __future__ import annotations

from voyage.config import ProjectConfig, Unset, UnsetType, is_provided, resolve_config


def test_unset_is_falsy_singleton() -> None:
    assert not Unset
    assert repr(Unset) == "Unset"
    assert isinstance(Unset, UnsetType)


def test_is_provided_treats_both_absences_as_absent() -> None:
    assert not is_provided(Unset)
    assert not is_provided(None)
    assert is_provided(0)
    assert is_provided("")
    assert is_provided(False)
    assert is_provided("qwen")


def test_resolve_config_accepts_unset_like_none() -> None:
    base = ProjectConfig(style="probe")
    via_none = resolve_config(base, blocks=None, take_seconds=None)
    via_unset = resolve_config(base, blocks=Unset, take_seconds=Unset)
    assert via_none == via_unset == base


def test_resolve_config_explicit_value_still_wins() -> None:
    base = ProjectConfig(style="probe")
    assert resolve_config(base, blocks=2).video.blocks_per_segment == 2


def test_tui_optional_int_emits_unset() -> None:
    from voyage.tui_state import GenerateFormState, to_generate_namespace

    namespace = to_generate_namespace(
        GenerateFormState(style="x", name="v", duration="5s", blocks="", take_seconds="")
    )
    assert namespace.blocks is Unset
    assert namespace.take_seconds is Unset

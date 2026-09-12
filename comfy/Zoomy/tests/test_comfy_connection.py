"""Tests for the ComfyUI connection's statistics parsing."""

from __future__ import annotations

from typing import Any, Literal

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from zoomy.comfy_connection import (
    SystemStatistics,
    _parse_system_statistics,
    _rejected_workflow_error,
)
from zoomy.errors import ComfyConnectionError, ZoomyError

# Wire-transportable text: NUL-free (matching the settings domain) and
# surrogate-free (lone surrogates are not UTF-8 encodable, so httpx refuses
# to build a body with them — like non-finite floats, they can never arrive
# on the wire and must not be generated for response bodies).
_SURROGATE_CATEGORY: tuple[Literal["Cs"], ...] = ("Cs",)
wire_characters = st.characters(
    blacklist_characters="\x00", blacklist_categories=_SURROGATE_CATEGORY
)

json_atom = (
    st.none()
    | st.booleans()
    | st.integers()
    # Non-finite floats are not JSON-transportable (httpx refuses to encode
    # them), so bodies under test stay within what can arrive on the wire.
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(alphabet=wire_characters, max_size=20)
)
json_value = st.recursive(
    json_atom,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(alphabet=wire_characters, max_size=8), children, max_size=3)
    ),
    max_leaves=8,
)

# Payloads shaped like the real endpoints but carrying arbitrary values,
# including NaN and infinities, which plain fuzzing almost never places
# under the exact keys the parser reads.
byte_like = json_atom
system_payload = st.fixed_dictionaries(
    {},
    optional={
        "system": st.fixed_dictionaries(
            {}, optional={"ram_free": byte_like, "ram_total": byte_like}
        ),
        "devices": st.lists(
            st.fixed_dictionaries({}, optional={"vram_free": byte_like, "vram_total": byte_like}),
            max_size=3,
        ),
        "error": json_value,
        "node_errors": json_value,
    },
)


def _full_payload() -> dict[str, Any]:
    """Build a well-formed /system_stats body with one device."""
    return {
        "system": {"ram_free": 12000000000, "ram_total": 32000000000},
        "devices": [{"vram_free": 9000000000, "vram_total": 16000000000}],
    }


def test_parse_system_statistics_reads_memory_figures() -> None:
    """A full payload yields system RAM plus the first device's VRAM."""
    statistics = _parse_system_statistics(_full_payload())
    assert statistics == SystemStatistics(
        system_memory_free_bytes=12000000000,
        system_memory_total_bytes=32000000000,
        video_memory_free_bytes=9000000000,
        video_memory_total_bytes=16000000000,
    )


def test_parse_system_statistics_tolerates_missing_devices() -> None:
    """Without devices the video figures stay None instead of failing."""
    payload = {"system": {"ram_free": 1, "ram_total": 2}, "devices": []}
    statistics = _parse_system_statistics(payload)
    assert statistics.video_memory_free_bytes is None
    assert statistics.video_memory_total_bytes is None
    assert statistics.system_memory_free_bytes == 1


def test_parse_system_statistics_rejects_malformed_payloads() -> None:
    """Garbage shapes raise a connection error rather than crashing."""
    with pytest.raises(ComfyConnectionError, match="unexpected shape"):
        _parse_system_statistics(["not", "a", "dict"])
    with pytest.raises(ComfyConnectionError, match="system section"):
        _parse_system_statistics({"devices": []})
    with pytest.raises(ComfyConnectionError, match="ram_free"):
        _parse_system_statistics({"system": {"ram_free": "lots", "ram_total": 2}})


def test_parse_system_statistics_rejects_non_finite_counts() -> None:
    """NaN and infinite byte counts are rejected, not leaked as ValueError."""
    for bad_value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ComfyConnectionError, match="finite"):
            _parse_system_statistics({"system": {"ram_free": bad_value, "ram_total": 8}})


@given(payload=json_value)
def test_parse_system_statistics_never_leaks_unexpected_errors(payload: Any) -> None:
    """Any JSON body yields valid figures or a connection error, never a crash."""
    try:
        statistics = _parse_system_statistics(payload)
    except ComfyConnectionError:
        return
    assert isinstance(statistics.system_memory_free_bytes, int)
    assert isinstance(statistics.system_memory_total_bytes, int)
    video_free = statistics.video_memory_free_bytes
    video_total = statistics.video_memory_total_bytes
    assert video_free is None or isinstance(video_free, int)
    assert video_total is None or isinstance(video_total, int)


@given(payload=system_payload)
def test_parse_system_statistics_survives_shaped_garbage(payload: Any) -> None:
    """Right keys with wrong-typed values still yield figures or an error."""
    try:
        statistics = _parse_system_statistics(payload)
    except ComfyConnectionError:
        return
    assert isinstance(statistics.system_memory_free_bytes, int)
    assert isinstance(statistics.system_memory_total_bytes, int)


@given(payload=json_value)
def test_rejection_parsing_never_crashes(payload: Any) -> None:
    """Any 400 body translates to a rejection error with a message."""
    response = httpx.Response(400, json=payload)
    error = _rejected_workflow_error(response)
    assert isinstance(error, ZoomyError)
    assert str(error)


def test_rejection_parsing_handles_non_json_bodies() -> None:
    """A non-JSON 400 still produces a usable rejection error."""
    error = _rejected_workflow_error(httpx.Response(400, content=b"not json"))
    assert "unknown validation failure" in str(error)


def test_rejection_parsing_handles_lone_surrogate_bodies() -> None:
    """Surrogate bytes arrive raw on the wire and must not crash parsing."""
    error = _rejected_workflow_error(httpx.Response(400, content=b'{"error": "\xed\xa0\x80"}'))
    assert isinstance(error, ZoomyError)
    assert str(error)

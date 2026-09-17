"""Tests for the Gradio interface assembly."""

from __future__ import annotations

import math
import re
import socket
from typing import TYPE_CHECKING

import gradio as gr
import httpx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from zoomy.engine_protocol import EngineStatistics
from zoomy.errors import ZoomyError
from zoomy.family_catalog import FAMILY_CATALOG, find_family
from zoomy.frame_repository import FrameRepository, SequenceStatistics
from zoomy.interface import (
    FamilyPanelWiring,
    InterfaceContext,
    _append_log_entry,
    _as_durations,
    _as_string_list,
    _bind_clear_frames,
    _bind_finalize,
    _bind_refresh_previews,
    _bind_refresh_status,
    _coerce_frame_size,
    _coerce_to_float,
    _draw_family_panel,
    _format_bytes,
    _format_timestamp,
    _parse_panel_submission,
    _parse_target_seconds,
    _record_frame_duration,
    _render_statistics_line,
    _safe_system_statistics,
    _selected_lora_names,
    build_application,
)
from zoomy.rendering import RenderEnvironment
from zoomy.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from PIL.Image import Image

    from zoomy.engine_protocol import FinalizeRequest, FrameRenderRequest, ProgressUpdate

BYTE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?) (B|KiB|MiB|GiB|TiB)$")
TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}$")
# Log lines never contain line boundaries (splitlines() would count them as
# several lines), so the alphabet excludes every Unicode line-boundary class.
SINGLE_LINE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Zl", "Zp")), max_size=50
)


def _example_statistics(tmp_path: Path) -> SequenceStatistics:
    """Build canned statistics with frames, video, and timestamps."""
    return SequenceStatistics(
        sequence_key="z_image",
        frame_count=2,
        frames_bytes=2048,
        recent_frame_paths=(),
        video_path=tmp_path / "z_image_00001-audio.mp4",
        video_bytes=4096,
        video_modified_timestamp=200.0,
    )


def _example_system() -> EngineStatistics:
    """Build canned engine statistics with RAM and VRAM figures."""
    return EngineStatistics(
        system_memory_free_bytes=12000000000,
        system_memory_total_bytes=32000000000,
        video_memory_free_bytes=9000000000,
        video_memory_total_bytes=16000000000,
    )


class StubEngine:
    """Test double standing in for the in-process generation engine."""

    def render_frame(self, request: FrameRenderRequest) -> Image:  # noqa: ARG002
        """Refuse frame renders; no test drives generation here."""
        raise ZoomyError("StubEngine never renders")

    def finalize_sequence(
        self,
        request: FinalizeRequest,  # noqa: ARG002
    ) -> Iterator[ProgressUpdate]:
        """Yield nothing; no test drives finalization here."""
        return
        yield  # Make this a generator even though it never yields.

    def request_interrupt(self) -> None:
        """Do nothing; no test drives interrupts through this double."""

    def is_ready(self) -> bool:
        """Report a ready engine."""
        return True

    def engine_statistics(self) -> EngineStatistics:
        """Report canned memory figures."""
        return _example_system()


class _FailingStatisticsEngine(StubEngine):
    """Test double whose statistics call blows up like a sick backend."""

    def engine_statistics(self) -> EngineStatistics:
        """Raise outside the ZoomyError hierarchy, as torch errors do."""
        raise RuntimeError("CUDA error: out of memory")


def _example_settings(output_directory: str) -> Settings:
    """Build settings pointing at the test output directory."""
    return Settings(
        models_directory="/models",
        seed_directory="/seed",
        music_project_directory="/music-project",
        output_directory=output_directory,
        interface_address="127.0.0.1",
        interface_port=7861,
        cuda_device="cuda:0",
    )


def test_build_application_returns_blocks(tmp_path: Path) -> None:
    """The full panel assembles without launching or touching the network."""
    settings = _example_settings(str(tmp_path))
    application = build_application(
        settings=settings,
        engine=StubEngine(),
        catalog=FAMILY_CATALOG,
        repository=FrameRepository(tmp_path),
    )
    assert isinstance(application, gr.Blocks)


def _wiring_context(tmp_path: Path) -> InterfaceContext:
    """Build handler wiring over stub components inside a Blocks scope."""
    engine = StubEngine()
    repository = FrameRepository(tmp_path)
    environment = RenderEnvironment(engine=engine, repository=repository)
    with gr.Blocks():
        return InterfaceContext(
            settings=_example_settings(str(tmp_path)),
            engine=engine,
            catalog=FAMILY_CATALOG,
            repository=repository,
            environment=environment,
            family_dropdown=gr.Dropdown(choices=["z_fast"], value="z_fast"),
            health_badge=gr.HTML(value="badge"),
            stats_line=gr.Markdown(value="stats"),
            status_markdown=gr.Markdown(value="status"),
            log_textbox=gr.Textbox(value="log"),
            log_state=gr.State(value=[]),
            durations_state=gr.State(value=(0, 0.0, 0.0)),
            confirmation_state=gr.State(value=False),
            preview_image=gr.Image(),
            gallery=gr.Gallery(),
            preview_video=gr.Video(),
            finalize_button=gr.Button(value="Finalize"),
            interrupt_button=gr.Button(value="Interrupt"),
            refresh_button=gr.Button(value="Refresh"),
            clear_frames_button=gr.Button(value="Clear"),
        )


def test_refresh_status_reports_unknown_family(tmp_path: Path) -> None:
    """A bad dropdown key yields an error line, never a traceback."""
    badge, line = _bind_refresh_status(_wiring_context(tmp_path))("no-such-family", (0, 0.0, 0.0))
    assert "Engine offline" in badge
    assert line.startswith("**Error:**")
    assert "no-such-family" in line


def test_refresh_previews_reports_unknown_family(tmp_path: Path) -> None:
    """A bad dropdown key disarms previews with an error, never a raise."""
    message, *_rest = _bind_refresh_previews(_wiring_context(tmp_path))(
        "no-such-family", (0, 0.0, 0.0)
    )
    assert message.startswith("**Error:**")
    assert "no-such-family" in message


def test_finalize_reports_unknown_family(tmp_path: Path) -> None:
    """A bad dropdown key yields one error update, never a traceback."""
    updates = list(_bind_finalize(_wiring_context(tmp_path))("no-such-family", [], (0, 0.0, 0.0)))
    assert len(updates) == 1
    (message, _log, _video, _stats, _gallery) = updates[0]
    assert isinstance(message, str)
    assert message.startswith("**Error:**")
    assert "no-such-family" in message


def test_clear_frames_reports_unknown_family(tmp_path: Path) -> None:
    """A bad dropdown key refuses the delete with an error, never a raise."""
    _armed, _button, message, _preview, _stats, _gallery = _bind_clear_frames(
        _wiring_context(tmp_path)
    )(False, "no-such-family", (0, 0.0, 0.0))  # noqa: FBT003
    # Positional: clear_frames takes *values, mirroring Gradio's payload order.
    assert isinstance(message, str)
    assert message.startswith("**Error:**")
    assert "no-such-family" in message


def test_clear_frames_reports_undeletable_directory(tmp_path: Path) -> None:
    """A failed delete yields an error update, never a traceback."""
    context = _wiring_context(tmp_path)
    (tmp_path / "z_image").write_bytes(b"not a directory")
    _armed, _button, message, _preview, _stats, _gallery = _bind_clear_frames(context)(
        True,  # noqa: FBT003
        "z_fast",
        (0, 0.0, 0.0),
    )
    assert isinstance(message, str)
    assert message.startswith("**Error:**")
    assert "z_image" in message


def test_format_bytes_uses_binary_units() -> None:
    """Byte counts render with the right unit and one decimal."""
    assert _format_bytes(0) == "0 B"
    assert _format_bytes(512) == "512 B"
    assert _format_bytes(1024) == "1.0 KiB"
    assert _format_bytes(1536) == "1.5 KiB"
    assert _format_bytes(5 * 1024 * 1024) == "5.0 MiB"
    assert _format_bytes(2 * 1024 * 1024 * 1024) == "2.0 GiB"


def test_render_statistics_line_shows_everything(tmp_path: Path) -> None:
    """The full line covers frames, disk, durations, video, and memory."""
    line = _render_statistics_line(
        _example_statistics(tmp_path), _example_system(), (2, 27.0, 13.0)
    )
    assert "**2** frames" in line
    assert "2.0 KiB" in line
    assert "last frame 13 s" in line
    assert "avg 13.5 s" in line
    assert "z_image_00001-audio.mp4" in line
    assert "4.0 KiB" in line
    assert "8.4 / 14.9 GiB" in line
    assert "11.2 / 29.8 GiB" in line


def test_render_statistics_line_handles_empty_state() -> None:
    """No frames, video, or system yields placeholders, never crashes."""
    statistics = SequenceStatistics(
        sequence_key="z_image",
        frame_count=0,
        frames_bytes=0,
        recent_frame_paths=(),
        video_path=None,
        video_bytes=None,
        video_modified_timestamp=None,
    )
    line = _render_statistics_line(statistics, None, (0, 0.0, 0.0))
    assert "**0** frames" in line
    assert "no video yet" in line
    assert "engine offline" in line
    assert "last frame" not in line


def test_safe_system_statistics_treats_backend_errors_as_offline(
    tmp_path: Path,
) -> None:
    """A non-ZoomyError from the engine yields an offline line, never a raise."""
    system = _safe_system_statistics(_FailingStatisticsEngine())
    assert system is None
    line = _render_statistics_line(_example_statistics(tmp_path), system, (0, 0.0, 0.0))
    assert "engine offline" in line


def test_parse_panel_submission_with_negative_prompt() -> None:
    """Checkbox, sliders, and both prompts parse in order with the count."""
    family = find_family(FAMILY_CATALOG, "z_fast")
    submission = _parse_panel_submission(
        family, [["Chalkboard"], 0.5, 0.9, "a prompt", "a negative"], has_negative_prompt=True
    )
    assert [lora.display_name for lora, _ in submission.lora_selections] == ["Chalkboard"]
    assert [strength for _, strength in submission.lora_selections] == [0.5]
    assert submission.prompt_text == "a prompt"
    assert submission.negative_text == "a negative"
    assert submission.consumed_count == 5


def test_parse_panel_submission_without_negative_prompt() -> None:
    """Families without a negative consume one fewer value."""
    family = find_family(FAMILY_CATALOG, "ernie_turbo")
    submission = _parse_panel_submission(family, [[], 1.0, "a prompt"], has_negative_prompt=False)
    assert submission.lora_selections == ()
    assert submission.prompt_text == "a prompt"
    assert submission.negative_text is None
    assert submission.consumed_count == 3


def test_parse_target_seconds_accepts_only_positive_durations() -> None:
    """Blank, zero, negative, and non-numeric durations all mean unlimited."""
    assert _parse_target_seconds(None) is None
    assert _parse_target_seconds(0) is None
    assert _parse_target_seconds(-2.5) is None
    assert _parse_target_seconds(value=True) is None
    assert _parse_target_seconds("long") is None
    assert _parse_target_seconds(2.5) == 2.5
    assert _parse_target_seconds(10) == 10.0


def test_parse_target_seconds_rejects_non_finite_numbers() -> None:
    """Infinities and NaN mean unlimited instead of sizing an endless loop."""
    assert _parse_target_seconds(float("inf")) is None
    assert _parse_target_seconds(float("-inf")) is None
    assert _parse_target_seconds(float("nan")) is None


def test_coerce_frame_size_falls_back_to_the_default() -> None:
    """Blank or garbage geometry restores the default; counts pass through."""
    assert _coerce_frame_size(None, 1376) == 1376
    assert _coerce_frame_size(0, 1376) == 1376
    assert _coerce_frame_size("wide", 1376) == 1376
    assert _coerce_frame_size(value=True, default_pixels=1376) == 1376
    assert _coerce_frame_size(512, 1376) == 512
    assert _coerce_frame_size(512.9, 1376) == 512


def test_append_log_entry_trims_and_shows_the_tail() -> None:
    """History stays bounded while the visible text shows recent lines."""
    entries, visible_text = _append_log_entry(["first"], "second")
    assert entries == ["first", "second"]
    assert visible_text == "first\nsecond"
    long_history = [f"line {number}" for number in range(205)]
    trimmed_entries, trimmed_text = _append_log_entry(long_history, "last")
    assert len(trimmed_entries) == 200
    assert trimmed_text.splitlines()[-1] == "last"
    assert len(trimmed_text.splitlines()) == 8


def test_as_durations_accepts_only_well_formed_tuples() -> None:
    """Garbage state payloads fall back to zeros instead of crashing."""
    assert _as_durations((2, 27.0, 13.0)) == (2, 27.0, 13.0)
    assert _as_durations(None) == (0, 0.0, 0.0)
    assert _as_durations([2, 27.0, 13.0]) == (0, 0.0, 0.0)
    assert _as_durations((2, "slow", 13.0)) == (0, 0.0, 0.0)


def test_record_frame_duration_accumulates() -> None:
    """Each completed frame bumps the count, total, and last figure."""
    assert _record_frame_duration((0, 0.0, 0.0), 14.0) == (1, 14.0, 14.0)
    assert _record_frame_duration((2, 27.0, 13.0), 15.0) == (3, 42.0, 15.0)


@given(byte_count=st.integers(min_value=0, max_value=10**15))
def test_format_bytes_matches_unit_pattern(byte_count: int) -> None:
    """Every byte count formats as a non-negative number plus a valid unit."""
    match = BYTE_PATTERN.match(_format_bytes(byte_count))
    assert match is not None
    assert float(match.group(1)) >= 0


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_render_statistics_line_contract(tmp_path: Path, data: st.DataObject) -> None:
    """The stats line always names the count and degrades gracefully."""
    frame_count = data.draw(st.integers(min_value=0, max_value=200))
    frames_bytes = data.draw(st.integers(min_value=0, max_value=10**12))
    has_video = data.draw(st.booleans())
    has_system = data.draw(st.booleans())
    rendered_frames = data.draw(st.integers(min_value=0, max_value=500))
    total_seconds = data.draw(st.floats(min_value=0, max_value=1e6, allow_nan=False))
    last_seconds = data.draw(st.floats(min_value=0, max_value=1e6, allow_nan=False))
    statistics = SequenceStatistics(
        sequence_key="z_image",
        frame_count=frame_count,
        frames_bytes=frames_bytes,
        recent_frame_paths=(),
        video_path=tmp_path / "video.mp4" if has_video else None,
        video_bytes=4096 if has_video else None,
        video_modified_timestamp=200.0 if has_video else None,
    )
    system = _example_system() if has_system else None
    line = _render_statistics_line(
        statistics, system, (rendered_frames, total_seconds, last_seconds)
    )
    assert str(frame_count) in line
    assert "frames" in line
    assert ("no video yet" in line) == (not has_video)
    assert ("engine offline" in line) == (not has_system)
    assert ("last frame" in line) == (rendered_frames > 0)


@given(data=st.data())
def test_panel_submission_roundtrip(data: st.DataObject) -> None:
    """Panel values parse back to the selected styles, strengths, and texts."""
    family = data.draw(st.sampled_from(FAMILY_CATALOG))
    names = [lora.display_name for lora in family.loras]
    selected = data.draw(st.lists(st.sampled_from(names), unique=True)) if names else []
    strengths = data.draw(
        st.lists(
            st.floats(min_value=0, max_value=2, allow_nan=False),
            min_size=len(names),
            max_size=len(names),
        )
    )
    prompt = data.draw(st.text(max_size=200))
    values: list[object] = [selected, *strengths, prompt]
    has_negative = family.negative_prompt is not None
    if has_negative:
        negative = data.draw(st.text(max_size=200))
        values.append(negative)
    submission = _parse_panel_submission(family, values, has_negative_prompt=has_negative)
    assert {lora.display_name for lora, _ in submission.lora_selections} == set(selected)
    assert {lora.display_name: strength for lora, strength in submission.lora_selections} == {
        name: strengths[names.index(name)] for name in selected
    }
    assert submission.prompt_text == prompt
    assert submission.consumed_count == len(values)


@given(value=st.integers() | st.floats() | st.text(max_size=30) | st.booleans() | st.none())
def test_parse_target_seconds_contract(value: object) -> None:
    """Durations are positive floats or None; nothing else escapes, nothing crashes."""
    target = _parse_target_seconds(value)
    if isinstance(value, bool):
        assert target is None
    elif isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        assert target == float(value)
    else:
        assert target is None


@given(entries=st.lists(SINGLE_LINE_TEXT, max_size=300), message=SINGLE_LINE_TEXT)
def test_append_log_entry_bounds_history(entries: list[str], message: str) -> None:
    """History stays bounded while the message always lands on the last line."""
    updated, visible_text = _append_log_entry(entries, message)
    assert len(updated) <= 200
    assert updated[-1] == message
    assert visible_text == "\n".join(updated[-8:])
    assert len(visible_text.splitlines()) <= 8


@given(
    value=st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=30)
    | st.booleans()
    | st.none()
    | st.lists(st.integers(), max_size=5)
)
def test_ui_payload_coercions_never_crash(value: object) -> None:
    """Gradio payload coercions always return sane shapes, never raise."""
    durations = _as_durations(value)
    assert len(durations) == 3
    assert isinstance(durations[0], int)
    assert durations[0] >= 0
    assert isinstance(_coerce_to_float(value), float)
    assert all(isinstance(name, str) for name in _selected_lora_names(value))
    assert isinstance(_as_string_list(value), list)


@given(epoch_seconds=st.floats(min_value=0, max_value=4102444800, allow_nan=False))
def test_format_timestamp_matches_clock_pattern(epoch_seconds: float) -> None:
    """Finite timestamps always render as zero-padded HH:MM."""
    assert TIMESTAMP_PATTERN.match(_format_timestamp(epoch_seconds)) is not None


def _test_wiring(tmp_path: Path) -> FamilyPanelWiring:
    """Build panel wiring with real components inside the active Blocks."""
    engine = StubEngine()
    repository = FrameRepository(tmp_path)
    return FamilyPanelWiring(
        settings=_example_settings(str(tmp_path)),
        engine=engine,
        catalog=FAMILY_CATALOG,
        repository=repository,
        environment=RenderEnvironment(engine=engine, repository=repository),
        status_markdown=gr.Markdown(),
        log_textbox=gr.Textbox(),
        log_state=gr.State(value=[]),
        preview_image=gr.Image(),
        stats_line=gr.Markdown(),
        gallery=gr.Gallery(),
        preview_video=gr.Video(),
        durations_state=gr.State(value=(0, 0.0, 0.0)),
    )


def test_draw_family_panel_for_every_family(tmp_path: Path) -> None:
    """Drawing each panel executes all component and event wiring.

    Construct-only tests never run render functions, so attribute errors on
    event methods (like a missing Checkbox.unselect) slip through to page
    load. Drawing every family headlessly closes that gap.
    """
    with gr.Blocks():
        wiring = _test_wiring(tmp_path)
        for family in FAMILY_CATALOG:
            with gr.Row():
                _draw_family_panel(wiring, family.key)


def _free_port() -> int:
    """Return a currently-free localhost port for the smoke launch."""
    with socket.socket() as socket_handle:
        socket_handle.bind(("127.0.0.1", 0))
        return int(socket_handle.getsockname()[1])


def test_application_serves_front_page(tmp_path: Path) -> None:
    """Launching serves HTTP 200 (catches launch/config regressions)."""
    settings = _example_settings(str(tmp_path))
    application = build_application(
        settings=settings,
        engine=StubEngine(),
        catalog=FAMILY_CATALOG,
        repository=FrameRepository(tmp_path),
    )
    _, url, _ = application.launch(
        server_name="127.0.0.1",
        server_port=_free_port(),
        allowed_paths=[str(tmp_path)],
        show_error=False,
        inbrowser=False,
        ssr_mode=False,
        quiet=True,
        prevent_thread_lock=True,
    )
    try:
        # The launch URL carries a trailing slash; joining paths naively
        # produces double slashes, which the server answers with 404.
        response = httpx.get(f"{url.rstrip('/')}/", timeout=10.0)
    finally:
        application.close()
    assert response.status_code == 200

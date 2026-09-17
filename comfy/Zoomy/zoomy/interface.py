"""Gradio web interface for the zoomy control panel.

Layout: a header row (family selector + reachability badge), a live
statistics line, a short status line, then a main row with the per-family
controls on the left and previews (latest frame, recent-frames gallery,
finalized video) on the right, followed by the session log and the global
action buttons.

Ordering note: the per-family panel is drawn by ``@gr.render`` whenever the
family dropdown changes. That decorated function executes immediately at app
build time, so every component its event wiring references is created before
the render block in source order — hence the two wiring dataclasses.

Loop control: checking the Loop box starts :func:`generate_video`, which
renders frame after frame until the optional target duration, a graceful
stop request (unchecking the box), or exhausted per-frame retries, then
runs the finalize stages automatically (after a reached target always,
after a manual stop when the auto-finalize box is checked). Graceful stops
flow through a module-level ``threading.Event`` (see
:func:`request_loop_stop`); the forceful Interrupt button aborts the
running engine job immediately, which ends the loop as a side effect.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import gradio as gr

from zoomy.engine_protocol import (
    FRAME_HEIGHT_PIXELS,
    FRAME_WIDTH_PIXELS,
    FinalizeRequest,
    FrameRenderRequest,
)
from zoomy.errors import ZoomyError
from zoomy.family_catalog import find_family
from zoomy.rendering import (
    RenderEnvironment,
    VideoGenerationOptions,
    finalize_video,
    generate_video,
    render_next_frame,
    request_loop_stop,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Any

    from zoomy.engine_protocol import EngineProtocol, EngineStatistics
    from zoomy.family_catalog import FamilyDefinition, LoraDefinition
    from zoomy.frame_repository import FrameRepository, SequenceStatistics
    from zoomy.settings import Settings

MAXIMUM_LORA_STRENGTH = 2.0
LORA_STRENGTH_STEP = 0.05
MAXIMUM_SEED = 2**48
STATUS_TIMER_SECONDS = 10.0
CLEAR_FRAMES_LABEL = "Clear frames"
CONFIRM_CLEAR_FRAMES_LABEL = "Confirm: clear all frames"
MAXIMUM_LOG_LINES = 200
VISIBLE_LOG_LINES = 8
DURATIONS_TUPLE_LENGTH = 3
LOG_LINES_HEIGHT = 8
GALLERY_COLUMNS = 4
GALLERY_ROWS = 2
BYTES_PER_UNIT = 1024.0
BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")

_random_generator = random.SystemRandom()


@dataclass(frozen=True, slots=True)
class FamilyPanelWiring:
    """Collaborators and outputs the per-family panel buttons target."""

    settings: Settings
    engine: EngineProtocol
    catalog: Sequence[FamilyDefinition]
    repository: FrameRepository
    environment: RenderEnvironment
    status_markdown: gr.Markdown
    log_textbox: gr.Textbox
    log_state: gr.State
    preview_image: gr.Image
    stats_line: gr.Markdown
    gallery: gr.Gallery
    preview_video: gr.Video
    durations_state: gr.State


@dataclass(frozen=True, slots=True)
class PanelSubmission:
    """Parsed values from one per-family panel submission."""

    lora_selections: tuple[tuple[LoraDefinition, float], ...]
    prompt_text: str
    negative_text: str | None
    consumed_count: int


@dataclass(frozen=True, slots=True)
class InterfaceContext:
    """Every collaborator and component the outer event wiring needs."""

    settings: Settings
    engine: EngineProtocol
    catalog: Sequence[FamilyDefinition]
    repository: FrameRepository
    environment: RenderEnvironment
    family_dropdown: gr.Dropdown
    health_badge: gr.HTML
    stats_line: gr.Markdown
    status_markdown: gr.Markdown
    log_textbox: gr.Textbox
    log_state: gr.State
    durations_state: gr.State
    confirmation_state: gr.State
    preview_image: gr.Image
    gallery: gr.Gallery
    preview_video: gr.Video
    finalize_button: gr.Button
    interrupt_button: gr.Button
    refresh_button: gr.Button
    clear_frames_button: gr.Button


def build_application(
    settings: Settings,
    engine: EngineProtocol,
    catalog: Sequence[FamilyDefinition],
    repository: FrameRepository,
) -> gr.Blocks:
    """Assemble the whole zoomy control panel as an unlaunched Gradio app."""
    with gr.Blocks(title="Zoomy") as application:
        context = _create_components(settings, engine, catalog, repository)
        _wire_events(context)
    return cast("gr.Blocks", application)


def _create_components(
    settings: Settings,
    engine: EngineProtocol,
    catalog: Sequence[FamilyDefinition],
    repository: FrameRepository,
) -> InterfaceContext:
    """Create every component in layout order and bundle the wiring context."""
    environment = RenderEnvironment(engine=engine, repository=repository)
    gr.Markdown("# Zoomy — infinite zoom control panel")
    with gr.Row():
        with gr.Column(scale=3):
            family_dropdown = gr.Dropdown(
                choices=[(family.display_name, family.key) for family in catalog],
                value=catalog[0].key,
                label="Model family",
            )
        with gr.Column(scale=1):
            health_badge = gr.HTML(value=_render_health_badge(is_reachable=engine.is_ready()))
    stats_line = gr.Markdown(value=_initial_statistics(catalog[0], repository, engine))
    status_markdown = gr.Markdown("Ready.")
    log_textbox = gr.Textbox(label="Session log", lines=LOG_LINES_HEIGHT, interactive=False)
    log_state: gr.State = gr.State(value=[])
    durations_state = gr.State(value=(0, 0.0, 0.0))
    confirmation_state = gr.State(value=False)
    with gr.Row():
        with gr.Column(scale=3):
            preview_image = gr.Image(label="Latest frame", interactive=False, type="filepath")
            gallery = gr.Gallery(
                label="Recent frames",
                columns=GALLERY_COLUMNS,
                rows=GALLERY_ROWS,
                object_fit="cover",
            )
            preview_video = gr.Video(label="Finalized video")
        with gr.Column(scale=2):
            panel_wiring = FamilyPanelWiring(
                settings=settings,
                engine=engine,
                catalog=catalog,
                repository=repository,
                environment=environment,
                status_markdown=status_markdown,
                log_textbox=log_textbox,
                log_state=log_state,
                preview_image=preview_image,
                stats_line=stats_line,
                gallery=gallery,
                preview_video=preview_video,
                durations_state=durations_state,
            )

            @gr.render(inputs=[family_dropdown])  # type: ignore[untyped-decorator]
            def family_panel(family_key: str) -> None:
                """Draw the per-family controls: LoRAs, prompts, render, loop."""
                _draw_family_panel(panel_wiring, family_key)

    with gr.Row():
        finalize_button = gr.Button("Finalize video")
        interrupt_button = gr.Button("Interrupt now (ends loop immediately)")
        refresh_button = gr.Button("Refresh previews")
        clear_frames_button = gr.Button(CLEAR_FRAMES_LABEL)
    return InterfaceContext(
        settings=settings,
        engine=engine,
        catalog=catalog,
        repository=repository,
        environment=environment,
        family_dropdown=family_dropdown,
        health_badge=health_badge,
        stats_line=stats_line,
        status_markdown=status_markdown,
        log_textbox=log_textbox,
        log_state=log_state,
        durations_state=durations_state,
        confirmation_state=confirmation_state,
        preview_image=preview_image,
        gallery=gallery,
        preview_video=preview_video,
        finalize_button=finalize_button,
        interrupt_button=interrupt_button,
        refresh_button=refresh_button,
        clear_frames_button=clear_frames_button,
    )


def _wire_events(context: InterfaceContext) -> None:
    """Attach every timer, change, and click handler.

    Gradio generates component event methods (tick/change/click) dynamically,
    so its type stubs omit them; each ignore below marks exactly one such
    call, verified present at runtime against gradio 6.26.0.
    """
    preview_outputs = [
        context.status_markdown,
        context.preview_image,
        context.preview_video,
        context.stats_line,
        context.gallery,
        context.confirmation_state,
        context.clear_frames_button,
    ]
    status_timer = gr.Timer(value=STATUS_TIMER_SECONDS)
    status_timer.tick(  # type: ignore[attr-defined]
        _bind_refresh_status(context),
        inputs=[context.family_dropdown, context.durations_state],
        outputs=[context.health_badge, context.stats_line],
        queue=False,
    )
    context.family_dropdown.change(  # type: ignore[attr-defined]
        _bind_refresh_previews(context),
        inputs=[context.family_dropdown, context.durations_state],
        outputs=preview_outputs,
        queue=False,
    )
    context.finalize_button.click(  # type: ignore[attr-defined]
        _bind_finalize(context),
        inputs=[context.family_dropdown, context.log_state, context.durations_state],
        outputs=[
            context.status_markdown,
            context.log_textbox,
            context.preview_video,
            context.stats_line,
            context.gallery,
        ],
    )
    context.interrupt_button.click(  # type: ignore[attr-defined]
        _bind_interrupt(context),
        inputs=None,
        outputs=[context.status_markdown],
        queue=False,
    )
    context.refresh_button.click(  # type: ignore[attr-defined]
        _bind_refresh_previews(context),
        inputs=[context.family_dropdown, context.durations_state],
        outputs=preview_outputs,
        queue=False,
    )
    context.clear_frames_button.click(  # type: ignore[attr-defined]
        _bind_clear_frames(context),
        inputs=[
            context.confirmation_state,
            context.family_dropdown,
            context.durations_state,
        ],
        outputs=[
            context.confirmation_state,
            context.clear_frames_button,
            context.status_markdown,
            context.preview_image,
            context.stats_line,
            context.gallery,
        ],
        queue=False,
    )


def _bind_refresh_status(
    context: InterfaceContext,
) -> Callable[[str, object], tuple[str, str]]:
    """Create the periodic handler that refreshes badge and statistics."""

    def refresh_status(family_key: str, durations_value: object) -> tuple[str, str]:
        """Poll reachability, memory, and sequence figures for the line."""
        try:
            family = find_family(context.catalog, family_key)
        except ZoomyError as failure:
            return (
                _render_health_badge(is_reachable=False),
                f"**Error:** {failure}",
            )
        durations = _as_durations(durations_value)
        system = _safe_system_statistics(context.engine)
        statistics = context.repository.sequence_statistics(family.sequence_key)
        return (
            _render_health_badge(is_reachable=system is not None),
            _render_statistics_line(statistics, system, durations),
        )

    return refresh_status


def _bind_refresh_previews(
    context: InterfaceContext,
) -> Callable[[str, object], tuple[str, object, object, str, object, bool, object]]:
    """Create the handler that reloads previews, stats, and gallery."""

    def refresh_previews(
        family_key: str, durations_value: object
    ) -> tuple[str, object, object, str, object, bool, object]:
        """Resolve the newest artifacts and disarm a pending clear-confirm."""
        try:
            family = find_family(context.catalog, family_key)
        except ZoomyError as failure:
            message = f"**Error:** {failure}"
            return (
                message,
                None,
                None,
                message,
                [],
                False,
                gr.update(value=CLEAR_FRAMES_LABEL),
            )
        durations = _as_durations(durations_value)
        system = _safe_system_statistics(context.engine)
        statistics = context.repository.sequence_statistics(family.sequence_key)
        recent_paths = [str(path) for path in statistics.recent_frame_paths]
        return (
            f"Refreshed previews for sequence *{family.sequence_key}*.",
            recent_paths[-1] if recent_paths else None,
            statistics.video_path,
            _render_statistics_line(statistics, system, durations),
            recent_paths,
            False,
            gr.update(value=CLEAR_FRAMES_LABEL),
        )

    return refresh_previews


def _bind_interrupt(context: InterfaceContext) -> Callable[[], str]:
    """Create the handler that aborts the running job immediately."""

    def interrupt_running_job() -> str:
        """Interrupt the engine now and end any running loop with it."""
        request_loop_stop()
        try:
            context.engine.request_interrupt()
        except ZoomyError as failure:
            return f"**Error:** {failure}"
        return "Interrupt requested; the running job stops immediately."

    return interrupt_running_job


def _bind_finalize(
    context: InterfaceContext,
) -> Callable[[str, object, object], Iterator[tuple[object, object, object, str, object]]]:
    """Create the generator that runs the finalize workflow for a family."""

    def finalize_sequence(
        family_key: str, log_value: object, durations_value: object
    ) -> Iterator[tuple[object, object, object, str, object]]:
        """Render the finalized video and stream status updates."""
        try:
            family = find_family(context.catalog, family_key)
        except ZoomyError as failure:
            message = f"**Error:** {failure}"
            log_entries, log_text = _append_log_entry(_as_string_list(log_value), message)
            yield (message, log_text, gr.update(), message, [])
            return
        log_entries = _as_string_list(log_value)
        durations = _as_durations(durations_value)
        request = FinalizeRequest(
            family=family,
            frame_count=context.repository.frame_count(family.sequence_key),
        )
        try:
            for update in finalize_video(
                request,
                context.environment,
            ):
                log_entries, log_text = _append_log_entry(log_entries, update.message)
                yield (
                    update.message,
                    log_text,
                    update.video_path if update.video_path is not None else gr.update(),
                    _fresh_statistics(context.repository, context.engine, family, durations),
                    _fresh_gallery(context.repository, family),
                )
        except ZoomyError as failure:
            message = f"**Error:** {failure}"
            log_entries, log_text = _append_log_entry(log_entries, message)
            yield (
                message,
                log_text,
                gr.update(),
                _fresh_statistics(context.repository, context.engine, family, durations),
                _fresh_gallery(context.repository, family),
            )

    return finalize_sequence


def _bind_clear_frames(
    context: InterfaceContext,
) -> Callable[..., tuple[object, object, object, object, str, object]]:
    """Create the handler that clears frames after a two-click confirmation."""

    def clear_frames(
        *values: object,
    ) -> tuple[object, object, object, object, str, object]:
        """Arm on the first click, delete the sequence directory on the second."""
        confirmation_armed = isinstance(values[0], bool) and values[0]
        try:
            family = find_family(context.catalog, str(values[1]))
        except ZoomyError as failure:
            message = f"**Error:** {failure}"
            return (
                False,
                gr.update(value=CLEAR_FRAMES_LABEL),
                message,
                None,
                message,
                [],
            )
        durations = _as_durations(values[2])
        if not confirmation_armed:
            return (
                True,
                gr.update(value=CONFIRM_CLEAR_FRAMES_LABEL),
                (
                    "Click **Confirm** again to permanently delete every frame "
                    f"of sequence *{family.sequence_key}*."
                ),
                gr.update(),
                _fresh_statistics(context.repository, context.engine, family, durations),
                _fresh_gallery(context.repository, family),
            )
        try:
            context.repository.clear_frames(family.sequence_key)
        except (OSError, ZoomyError) as failure:
            message = f"**Error:** Could not clear sequence *{family.sequence_key}*: {failure}"
            return (
                False,
                gr.update(value=CLEAR_FRAMES_LABEL),
                message,
                None,
                message,
                [],
            )
        return (
            False,
            gr.update(value=CLEAR_FRAMES_LABEL),
            f"Cleared all frames of sequence *{family.sequence_key}*.",
            None,
            _fresh_statistics(context.repository, context.engine, family, durations),
            _fresh_gallery(context.repository, family),
        )

    return clear_frames


def _loop_toggled(*values: object) -> str:
    """React to a loop checkbox toggle without queueing.

    Checking is a no-op here (the queued generator does the work); unchecking
    sets the stop flag immediately, which is what ends a running loop.
    """
    if isinstance(values[0], bool) and values[0]:
        return "Loop starting…"
    request_loop_stop()
    return "Stopping the loop after the current frame… (Interrupt now ends it immediately.)"


def _fresh_statistics(
    repository: FrameRepository,
    engine: EngineProtocol,
    family: FamilyDefinition,
    durations: tuple[int, float, float],
) -> str:
    """Recompute the statistics line with live system figures."""
    statistics = repository.sequence_statistics(family.sequence_key)
    return _render_statistics_line(statistics, _safe_system_statistics(engine), durations)


def _fresh_gallery(repository: FrameRepository, family: FamilyDefinition) -> list[str]:
    """Return the current gallery strip paths for one family."""
    statistics = repository.sequence_statistics(family.sequence_key)
    return [str(path) for path in statistics.recent_frame_paths]


def _draw_family_panel(wiring: FamilyPanelWiring, family_key: str) -> None:
    """Draw the LoRA, prompt, render, and loop controls for one family."""
    family = find_family(wiring.catalog, family_key)
    selected_loras = gr.CheckboxGroup(
        choices=[lora.display_name for lora in family.loras],
        value=[lora.display_name for lora in family.loras if lora.selected_by_default],
        label="LoRA styles",
    )
    strength_sliders = [
        gr.Slider(
            minimum=0.0,
            maximum=MAXIMUM_LORA_STRENGTH,
            step=LORA_STRENGTH_STEP,
            value=lora.default_strength,
            label=f"{lora.display_name} strength",
        )
        for lora in family.loras
    ]
    prompt_textbox = gr.Textbox(value=family.default_prompt, lines=6, max_lines=12, label="Prompt")
    has_negative_prompt = family.negative_prompt is not None
    negative_textbox: gr.Textbox | None = None
    if has_negative_prompt:
        negative_textbox = gr.Textbox(
            value=family.negative_prompt, lines=2, max_lines=4, label="Negative prompt"
        )
    else:
        gr.Markdown("_This family uses zeroed negative conditioning._")
    with gr.Row():
        render_button = gr.Button("Render next frame", variant="primary")
        loop_checkbox = gr.Checkbox(label="Loop frames (uncheck to stop after current frame)")
    # NOTE: no minimum= on the numbers below. A blank box submits 0 (which
    # means unlimited for the duration, or the default size for the frame
    # geometry), and Gradio validates minimum= in preprocess — before the
    # handler runs — so any minimum would reject blank inputs outright.
    duration_seconds = gr.Number(
        label="Target video duration, seconds (blank = run until stopped)", precision=1
    )
    finalize_checkbox = gr.Checkbox(label="Finalize video when the loop ends", value=True)
    with gr.Row():
        frame_width = gr.Number(value=FRAME_WIDTH_PIXELS, label="Frame width", precision=0)
        frame_height = gr.Number(value=FRAME_HEIGHT_PIXELS, label="Frame height", precision=0)
    render_inputs: list[Any] = [selected_loras, *strength_sliders, prompt_textbox]
    if negative_textbox is not None:
        render_inputs.append(negative_textbox)
    render_button.click(  # type: ignore[attr-defined]
        _create_render_handler(
            wiring=wiring, family=family, has_negative_prompt=has_negative_prompt
        ),
        inputs=[
            *render_inputs,
            wiring.log_state,
            wiring.durations_state,
            frame_width,
            frame_height,
        ],
        outputs=[
            wiring.status_markdown,
            wiring.log_textbox,
            wiring.preview_image,
            wiring.stats_line,
            wiring.gallery,
            wiring.durations_state,
        ],
    )
    # Both listeners below share the checkbox change event: the generator runs
    # queued and does the rendering, while the plain handler runs unqueued so
    # an uncheck stops the loop immediately even mid-frame. A programmatic
    # uncheck (loop end) fires no event, so no phantom loop can start.
    loop_checkbox.change(  # type: ignore[attr-defined]
        _create_loop_handler(wiring=wiring, family=family, has_negative_prompt=has_negative_prompt),
        inputs=[
            loop_checkbox,
            *render_inputs,
            duration_seconds,
            finalize_checkbox,
            frame_width,
            frame_height,
            wiring.log_state,
            wiring.durations_state,
        ],
        outputs=[
            wiring.status_markdown,
            wiring.log_textbox,
            wiring.preview_image,
            wiring.stats_line,
            wiring.gallery,
            wiring.durations_state,
            loop_checkbox,
            wiring.preview_video,
        ],
    )
    loop_checkbox.change(  # type: ignore[attr-defined]
        _loop_toggled,
        inputs=[loop_checkbox],
        outputs=[wiring.status_markdown],
        queue=False,
    )


def _create_render_handler(
    *,
    wiring: FamilyPanelWiring,
    family: FamilyDefinition,
    has_negative_prompt: bool,
) -> Callable[..., Iterator[tuple[str, str, object, str, object, tuple[int, float, float]]]]:
    """Build the generator Gradio calls for one Render-next-frame click."""

    def render_frame(
        *values: object,
    ) -> Iterator[tuple[str, str, object, str, object, tuple[int, float, float]]]:
        """Queue one frame render and stream status updates."""
        submission = _parse_panel_submission(
            family, values, has_negative_prompt=has_negative_prompt
        )
        rest = values[submission.consumed_count :]
        log_entries = _as_string_list(rest[0])
        durations = _as_durations(rest[1])
        frame_width = _coerce_frame_size(rest[2], FRAME_WIDTH_PIXELS)
        frame_height = _coerce_frame_size(rest[3], FRAME_HEIGHT_PIXELS)
        request = FrameRenderRequest(
            family=family,
            prompt=submission.prompt_text,
            negative_prompt=submission.negative_text,
            frame_count=wiring.repository.frame_count(family.sequence_key),
            lora_selections=submission.lora_selections,
            seed=_random_generator.randrange(MAXIMUM_SEED),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        try:
            for update in render_next_frame(request, wiring.environment):
                log_entries, log_text = _append_log_entry(log_entries, update.message)
                preview_value: object = gr.update()
                gallery_value: object = gr.update()
                if update.frame_path is not None and update.elapsed_seconds is not None:
                    durations = _record_frame_duration(durations, update.elapsed_seconds)
                    preview_value = update.frame_path
                    gallery_value = _fresh_gallery(wiring.repository, family)
                yield (
                    update.message,
                    log_text,
                    preview_value,
                    _fresh_statistics(wiring.repository, wiring.engine, family, durations),
                    gallery_value,
                    durations,
                )
        except ZoomyError as failure:
            message = f"**Error:** {failure}"
            _, log_text = _append_log_entry(log_entries, message)
            yield (
                message,
                log_text,
                gr.update(),
                _fresh_statistics(wiring.repository, wiring.engine, family, durations),
                _fresh_gallery(wiring.repository, family),
                durations,
            )

    return render_frame


def _create_loop_handler(
    *,
    wiring: FamilyPanelWiring,
    family: FamilyDefinition,
    has_negative_prompt: bool,
) -> Callable[
    ..., Iterator[tuple[str, str, object, str, object, tuple[int, float, float], object, object]]
]:
    """Build the generator Gradio calls for one Loop-frames change."""

    def loop_frames(
        *values: object,
    ) -> Iterator[tuple[str, str, object, str, object, tuple[int, float, float], object, object]]:
        """Render frames until the duration, a stop, or failed retries, then finalize."""
        submission = _parse_panel_submission(
            family, values[1:], has_negative_prompt=has_negative_prompt
        )
        rest = values[1 + submission.consumed_count :]
        target_seconds = _parse_target_seconds(rest[0])
        finalize_on_stop = rest[1] is True
        frame_width = _coerce_frame_size(rest[2], FRAME_WIDTH_PIXELS)
        frame_height = _coerce_frame_size(rest[3], FRAME_HEIGHT_PIXELS)
        log_entries = _as_string_list(rest[4])
        durations = _as_durations(rest[5])
        if not (isinstance(values[0], bool) and values[0]):
            message = "Loop is off — check the box to start rendering."
            _, log_text = _append_log_entry(log_entries, message)
            yield (
                message,
                log_text,
                gr.update(),
                _fresh_statistics(wiring.repository, wiring.engine, family, durations),
                _fresh_gallery(wiring.repository, family),
                durations,
                gr.update(),
                gr.update(),
            )
            return

        def request_factory(frame_count: int) -> FrameRenderRequest:
            """Build one loop frame request with a fresh seed."""
            return FrameRenderRequest(
                family=family,
                prompt=submission.prompt_text,
                negative_prompt=submission.negative_text,
                frame_count=frame_count,
                lora_selections=submission.lora_selections,
                seed=_random_generator.randrange(MAXIMUM_SEED),
                frame_width=frame_width,
                frame_height=frame_height,
            )

        options = VideoGenerationOptions(
            target_seconds=target_seconds, finalize_on_stop=finalize_on_stop
        )
        system = _safe_system_statistics(wiring.engine)
        if target_seconds is None:
            start_message = "Loop started — rendering until stopped…"
        else:
            start_message = f"Loop started — rendering a {target_seconds:.1f} s video…"
        last_message = start_message
        log_entries, log_text = _append_log_entry(log_entries, start_message)
        statistics = wiring.repository.sequence_statistics(family.sequence_key)
        yield (
            start_message,
            log_text,
            gr.update(),
            _render_statistics_line(statistics, system, durations),
            [str(path) for path in statistics.recent_frame_paths],
            durations,
            gr.update(),
            gr.update(),
        )
        try:
            for update in generate_video(family, request_factory, wiring.environment, options):
                last_message = update.message
                log_entries, log_text = _append_log_entry(log_entries, update.message)
                preview_value: object = gr.update()
                gallery_value: object = gr.update()
                video_value: object = gr.update()
                if update.frame_path is not None and update.elapsed_seconds is not None:
                    durations = _record_frame_duration(durations, update.elapsed_seconds)
                    preview_value = update.frame_path
                    gallery_value = _fresh_gallery(wiring.repository, family)
                if update.video_path is not None:
                    video_value = update.video_path
                yield (
                    update.message,
                    log_text,
                    preview_value,
                    _fresh_statistics(wiring.repository, wiring.engine, family, durations),
                    gallery_value,
                    durations,
                    gr.update(),
                    video_value,
                )
        except ZoomyError as failure:
            last_message = f"**Error:** {failure}"
            _, log_text = _append_log_entry(log_entries, last_message)
        final_statistics = wiring.repository.sequence_statistics(family.sequence_key)
        yield (
            last_message,
            log_text,
            gr.update(),
            _render_statistics_line(final_statistics, system, durations),
            [str(path) for path in final_statistics.recent_frame_paths],
            durations,
            gr.update(value=False),
            gr.update(),
        )

    return loop_frames


def _parse_panel_submission(
    family: FamilyDefinition, values: Sequence[object], *, has_negative_prompt: bool
) -> PanelSubmission:
    """Parse the checkbox, sliders, and prompt boxes of one panel submission."""
    selected_names = _selected_lora_names(values[0])
    lora_count = len(family.loras)
    strengths = [_coerce_to_float(value) for value in values[1 : 1 + lora_count]]
    prompt_text = str(values[1 + lora_count])
    negative_text: str | None = None
    consumed_count = 2 + lora_count
    if has_negative_prompt:
        negative_text = str(values[2 + lora_count])
        consumed_count = 3 + lora_count
    selections = tuple(
        (lora, strength)
        for (lora, strength) in zip(family.loras, strengths, strict=True)
        if lora.display_name in selected_names
    )
    return PanelSubmission(
        lora_selections=selections,
        prompt_text=prompt_text,
        negative_text=negative_text,
        consumed_count=consumed_count,
    )


def _parse_target_seconds(value: object) -> float | None:
    """Interpret the loop duration input; blank, zero, or negative is unlimited.

    Non-finite floats fall in the unlimited bucket too: arithmetic on
    infinity would size an unbounded frame loop instead of answering.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def _coerce_frame_size(value: object, default_pixels: int) -> int:
    """Interpret a frame geometry input, falling back to the default size.

    Blank or unusable boxes submit 0 or text; the engine's alignment check
    still refuses sizes like 511, so this only restores the default.
    """
    if isinstance(value, bool):
        return default_pixels
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, float) and math.isfinite(value) and value >= 1:
        return int(value)
    return default_pixels


def _initial_statistics(
    family: FamilyDefinition, repository: FrameRepository, engine: EngineProtocol
) -> str:
    """Render the statistics line for the initial page load."""
    statistics = repository.sequence_statistics(family.sequence_key)
    return _render_statistics_line(statistics, _safe_system_statistics(engine), (0, 0.0, 0.0))


def _safe_system_statistics(engine: EngineProtocol) -> EngineStatistics | None:
    """Fetch engine statistics, returning None when the engine is offline.

    ``EngineProtocol`` is structural, so any implementation may raise errors
    outside the ``ZoomyError`` hierarchy (torch ``RuntimeError``,
    ``OSError``, ...); all of them mean "no figures right now". Only
    ``Exception`` is caught — ``BaseException`` still propagates, so
    interrupts and cancellations are never swallowed.
    """
    try:
        return engine.engine_statistics()
    except Exception:  # noqa: BLE001
        return None


def _render_statistics_line(
    statistics: SequenceStatistics,
    system: EngineStatistics | None,
    durations: tuple[int, float, float],
) -> str:
    """Format the live statistics line for one sequence."""
    segments = [
        f"**{statistics.frame_count}** frames",
        f"**{_format_bytes(statistics.frames_bytes)}** on disk",
    ]
    rendered_frames, total_seconds, last_seconds = durations
    if rendered_frames > 0:
        average_seconds = total_seconds / rendered_frames
        segments.append(f"last frame {last_seconds:.0f} s | avg {average_seconds:.1f} s")
    if statistics.video_path is not None and statistics.video_bytes is not None:
        video_details = f"**{statistics.video_path.name}** ({_format_bytes(statistics.video_bytes)}"
        if statistics.video_modified_timestamp is not None:
            video_details += f", {_format_timestamp(statistics.video_modified_timestamp)}"
        video_details += ")"
        segments.append(f"video {video_details}")
    else:
        segments.append("no video yet")
    if system is None:
        segments.append("engine offline")
    else:
        segments.append(
            f"VRAM **{_gibibytes(system.video_memory_free_bytes)} / "
            f"{_gibibytes(system.video_memory_total_bytes)} GiB** | "
            f"RAM {_gibibytes(system.system_memory_free_bytes)} / "
            f"{_gibibytes(system.system_memory_total_bytes)} GiB"
        )
    return " | ".join(segments)


def _format_bytes(byte_count: int) -> str:
    """Format a byte count with binary units ("45.2 MiB", "512 B")."""
    size = float(byte_count)
    for unit in BYTE_UNITS:
        if size < BYTES_PER_UNIT or unit == BYTE_UNITS[-1]:
            if unit == "B":
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= BYTES_PER_UNIT
    return f"{size:.1f} {BYTE_UNITS[-1]}"


def _gibibytes(byte_count: int | None) -> str:
    """Format bytes as GiB with one decimal, or "n/a" when unknown."""
    if byte_count is None:
        return "n/a"
    return f"{byte_count / BYTES_PER_UNIT**3:.1f}"


def _format_timestamp(epoch_seconds: float) -> str:
    """Format epoch seconds as a short local HH:MM stamp."""
    return time.strftime("%H:%M", time.localtime(epoch_seconds))


def _append_log_entry(entries: list[str], message: str) -> tuple[list[str], str]:
    """Append a message, returning the trimmed history and its visible text."""
    updated_entries = [*entries, message][-MAXIMUM_LOG_LINES:]
    return updated_entries, "\n".join(updated_entries[-VISIBLE_LOG_LINES:])


def _as_string_list(value: object) -> list[str]:
    """Coerce a log State payload to a string list, defaulting to empty."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _as_durations(value: object) -> tuple[int, float, float]:
    """Coerce a durations State payload to (frames, total, last) seconds."""
    if (
        isinstance(value, tuple)
        and len(value) == DURATIONS_TUPLE_LENGTH
        and isinstance(value[0], int)
        and isinstance(value[1], (int, float))
        and isinstance(value[2], (int, float))
    ):
        return (value[0], float(value[1]), float(value[2]))
    return (0, 0.0, 0.0)


def _record_frame_duration(
    durations: tuple[int, float, float], elapsed_seconds: float
) -> tuple[int, float, float]:
    """Fold one completed frame time into the running session durations."""
    rendered_frames, total_seconds, _previous_last = durations
    return (rendered_frames + 1, total_seconds + elapsed_seconds, elapsed_seconds)


def _render_health_badge(*, is_reachable: bool) -> str:
    """Format the engine readiness badge as a colored status line."""
    if is_reachable:
        return '<span style="color: #16a34a; font-weight: bold;">Engine ready</span>'
    return '<span style="color: #dc2626; font-weight: bold;">Engine offline</span>'


def _selected_lora_names(value: object) -> set[str]:
    """Extract the checked LoRA display names from a CheckboxGroup payload.

    Only genuine strings are kept: anything else matches no LoRA, so it is
    dropped rather than stringified into a phantom selection.
    """
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _coerce_to_float(value: object) -> float:
    """Coerce a Gradio slider payload to a float, defaulting to zero.

    Booleans and non-finite numbers are payload garbage, not strengths, so
    they take the documented zero default like every other non-numeric
    value.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return float(value)

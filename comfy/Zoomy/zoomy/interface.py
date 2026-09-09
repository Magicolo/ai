"""Gradio web interface for the zoomy control panel.

Layout note: the per-family panel is drawn by ``@gr.render`` whenever the
family dropdown changes. That decorated function executes immediately at app
build time, so every component its event wiring references is created before
the render block in source order — hence the two wiring dataclasses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import gradio as gr

from zoomy.errors import ZoomyError
from zoomy.family_catalog import find_family
from zoomy.finalize_workflow import FinalizeRequest
from zoomy.frame_workflow import FrameRenderRequest
from zoomy.rendering import finalize_video, render_next_frame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Any

    from zoomy.comfy_connection import ConnectionProtocol
    from zoomy.family_catalog import FamilyDefinition
    from zoomy.frame_repository import FrameRepository
    from zoomy.settings import Settings

MAXIMUM_LORA_STRENGTH = 2.0
LORA_STRENGTH_STEP = 0.05
MAXIMUM_SEED = 2**48
STATUS_TIMER_SECONDS = 10.0
CLEAR_FRAMES_LABEL = "Clear frames"
CONFIRM_CLEAR_FRAMES_LABEL = "Confirm: clear all frames"

_random_generator = random.SystemRandom()


@dataclass(frozen=True, slots=True)
class FamilyPanelWiring:
    """Collaborators and outputs the per-family panel's render button targets."""

    settings: Settings
    connection: ConnectionProtocol
    catalog: Sequence[FamilyDefinition]
    repository: FrameRepository
    status_markdown: gr.Markdown
    preview_image: gr.Image
    frame_counter: gr.Markdown


@dataclass(frozen=True, slots=True)
class InterfaceContext:
    """Every collaborator and component the outer event wiring needs."""

    settings: Settings
    connection: ConnectionProtocol
    catalog: Sequence[FamilyDefinition]
    repository: FrameRepository
    family_dropdown: gr.Dropdown
    health_badge: gr.HTML
    frame_counter: gr.Markdown
    status_markdown: gr.Markdown
    confirmation_state: gr.State
    preview_image: gr.Image
    preview_video: gr.Video
    finalize_button: gr.Button
    interrupt_button: gr.Button
    refresh_button: gr.Button
    clear_frames_button: gr.Button


def build_application(
    settings: Settings,
    connection: ConnectionProtocol,
    catalog: Sequence[FamilyDefinition],
    repository: FrameRepository,
) -> gr.Blocks:
    """Assemble the whole zoomy control panel as an unlaunched Gradio app."""
    with gr.Blocks(title="Zoomy") as application:
        context = _create_components(settings, connection, catalog, repository)
        _wire_events(context)
    return cast("gr.Blocks", application)


def _create_components(
    settings: Settings,
    connection: ConnectionProtocol,
    catalog: Sequence[FamilyDefinition],
    repository: FrameRepository,
) -> InterfaceContext:
    """Create every component in layout order and bundle the wiring context."""
    gr.Markdown("# Zoomy — infinite zoom control panel")
    with gr.Row():
        with gr.Column(scale=3):
            family_dropdown = gr.Dropdown(
                choices=[(family.display_name, family.key) for family in catalog],
                value=catalog[0].key,
                label="Model family",
            )
        with gr.Column(scale=1):
            health_badge = gr.HTML(
                value=_render_health_badge(is_reachable=connection.is_reachable())
            )
        with gr.Column(scale=1):
            frame_counter = gr.Markdown(value=_render_frame_counter(repository, catalog[0]))
    status_markdown = gr.Markdown("Ready.")
    confirmation_state = gr.State(value=False)
    with gr.Row():
        with gr.Column(scale=3):
            preview_image = gr.Image(label="Latest frame", interactive=False, type="filepath")
            preview_video = gr.Video(label="Finalized video")
        with gr.Column(scale=2):
            panel_wiring = FamilyPanelWiring(
                settings=settings,
                connection=connection,
                catalog=catalog,
                repository=repository,
                status_markdown=status_markdown,
                preview_image=preview_image,
                frame_counter=frame_counter,
            )

            @gr.render(inputs=[family_dropdown])  # type: ignore[untyped-decorator]
            def family_panel(family_key: str) -> None:
                """Draw the per-family controls: LoRA styles, prompts, render button."""
                _draw_family_panel(panel_wiring, family_key)

    with gr.Row():
        finalize_button = gr.Button("Finalize video")
        interrupt_button = gr.Button("Interrupt running job")
        refresh_button = gr.Button("Refresh previews")
        clear_frames_button = gr.Button(CLEAR_FRAMES_LABEL)
    return InterfaceContext(
        settings=settings,
        connection=connection,
        catalog=catalog,
        repository=repository,
        family_dropdown=family_dropdown,
        health_badge=health_badge,
        frame_counter=frame_counter,
        status_markdown=status_markdown,
        confirmation_state=confirmation_state,
        preview_image=preview_image,
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
        context.frame_counter,
        context.confirmation_state,
        context.clear_frames_button,
    ]
    status_timer = gr.Timer(value=STATUS_TIMER_SECONDS)
    status_timer.tick(  # type: ignore[attr-defined]
        _bind_refresh_status(context),
        inputs=[context.family_dropdown],
        outputs=[context.health_badge, context.frame_counter],
        queue=False,
    )
    context.family_dropdown.change(  # type: ignore[attr-defined]
        _bind_refresh_previews(context),
        inputs=[context.family_dropdown],
        outputs=preview_outputs,
        queue=False,
    )
    context.finalize_button.click(  # type: ignore[attr-defined]
        _bind_finalize(context),
        inputs=[context.family_dropdown],
        outputs=[context.status_markdown, context.preview_video, context.frame_counter],
    )
    context.interrupt_button.click(  # type: ignore[attr-defined]
        _bind_interrupt(context),
        inputs=None,
        outputs=[context.status_markdown],
        queue=False,
    )
    context.refresh_button.click(  # type: ignore[attr-defined]
        _bind_refresh_previews(context),
        inputs=[context.family_dropdown],
        outputs=preview_outputs,
        queue=False,
    )
    context.clear_frames_button.click(  # type: ignore[attr-defined]
        _bind_clear_frames(context),
        inputs=[context.confirmation_state, context.family_dropdown],
        outputs=[
            context.confirmation_state,
            context.clear_frames_button,
            context.status_markdown,
            context.preview_image,
            context.frame_counter,
        ],
        queue=False,
    )


def _bind_refresh_status(
    context: InterfaceContext,
) -> Callable[[str], tuple[str, str]]:
    """Create the periodic handler that refreshes badge and frame counter."""

    def refresh_status(family_key: str) -> tuple[str, str]:
        """Poll ComfyUI reachability and reformat the frame counter."""
        family = find_family(context.catalog, family_key)
        return (
            _render_health_badge(is_reachable=context.connection.is_reachable()),
            _render_frame_counter(context.repository, family),
        )

    return refresh_status


def _bind_refresh_previews(
    context: InterfaceContext,
) -> Callable[[str], tuple[str, object, object, str, bool, object]]:
    """Create the handler that reloads the latest frame and video previews."""

    def refresh_previews(
        family_key: str,
    ) -> tuple[str, object, object, str, bool, object]:
        """Resolve the newest artifacts and disarm a pending clear-confirm."""
        family = find_family(context.catalog, family_key)
        latest_frame = context.repository.latest_frame_path(family.sequence_key)
        latest_video = context.repository.latest_video_path(family.sequence_key)
        return (
            f"Refreshed previews for sequence *{family.sequence_key}*.",
            latest_frame,
            latest_video,
            _render_frame_counter(context.repository, family),
            False,
            gr.update(value=CLEAR_FRAMES_LABEL),
        )

    return refresh_previews


def _bind_interrupt(context: InterfaceContext) -> Callable[[], str]:
    """Create the handler that asks ComfyUI to stop the running job."""

    def interrupt_running_job() -> str:
        """Forward an interrupt request to ComfyUI."""
        try:
            context.connection.interrupt()
        except ZoomyError as failure:
            return f"**Error:** {failure}"
        return "Interrupt requested; the running job will stop shortly."

    return interrupt_running_job


def _bind_finalize(
    context: InterfaceContext,
) -> Callable[[str], Iterator[tuple[object, object, str]]]:
    """Create the generator that runs the finalize workflow for a family."""

    def finalize_sequence(family_key: str) -> Iterator[tuple[object, object, str]]:
        """Render the finalized video and stream status updates."""
        family = find_family(context.catalog, family_key)
        request = FinalizeRequest(
            family=family,
            frame_count=context.repository.frame_count(family.sequence_key),
            output_directory=context.settings.output_directory,
        )
        try:
            for update in finalize_video(
                request,
                context.connection,
                context.repository,
                operation_timeout_seconds=context.settings.operation_timeout_seconds,
                poll_interval_seconds=context.settings.poll_interval_seconds,
            ):
                yield (
                    update.message,
                    update.video_path if update.video_path is not None else gr.update(),
                    _render_frame_counter(context.repository, family),
                )
        except ZoomyError as failure:
            yield (
                f"**Error:** {failure}",
                gr.update(),
                _render_frame_counter(context.repository, family),
            )

    return finalize_sequence


def _bind_clear_frames(
    context: InterfaceContext,
) -> Callable[..., tuple[object, object, object, object, str]]:
    """Create the handler that clears frames after a two-click confirmation."""

    def clear_frames(*values: object) -> tuple[object, object, object, object, str]:
        """Arm on the first click, delete the sequence directory on the second."""
        confirmation_armed = isinstance(values[0], bool) and values[0]
        family = find_family(context.catalog, str(values[1]))
        if not confirmation_armed:
            return (
                True,
                gr.update(value=CONFIRM_CLEAR_FRAMES_LABEL),
                (
                    "Click **Confirm** again to permanently delete every frame "
                    f"of sequence *{family.sequence_key}*."
                ),
                gr.update(),
                _render_frame_counter(context.repository, family),
            )
        context.repository.clear_frames(family.sequence_key)
        return (
            False,
            gr.update(value=CLEAR_FRAMES_LABEL),
            f"Cleared all frames of sequence *{family.sequence_key}*.",
            None,
            _render_frame_counter(context.repository, family),
        )

    return clear_frames


def _draw_family_panel(wiring: FamilyPanelWiring, family_key: str) -> None:
    """Draw the LoRA, prompt, and render controls for one family."""
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
    render_button = gr.Button("Render next frame", variant="primary")
    handler_inputs: list[Any] = [selected_loras, *strength_sliders, prompt_textbox]
    if negative_textbox is not None:
        handler_inputs.append(negative_textbox)
    render_button.click(  # type: ignore[attr-defined]
        _create_render_handler(
            wiring=wiring, family=family, has_negative_prompt=has_negative_prompt
        ),
        inputs=handler_inputs,
        outputs=[wiring.status_markdown, wiring.preview_image, wiring.frame_counter],
    )


def _create_render_handler(
    *,
    wiring: FamilyPanelWiring,
    family: FamilyDefinition,
    has_negative_prompt: bool,
) -> Callable[..., Iterator[tuple[object, object, str]]]:
    """Build the generator Gradio calls for one Render-next-frame click."""
    lora_count = len(family.loras)

    def render_frame(*values: object) -> Iterator[tuple[object, object, str]]:
        """Queue one frame render and stream status updates."""
        selected_names = _selected_lora_names(values[0])
        strengths = [_coerce_to_float(value) for value in values[1 : 1 + lora_count]]
        prompt_text = str(values[1 + lora_count])
        negative_text: str | None = None
        if has_negative_prompt:
            negative_text = str(values[2 + lora_count])
        lora_selections = tuple(
            (lora, strength)
            for (lora, strength) in zip(family.loras, strengths, strict=True)
            if lora.display_name in selected_names
        )
        request = FrameRenderRequest(
            family=family,
            prompt=prompt_text,
            negative_prompt=negative_text,
            frame_count=wiring.repository.frame_count(family.sequence_key),
            lora_selections=lora_selections,
            seed=_random_generator.randrange(MAXIMUM_SEED),
            output_directory=wiring.settings.output_directory,
        )
        try:
            for update in render_next_frame(
                request,
                wiring.connection,
                wiring.repository,
                operation_timeout_seconds=wiring.settings.operation_timeout_seconds,
                poll_interval_seconds=wiring.settings.poll_interval_seconds,
            ):
                yield (
                    update.message,
                    update.frame_path if update.frame_path is not None else gr.update(),
                    _render_frame_counter(wiring.repository, family),
                )
        except ZoomyError as failure:
            yield (
                f"**Error:** {failure}",
                gr.update(),
                _render_frame_counter(wiring.repository, family),
            )

    return render_frame


def _render_health_badge(*, is_reachable: bool) -> str:
    """Format the ComfyUI reachability badge as a colored status line."""
    if is_reachable:
        return '<span style="color: #16a34a; font-weight: bold;">ComfyUI online</span>'
    return '<span style="color: #dc2626; font-weight: bold;">ComfyUI unreachable</span>'


def _render_frame_counter(repository: FrameRepository, family: FamilyDefinition) -> str:
    """Format the frame-count line shown beside the family selector."""
    frame_count = repository.frame_count(family.sequence_key)
    return f"**{frame_count}** frames in *{family.sequence_key}*"


def _selected_lora_names(value: object) -> set[str]:
    """Extract the checked LoRA display names from a CheckboxGroup payload."""
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _coerce_to_float(value: object) -> float:
    """Coerce a Gradio slider payload to a float, defaulting to zero."""
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0

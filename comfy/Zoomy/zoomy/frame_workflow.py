"""Builder for the per-frame render workflow.

One run produces the next zoom frame: take the previous frame (or the
cold-start seed image), crop a border off every edge, rescale the crop back to
full size (the zoom dive, ~1.5% per frame at 1376x768), then run an img2img
pass that re-renders the enlarged center in the selected style.

The zoom crop geometry lives here as constants shared by every family:

    crop size = frame size - 2 * CROP_BORDER_PIXELS  (1356x748 at 1376x768)
    zoom factor per frame = 1376 / 1356 ≈ 1.0147
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zoomy.graph import ComfyWorkflow, NodeReference

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zoomy.family_catalog import FamilyDefinition, LoraDefinition

FRAME_WIDTH_PIXELS = 1376
FRAME_HEIGHT_PIXELS = 768
CROP_BORDER_PIXELS = 10
RESCALE_METHOD = "bicubic"
RESCALE_CROP_MODE = "disabled"
BASE_MODEL_WEIGHT_DTYPE = "default"


@dataclass(frozen=True, slots=True)
class FrameRenderRequest:
    """Everything the frame workflow builder needs for one frame.

    Attributes:
        family: The model family to render with.
        prompt: Positive prompt text for this frame.
        negative_prompt: Negative prompt text; only used when the family
            defines one (families without a negative use zeroed conditioning).
        frame_count: Frames already in the sequence; zero means cold start.
        lora_selections: Selected LoRA styles with strengths, applied in this
            order; the chain is model+clip through every entry.
        seed: Seed for the KSampler; the interface draws a fresh one per frame.
        output_directory: ComfyUI-side path of the output directory (the
            previous-frame loader and the save prefix derive paths from it).
    """

    family: FamilyDefinition
    prompt: str
    negative_prompt: str | None
    frame_count: int
    lora_selections: Sequence[tuple[LoraDefinition, float]]
    seed: int
    output_directory: str


def build_frame_workflow(request: FrameRenderRequest) -> dict[str, dict[str, Any]]:
    """Build the API-format workflow that renders one next frame.

    Cold start (``frame_count == 0``) loads the family's seed image from
    ComfyUI's input directory; otherwise the loader reads only the newest
    frame from the sequence directory.
    """
    workflow = ComfyWorkflow()
    source_image = _add_source_image(workflow, request)
    model, text_encoder = _add_model_loaders(workflow, request.family)
    model, clip = _apply_lora_chain(workflow, request.lora_selections, model, text_encoder)
    model = _apply_model_shift(workflow, request.family.model_shift, model)
    positive_conditioning = workflow.add(
        "encode_prompt",
        "CLIPTextEncode",
        {"clip": clip, "text": request.prompt},
    )
    negative_conditioning = _add_negative_conditioning(
        workflow,
        request.family.negative_prompt,
        request.negative_prompt,
        clip,
        positive_conditioning,
    )
    frame_latent = _add_zoom_chain(workflow, source_image)
    sampled_latent = workflow.add(
        "sample_next_frame",
        "KSampler",
        {
            "model": model,
            "positive": positive_conditioning,
            "negative": negative_conditioning,
            "latent_image": frame_latent,
            "seed": request.seed,
            "steps": request.family.sampler_steps,
            "cfg": request.family.classifier_free_guidance,
            "sampler_name": request.family.sampler_name,
            "scheduler": request.family.scheduler_name,
            "denoise": request.family.denoise_strength,
        },
    )
    decoded_frame = workflow.add(
        "decode_next_frame",
        "VAEDecode",
        {"samples": sampled_latent, "vae": _autoencoder_reference(workflow)},
    )
    workflow.add(
        "save_next_frame",
        "SaveImage",
        {
            "filename_prefix": f"Zoomy/{request.family.sequence_key}/frame",
            "images": decoded_frame,
        },
    )
    return workflow.build()


def _add_source_image(workflow: ComfyWorkflow, request: FrameRenderRequest) -> NodeReference:
    """Load the previous frame, or the cold-start seed image when empty."""
    if request.frame_count == 0:
        return workflow.add(
            "load_input_image",
            "LoadImage",
            {"image": request.family.cold_start_image},
        )
    return workflow.add(
        "load_previous_frame",
        "VHS_LoadImagesPath",
        {
            "directory": f"{request.output_directory}/Zoomy/{request.family.sequence_key}",
            "image_load_cap": 1,
            "skip_first_images": request.frame_count - 1,
            "select_every_nth": 1,
        },
    )


def _add_model_loaders(
    workflow: ComfyWorkflow, family: FamilyDefinition
) -> tuple[NodeReference, NodeReference]:
    """Load the base diffusion model, text encoder, and autoencoder."""
    base_model = workflow.add(
        "load_base_model",
        "UNETLoader",
        {"unet_name": family.base_model_file, "weight_dtype": BASE_MODEL_WEIGHT_DTYPE},
    )
    text_encoder = workflow.add(
        "load_text_encoder",
        "CLIPLoader",
        {"clip_name": family.text_encoder_file, "type": family.text_encoder_type},
    )
    workflow.add("load_autoencoder", "VAELoader", {"vae_name": family.autoencoder_file})
    return base_model, text_encoder


def _apply_lora_chain(
    workflow: ComfyWorkflow,
    lora_selections: Sequence[tuple[LoraDefinition, float]],
    model: NodeReference,
    clip: NodeReference,
) -> tuple[NodeReference, NodeReference]:
    """Chain every selected LoRA onto the model and text encoder.

    LoraLoader outputs ``(MODEL, CLIP)``, so the model reference stays on slot
    zero while the clip reference moves to slot one of each applied node. With
    no selections the originals pass through untouched.
    """
    for index, (lora, strength) in enumerate(lora_selections):
        applied = workflow.add(
            f"apply_lora_{index}",
            "LoraLoader",
            {
                "model": model,
                "clip": clip,
                "lora_name": lora.file_name,
                "strength_model": strength,
                "strength_clip": strength,
            },
        )
        model = applied
        clip = NodeReference(key=applied.key, output_slot=1)
    return model, clip


def _apply_model_shift(
    workflow: ComfyWorkflow, model_shift: float | None, model: NodeReference
) -> NodeReference:
    """Apply the sampling shift when the family defines one, else pass through."""
    if model_shift is None:
        return model
    return workflow.add(
        "apply_model_shift",
        "ModelSamplingAuraFlow",
        {"model": model, "shift": model_shift},
    )


def _add_negative_conditioning(
    workflow: ComfyWorkflow,
    family_negative_prompt: str | None,
    request_negative_prompt: str | None,
    clip: NodeReference,
    positive_conditioning: NodeReference,
) -> NodeReference:
    """Encode the negative prompt, or zero out conditioning for families without one."""
    if family_negative_prompt is None:
        return workflow.add(
            "zero_negative_prompt",
            "ConditioningZeroOut",
            {"conditioning": positive_conditioning},
        )
    return workflow.add(
        "encode_negative_prompt",
        "CLIPTextEncode",
        {"clip": clip, "text": request_negative_prompt or ""},
    )


def _add_zoom_chain(workflow: ComfyWorkflow, source_image: NodeReference) -> NodeReference:
    """Crop the border, rescale back to full size, and encode the pixels."""
    cropped_frame = workflow.add(
        "crop_previous_frame",
        "ImageCrop",
        {
            "image": source_image,
            "width": FRAME_WIDTH_PIXELS - 2 * CROP_BORDER_PIXELS,
            "height": FRAME_HEIGHT_PIXELS - 2 * CROP_BORDER_PIXELS,
            "x": CROP_BORDER_PIXELS,
            "y": CROP_BORDER_PIXELS,
        },
    )
    rescaled_frame = workflow.add(
        "rescale_cropped_frame",
        "ImageScale",
        {
            "image": cropped_frame,
            "upscale_method": RESCALE_METHOD,
            "width": FRAME_WIDTH_PIXELS,
            "height": FRAME_HEIGHT_PIXELS,
            "crop": RESCALE_CROP_MODE,
        },
    )
    return workflow.add(
        "encode_frame_pixels",
        "VAEEncode",
        {"pixels": rescaled_frame, "vae": _autoencoder_reference(workflow)},
    )


def _autoencoder_reference(workflow: ComfyWorkflow) -> NodeReference:
    """Return the reference to the already-added autoencoder loader node.

    A tiny indirection that keeps the node key in one place; the loader is
    always added by :func:`_add_model_loaders` before this is used.
    """
    del workflow
    return NodeReference(key="load_autoencoder")

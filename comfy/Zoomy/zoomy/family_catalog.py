"""Catalog of model families with every rendering and audio parameter.

Each :class:`FamilyDefinition` carries everything the frame and finalize
workflow builders need: model file names, sampler settings, prompts, LoRA
styles, and the cold-start seed image. Adding a family is additive — append
one entry to :data:`FAMILY_CATALOG` and the interface picks it up.

Families that share a ``sequence_key`` write into the same frame directory and
video namespace, so they continue one zoom sequence (that is why both
Juggernaut Z speed variants share ``z_image``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zoomy.errors import ZoomyError

if TYPE_CHECKING:
    from collections.abc import Sequence

ERNIE_C64_PROMPT = (
    "c64style pixel art, Commodore 64 16-color palette, chunky blocky pixels, "
    "heavy crosshatch dithering, CRT scanlines, phosphor glow, wildly psychedelic "
    "experimental infinite zoom dive: a cathedral of melting neon oscilloscope "
    "waves, kaleidoscopic mushrooms with blinking eyeball caps, a holographic "
    "jellyfish marching band playing glowing tubas, chrome skeleton astronauts "
    "riding flaming unicycles, fractal peacocks with recursive tail-eyes, rivers "
    "of liquid chrome mercury, giant floating CRT monitors each showing a smaller "
    "copy of this same scene, candy-striped tentacles juggling disco-ball suns, "
    "lava-lamp volcanoes erupting rainbow static, checkerboard skies glitching "
    "into sprites, pixel dragons built from Tetris blocks, cassette tapes raining "
    "upward, electric aurora curtains, melting clocks dripping over neon "
    "pyramids, infinite recursion, ultra detailed, vibrant, crazy, weird, "
    "hypnotic, 8-bit demoscene energy"
)

ERNIE_MUSIC_PROMPT = (
    "Modern experimental electroacoustic composition: granular-synthesized glass "
    "and ice textures, bowed cymbals and prepared-piano resonances, modular "
    "feedback drones, glitch percussion from bitcrushed sine bursts, ghostly tape "
    "loops, deep sub-bass swells, spectral-freeze vocal wisps without words, "
    "sparse metallic pings in vast reverb, meditative, spacious, seamless loop "
    "feel"
)

ERNIE_SOUND_EFFECT_PROMPT = (
    "experimental electroacoustic soundscape locked to a continuous inward dive: "
    "bowed-metal groans bending downward, granular glass chimes scattering past, "
    "glitch clicks and voltage crackles, tape-rewind whistles accelerating, deep "
    "sub drops, resonant sine pings swooshing inward, airy filtered-noise risers, "
    "ceaseless smooth acceleration"
)

Z_IMAGE_POSITIVE_PROMPT = (
    "A breathtaking endless forward dive through a living psychedelic cathedral "
    "of impossible architecture. Towering arches of blooming stained glass and "
    "polished chrome curve overhead while rivers of liquid rainbow light flow "
    "ever inward past the viewer. Gentle jellyfish with glowing carnival "
    "lanterns drift by alongside clockwork birds with peacock feathers of molten "
    "gold, and smiling mushroom houses with warm lit windows spiral into the "
    "glowing distance. Every surface teems with intricate detail: fractal "
    "carvings, floating candles, twisted vines of neon flowers, shimmering "
    "portals of light. The perspective rushes continuously toward a radiant "
    "vanishing point, dreamlike, vibrant, wildly imaginative, ultra-detailed "
    "fantasy illustration."
)

Z_IMAGE_NEGATIVE_PROMPT = (
    "text, writing, subtitle, watermark, logo, blurry, low quality, jpeg "
    "artifacts, grainy, bad anatomy, deformed hands, facial distortion, "
    "oversaturated, visual artifacts, quality degradation"
)

Z_IMAGE_MUSIC_PROMPT = (
    "Experimental electroacoustic journey: bowed glass harmonics, granular "
    "string textures, deep sub-bass swells, spectral choir pads, prepared piano "
    "motifs, shimmering metallic percussion, slow-evolving drone layers, "
    "delicate music-box arpeggios, airy flute phrases over warm analog pads, "
    "cinematic and dreamlike"
)

Z_IMAGE_SOUND_EFFECT_PROMPT = (
    "Continuous forward-motion soundscape: deep airy whooshes rushing past the "
    "listener, low resonant rumble swelling and receding, sparkling chimes, "
    "soft wind currents, distant bell tones and shimmering transitions "
    "matching a fast inward dive"
)

SOUND_EFFECT_NEGATIVE_PROMPT = "speech, voice, vocals, singing, music, melody, drums, beat"


@dataclass(frozen=True, slots=True)
class LoraDefinition:
    """One selectable LoRA style attachment."""

    file_name: str
    display_name: str
    default_strength: float
    selected_by_default: bool


@dataclass(frozen=True, slots=True)
class FamilyDefinition:
    """One model family plus every parameter its render and finalize runs need.

    Attributes:
        key: Stable identifier used in environment-like contexts (dropdown
            values, logs).
        display_name: Human-friendly label shown in the interface.
        sequence_key: Names the frame directory under the output directory
            and the video prefix ``<sequence_key>``; families sharing this value
            contribute to one zoom sequence.
        base_model_file: Diffusion model file for ``UNETLoader``.
        text_encoder_file: Text encoder file for ``CLIPLoader``.
        text_encoder_type: Loader type string (``flux2``, ``lumina2``).
        autoencoder_file: Autoencoder file for ``VAELoader``.
        model_shift: Shift for ``ModelSamplingAuraFlow``, or ``None`` to skip
            that node entirely (Ernie does not use it).
        sampler_name: ``KSampler`` sampler name.
        scheduler_name: ``KSampler`` scheduler name.
        sampler_steps: ``KSampler`` step count.
        classifier_free_guidance: ``KSampler`` cfg value.
        denoise_strength: ``KSampler`` denoise for the img2img restore pass.
        default_prompt: Prefilled positive prompt text.
        negative_prompt: Prefilled negative prompt text; ``None`` means the
            family uses zeroed negative conditioning instead.
        cold_start_image: Input image file name used when the sequence has no
            frames yet (must live in ComfyUI's input directory).
        music_prompt: ACE-Step tags for the finalize music bed.
        sound_effect_prompt: MMAudio prompt for video-synced sound effects.
        sound_effect_negative_prompt: MMAudio negative prompt.
        loras: LoRA styles offered for this family, applied in selection order.
    """

    key: str
    display_name: str
    sequence_key: str
    base_model_file: str
    text_encoder_file: str
    text_encoder_type: str
    autoencoder_file: str
    model_shift: float | None
    sampler_name: str
    scheduler_name: str
    sampler_steps: int
    classifier_free_guidance: float
    denoise_strength: float
    default_prompt: str
    negative_prompt: str | None
    cold_start_image: str
    music_prompt: str
    sound_effect_prompt: str
    sound_effect_negative_prompt: str
    loras: tuple[LoraDefinition, ...]


FAMILY_CATALOG: tuple[FamilyDefinition, ...] = (
    FamilyDefinition(
        key="ernie_turbo",
        display_name="Ernie Image Turbo",
        sequence_key="ernie_turbo",
        base_model_file="ernie-image-turbo.safetensors",
        text_encoder_file="ministral-3-3b.safetensors",
        text_encoder_type="flux2",
        autoencoder_file="flux2-vae.safetensors",
        model_shift=None,
        sampler_name="euler",
        scheduler_name="simple",
        sampler_steps=8,
        classifier_free_guidance=1.0,
        denoise_strength=0.60,
        default_prompt=ERNIE_C64_PROMPT,
        negative_prompt=None,
        cold_start_image="ernie_zoom_seed.png",
        music_prompt=ERNIE_MUSIC_PROMPT,
        sound_effect_prompt=ERNIE_SOUND_EFFECT_PROMPT,
        sound_effect_negative_prompt=SOUND_EFFECT_NEGATIVE_PROMPT,
        loras=(
            LoraDefinition(
                file_name="c64style_ernie.safetensors",
                display_name="Ernie C64",
                default_strength=1.0,
                selected_by_default=True,
            ),
        ),
    ),
    FamilyDefinition(
        key="z_fast",
        display_name="Juggernaut Z (Fast)",
        sequence_key="z_image",
        base_model_file="juggernautZ_v10FastBy.safetensors",
        text_encoder_file="qwen_3_4b_fp8_mixed.safetensors",
        text_encoder_type="lumina2",
        autoencoder_file="ae.safetensors",
        model_shift=3.0,
        sampler_name="ddim",
        scheduler_name="normal",
        sampler_steps=6,
        classifier_free_guidance=1.0,
        denoise_strength=0.60,
        default_prompt=Z_IMAGE_POSITIVE_PROMPT,
        negative_prompt=Z_IMAGE_NEGATIVE_PROMPT,
        cold_start_image="z_zoom_seed.png",
        music_prompt=Z_IMAGE_MUSIC_PROMPT,
        sound_effect_prompt=Z_IMAGE_SOUND_EFFECT_PROMPT,
        sound_effect_negative_prompt=SOUND_EFFECT_NEGATIVE_PROMPT,
        loras=(
            LoraDefinition(
                file_name="Chalkboard01-1_CE_ZIMG_AIT4k.safetensors",
                display_name="Chalkboard",
                default_strength=0.85,
                selected_by_default=False,
            ),
            LoraDefinition(
                file_name="ClayArt01a_CE_ZIMG_AIT3k.safetensors",
                display_name="Clay Art",
                default_strength=0.85,
                selected_by_default=False,
            ),
        ),
    ),
    FamilyDefinition(
        key="z_quality",
        display_name="Juggernaut Z (Quality)",
        sequence_key="z_image",
        base_model_file="juggernautZ_v10ByRundiffusion.safetensors",
        text_encoder_file="qwen_3_4b_fp8_mixed.safetensors",
        text_encoder_type="lumina2",
        autoencoder_file="ae.safetensors",
        model_shift=3.0,
        sampler_name="res_multistep",
        scheduler_name="beta",
        sampler_steps=22,
        classifier_free_guidance=4.0,
        denoise_strength=0.60,
        default_prompt=Z_IMAGE_POSITIVE_PROMPT,
        negative_prompt=Z_IMAGE_NEGATIVE_PROMPT,
        cold_start_image="z_zoom_seed.png",
        music_prompt=Z_IMAGE_MUSIC_PROMPT,
        sound_effect_prompt=Z_IMAGE_SOUND_EFFECT_PROMPT,
        sound_effect_negative_prompt=SOUND_EFFECT_NEGATIVE_PROMPT,
        loras=(
            LoraDefinition(
                file_name="Chalkboard01-1_CE_ZIMG_AIT4k.safetensors",
                display_name="Chalkboard",
                default_strength=0.85,
                selected_by_default=False,
            ),
            LoraDefinition(
                file_name="ClayArt01a_CE_ZIMG_AIT3k.safetensors",
                display_name="Clay Art",
                default_strength=0.85,
                selected_by_default=False,
            ),
        ),
    ),
)


def find_family(catalog: Sequence[FamilyDefinition], family_key: str) -> FamilyDefinition:
    """Return the family with the given key.

    Raises:
        ZoomyError: If no catalog entry matches ``family_key``.
    """
    for family in catalog:
        if family.key == family_key:
            return family
    available = ", ".join(family.key for family in catalog)
    raise ZoomyError(f"Unknown model family {family_key!r}; available families: {available}")

import sys, os, torch, gradio
from PIL import Image
from streamdiffusion.image_utils import postprocess_image
sys.path.append("/stream")

from utils.wrapper import StreamDiffusionWrapper

def generate():
    stream = StreamDiffusionWrapper(
        mode="txt2img",
        width=512,
        height=512,
        model_id_or_path="SimianLuo/LCM_Dreamshaper_v7",
        lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
        vae_id="madebyollin/taesd",
        acceleration="xformers",
        t_index_list=[0, 16, 32, 45],
        warmup=10,
        use_safety_checker=False,
        cfg_type="none",
    )

    prompt = "(oil painting, ultra detailed, surreal, conceptual, abstract, hyper realistic) Lighthouse in a melting spinning hurricane of thunder elementals."
    stream.prepare(prompt, "worst quality, blurry, simple", num_inference_steps=50, guidance_scale=1.25)
    while True: yield stream(prompt)

ui = gradio.Interface(fn=generate, inputs=None, outputs="image", title="T2I")
ui.launch(server_name="0.0.0.0")
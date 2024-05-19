import gradio, core, numpy
from PIL import Image

def run(model, positive, negative, sampler, scheduler, width, height, steps, guide, denoise, zoom):
    state = {
        "width": width,
        "height": height,
        "zoom": zoom,
        "steps": steps,
        "guide": guide,
        "denoise": denoise,
        "model": model,
        "positive": positive,
        "negative": negative,
        "sampler": sampler,
        "scheduler": scheduler,
    }

    for images in core.run(state):
        yield [
            Image.fromarray(numpy.clip(255.0 * image.cpu().numpy(), 0, 255).astype(numpy.uint8)) 
            for image in images
        ]

inputs = [
    gradio.Dropdown(choices = ["dreamshaperXL_v21TurboDPMSDE.safetensors"], value = "dreamshaperXL_v21TurboDPMSDE.safetensors", label = "Model"),
    gradio.Textbox(value = "(ultra detailed, oil painting, abstract, conceptual, hyper realistic, vibrant) Everything is a 'TCHOO TCHOO' train. Flesh organic locomotive speeding on vast empty nebula tracks. Eternal spiral railways in the cosmos. Coal ember engine of intricate fusion. Unholy desecrated church station. Runic glyphs neon 'TCHOO' engravings. Darkness engulfed black hole pentagram. Blood magic eldritch rituals to summon whimsy hellish trains of wonder. Everything is a 'TCHOO TCHOO' train.", label = "Positive"),
    gradio.Textbox(value = "(blurry, worst quality, low detail, monochrome, simple, centered)", label = "Negative"),
    gradio.Dropdown(choices = ["euler", "lcm"], value = "euler", label = "Sampler"),
    gradio.Dropdown(choices = ["normal", "karras", "sgm_uniform", "simple", "ddim_uniform"], value = "ddim_uniform", label = "Scheduler"),
    gradio.Slider(minimum = 256, maximum = 4096, value = 800, step = 8, label = "Width"),
    gradio.Slider(minimum = 256, maximum = 4096, value = 600, step = 8, label = "Height"),
    gradio.Slider(minimum = 1, maximum = 50, value = 6, step = 1, label = "Steps"),
    gradio.Slider(minimum = 1, maximum = 10, value = 3, step = 0.01, label = "Guide"),
    gradio.Slider(minimum = 0.0, maximum = 1.0, value = 0.6, step = 0.01, label = "Denoise"),
    gradio.Slider(minimum = 1, maximum = 256, value = 24, step = 1, label = "Zoom"),
]
output = gradio.Gallery(label="Output", height=1024, columns=4)
ui = gradio.Interface(fn=run, inputs=inputs, outputs=output, title="Comfy", allow_flagging="never")
ui.launch(server_name="0.0.0.0")
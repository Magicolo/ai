import os, random, sys, torch, time, asyncio, queue, threading
sys.path.append("/comfy")
import execution, server
from nodes import (
    ImageBatch,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    ImageScale,
    EmptyLatentImage,
    KSampler,
    NODE_CLASS_MAPPINGS,
    VAEEncode,
    ControlNetLoader,
    ControlNetApplyAdvanced,
    VAEDecode,
    SaveImage,
    init_custom_nodes
)

def seed(): return random.randint(1, 2**64)

def generate(state, send):
    initialize()
    with torch.inference_mode():
        (model, clip, vae) = CheckpointLoaderSimple().load_checkpoint(ckpt_name=state['model'])
        (positive,) = CLIPTextEncode().encode(text=state['positive'], clip=clip)
        (negative,) = CLIPTextEncode().encode(text=state['negative'], clip=clip)
        (latent,) = EmptyLatentImage().generate(width=state['width'], height=state['height'], batch_size=1)
        sampler = KSampler()
        encoder = VAEEncode()
        decoder = VAEDecode()
        cropper = NODE_CLASS_MAPPINGS["ImageCrop"]()
        scaler = ImageScale()
        batcher = ImageBatch()
        old = None

        while True:
            try:
                now = time.time()
                if old is None:
                    (sampled,) = sampler.sample(seed=seed(), steps=state['steps'], cfg=state['guide'], sampler_name=state['sampler'], scheduler=state['scheduler'], denoise=1.0, model=model, positive=positive, negative=negative, latent_image=latent)
                    (decoded,) = decoder.decode(samples=sampled, vae=vae)
                    old = decoded
                
                (cropped,) = cropper.crop(width=state['width'] - state['zoom'] * 2, height=state['height'] - state['zoom'], x=state['zoom'] * 2, y=state['zoom'], image=old)
                (scaled,) = scaler.upscale(upscale_method="nearest-exact", width=state['width'], height=state['height'], crop="disabled", image=cropped)
                (encoded,) = encoder.encode(pixels=scaled, vae=vae)
                (sampled,) = sampler.sample(seed=seed(), steps=state['steps'], cfg=state['guide'], sampler_name=state['sampler'], scheduler=state['scheduler'], denoise=state['denoise'], model=model, positive=positive, negative=negative, latent_image=encoded)
                (decoded,) = decoder.decode(samples=sampled, vae=vae)
                (batched,) = batcher.batch(old, decoded)
                send.put(batched, block=False)
                old = decoded
                print(f"Generate: {time.time() - now}")
            except Exception as error: print(f"Generate Error: {error}")

def interpolate(state, receive, send):
    initialize()
    with torch.inference_mode():
        # film = NODE_CLASS_MAPPINGS["FILM VFI"]()
        rife = NODE_CLASS_MAPPINGS["RIFE VFI"]()

        while True:
            try:
                now = time.time()
                batched = receive.get(block=True)
                size = receive.qsize()
                # (interpolated,) = film.vfi(ckpt_name="film_net_fp32.pt", frames=batched, clear_cache_after_n_frames=100, multiplier=4)
                (interpolated,) = rife.vfi(ckpt_name="rife49.pth", frames=batched, clear_cache_after_n_frames=100, multiplier=max(18 - size, 2), fast_mode=True, ensemble=True, scale_factor=4.0)
                print(f"Interpolate: {time.time() - now}")
                send.put(interpolated, block=False)
            except Exception as error: print(f"Interpolate Error: {error}")

def initialize():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    instance = server.PromptServer(loop)
    execution.PromptQueue(instance)
    init_custom_nodes()

def run(state):
    initialize()
    saver = SaveImage()
    bridge = queue.SimpleQueue()
    output = queue.SimpleQueue()
    threading.Thread(target=generate, args=(state, bridge)).start()
    threading.Thread(target=interpolate, args=(state, bridge, output)).start()

    while True:
        try:
            now = time.time()
            images = output.get()[:-1]
            saver.save_images(images, filename_prefix="Train")
            print(f"Root: {time.time() - now}")
            yield images
        except Exception as error: print(f"Root Error: {error}")
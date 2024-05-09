import torch, cv2, numpy, gradio, threading, time
from PIL import Image
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

frame = None

def stream_camera(camera):
    global frame

    count = 0
    while True:
        count += 1
        print(f"Read frame '{count}' from camera.")
        success, read = camera.read()
        if success:
            print(f"Success reading frame '{count}'.")
            frame = read
        else:
            print(f"Could not read frame '{count}'.")
            time.sleep(0.1)

def generate():
    global frame

    prompt = "A curious firefighter walking in a rainy hellscape."
    steps = 50
    width = 512
    height = 512

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(device=device, dtype=torch.float16)
    stream = StreamDiffusion(pipe, t_index_list=[32, 45], torch_dtype=pipe.dtype, do_add_noise=False)
    stream.load_lcm_lora()
    stream.fuse_lora()
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    stream.prepare(prompt = prompt, num_inference_steps=steps, guidance_scale=0)
    stream.enable_similar_image_filter()

    print("Start video capture.")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    if camera.isOpened():
        print("Camera stream opened.")
    else:
        print("Error: Could not open camera stream.")
        exit()

    print("Read from camera.")
    threading.Thread(target=stream_camera, args=(camera,)).start()

    while frame is None:
        print("Waiting for first frame.")
        time.sleep(0.25)

    print("Start image generation.")
    while True:
        cutout = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        print(f"Generate image from frame.")
        output = stream(cutout)
        print(f"Post process image.")
        image = postprocess_image(output, output_type="pil")[0]
        image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        yield image

    camera.release()
    cv2.destroyAllWindows()

ui = gradio.Interface(fn=generate, inputs=None, outputs="image", title="Camera")
ui.launch(server_name="0.0.0.0")
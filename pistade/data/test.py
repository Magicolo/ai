import base64, argparse
import ollama, bark, scipy
import resemble_enhance, resemble_enhance.enhancer, resemble_enhance.enhancer.inference

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="/data/me.jpg", help="Path to an image.")
parser.add_argument(
    "--prompt",
    default="Write a mean joke about the appearance of the person in the image with lots of emojis and styling text. Answer with a single sentence.",
    help="Prompt for the LLM.",
)
parser.add_argument("--model", default="llava", help="Model.")
parser.add_argument("--host", default="http://ollama:11434", help="Host.")
arguments = parser.parse_args()

bark.preload_models()
client = ollama.Client(host=arguments.host)
enhance = resemble_enhance.enhancer.inference.load_enhancer(None, "cuda")

with open(arguments.image, "rb") as file:
    image = file.read()
    image = base64.b64encode(image).decode("utf-8")

while True:
    prompt = arguments.prompt
    result = client.generate(arguments.model, prompt=prompt, images=[image])
    text = result["response"]
    print(text)
    _, raw = bark.generate_audio(text, silent=True, output_full=True)
    # process = resemble_enhance.inference.inference(enhance, raw, bark.SAMPLE_RATE, "cuda")
    scipy.io.wavfile.write("/dump/raw.wav", rate=bark.SAMPLE_RATE, data=raw)
    # scipy.io.wavfile.write("/dump/process.wav", rate=bark.SAMPLE_RATE, data=process)
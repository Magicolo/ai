import os, scipy, torch, hashlib, shutil, time
from bark import SAMPLE_RATE, generate_audio, preload_models

preload_models()
cache = "/cache"
os.mkdir(cache)

while True:
    try:
        with open('/data/transcribe.txt', 'r') as file:
            text = file.read()

        hash = hashlib.sha1(text.encode('utf-8')).hexdigest()
        path = f"{cache}/{hash}"
        expire = time.time() - 3600

        if not os.path.exists(path) or os.path.getctime(path) < expire:
            audio = generate_audio(text)
            scipy.io.wavfile.write(path, rate=SAMPLE_RATE, data=audio)
            print("Generate: ", text)

        shutil.copy(path, "/data/generate.wav")
    except Exception as exception:
        print("Error: ", exception)
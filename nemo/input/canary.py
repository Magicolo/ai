import sounddevice
import numpy
import wave
import os
from nemo.collections.asr.models import EncDecMultiTaskModel
from contextlib import redirect_stdout, redirect_stderr

channels = 1
sample_rate = 16000
duration = 10

def mute(scope):
    with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        return scope()

data = None
model = mute(lambda: EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b"))
samples = sample_rate * channels * duration

def callback(input, frames, time, status):
    global data, samples

    if data is None:
        data = input
    else:
        skip = max(len(data) + len(input) - samples, 0)
        trim = data if skip == 0 else data[skip:]
        data = numpy.append(trim, input)

with sounddevice.InputStream(samplerate=sample_rate, channels=channels, dtype=numpy.int16, callback=callback):
    while data is None:
        sounddevice.sleep(1)

    while True:
        try:
            with wave.open("/input/record.wav", 'wb') as file:
                file.setsampwidth(2)
                file.setnchannels(channels)
                file.setframerate(sample_rate)
                file.writeframes(data.tobytes())
            text = mute(lambda: model.transcribe("/input/manifest.json"))
            print(text)
        except Exception as exception:
            print("ERROR: ", exception)

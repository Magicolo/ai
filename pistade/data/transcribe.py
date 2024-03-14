import os, wave, numpy, sounddevice
from nemo.collections.asr.models import EncDecMultiTaskModel
from contextlib import redirect_stdout, redirect_stderr

channels = 1
sample_rate = 16000
duration = 10
wait = duration / 5
data = None
model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
samples = sample_rate * channels * duration

def callback(input, frames, time, status):
    global data, samples

    if data is None:
        data = input
    else:
        skip = max(len(data) + len(input) - samples, 0)
        trim = data if skip == 0 else data[skip:]
        data = numpy.append(trim, input)

with sounddevice.InputStream(samplerate=sample_rate, channels=channels, dtype=numpy.int16, callback=callback) as stream:
    while data is None:
        sounddevice.sleep(1)

    while True:
        time = stream.time
        try:
            with wave.open("/data/record.wav", 'wb') as file:
                file.setsampwidth(2)
                file.setnchannels(channels)
                file.setframerate(sample_rate)
                file.writeframes(data.tobytes())
            text = model.transcribe("/data/manifest.json")
            with open('/data/transcribe.txt', 'w') as file:
                for line in text:
                    file.write(line)
            print("Transcribe: ", text)
            elapsed = stream.time - time
        except Exception as exception:
            print("Error: ", exception)
        sounddevice.sleep(int(max(0, wait - elapsed) * 1000))

import whisper
import sounddevice
import numpy
import wave

def record(filename, duration=5, sample_rate=44100):
    audio_data = sounddevice.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=2,
        dtype=numpy.int16,
    )
    sounddevice.wait()

    # Save audio to WAV file
    with wave.open(filename, "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(audio_data.tobytes())

model = whisper.load_model("large-v3")
while True:
    # record("./test.wav")
    result = model.transcribe("./test.wav")
    print(result["text"])

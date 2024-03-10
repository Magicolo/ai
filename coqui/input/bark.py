import sys, io, os, datetime
sys.path.append('~/TTS')
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

manager = ModelManager()
model, _, _ = manager.download_model("tts_models/multilingual/multi-dataset/bark")
configuration = os.path.join(model, "config.json")
print(f" > Model: {model}")
print(f" > Configuration: {configuration}")
synthesizer = Synthesizer(tts_checkpoint=model, tts_config_path=configuration)
print(f" > Model loaded.")
print(f" > Waiting for input...")

for text in sys.stdin:
    waves = synthesizer.tts(text, speaker_name=None, language_name="fr", speaker_wav="/input/test.wav")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"/output/speech-{timestamp}.wav", 'wb') as file:
        synthesizer.save_wav(waves, file)

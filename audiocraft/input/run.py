import sys, time, numpy

sys.path.append("/audiocraft")

from pydub import AudioSegment
from audiocraft.models import MusicGen, MAGNeT

duration = 10
chunk = 100
# model = MAGNeT.get_pretrained("facebook/magnet-medium-10secs")
# model = MAGNeT.get_pretrained("facebook/audio-magnet-medium")
# model = MusicGen.get_pretrained("facebook/musicgen-large")
# model.set_generation_params(duration=duration)
rate = model.sample_rate
trim = duration * rate // chunk
fade = duration * 1000 // chunk
prompt = "Earthquake. Rumble. Rubbles."
audio = AudioSegment.silent(duration=fade)
old = model.generate([prompt])

for i in range(1):
    now = time.time()
    new = model.generate_continuation(old[:, :, -trim:], model.sample_rate, [prompt])
    old = new
    new = new.squeeze().cpu().numpy()
    new = (new * 32767).astype(numpy.int16)

    duration = audio.duration_seconds
    segment = AudioSegment(new.tobytes(), frame_rate=rate, sample_width=2, channels=1)
    audio = audio.append(segment, crossfade=fade)
    audio.export("/input/test.wav", format="wav")
    elapsed = time.time() - now
    print(
        f"Done({i}) after '{elapsed}' with speed '{(audio.duration_seconds - duration) / elapsed}'."
    )

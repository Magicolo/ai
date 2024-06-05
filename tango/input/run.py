import sys

sys.path.append("/tango")

import soundfile
from tango import Tango

tango = Tango("declare-lab/tango2-full")
audio = tango.generate("An audience cheering and clapping")
soundfile.write(f"/input/test.wav", audio, samplerate=16000)

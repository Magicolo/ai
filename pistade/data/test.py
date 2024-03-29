import bark, scipy

bark.preload_models()

while True:
    text = input("> ")
    _, samples = bark.generate_audio(text, silent=True, output_full=True)
    scipy.io.wavfile.write("/data/test.wav", rate=bark.SAMPLE_RATE, data=samples)
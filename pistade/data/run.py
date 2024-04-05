import os, time, random, threading, math, re, json, argparse
import whisperx, whisperx.audio, ollama, numpy, sounddevice, scipy, scipy.signal, bark, torch, torchaudio
import pydub, pydub.playback, pydub.effects
import resemble_enhance, resemble_enhance.enhancer, resemble_enhance.enhancer.inference

parser = argparse.ArgumentParser()
parser.add_argument("--synthesize", action="store_true", help="Synthesize audio input.")
arguments = parser.parse_args()
print(f"> Run with: {arguments}")

audio_lock = threading.Lock()
audio = None
synthesize = arguments.synthesize

print(f"> Clear dump.")
for file in os.listdir("/dump"):
    os.remove(f"/dump/{file}")

print(f"> Load Ollama.")
ollama_model = (
    "gemma:2b-instruct"  # llava # gemma:2b-instruct-fp16 # gemma:7b-instruct-q3_K_L
)
ollama_client = ollama.Client(host="http://ollama:11434")
ollama_client.pull(ollama_model)

print(f"> Load Enhance.")
enhance_device = "cpu"
enhance_model = resemble_enhance.enhancer.inference.load_enhancer(None, enhance_device)
enhance_lock = threading.Lock()

print(f"> Load Bark.")
bark.preload_models()
bark_rate = bark.SAMPLE_RATE
bark_lock = threading.Lock()

print(f"> Load Whisper.")
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_rate = whisperx.audio.SAMPLE_RATE
whisper_model = whisperx.load_model("large-v3", whisper_device)
whisper_align = {}
whisper_lock = threading.Lock()
# with open("/data/token", "r") as file:
#     whisper_diarize = whisperx.DiarizationPipeline(use_auth_token=file.read(), device=whisper_device)


def unique_path(folder, name, extension):
    counter = 0
    while True:
        current = f"{folder}/{name}_{counter}.{extension}"
        if os.path.exists(current):
            counter += 1
        else:
            return current


def dump(name, value, rate=None):
    if isinstance(value, str):
        with open(unique_path("/dump/", name, "txt"), "w") as file:
            file.write(value)
    elif isinstance(value, pydub.AudioSegment):
        value.export(unique_path("/dump/", name, "wav"), format="wav")
    elif rate is None:
        with open(unique_path("/dump/", name, "json"), "w") as file:
            json.dump(value, file, indent=4)
    elif torch.is_tensor(value):
        torchaudio.save(unique_path("/dump/", name, "wav"), value, rate)
    else:
        scipy.io.wavfile.write(
            unique_path("/dump/", name, "wav"), rate=rate, data=value
        )


def whisper(samples):
    global whisper_model, whisper_lock, whisper_align, whisper_device

    with whisper_lock:
        result = whisper_model.transcribe(samples)
        language = result["language"]

        if language in whisper_align:
            align_model, align_meta = whisper_align[language]
        elif language in ["fr", "en", "de", "es", "it"]:
            print(f"> Load align model for '{language}'.")
            align_model, align_meta = whisperx.load_align_model(
                language_code=language, device=whisper_device
            )
            whisper_align[language] = align_model, align_meta
        else:
            return None

        segments = result["segments"]
        return whisperx.align(
            segments, align_model, align_meta, samples, whisper_device
        )


def generate(prompt, rate):
    global bark_rate, bark_lock

    print(f"> Generate audio text with prompt '{prompt}'.")
    response = ollama_client.generate(ollama_model, prompt=prompt)
    text = response["response"]
    print(f"> Synthesize audio for '{text}'.")
    with bark_lock:
        history, samples = bark.generate_audio(text, silent=True, output_full=True)
    # TODO: Use 'history' for more coherence in the generated audio?
    print(f"> Resample synthesized audio.")
    samples = scipy.signal.resample(
        samples, int(len(samples) * float(rate) / float(bark_rate))
    )
    return text, history, samples


def enhance(audio, rate):
    global enhance_device, enhance_model, enhance_lock

    with enhance_lock:
        return resemble_enhance.inference.inference(
            enhance_model, audio, rate, enhance_device
        )


def take(rate):
    global audio_data, audio_lock

    if audio_data is None or len(audio_data) < rate:
        return None
    else:
        with audio_lock:
            audio = audio_data
            audio_data = audio_data[math.ceil(len(audio_data) * 0.1) : rate * 10]

        meta = numpy.iinfo(numpy.int16)
        multiplier = 0.5 / (meta.max - meta.min)
        return numpy.multiply(audio, multiplier, dtype=numpy.float32)


def word_path(name):
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w\s]", "", name)
    return f"/data/words/{name}.wav"


def extract(audio, result, rate):
    for segment in result["segments"]:
        text = segment["text"]
        print(f"> Transcribe '{text}'.")

        words = segment["words"]
        for i in range(len(words)):
            name = None
            start = None

            for word in words[i:]:
                if (
                    "score" not in word
                    or "start" not in word
                    or "end" not in word
                    or "word" not in word
                ):
                    break
                if word["score"] < 0.75:
                    break

                name = word["word"] if name is None else name + " " + word["word"]
                start = start or word["start"]
                end = word["end"]
                data = audio[int(start * rate) : int(word["end"] * rate)]
                scipy.io.wavfile.write(word_path(name), rate=rate, data=data)


def record():
    global synthesize, whisper_rate

    def callback(input, frames, time, status):
        global audio_data, audio_lock

        with audio_lock:
            audio_data = (
                input if audio_data is None else numpy.append(audio_data, input)
            )

    if synthesize:
        return

    with sounddevice.InputStream(
        samplerate=whisper_rate, channels=1, dtype=numpy.int16, callback=callback
    ) as stream:
        while stream.active:
            time.sleep(1)


def transcribe():
    global synthesize, whisper_rate

    while True:
        if synthesize:
            _, _, samples = generate(
                f"Écris une courte phrase créative et poétique en français sans introduction.",
                whisper_rate,
            )
        else:
            samples = take(whisper_rate)

        if samples is None:
            time.sleep(0.1)
            continue
        result = whisper(samples)
        if result is None:
            time.sleep(0.1)
            continue
        extract(samples, result, whisper_rate)
        dump("transcribe-audio", samples, rate=whisper_rate)
        dump("transcribe-result", result)

        # diarize_segments = diarize_model(record)
        # result = whisperx.assign_word_speakers(diarize_segments, result)
        # print(diarize_segments)
        # print(result["segments"])


def speak():
    global whisper_rate

    folder = "/data/words"
    fade = 10

    while True:
        words = set()
        for file in os.listdir(folder):
            [name, *_] = os.path.splitext(file)
            for split in name.split("_"):
                words.add(split)
        words = random.sample(words, min(len(words), 100))

        prompt = f"Écris une phrase composée uniquement des mots dans la liste suivante: {words}."
        text, history, samples = generate(prompt, whisper_rate)
        scipy.io.wavfile.write(f"/data/speak.wav", rate=whisper_rate, data=samples)
        print(f"> Transcribe generated audio.")
        result = whisper(samples)
        if result is None:
            continue

        print(f"> Mutate generated audio.")
        audio = pydub.AudioSegment.from_wav("/data/speak.wav")
        # TODO: Find out why creating the 'AudioSegment' leads to massive clipping and distortion.
        # audio = pydub.AudioSegment(samples.tobytes(), sample_width=samples.itemsize, channels=1, frame_rate=whisper_rate)
        audio = pydub.effects.normalize(audio)
        for segment in result["segments"]:
            words = segment["words"]
            index = 0

            while index < len(words):
                segment = None
                name = None
                start = None
                end = None

                for word in words[index:]:
                    if (
                        "score" not in word
                        or "start" not in word
                        or "end" not in word
                        or "word" not in word
                    ):
                        break
                    if word["score"] < 0.5:
                        print("> Word score too low.")
                        break

                    name = word["word"] if name is None else name + " " + word["word"]
                    path = word_path(name)
                    if os.path.exists(path):
                        segment = path
                    else:
                        print(f"> Word '{name}' at path '{path}' not found.")
                        break

                    index += 1
                    start = start or word["start"]
                    end = word["end"]

                if segment is None or start is None or end is None:
                    index += 1
                    continue
                segment = pydub.AudioSegment.from_wav(segment)
                start *= 1000
                end *= 1000
                speed = len(segment) / float(end - start)
                try:
                    segment = pydub.effects.speedup(
                        segment, playback_speed=speed, crossfade=10
                    )
                except Exception as error:
                    print("> Speedup error:", error)
                try:
                    segment = pydub.effects.normalize(segment)
                except Exception as error:
                    print("> Normalize error:", error)
                try:
                    audio = (
                        audio[:start]
                        .append(segment, crossfade=fade)
                        .append(audio[end:], crossfade=fade)
                    )
                except Exception as error:
                    print("> Append error:", error)
                print(f"> Replace word '{name}' in generated audio.")

        # TODO: Find a way to convert the AudioSegment directly to a tensor.
        audio.export("/data/speak.wav", format="wav")
        audio, rate = torchaudio.load("/data/speak.wav")
        audio = audio.squeeze()
        audio, rate = enhance(audio, rate)
        extract(samples, result, whisper_rate)
        dump("speak-prompt", prompt)
        dump("speak-text", text)
        dump("speak-result", result)
        dump("speak-raw", samples, rate=whisper_rate)
        dump("speak-process", audio.unsqueeze(0), rate=rate)

        # print(f"> Compose sentence: {text}")
        # words = [word for word in re.split(r'\W+', text) if word != '']
        # audio = pydub.AudioSegment.silent(duration=fade)
        # for word in words:
        #     path = f"{folder}/{word}.wav"
        #     if not os.path.exists(path):
        #         continue

        #     segment = pydub.AudioSegment.from_wav(path)
        #     segment = pydub.effects.normalize(segment)
        #     audio = audio.append(segment, crossfade=min(len(segment), fade))
        # print(f"> Play: {words}")
        # audio.export("/data/speak.wav", format="wav")
        # pydub.playback.play(audio)

        # TODO: Reduce noise from audio using 'resemblance-ai' model.
        # TODO: Trim silence from beginning and end of audio.


threads = [
    # threading.Thread(target=record),
    # threading.Thread(target=transcribe),
    threading.Thread(target=speak),
]

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

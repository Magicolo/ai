from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch, torchaudio, scipy

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
sample_rate = model.config.sampling_rate

# # from text
# text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt").to(device)
# audio_array_from_text = model.generate(**text_inputs, tgt_lang="fra")[0].cpu().numpy().squeeze()
# scipy.io.wavfile.write("/data/from_text.wav", rate=sample_rate, data=audio_array_from_text)

while True:
    # # from audio
    audio, orig_freq =  torchaudio.load("/data/test.wav")
    audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio, return_tensors="pt").to(device)
    # audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="fra")[0].cpu().numpy().squeeze()
    # scipy.io.wavfile.write("/data/from_audio.wav", rate=sample_rate, data=audio_array_from_text)
    
    output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    print(f"Translation from audio: {translated_text_from_audio}")
import sounddevice as sd
import numpy as np
import joblib
import librosa

from extract_features import extract_features

model = joblib.load("/models/svm_model.pkl")
scaler = joblib.load("/models/scaler.pkl")
le = joblib.load("/models/label_encoder.pkl")

sr = 16000

def listen(seconds=2):
    print("Listening...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()

    audio = audio.flatten()
    return audio

while True:
    signal = listen()
    librosa.output.write_wav("temp.wav", signal, sr)

    feats = extract_features("temp.wav")
    feats_scaled = scaler.transform([feats])

    pred = model.predict(feats_scaled)
    emotion = le.inverse_transform(pred)[0]

    print("Emotion detected:", emotion)

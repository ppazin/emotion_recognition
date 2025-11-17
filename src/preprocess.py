import librosa
import numpy as np

def preprocess_audio(path, sr=16000):
    signal, _ = librosa.load(path, sr=sr, mono=True)

    signal, _ = librosa.effects.trim(signal)

    if signal.size == 0:
        signal, _ = librosa.load(path, sr=sr, mono=True)

    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    return signal

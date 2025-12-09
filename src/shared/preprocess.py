import librosa
import numpy as np
import scipy.signal as signal


def preprocess_audio(path, sr=16000):

    y, _ = librosa.load(path, sr=sr, mono=True)

    if y.size == 0:
        y, _ = librosa.load(path, sr=sr, mono=True)

    y, _ = librosa.effects.trim(y, top_db=30)

    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    try:
        y_harmonic, _ = librosa.effects.hpss(y)
        y = y_harmonic
    except:
        pass

    return y

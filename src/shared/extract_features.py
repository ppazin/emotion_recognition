import numpy as np
import librosa
from emotion_recognition.src.shared.preprocess import preprocess_audio

def extract_features(path):
    signal = preprocess_audio(path)

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=80)
    mfcc_median = np.median(mfcc.T, axis=0)

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    delta_median = np.median(delta.T, axis=0)

    # Delta-Delta
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_median = np.median(delta2.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=16000)
    chroma_median = np.median(chroma.T, axis=0)

    # RMSE (energy)
    rmse = librosa.feature.rms(y=signal)
    rmse_median = np.median(rmse.T, axis=0)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=signal, sr=16000)
    contrast_median = np.median(contrast.T, axis=0)

    # final feature vector
    features = np.concatenate([
        mfcc_median,
        delta_median,
        delta2_median,
        chroma_median,
        rmse_median,
        contrast_median
    ])

    return features

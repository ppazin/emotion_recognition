import numpy as np
import librosa
import parselmouth
from src.shared.preprocess import preprocess_audio


def stats(x):
    return np.concatenate([
        np.mean(x, axis=1),
        np.std(x, axis=1),
        np.min(x, axis=1),
        np.max(x, axis=1),
    ])


def compute_jitter_shimmer(signal, sr):
    try:
        snd = parselmouth.Sound(signal, sampling_frequency=sr)
        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 300
        )

        jitter = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 75, 300, 1.3
        )

        shimmer = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)", 
            0, 0, 75, 300, 1.3, 1.6
        )

        return jitter, shimmer
    except:
        return 0.0, 0.0


def extract_features(path):
    signal = preprocess_audio(path)
    sr = 16000

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_stats = stats(mfcc)
    delta_stats = stats(delta)
    delta2_stats = stats(delta2)

    # Chroma + Contrast
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_stats = stats(chroma)

    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    contrast_stats = stats(contrast)

    # Energy
    rmse = librosa.feature.rms(y=signal)
    rmse_stats = np.array([
        np.mean(rmse),
        np.std(rmse),
        np.min(rmse),
        np.max(rmse)
    ])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    zcr_stats = np.array([
        np.mean(zcr),
        np.std(zcr),
        np.min(zcr),
        np.max(zcr)
    ])

    # PITCH + DYNAMICS
    try:
        f0, voiced_flag = librosa.pyin(
            signal,
            fmin=50,
            fmax=350,
            sr=sr
        )
        f0 = f0[~np.isnan(f0)]

        if len(f0) > 0:
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
            pitch_min = np.min(f0)
            pitch_max = np.max(f0)
            pitch_range = pitch_max - pitch_min

            # pitch slope (trend)
            x = np.arange(len(f0))
            slope = np.polyfit(x, f0, 1)[0]
            pitch_stats = np.array([pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range, slope])
        else:
            raise ValueError()

    except:
        pitch_stats = np.zeros(6)

    # Jitter + Shimmer
    jitter, shimmer = compute_jitter_shimmer(signal, sr)
    voice_quality_stats = np.array([jitter, shimmer])

    # Spectral centroid + bandwidth
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)

    centroid_stats = np.array([
        np.mean(centroid),
        np.std(centroid),
        np.min(centroid),
        np.max(centroid)
    ])

    bandwidth_stats = np.array([
        np.mean(bandwidth),
        np.std(bandwidth),
        np.min(bandwidth),
        np.max(bandwidth)
    ])

    # FINAL FEATURE VECTOR
    features = np.concatenate([
        mfcc_stats,
        delta_stats,
        delta2_stats,
        chroma_stats,
        contrast_stats,
        rmse_stats,
        zcr_stats,
        pitch_stats,
        voice_quality_stats,
        centroid_stats,
        bandwidth_stats,
    ])

    return features

import librosa
import numpy as np


def extract_temporal(audio: np.ndarray, sr: int) -> dict:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    rms          = librosa.feature.rms(y=audio)[0]
    rms_mean     = float(np.mean(rms))
    rms_variance = float(np.var(rms))

    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)[0]))

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets    = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration  = len(audio) / sr
    transient_density = float(len(onsets) / (duration + 1e-9))

    return {
        "rms_mean":          rms_mean,
        "rms_variance":      rms_variance,
        "zcr_mean":          zcr_mean,
        "transient_density": transient_density,
    }

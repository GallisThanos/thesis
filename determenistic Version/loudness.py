import pyloudnorm as pyln
import numpy as np
import librosa
from scipy.stats import trim_mean

from config import Config


def extract_loudness(audio: np.ndarray, sr: int, config: Config) -> dict:
    meter = pyln.Meter(sr)

    if audio.ndim > 1:
        audio_mono = np.sqrt((audio[0] ** 2 + audio[1] ** 2) / 2)
    else:
        audio_mono = audio.copy()

    # Relative silence trim: threshold relative to peak
    peak_db = 20 * np.log10(np.max(np.abs(audio_mono)) + 1e-9)
    relative_trim_db = min(config.trim_db, abs(peak_db) + 30)
    audio_mono, _ = librosa.effects.trim(audio_mono, top_db=relative_trim_db)

    integrated_lufs = meter.integrated_loudness(audio_mono)

    frame_size = int(sr * max(config.momentary_window, 0.4))
    values = []
    for start in range(0, len(audio_mono) - frame_size + 1, frame_size):
        frame = audio_mono[start : start + frame_size]
        lufs = meter.integrated_loudness(frame)
        if np.isfinite(lufs):
            values.append(lufs)

    if values:
        if config.avg_method == "mean":
            momentary_lufs = float(np.mean(values))
        elif config.avg_method == "median":
            momentary_lufs = float(np.median(values))
        else:
            momentary_lufs = float(trim_mean(values, 0.1))
    else:
        momentary_lufs = float(integrated_lufs)

    peak = np.max(np.abs(audio_mono))
    rms  = np.sqrt(np.mean(audio_mono ** 2))
    crest_factor = 20 * np.log10(peak / (rms + 1e-9))

    return {
        "integrated_lufs": float(integrated_lufs),
        "momentary_lufs":  momentary_lufs,
        "crest_factor_db": float(crest_factor),
    }

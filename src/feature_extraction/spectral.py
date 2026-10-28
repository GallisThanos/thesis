import librosa
import numpy as np


def extract_spectral(audio: np.ndarray, sr: int) -> dict:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))

    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    low_energy = float(np.sum(fft[(freqs >= 20) & (freqs < 250)]))
    mid_energy = float(np.sum(fft[(freqs >= 250) & (freqs < 4000)]))
    high_energy = float(np.sum(fft[(freqs >= 4000)]))
    total = low_energy + mid_energy + high_energy + 1e-9

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "low_energy_ratio": low_energy / total,
        "mid_energy_ratio": mid_energy / total,
        "high_energy_ratio": high_energy / total,
    }

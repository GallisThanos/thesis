import librosa
import numpy as np


def load_audio(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(path, sr=sr, mono=False)
    return audio, sample_rate

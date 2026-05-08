import numpy as np


def extract_stereo(audio: np.ndarray) -> dict:
    if audio.ndim < 2:
        return {
            "stereo_width":   0.0,
            "correlation":    1.0,
            "mid_side_ratio": 0.0,
        }

    left, right = audio[0], audio[1]

    correlation   = float(np.corrcoef(left, right)[0, 1])
    width         = float(np.std(left - right))

    mid  = (left + right) / 2
    side = (left - right) / 2
    mid_side_ratio = float(np.sum(side ** 2) / (np.sum(mid ** 2) + 1e-9))

    return {
        "stereo_width":   width,
        "correlation":    correlation,
        "mid_side_ratio": mid_side_ratio,
    }

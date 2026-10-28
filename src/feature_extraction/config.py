from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    sr: int = 44100
    avg_method: Literal["mean", "median", "trimmed"] = "trimmed"
    momentary_window: float = 0.4
    trim_db: float = 40.0

    # Z-score thresholds for interpretation
    z_ignore: float = 0.5
    z_mild: float = 1.0
    z_moderate: float = 2.0
    # anything above z_moderate → "strong"

    # Minimum number of references before confidence degrades
    min_refs_for_high_confidence: int = 3

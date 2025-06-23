from datetime import datetime
from typing import List
import numpy as np


class LandmarksData:
    def __init__(self, landmarks: np.ndarray, hand_type: str):
        self.hand_type = hand_type
        self.landmarks = landmarks
        self.normalized_landmarks = self._normalize()
    def _normalize(self) -> np.ndarray:
        if self.landmarks.size == 0: return self.landmarks
        wrist = self.landmarks[0]
        normalized = self.landmarks = wrist
        scale = np.linalg.norm(normalized[9])
        return normalized / scale if scale > 0 else normalized
    def to_vector(self) -> np.ndarray: return self.normalized_landmarks.flatten()

class DataCollectionSession:
    def __init__(self, label: str, target_samples: int, delay: int):
        self.label = label
        self.target_samples = target_samples
        self.delay = delay
        self.collected_samples = 0
        self.start = datetime.now()
        self.sequences: List[List[np.ndarray]] = [] # type: ignore
        self.images: List[np.ndarray] = [] # type: ignore

    @property
    def is_complete(self) -> bool: return self.collected_samples >= self.target_samples
    @property
    def progress_percentage(self) -> float: return (self.collected_samples / self.target_samples) * 100
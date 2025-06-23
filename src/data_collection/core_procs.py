from datetime import datetime
import logging
import os
from data_collection.config import Config
import mediapipe as mp
import numpy as np
from typing import Tuple, Any, List, Optional
from data_collection.data_models import LandmarksData
import cv2

class HandProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands # type: ignore
        self.mp_drawing = mp.solutions.drawing_utils # type: ignore
        self.hands = self.mp_hands.Hands(max_num_hands = config.MAX_NUM_HANDS, min_detection_confidence = config.MIN_DETECTION_CONFIDENCE, min_tracking_confidence = config.MIN_TRACKING_CONFIDENCE)

    def process_frame(self, frame: np.ndarray) -> Tuple[Any, List[LandmarksData]]: # type: ignore
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        results = self.hands.process(rgb_frame)
        hand_landmarks = []
        if results.multi_handedness and results.multi_hand_landmarks:
            for index, hand_info in enumerate(results.multi_handedness):
                hand_label = hand_info.classification[0].label
                landmarks = results.multi_hand_landmarks[index]
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                hand_landmarks.append(LandmarksData(landmark_array, hand_label))
        return results, hand_landmarks
    
    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
    
    def landmarks_to_vector(self, hand_landmarks: List[LandmarksData]) -> np.ndarray:
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))
        for hand in hand_landmarks:
            if hand.hand_type == 'Left':
                left_hand = hand.normalized_landmarks
            else:
                right_hand = hand.normalized_landmarks
        return np.concatenate([left_hand.flatten(), right_hand.flatten()])
    
class CameraManager:
    def __init__(self, config: Config):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None # type: ignore
        self.is_opened = False

    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.config.CAMERA_IDX)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
                self.is_opened = True
                return True
        except Exception as e:
            logging.error(f"ERROR: Failed to init camera: {e}")
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]: # type: ignore
        if not self.is_opened or not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # da mirror epeks
        return ret, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
        self.is_opened = False


class FileManager:
    def __init__(self, config: Config):
        self.config = config
        self._create_necessary_dirs()

    def _create_necessary_dirs(self):
        for subdir_path in [self.config.landmarks_path, self.config.images_path]:
            for label in self.config.LABELS:
                os.makedirs(os.path.join(subdir_path, label), exist_ok=True)

    def frame_image_save(self, frame: np.ndarray, label: str) -> str:
        file_name = f"{label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        file_path = os.path.join(self.config.images_path, label, file_name)

        try:
            cv2.imwrite(file_path, frame)
            logging.info(f"INFO: Image saved: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"ERROR: Failed to save image: {e}")
            return ""
        
    def sequence_save(self, sequence: List[np.ndarray], label: str) -> str:
        file_name = f"{label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
        file_path = os.path.join(self.config.landmarks_path, label, file_name)
        
        try:
            np.save(file_path, np.array(sequence))
            logging.info(f"INFO: Saved sequence: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"ERROR: Failed to save sequence: {e}")
            return ""
        

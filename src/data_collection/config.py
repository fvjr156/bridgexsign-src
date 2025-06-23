import os
import json
from typing import Dict, Any

import cv2

class Config:
    
    def __init__(self, config_path: str = "./config.json"): 
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, self.config_path)
        JSONDATA: Dict[str, Any] = {}
        try:
            with open(config_path, "r") as f:
                JSONDATA = json.load(f)
            
            self.LABELS = JSONDATA["labels"]
            self.DATA_PATH = JSONDATA["data"]
            self.LANDMARKS_PATH = JSONDATA["landmarks"]
            self.IMAGES_PATH = JSONDATA["images"]
            self.CAMERA_IDX = JSONDATA["camera"]["index"]
            self.CAMERA_WIDTH = JSONDATA["camera"]["width"]
            self.CAMERA_HEIGHT = JSONDATA["camera"]["height"]
            self.SEQUENCE_LENGTH = JSONDATA["collection"]["sequence_len"]
            self.DEFAULT_SAMPLES = JSONDATA["collection"]["def_samples"]
            self.DEFAULT_DELAY = JSONDATA["collection"]["def_delay"]
            self.MAX_NUM_HANDS = JSONDATA["mediapipe"]["max_hands"]
            self.MIN_DETECTION_CONFIDENCE = JSONDATA["mediapipe"]["min_detection_confidence"]
            self.MIN_TRACKING_CONFIDENCE = JSONDATA["mediapipe"]["min_tracking_confidence"]

            self.COUNTDOWN_COLOR = (0, 0, 255) # palitan mo color pag d bagay sa UI
            self.COLLECTING_COLOR = (0, 255, 0)
            self.FONT = cv2.FONT_HERSHEY_SIMPLEX
            self.FONT_SCALE = 0.8
            self.FONT_THICKNESS = 2

        except Exception as e:
            raise ValueError(f"ERROR: Failed to load config: {e}")
        
    @property
    def landmarks_path(self) -> str: return self.LANDMARKS_PATH
    @property
    def images_path(self) -> str: return self.IMAGES_PATH
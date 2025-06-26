import cv2
import numpy as np
from typing import Optional, Tuple, Any

class Camera:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self._IDX = 0
        self._CAM_W = 540
        self._CAM_H = 720

    def initialize(self):
        self.cap = cv2.VideoCapture(self._IDX)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._CAM_H)
            self.is_opened = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Any]: # type: ignore
        if not self.is_opened or not self.cap:
            return False, None, None
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame, image # type: ignore
    
    def release(self):
        if self.cap:
            self.cap.release()
        self.is_opened = False

    def putText(self, frame, text, point, font, size, color, thickness):
        cv2.putText(frame, text, point, font, size, color, thickness)
   
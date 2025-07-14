import cv2
import numpy as np
from typing import Optional, Tuple, Any

class Camera:
    def __init__(self):
        self.video_capture: Optional[cv2.VideoCapture] = None
        self._IDX = 0 # 1 for obs
        # self._CAM_W = 540
        # self._CAM_H = 720
        self._CAM_W = 768
        self._CAM_H = 432

    def initialize(self):
        self.video_capture = cv2.VideoCapture(self._IDX)
        if self.video_capture.isOpened():
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._CAM_W)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._CAM_H)
            self.is_opened = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Any]: # type: ignore
        if not self.is_opened or not self.video_capture:
            return False, None, None
        
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame, image # type: ignore
    
    def release(self):
        if self.video_capture:
            self.video_capture.release()
        self.is_opened = False

    def putText(self, frame, text, point, font, size, color, thickness):
        cv2.putText(frame, text, point, font, size, color, thickness)
   
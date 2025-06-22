import logging
import threading
import time

import cv2
import numpy as np
from observers import Subject
from config import Config
from core_procs import HandProcessor, CameraManager, FileManager
from data_models import DataCollectionSession
from typing import Optional, List

class DataCollectionEngine(Subject): # data_collection's most important code
    def __init__(self, config: Config):
        super().__init__() # Subject .__init__()
        self.config = config
        self.hand_proc = HandProcessor(config)
        self.cam_mgr = CameraManager(config)
        self.file_mgr = FileManager(config)

        self.current_session: Optional[DataCollectionSession] = None
        self.current_sequence: List[np.ndarray] = []
        self.is_collecting = False
        self.stop_request = False

    def start_session(self, label: str, samples: int, delay: int) -> bool:
        if self.is_collecting:
            return False
        
        if not self.cam_mgr.initialize():
            self.notifyAllObservers('error', 'Failed to initialize camera!')
            return False
        
        self.current_session = DataCollectionSession(label, samples, delay)
        self.is_collecting = True
        self.stop_request = False

        threading.Thread(target=self._collection_loop, daemon=True).start()
        return True
    
    def stop_session(self):
        self.stop_request = True
    
    def _collection_loop(self):
        try:
            while (self.is_collecting and not self.stop_request and not self.current_session.is_complete):
                self._collect_single_sample()
        except Exception as e:
            logging.error(f"ERROR: Collection error: {e}")
            self.notifyAllObservers('error', str(e))
        finally:
            self._cleanup() 

    def _collect_single_sample(self):
        self._countdown()

        if self.stop_request:
            return
        
        self._sequence_collection()

        if len(self.current_session) >= self.config.SEQUENCE_LENGTH:
            self._current_sample_save()
            self.current_session.collected_samples += 1
            self.notifyAllObservers('sample_completed', self.current_session)

    def _countdown(self):
        for i in range(self.current_session.delay, 0, -1):
            if self.stop_request:
                return
            ret, frame = self.cam_mgr.read_frame()
            if not ret:
                continue

            self._countdown_text_draw(frame, i)
            self.notifyAllObservers('frame_update', frame)
            time.sleep(1) # 1s

    def _sequence_collection(self):
        self.current_sequence = []
        image_captured = False

        while len(self.current_sequence) < self.config.SEQUENCE_LENGTH:
            if self.stop_request:
                return
            ret, frame = self.cam_mgr.read_frame()
            if not ret:
                continue

            results, hand_landmarks = self.hand_proc.process_frame(frame)

            if hand_landmarks: 
                if not image_captured:
                    # naga-capture laang sa first detection
                    threading.Thread(
                        target=self._image_save_async,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()
                    image_captured = True
                
                landmark_vector = self.hand_proc.landmarks_to_vector(hand_landmarks)
                self.current_sequence.append(landmark_vector)
            
            # drawinf hands and progs
            frame = self.hand_proc.draw_landmarks(frame, results)
            self._collection_progress_draw(frame)
            self.notifyAllObservers('frame_update', frame)
            
            cv2.waitKey(1)

    def _image_save_async(self, frame: np.ndarray):
        self.file_mgr.frame_image_save(frame, self.current_session.label)

    def _current_sample_save(self):
        threading.Thread(
            target=self._sequence_save_async,
            args=(self.current_sequence.copy(),),
            daemon=True
        ).start()

    def _sequence_save_async(self, sequence: List[np.ndarray]):
        self.file_mgr.sequence_save(sequence, self.current_session.label)

    def _countdown_text_draw(self, frame: np.ndarray, countdown: int):
        text = f"Starting in {countdown}s..."
        cv2.putText(
            frame,
            text,
            (30, 50),
            self.config.FONT,
            1.0,
            self.config.COUNTDOWN_COLOR,
            self.config.FONT_THICKNESS
        )
        
    def _collection_progress_draw(self, frame: np.ndarray):
        prog_txt = f"Collecting {self.current_session.label}: {len(self.current_sequence)}/{self.config.SEQUENCE_LENGTH}"
        cv2.putText(
            frame, 
            prog_txt, 
            (10, 40), 
            self.config.FONT, 
            self.config.FONT_SCALE, 
            self.config.COLLECTING_COLOR, 
            self.config.FONT_THICKNESS
        )

        session_prog = f"Sample {self.current_session.collected_samples + 1}/{self.current_session.target_samples}"
        cv2.putText(
            frame,
            session_prog,
            (10, 70),
            self.config.FONT,
            self.config.FONT_SCALE,
            self.config.COLLECTING_COLOR,
            self.config.FONT_THICKNESS
        )

    def _cleanup(self):
        self.cam_mgr.release()
        cv2.destroyAllWindows()
        self.is_collecting = False
        self.notifyAllObservers('session_completed', self.current_session)



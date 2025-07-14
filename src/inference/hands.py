from typing import List, Any
import cv2
import numpy as np
import mediapipe as mp

class Hands:
    def __init__(self):
        self.finger_angles: List[Any] = []
        self.LEFT_HAND_COLOR = (255, 0, 0)
        self.RIGHT_HAND_COLOR = (0, 0, 255)
        self.mp_hands = mp.solutions.hands # type: ignore
        self.mp_drawing = mp.solutions.drawing_utils # type: ignore

        self.hands = self.mp_hands.Hands(
            max_num_hands = 2, min_tracking_confidence = 0.8, min_detection_confidence = 0.8
        )

    @property
    def get_finger_angles(self) -> List[Any]:
        return self.finger_angles

    def draw(self, frame, results):
        # Hand Landmarks
        # draw() will draw hand landmarks for both hands

        if results.multi_hand_landmarks:
            for idx, lm in enumerate(results.multi_hand_landmarks):
                landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark])
                hand_label = results.multi_handedness[idx].classification[0].label
                self._draw2(frame, landmarks, hand_label)
                self.mp_drawing.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                        color = self.LEFT_HAND_COLOR if (hand_label == 'Left') else self.RIGHT_HAND_COLOR,
                        thickness = 2, circle_radius = 2
                    )
                )

    def _draw2(self, frame, landmarks, hand_label=None):
        # _draw2() will draw vectors and angles on individual hands

        h, w, _ = frame.shape
        points = (landmarks[:, :2] * [w, h]).astype(int)

        # Palm Normal Vectors
        palm_normal = self._compute_palm_normal_vector(landmarks, hand_label)
        wrist = points[0]
        end_point = (wrist + (palm_normal[:2] * 100)).astype(int)
        cv2.arrowedLine(frame, wrist, tuple(end_point), (0, 255, 0), 2)

        pitch, yaw = self._palm_orientation_angles(palm_normal)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
       
        # Finger Angles
        self.finger_angles = self._compute_finger_angles(landmarks)
        for i, angle in enumerate(self.finger_angles): # type: ignore
            cv2.putText(frame, f"F{i+1}:{angle:.1f}", (10, 90 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (39, 187, 255), 2)
        

    def _compute_palm_normal_vector(self, landmarks, hand_label=None):
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        normal = np.cross(v1, v2)

        # cross product order matters for direction, so to account for chirality, when hand_label = 'Left', if-flip ko lang
        if hand_label == 'Left':
            normal = -normal

        return normal / (np.linalg.norm(normal) + 1e-6)

    def _palm_orientation_angles(self, normal):
        pitch = np.arcsin(normal[1])
        yaw = np.arctan2(normal[0], normal[2])
        return np.degrees(pitch), np.degrees(yaw)
    
    def _compute_angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _compute_finger_angles(self, landmarks):
        indices = [
            (1, 2, 3),   # Thumb
            (5, 6, 7),   # Index
            (9, 10, 11), # Middle
            (13, 14, 15),# Ring
            (17, 18, 19) # Pinky
        ]
        return [self._compute_angle(landmarks[a], landmarks[b], landmarks[c]) for a, b, c in indices]

    def extract_all_hand_features(self, results, image_shape):
        left_hand = np.zeros(73, dtype=np.float32)
        right_hand = np.zeros(73, dtype=np.float32)

        if not results.multi_hand_landmarks:
            return np.concatenate([left_hand, right_hand])  # when no hands

        for idx, lm in enumerate(results.multi_hand_landmarks):
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark])
            label = results.multi_handedness[idx].classification[0].label
            features = self._extract_hand_features(landmarks, label)
            assert features.shape == (73,), f"{label} hand feature shape invalid: {features.shape}"

            if label == "Left":
                left_hand = features
            elif label == "Right":
                right_hand = features

        return np.concatenate([left_hand, right_hand])

    def _extract_hand_features(self, landmarks, hand_label):
            # the hand features:
            # - 21 landmarks * 3 = 63
            # - palm normal vector = 3
            # - pitch + yaw = 2
            # - 5 finger angles = 5
            flattened_landmark = landmarks.flatten()  # 63
            normal = self._compute_palm_normal_vector(landmarks, hand_label)  # 3
            pitch, yaw = self._palm_orientation_angles(normal)  # 2
            finger_angles = self._compute_finger_angles(landmarks)  # 5

            return np.concatenate([     # this retuens:
                flattened_landmark,     # (63,)
                normal,                 # (3,)
                [pitch, yaw],           # (2,)
                finger_angles           # (5,)
            ])


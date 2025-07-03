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
            max_num_hands = 2, min_tracking_confidence = 0.5, min_detection_confidence = 0.7
        )

    @property
    def get_finger_angles(self) -> List[Any]:
        return self.finger_angles

    def draw(self, frame, results):
        # draw and _draw2 will display hand landmarks (different colors), then angles and vectors for each hand present

        def _draw2(frame, landmarks, hand_label = None):
            h, w, _ = frame.shape
            points = (landmarks[:, :2] * [w, h]).astype(int)

            palm_normal = self._compute_palm_normal_vector(landmarks, hand_label)
            wrist = points[0]
            end_pt = (wrist + (palm_normal[:2] * 100)).astype(int)
            cv2.arrowedLine(frame, wrist, tuple(end_pt), (0, 255, 0), 2)

        if not results.multi_hand_landmarks:
            return
        
        hand_inf = []

        for idx, lm in enumerate(results.multi_hand_landmarks):
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark])
            hand_label = results.multi_handedness[idx].classification[0].label

            pitch, yaw = self._palm_orientation_angles(self._compute_palm_normal_vector(landmarks, hand_label))
            finger_angles = self._compute_finger_angles(landmarks)

            hand_inf.append({
                "label": hand_label,
                "landmarks": landmarks,
                "pitch": pitch,
                "yaw": yaw,
                "finger_angles": finger_angles
            })

            _draw2(frame, landmarks, hand_label)

            self.mp_drawing.draw_landmarks(
                frame, lm, self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                        color = self.LEFT_HAND_COLOR if (hand_label == 'Left') else self.RIGHT_HAND_COLOR,
                        thickness = 2, circle_radius = 2
                )
            )

        hand_inf.sort(key = lambda x: 
                                x["label"])

        x, y, spacing = 10, 50, 20

        pitch_text = "Pitch: " + ", ".join(f"{h['pitch']:.1f}" for h in hand_inf)
        yaw_text = "Yaw: " + ", ".join(f"{h['yaw']:.1f}" for h in hand_inf)
        cv2.putText(frame, pitch_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, yaw_text, (x, y + spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        for i in range(5):
            angles = ", ".join(f"{h['finger_angles'][i]:.1f}" for h in hand_inf)
            cv2.putText(frame, f"F{i+1}: {angles}", (x, y + spacing * (i + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (39, 187, 255), 2)

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
        frame_vector = []

        if not results.multi_hand_landmarks:
            return np.zeros(146, dtype = np.float32)
        
        hands_data = []
        for idx, lm in enumerate(results.multi_hand_landmarks):
            landmarks = np.array([[l.x, l.y, l.z] for l in lm.landmark])
            label = results.multi_handedness[idx].classification[0].label
            hands_data.append((label, landmarks))

        sorted_hands_data = sorted(hands_data, key = lambda x: x[0]) # hands data always ordered -> Left, Right

        for label, landmarks in sorted_hands_data:
            features = self._extract_hand_features(landmarks, label)
            frame_vector.append(features)

        while len(frame_vector) < 2:
            frame_vector.append(np.zeros_like(frame_vector[0]))

        return np.concatenate(frame_vector)

    def _extract_hand_features(self, landmarks, hand_label):
        flattened_landmark = landmarks.flatten()
        normal = self._compute_palm_normal_vector(landmarks, hand_label)
        pitch, yaw = self._palm_orientation_angles(normal)
        finger_angles = self._compute_finger_angles(landmarks)

        return np.concatenate([
            flattened_landmark,
            normal,
            [pitch, yaw],
            finger_angles
        ])

if __name__ == '__main__':
    for i in range(0, 100): print("DO NOT RUN THIS CODE!!! INSTEAD, RUN src/data_collection.py !!!")

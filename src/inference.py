import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from typing import Any
import logging

from inference.camera import Camera
from inference.hands import Hands
from inference.sequence import Sequence

#make a model with only 20 sequences per npy, not 30
#for faster inference, more static gestures friendly


class Inference:
    def __init__(self):
        self.interpreter: tf.lite.Interpreter = tf.lite.Interpreter(model_path='./models/asl_model_lstm_quant.tflite')
        # self.interpreter: tf.lite.Interpreter = tf.lite.Interpreter(model_path='./models/asl_model_lstm_quant.tflite')
        self.labels: list[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'hello', 'i', 'iloveyou', 'j', 'k', 'l', 'm', 'n', 'no', 'o', 'p', 'please', 'q', 'r', 's', 'sorry', 't', 'thankyou', 'u', 'v', 'w', 'x', 'y', 'yes', 'z']
        # Found 34 labels: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'hello', 'i', 'iloveyou', 'j', 'k', 'l', 'm', 'n', 'no', 'o', 'p', 'please', 'q', 'r', 's', 'sorry', 't', 'thankyou', 'u', 'urwelc', 'v', 'w', 'x', 'y', 'yes', 'z']
        self.camera: Camera = Camera()
        self.hands: Hands = Hands()
        self.sequence: Sequence = Sequence()

        self.SM_WND = 2
        self.AC_THR = 1
        self.output_buffer = deque(maxlen=self.SM_WND)
        self.prediction_buffer = deque(maxlen=self.AC_THR)
        self.last_confirmed_label = "Idle"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def main(self):
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()   # Shape: [1, 30, 146]
        output_details = self.interpreter.get_output_details() # Shape: [1, 43]

        self.camera.initialize()
        
        smoothed_out: Any = 0

        while True:
            ret, frame, image = self.camera.read_frame()
            if not ret:
                break

            results = self.hands.hands.process(image)
            hand_present = results.multi_hand_landmarks is not None
            draw_boundbox = False

            if hand_present:
                draw_boundbox = True
                # if u want to draw landmarks on hands: # self.hands.draw(frame, results)
                frame_features = self.hands.extract_all_hand_features(results, image.shape)
                self.sequence.append(frame_features)

                if self.sequence.is_full():
                    input_seq = np.expand_dims(self.sequence.get_sequence(), axis=0).astype(np.float32)
                    self.interpreter.set_tensor(input_details[0]['index'], input_seq)
                    self.interpreter.invoke()
                    output = self.interpreter.get_tensor(output_details[0]['index'])[0]

                    self.output_buffer.append(output)
                    smoothed_out = np.mean(self.output_buffer, axis=0)
                    prediction_index = int(np.argmax(smoothed_out))
                    self.prediction_buffer.append(prediction_index)
                                        
                    if len(self.prediction_buffer) == self.AC_THR:
                        most_common, freq = Counter(self.prediction_buffer).most_common(1)[0]
                        if freq >= self.AC_THR:
                            self.last_confirmed_label = self.labels[most_common]
                        self.prediction_buffer.clear()

                    self.sequence.reset()
                
            else:
                self.sequence.reset()
                self.output_buffer.clear()
                self.prediction_buffer.clear()
                self.last_confirmed_label = "Idle"

            if draw_boundbox and hand_present:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = np.array([[pt.x, pt.y] for pt in hand_landmarks.landmark])
                    h, w, _ = frame.shape # type: ignore
                    lm_px = (lm * [w, h]).astype(int)
                    x1, y1 = np.min(lm_px, axis=0)
                    x2, y2 = np.max(lm_px, axis=0)

                    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2) # type: ignore
                    label_txt = f"{self.last_confirmed_label}"
                    if self.last_confirmed_label != "Idle":
                        confidence = np.max(smoothed_out)
                        label_txt += f" ({confidence * 100:.1f}%)"
                    self.put_text_with_background(frame, label_txt, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), (0, 255, 0), 2) # type: ignore
            else:
                cv2.putText(frame, "Idle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2) # type: ignore

            cv2.imshow("inference", frame) # type: ignore

            key = cv2.waitKey(1) & 0xFF
            if key == ord('Q') or cv2.getWindowProperty("inference", cv2.WND_PROP_VISIBLE) < 1:
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def put_text_with_background(self, img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=0.0, text_color=(0, 0, 0),
                             bg_color=(0, 0, 0), thickness=0):
        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org

        top_left = (x, y - text_height - baseline)
        bottom_right = (x + text_width, y + baseline)

        cv2.rectangle(img, top_left, bottom_right, bg_color, thickness=cv2.FILLED) # type: ignore
        cv2.putText(img, text, org, font, font_scale, text_color, thickness) # type: ignore

if __name__ == '__main__':
    print("The app is being loaded. Please wait.")
    instance = Inference()
    instance.main()

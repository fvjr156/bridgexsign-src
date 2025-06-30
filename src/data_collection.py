import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import logging
from data_collection_src.camera import Camera
from data_collection_src.hands import Hands
from data_collection_src.sequence import Sequence
from data_collection_src.procs import UIProcess, FileProcs

# 's' to record sequence
# 'q' to quit
# '0' to count files

class Main:
    @staticmethod
    def main():
        collecting = False

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        camera = Camera()
        hands = Hands()
        frame_sequence = Sequence()

        camera.initialize()
        FileProcs.count_dirs()

        while True:
            ret, frame, image = camera.read_frame()
            if not ret:
                continue

            results = hands.hands.process(image)

            if collecting:
                frame_features = hands.extract_all_hand_features(results, image.shape)
                frame_sequence.append(frame_features)

                camera.putText(frame, f"Collecting: {len(frame_sequence.sequence)}/{frame_sequence.sequence_length}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (106, 255, 0), 2)

                if frame_sequence.is_full():
                    collecting = False
                    label: str|None = UIProcess.prompt_label()

                    if label == None:
                        logging.info(f"Label null. Sequence not saved.")
                    elif label == "":
                        logging.info(f"Label empty. Sequence not saved.")
                    else:
                        FileProcs.save_sequence(frame_sequence.get_sequence(), label)

            hands.draw(frame, results)
            cv2.imshow("demo", frame) # type: ignore

            # to quit, either shift+q or close window
            # to start collection, shift+s

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord(' '):
                frame_sequence.reset() # reset every before collection
                logging.info("Collecting sequence...")
                collecting = True
            elif key == ord('0'):
                FileProcs.count_dirs()
            elif key == ord('Q') or cv2.getWindowProperty("demo", cv2.WND_PROP_VISIBLE) < 1:
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Main.main()
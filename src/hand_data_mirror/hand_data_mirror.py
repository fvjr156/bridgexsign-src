import numpy as np
import os

class HandDataMirror:
    def __init__(self, input_path):
        self._input_path = input_path
        self._npy_files = self.scan_dir(input_path)

    @property
    def npy_files(self):
        return self._npy_files

    def scan_dir(self, dirpath: str, extname: str = ".npy") -> list[str]:
        files = []
        self._count_dirs()
        for root, dirs, f in os.walk(dirpath):
            for file in f:
                if file.endswith(extname):
                    files.append(os.path.join(root, file))
        return files
    
    def _count_dirs(self):
        folder = self._input_path
        os.makedirs(folder, exist_ok = True)
        print("===== EXISTING LABELS =====")
        f1: list[str] = os.listdir(folder)
        if len(f1) < 1:
            print ("None.")
        else:
            for label in os.listdir(folder):
                print(f"\"{label}\": {len([f for f in os.listdir(os.path.join(folder, label)) if os.path.isfile(os.path.join(folder, label, f))])}")
        print("===== END =====")

    def mirror_sequence(self, sequence: np.ndarray) -> np.ndarray:
        sequence_m = []

        for frame in sequence:
            hand_r = frame[:63].reshape((21,3))
            normal_r = frame[126:129]
            pitch_yaw_r = frame[129:131]
            angles_r = frame[131:136]

            landmarks_m = hand_r.copy()
            landmarks_m[:, 0] = 1.0 - landmarks_m[:, 0]

            normal_m = normal_r.copy()
            normal_m[0] *= -1
            normal_m[2] *= -1

            pitch_m = pitch_yaw_r[0]
            yaw_m = -pitch_yaw_r[1]

            features_m = np.concatenate([
                landmarks_m.flatten(),
                normal_m,
                [pitch_m, yaw_m],
                angles_r
            ])

            full_frame = np.concatenate([features_m, np.zeros_like(features_m)])
            sequence_m.append(full_frame)

        return np.array(sequence_m, dtype = np.float32)
    
    def preview_first_frame(self, path: str):
        data = np.load(path)
        if len(data.shape) != 2 or data.shape[1] != 146:
            print(f"Invalid shape: {data.shape}")
            return

        frame = data[0]

        # Split into two hand-slots
        hand1 = frame[:73]
        hand2 = frame[73:146]

        # Determine which one is real (not all zeros)
        if not np.all(hand1 == 0):
            print("🖐 Detected hand in slot 1 (left slot)")
            active_hand = hand1
        elif not np.all(hand2 == 0):
            print("🖐 Detected hand in slot 2 (right slot)")
            active_hand = hand2
        else:
            print("⚠️ No hand data found in this frame.")
            return

        # Extract components from active hand
        landmarks = active_hand[:63].reshape((21, 3))
        normal = active_hand[63:66]
        pitch, yaw = active_hand[66:68]
        angles = active_hand[68:73]

        print("===== ORIGINAL FRAME =====")
        print("Landmarks:\n", landmarks)
        print("Palm Normal:\n", normal)
        print(f"Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")
        print("Finger Angles:\n", angles)

        # Mirror logic
        landmarks_m = landmarks.copy()
        landmarks_m[:, 0] = 1.0 - landmarks_m[:, 0]

        normal_m = normal.copy()
        normal_m[0] *= -1
        normal_m[2] *= -1

        pitch_m = pitch
        yaw_m = -yaw

        print("\n===== MIRRORED FRAME =====")
        print("Landmarks:\n", landmarks_m)
        print("Palm Normal:\n", normal_m)
        print(f"Pitch: {pitch_m:.3f}, Yaw: {yaw_m:.3f}")
        print("Finger Angles:\n", angles)

    def preview_first_frame(self, input_file_path: str, frame_idx: int = 0):
        data = np.load(input_file_path)
        if len(data.shape) != 2 or data.shape[1] != 146:
            print(f"Input file has invalid shape: {data.shape}")
            return
        frame = data[frame_idx]

        hand1 = frame[:73]
        hand2 = frame[73:146]

        if not np.all(hand1 == 0):
            print("")


if __name__ == '__main__':
    instance = HandDataMirror(r'data/landmark_sequences')
    instance.preview_first_frame(instance.npy_files[305])

    """
        ideal pipeline:
        1. retrieve npy files from input path
        2. preview what will be changed
        3. when user typed confirm, apply batch mirroring
        4. name as <original>_flipped.npy
        5. save the mirrored npy file to the input path

        HandDataMirror.__init__(input_path: str, save_dir: str)
        HandDataMirror.preview_first_frame(path: str)
        HandDataMirror.batch_mirror(files: list[str])
        HandDataMirror.Utils.save_file(filename, content, save_dir)

        pipeline for saving:
            get file path
            get folder name from filepath
                example/a/a_00.npy get example/a
            name the new file as a_00_flipped.npy
            save to example/a
    """

    """
        batch mirror:
            get file path from path name example/a/a_00.npy
            get directory path example/a
            set _filename to

        
    """
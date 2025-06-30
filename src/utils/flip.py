from datetime import datetime, timedelta
import time
import numpy as np
import os
from fileprocs import FileProcs as fp
from tqdm import tqdm

class Flip:
    @staticmethod
    def scan_dirs(dirpath: str, extname: str) -> list[str]:
        files = []
        for root, dirs, f in os.walk(dirpath):
            for file in f:
                if file.endswith(extname):
                    files.append(os.path.join(root, file))
        return files

    # filename conflicts, do not use for now
    @staticmethod
    def flip_right_to_left(sequence_paths: list[str], out_root: str = './data/landmark_sequences_flipped_left'):
        for path in sequence_paths:
            sequence = np.load(path)

            flipped_sequence = []
            for frame in sequence:
                left = frame[:73]
                right = frame[73:]

                rh_landmarks = right[:63].reshape(-1, 3)
                rh_normal = right[63:66]
                rh_pitch, rh_yaw = right[66:68]
                rh_finger_angles = right[68:73]

                rh_landmarks[:, 0] = 1.0 - rh_landmarks[:, 0] # horiz flip (x to 1 - x)

                rh_normal[0] *= -1 # flip palm normal's x
                rh_yaw = -rh_yaw # invert yaw

                # za mirrored left hand
                new_left = np.concatenate([
                    rh_landmarks.flatten(),
                    rh_normal,
                    [rh_pitch, rh_yaw],
                    rh_finger_angles
                ])

                # then pad right hand with 0s
                new_right = np.zeros(73, dtype=np.float32)

                flipped_frame = np.concatenate([new_left, new_right])
                flipped_sequence.append(flipped_frame)

            flipped_sequence = np.array(flipped_sequence, dtype=np.float32)

            # then save the flipped sequence in a mirrored directory structure
            relative_path = os.path.relpath(path, './data/landmark_sequences')
            save_path = os.path.join(out_root, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, flipped_sequence)
            print(f"Saved flipped sequence: {save_path}")

    @staticmethod
    def flip_right_to_left_delayed(sequence_paths: list[str], out_root: str):
        print(f"How long will it take? {Flip.seconds_to_hhmmss(len(sequence_paths))}")
        print("THANK YOU FOR YOUR PATIENCE!!!")

        for path in tqdm(sequence_paths, desc="Flipping sequences", unit="file"):
            start_time = time.time()

            sequence = np.load(path)

            flipped_sequence = []
            for frame in sequence:
                left = frame[:73]
                right = frame[73:]

                rh_landmarks = right[:63].reshape(-1, 3)
                rh_normal = right[63:66]
                rh_pitch, rh_yaw = right[66:68]
                rh_finger_angles = right[68:73]

                rh_landmarks[:, 0] = 1.0 - rh_landmarks[:, 0]  # mirror x

                rh_normal[0] *= -1  # invert x-component of normal
                rh_yaw = -rh_yaw    # invert yaw angle

                new_left = np.concatenate([
                    rh_landmarks.flatten(),
                    rh_normal,
                    [rh_pitch, rh_yaw],
                    rh_finger_angles
                ])

                new_right = np.zeros(73, dtype=np.float32)

                flipped_frame = np.concatenate([new_left, new_right])
                flipped_sequence.append(flipped_frame)

            flipped_sequence = np.array(flipped_sequence, dtype=np.float32)

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            relative_path = os.path.relpath(path, './data/landmark_sequences')
            base, _ = os.path.splitext(os.path.basename(relative_path))
            dir_part = os.path.dirname(relative_path)
            filename_with_time = f"{base}_{timestamp}.npy"
            save_path = os.path.join(out_root, dir_part, filename_with_time)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, flipped_sequence)
            print(f"\n{save_path}")

            # wait for the remainder of the second before processing the next
            elapsed = time.time() - start_time
            time.sleep(max(0, 1.0 - elapsed))

    @staticmethod
    def seconds_to_hhmmss(n: int):
        td = timedelta(seconds=n)
        # timedelta's str format is "D days, HH:MM:SS" if days > 0, so format manually:
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02}"


if __name__ == '__main__':
    # paths = Flip.scan_dirs('./data/landmark_sequences', '.npy')
    # Flip.flip_right_to_left_delayed(paths, './data/landmark_sequences_flipped_left')
    flipped_paths = Flip.scan_dirs('./data/landmark_sequences_flipped_left', '.npy')
    fp.batch_move(flipped_paths, './data/landmark_sequences')
    fp.count_dirs()



import numpy as np
import tqdm
import time
from numpy_array_file import NumpyArrayFile

class HandMirror:
    
    def start_sequence_mirroring(self, file_objs: list[NumpyArrayFile]):

        start_time = time.time()
        for sequence in tqdm.tqdm(file_objs, desc="Mirroring sequences", colour = 'green', unit = 'files', ascii = True):
            sequence_original = sequence.sequence
            sequence_mirrored = self.safe_sequence_mirroring(sequence_original)
            sequence.set_new_sequence(sequence_mirrored)
            sequence.save_sequence()
        
        elapsed_time = time.time() - start_time
        out_text = f"Mirrored {len(file_objs)} sequences in {elapsed_time:.2f} seconds."
        print("-" * (len(out_text) + 4))
        print(f"| {out_text} |")
        print("-" * (len(out_text) + 4))
    
    def safe_sequence_mirroring(self, sequence: np.ndarray) -> np.ndarray:

        sequence_mirrored = []
        
        for frame in sequence:
            hand_left = frame[:73]
            hand_right = frame[73:]

            mirrored_l = np.zeros_like(hand_left)
            mirrored_r = np.zeros_like(hand_right)

            def mirror_hand(hand):
                landmarks = hand[:63].reshape((21, 3)).copy()
                normal = hand[63:66]
                pitch, yaw = hand[66:68]
                angles = hand[68:73]

                landmarks[:, 0] = 1.0 - landmarks[:, 0]
                normal[0] *= -1
                normal[2] *= -1
                yaw *= -1

                return np.concatenate([
                    landmarks.flatten(),
                    normal,
                    [pitch, yaw],
                    angles
                ])

            if not np.all(hand_left == 0):
                mirrored_r = mirror_hand(hand_left)
            if not np.all(hand_right == 0):
                mirrored_l = mirror_hand(hand_right)

            full_frame = np.concatenate([mirrored_l, mirrored_r])
            sequence_mirrored.append(full_frame)

        return np.array(sequence_mirrored, dtype=np.float32)

    def preview_mirroring_output(self, file_obj: NumpyArrayFile, frame_idx: int = 0):
        sequence_original = file_obj.sequence

        print(f"Preview file path: {file_obj.file_path}")        

        if frame_idx >= len(sequence_original):
            print(f"Frame index {frame_idx} is out of bounds. Sequence has {len(sequence_original)} frames.")
            return
        
        frame_original = sequence_original[frame_idx]
        sequence_mirrored = self.safe_sequence_mirroring(sequence_original)
        frame_mirrored = sequence_mirrored[frame_idx]

        print(f"\n===== PREVIEW FOR FRAME {frame_idx} =====")
        print(f"Original frame shape: {frame_original.shape}")
        print(f"Mirrored frame shape: {frame_mirrored.shape}")
        
        orig_left = frame_original[:73]
        orig_right = frame_original[73:]
        mir_left = frame_mirrored[:73]
        mir_right = frame_mirrored[73:]
        
        print(f"\nOriginal - Left hand non-zero: {not np.all(orig_left == 0)}")
        print(f"Original - Right hand non-zero: {not np.all(orig_right == 0)}")
        print(f"Mirrored - Left hand non-zero: {not np.all(mir_left == 0)}")
        print(f"Mirrored - Right hand non-zero: {not np.all(mir_right == 0)}")
        
        orig_X = orig_left[:63].reshape(21, 3)[:, 0]
        mir_X = mir_right[:63].reshape(21, 3)[:, 0]
        print("\nX-values: Original vs Mirrored (should be roughly opposite):")
        for i, (o, m) in enumerate(zip(orig_X, mir_X)):
            print(f"  LM{i}: {o:.4f} --> {-o:.4f} vs {m:.4f}")

        print("ORIGINAL:")
        print(frame_original)
        print("MIRRORED:")
        print(frame_mirrored)
        
        print("===== END PREVIEW =====\n")
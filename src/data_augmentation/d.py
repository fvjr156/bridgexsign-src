import os
import numpy as np
import time
import tqdm

class NumpyArrayFile:
    def __init__(self, file_path: str):
        file_path_split: tuple[str, str] = os.path.splitext(file_path)
        self._dir_path, self._file_name = file_path_split

        self._new_file_name: str = f"{os.path.splitext(self._file_name)[0]}_flipped.npy"
        self._new_file_path: str = os.path.join(self._dir_path, self._new_file_name)

        self._sequence: np.ndarray = np.load(file_path)
        self._new_sequence: np.ndarray
        pass

    def set_new_sequence(self, sequence: np.ndarray):
        self._new_sequence = sequence 

    @property
    def sequence(self): 
        return self._sequence

    @property
    def file_path(self):
        return os.path.join(self._dir_path, self._file_name)
    
    def save_sequence(self):
        np.save(self._new_file_path, self._new_sequence)
        print(f"{self._new_file_name} is saved successfully.")

class NumpyFileProcs:
    def __init__(self, input_directory_path):
        self._input_directory_path = input_directory_path
        self._npy_files = self.scan_dir(self._input_directory_path)

        self._file_objs: list[NumpyArrayFile] = []

    def scan_dir(self, directory_path: str, extension_name: str = ".npy") -> list[str]:
        scans = []
        self._count_dirs()
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(extension_name):
                    scans.append(os.path.join(root, file))

        out_text = f"{directory_path} has {len(scans)} file/s."
        print("-" * (len(out_text) + 4))
        print(f"| {out_text} |")
        print("-" * (len(out_text) + 4))
        return scans

    def _count_dirs(self):
        folder = self._input_directory_path
        os.makedirs(folder, exist_ok = True)
        print("===== EXISTING LABELS =====")
        f1: list[str] = os.listdir(folder)
        if len(f1) < 1:
            print ("None.")
        else:
            for label in os.listdir(folder):
                print(f"\"{label}\": {len([f for f in os.listdir(os.path.join(folder, label)) if os.path.isfile(os.path.join(folder, label, f))])}")
        print("===== END =====")

    def create_file_objs(self):
        start = time.time()
        self._file_objs = []
        for file_path in tqdm.tqdm(self._npy_files, desc="Creating file objects", colour = 'yellow', unit = ' files', ascii = True):
            self._file_objs.append(NumpyArrayFile(file_path))

        out_text = f"{len(self._file_objs)} file object/s created in {time.time() - start:.2f} second/s."
        print("-" * (len(out_text) + 4))
        print(f"| {out_text} |")
        print("-" * (len(out_text) + 4))

    @property
    def file_objs(self):
        return self._file_objs

class HandMirror:
    
    def start_sequence_mirroring(self, file_objs: list[NumpyArrayFile]):

        start_time = time.time()
        
        for file_obj in tqdm.tqdm(file_objs, desc="Mirroring sequences", 
                                colour='green', unit='files', ascii=True):
            original_sequence = file_obj.sequence
            
            mirrored_sequence = self.safe_sequence_mirroring(original_sequence)
            
            file_obj.set_new_sequence(mirrored_sequence)
            
            file_obj.save_sequence()
        
        elapsed_time = time.time() - start_time
        out_text = f"Mirrored {len(file_objs)} sequences in {elapsed_time:.2f} seconds."
        print("-" * (len(out_text) + 4))
        print(f"| {out_text} |")
        print("-" * (len(out_text) + 4))
    
    def safe_sequence_mirroring(self, sequence: np.ndarray) -> np.ndarray:
        sequence_mirrored = []

        for frame in sequence:
            hand_l = frame[:73]
            hand_r = frame[73:]

            mirrored_l = np.zeros_like(hand_l)
            mirrored_r = np.zeros_like(hand_r)

            def mirror_hand(hand):
                # Extract components
                landmarks = hand[:63].reshape((21, 3)).copy()
                normal = hand[63:66]
                pitch, yaw = hand[66:68]
                angles = hand[68:73]

                # Mirror landmarks: flip X coordinate
                landmarks[:, 0] = 1.0 - landmarks[:, 0]

                # Flip normal vector: negate x and z components
                normal[0] *= -1
                normal[2] *= -1

                # Negate yaw (x-z plane flip), pitch stays the same
                yaw *= -1

                # Reassemble mirrored feature vector
                return np.concatenate([
                    landmarks.flatten(),
                    normal,
                    [pitch, yaw],
                    angles
                ])

            # Mirror left hand if it's non-zero (becomes right hand in mirrored frame)
            if not np.all(hand_l == 0):
                mirrored_r = mirror_hand(hand_l)

            # Mirror right hand if it's non-zero (becomes left hand in mirrored frame)
            if not np.all(hand_r == 0):
                mirrored_l = mirror_hand(hand_r)

            # Combine to new mirrored frame: [left_slot, right_slot]
            full_frame = np.concatenate([mirrored_l, mirrored_r])
            sequence_mirrored.append(full_frame)

        return np.array(sequence_mirrored, dtype=np.float32)

    def preview_mirroring_output(self, file_obj: NumpyArrayFile, frame_idx: int = 0):
        original_sequence = file_obj.sequence

        print(f"Preview file's path: {file_obj.file_path}")
        
        if frame_idx >= len(original_sequence):
            print(f"Frame index {frame_idx} is out of bounds. Sequence has {len(original_sequence)} frames.")
            return
        
        original_frame = original_sequence[frame_idx]
        mirrored_sequence = self.safe_sequence_mirroring(original_sequence)
        mirrored_frame = mirrored_sequence[frame_idx]
        
        print(f"\n===== PREVIEW FOR FRAME {frame_idx} =====")
        print(f"Original frame shape: {original_frame.shape}")
        print(f"Mirrored frame shape: {mirrored_frame.shape}")
        
        # Show left and right hand data
        orig_left = original_frame[:73]
        orig_right = original_frame[73:]
        mir_left = mirrored_frame[:73]
        mir_right = mirrored_frame[73:]
        
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
        print(original_frame)
        print("MIRRORED:")
        print(mirrored_frame)
        
        print("===== END PREVIEW =====\n")

# test
if __name__ == "__main__":
    # init the file processor
    instance = NumpyFileProcs("data/landmark_sequences")
    instance.create_file_objs()
    
    # init the hand mirror
    hand_mirror = HandMirror()
    
    # preview
    if instance.file_objs:
        print("\n===== PREVIEW BEFORE MIRRORING =====")
        hand_mirror.preview_mirroring_output(instance.file_objs[0], frame_idx=0)
    
    if input("Want to start the process? (type 'confirm'):\t") == 'confirm':
        # this will start processs
        # hand_mirror.start_sequence_mirroring(instance.file_objs)
        print("\nAll sequences have been mirrored and saved!")
    else:
        print("\nUser cancel.")
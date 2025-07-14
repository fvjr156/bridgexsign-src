#
from google.colab import drive # type: ignore
drive.mount('/content/drive')
#
import os
import numpy
import time
from tqdm import tqdm

class NPArrayFile:
    def __init__(self, file_path: str):
        dir_path, file_name = os.path.split(file_path)
        f_name, f_ext = os.path.splitext(file_name)

        self._dir_path = dir_path
        self._file_name = file_name
        self._new_file_name = f"{f_name}_flipped{f_ext}"
        self._new_file_path = os.path.join(self._dir_path, self._new_file_name)

        self._sequence: numpy.ndarray = numpy.load(file_path)
        self._new_sequence: numpy.ndarray

    @property
    def sequence(self):
        return self._sequence
    
    @property
    def new_sequence(self):
        return self._new_sequence

    @property
    def file_path(self):
        return os.path.join(self._dir_path, self._file_name)
    
    def save_sequence(self):
        numpy.save(self._new_file_path, self._new_sequence)
    
    @new_sequence.setter
    def new_sequence(self, sequence: numpy.ndarray):
        self._new_sequence = sequence

class HandMirror:
    
    def start_sequence_mirroring(self, file_objs: list[NPArrayFile]):

        start_time = time.time()
        for sequence in tqdm(file_objs, desc="Mirroring sequences", colour = 'green', unit = 'files', ascii = True):
            sequence_original = sequence.sequence
            sequence_mirrored = self.safe_sequence_mirroring(sequence_original)
            sequence.new_sequence = sequence_mirrored
            sequence.save_sequence()
        
        elapsed_time = time.time() - start_time
        out_text = f"Mirrored {len(file_objs)} sequences in {elapsed_time:.2f} seconds."
        print("-" * (len(out_text) + 4))
        print(f"| {out_text} |")
        print("-" * (len(out_text) + 4))
    
    def safe_sequence_mirroring(self, sequence: numpy.ndarray) -> numpy.ndarray:

        sequence_mirrored = []
        
        for frame in sequence:
            hand_left = frame[:73]
            hand_right = frame[73:]

            mirrored_l = numpy.zeros_like(hand_left)
            mirrored_r = numpy.zeros_like(hand_right)

            def mirror_hand(hand):
                landmarks = hand[:63].reshape((21, 3)).copy()
                normal = hand[63:66]
                pitch, yaw = hand[66:68]
                angles = hand[68:73]

                landmarks[:, 0] = 1.0 - landmarks[:, 0]
                normal[0] *= -1
                normal[2] *= -1
                yaw *= -1

                return numpy.concatenate([
                    landmarks.flatten(),
                    normal,
                    [pitch, yaw],
                    angles
                ])

            if not numpy.all(hand_left == 0):
                mirrored_r = mirror_hand(hand_left)
            if not numpy.all(hand_right == 0):
                mirrored_l = mirror_hand(hand_right)

            full_frame = numpy.concatenate([mirrored_l, mirrored_r])
            sequence_mirrored.append(full_frame)

        return numpy.array(sequence_mirrored, dtype=numpy.float32)

    def preview_mirroring_output(self, file_obj: NPArrayFile, frame_idx: int = 0):
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
        
        print(f"\nOriginal - Left hand non-zero: {not numpy.all(orig_left == 0)}")
        print(f"Original - Right hand non-zero: {not numpy.all(orig_right == 0)}")
        print(f"Mirrored - Left hand non-zero: {not numpy.all(mir_left == 0)}")
        print(f"Mirrored - Right hand non-zero: {not numpy.all(mir_right == 0)}")
        
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

def scan_dir(directory_path: str, extension_name: str = ".npy") -> list[str]:
    scans = []
    count_dirs(directory_path)
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension_name):
                scans.append(os.path.join(root, file))

    out_text = f"{directory_path} has {len(scans)} file/s."
    print("-" * (len(out_text) + 4))
    print(f"| {out_text} |")
    print("-" * (len(out_text) + 4))
    return scans

def count_dirs(dirpath):
    folder = dirpath
    os.makedirs(folder, exist_ok = True)
    print("===== EXISTING LABELS =====")
    f1: list[str] = os.listdir(folder)
    if len(f1) < 1:
        print ("None.")
    else:
        for label in os.listdir(folder):
            print(f"\"{label}\": {len([f for f in os.listdir(os.path.join(folder, label)) if os.path.isfile(os.path.join(folder, label, f))])}")
    print("===== END =====")

def create_file_objects(files):
    start = time.time()
    objs = []
    for file_path in tqdm(files, desc="Creating file objects", colour = 'yellow', unit = ' files', ascii = True):
        objs.append(NPArrayFile(file_path))

    out_text = f"{len(objs)} file object/s created in {time.time() - start:.2f} second/s."
    print("-" * (len(out_text) + 4))
    print(f"| {out_text} |")
    print("-" * (len(out_text) + 4))

    return objs
#
_input_directory_path = ("data/landmark_sequences") # replace with actual drive path
_npy_files = scan_dir(_input_directory_path)
_file_objects: list[NPArrayFile] = create_file_objects(_npy_files)
hand_mirror = HandMirror()

if _file_objects:
    print("\n============ PREVIEW ============")
    hand_mirror.preview_mirroring_output(_file_objects[0])
    print("\n========== END PREVIEW ==========\n\n")

inp = "> Want to start the process? (type 'confirm'):\t"
print('-' * (len(inp) + 10))
if input(inp) == 'confirm':
    # this will start processs
    hand_mirror.start_sequence_mirroring(_file_objects)
    print("\nAll sequences have been mirrored and saved!")
else:
    print("\nUser cancel.")

    
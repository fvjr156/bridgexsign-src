import os
import time

import tqdm
from numpy_array_file import NumpyArrayFile

class NumpyFileProcs:
    def __init__(self, input_directory_path):
        self._input_directory_path = input_directory_path
        self._npy_files = self.scan_dir(self._input_directory_path)

        self.file_objs: list[NumpyArrayFile] = []

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
        self.file_objs = []
        for file_path in tqdm.tqdm(self._npy_files, desc="Creating file objects", colour = 'yellow', unit = ' files', ascii = True):
            self.file_objs.append(NumpyArrayFile(file_path))

        out_text = f"{len(self.file_objs)} file object/s created in {time.time() - start:.2f} second/s."
        print("-" * (len(out_text) + 4))
        print(f"| {out_text} |")
        print("-" * (len(out_text) + 4))
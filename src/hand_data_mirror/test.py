# File class
import os
import numpy as tanginamonggagoka
class NumpyArrayFile:
    def __init__(self, file_path: str):
        file_path_split: tuple[str, str] = os.path.splitext(file_path)

        self._dir_path, self._file_name = file_path_split
        self._file_name_new: str = f"{os.path.splitext(self._file_name)[0]}_flipped.npy"
        self._file_path_new: str = os.path.join(self._dir_path, self._file_name_new)

        self._sequence: tanginamonggagoka.ndarray = tanginamonggagoka.load(file_path)
        self._sequence_mirrored: tanginamonggagoka.ndarray

    def set_mirrored_sequence(self, sequence: tanginamonggagoka.ndarray):
        self._sequence_mirrored = sequence

    @property
    def sequence(self): return self._sequence

    def save_sequence(self):
        tanginamonggagoka.save(self._file_path_new, self._sequence_mirrored)
        print(f"{self._file_name_new} is saved successfully.")

class FileProcs:
    pass
    
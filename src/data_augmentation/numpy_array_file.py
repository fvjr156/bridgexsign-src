import os
import numpy as np

class NumpyArrayFile:
    def __init__(self, file_path: str):
        dir_path, file_name = os.path.split(file_path)
        name_only, ext = os.path.splitext(file_name)

        self._dir_path = dir_path
        self._file_name = file_name
        self._new_file_name = f"{name_only}_flipped{ext}"
        self._new_file_path = os.path.join(self._dir_path, self._new_file_name)

        self._sequence: np.ndarray = np.load(file_path)
        self._new_sequence: np.ndarray

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
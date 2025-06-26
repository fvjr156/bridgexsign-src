import numpy as np

class Sequence:
    def __init__(self):
        self.reset()
        self.sequence_length = 30

    def reset(self):
        self.sequence = []

    def append(self, frame_vector):
        self.sequence.append(frame_vector)
    
    def is_full(self):
        return len(self.sequence) >= self.sequence_length
    
    def get_sequence(self):
        return np.array(self.sequence)

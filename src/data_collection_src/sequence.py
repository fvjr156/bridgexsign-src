import numpy as np

class Sequence:
    def __init__(self):
        self.reset()
        self.sequence_length = 20

    def reset(self):
        self.sequence = []

    def append(self, frame_vector):
        self.sequence.append(frame_vector)
    
    def is_full(self):
        return len(self.sequence) >= self.sequence_length
    
    def get_sequence(self):
        return np.array(self.sequence)
    
if __name__ == '__main__':
    for i in range(0, 100): print("DO NOT RUN THIS CODE!!! INSTEAD, RUN src/data_collection.py !!!")

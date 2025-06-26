import os
import logging
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

class UIProcess:
    @staticmethod
    def prompt_label() -> str|None:
        root = tk.Tk()
        root.withdraw()
        label: str|None = simpledialog.askstring(title = "Label Input", prompt = "Enter ASL Label:")
        if label is None:
            return None
        root.destroy()
        return label
    
class FileProcs:
    @staticmethod
    def count_dirs():
        folder = os.path.join("data", "landmark_sequences")
        os.makedirs(folder, exist_ok = True)
        print("===== EXISTING LABELS =====")
        f1: list[str] = os.listdir(folder)
        if len(f1) < 1:
            print ("None.")
        else:
            for label in os.listdir(folder):
                print(f"\"{label}\": {len([f for f in os.listdir(os.path.join(folder, label)) if os.path.isfile(os.path.join(folder, label, f))])}")
        print("===== END =====")

    @staticmethod
    def save_sequence(sequence, label):
        folder = os.path.join("data", "landmark_sequences", label)
        os.makedirs(folder, exist_ok = True)
        filename = os.path.join(
            folder, 
            f"{label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
            )
        np.save(filename, sequence)
        logging.info(f"Saved: {label} => {filename}")
        logging.info(f"Size of {filename}: {os.path.getsize(filename)}")
        logging.info(f"File count for {folder}: {len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])}")
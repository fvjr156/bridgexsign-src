import os
import shutil

class FileProcs:
    @staticmethod
    def batch_move(file_paths: list[str], out_root: str, in_root: str = './data/landmark_sequences_flipped_left') -> None:
        if not os.path.isdir(out_root):
            raise NotADirectoryError(f"The output directory '{out_root}' does not exist or is not a directory.")

        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")

            relative_path = os.path.relpath(file_path, in_root)
            dest_path = os.path.join(out_root, relative_path)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            shutil.move(file_path, dest_path)
            print(f"Moved '{file_path}' to '{dest_path}'")

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

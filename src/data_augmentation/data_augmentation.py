from hand_mirror import HandMirror
from numpy_array_file import NumpyArrayFile
from numpy_file_procs import NumpyFileProcs

if __name__ == '__main__':
    file_procs_instance = NumpyFileProcs("data copy/landmark_sequences")
    file_procs_instance.create_file_objs()

    hand_mirror = HandMirror()

    if file_procs_instance.file_objs:
        print("\n============ PREVIEW ============")
        hand_mirror.preview_mirroring_output(file_procs_instance.file_objs[0])
        print("\n========== END PREVIEW ==========\n\n")

    inp = "> Want to start the process? (type 'confirm'):\t"
    print('-' * (len(inp) + 10))
    if input(inp) == 'confirm':
        # this will start processs
        hand_mirror.start_sequence_mirroring(file_procs_instance.file_objs)
        print("\nAll sequences have been mirrored and saved!")
    else:
        print("\nUser cancel.")
import os
from pathlib import Path

def main():
    # Get user input for the directory and output file name
    dir_path = input("Enter the relative path to the directory containing .py and .json files: ")
    output_file_name = input("Enter the name of the output file (e.g., dump.txt): ")

    directory = Path(dir_path)
    output_file = Path(output_file_name)

    if not directory.exists() or not directory.is_dir():
        print("Error: The specified directory does not exist or is not a directory.")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as writer:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix in ('.py', '.json'):
                        try:
                            writer.write(f"===== {file_path.name} =====\n")
                            with open(file_path, 'r', encoding='utf-8') as reader:
                                writer.write(reader.read())
                            writer.write("\n\n")  # Separate each file's content
                        except IOError as e:
                            print(f"Error processing file {file_path}: {e.strerror}")
                        except UnicodeDecodeError:
                            print(f"Error decoding file {file_path} (might be binary)")

            print(f"Code dump completed. Output written to {output_file.absolute()}")

    except IOError as e:
        print(f"Error creating/writing to output file: {e.strerror}")

if __name__ == "__main__":
    main()
import os
import shutil


def clear_output_directory(output_dir):
    """
    Removes the output directory if it exists and creates a new empty one.
    """
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except PermissionError as e:
            print(f"PermissionError while removing {output_dir}: {e}")
            # Try to remove files inside the directory
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception as e:
                        print(f"Could not remove file {file}: {e}")
    os.makedirs(output_dir, exist_ok=True)


def combine_folders(folder1, folder2, output_folder):
    """
    Combines all files from folder1 and folder2 into output_folder.
    """
    clear_output_directory(output_folder)
    for folder in [folder1, folder2]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                src = os.path.join(folder, file)
                dst = os.path.join(output_folder, file)
                shutil.copy(src, dst)
    print(f"Combined folders {folder1} and {folder2} into {output_folder}")

    def green_print(message):
        print("\033[92m" + message + "\033[0m")


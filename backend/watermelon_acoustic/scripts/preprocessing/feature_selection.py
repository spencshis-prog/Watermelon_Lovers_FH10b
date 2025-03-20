import os
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


def load_feature_data(folder):
    """
    Loads .npy feature files from folder (only files directly in folder, not in subfolders).
    Assumes naming format: <watermelonID>_<brix>_<index>.npy.
    Returns X, y, fnames.
    """
    data, labels, fnames = [], [], []
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path) and file.lower().endswith(".npy"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            try:
                brix_val = float(parts[1])
            except Exception as e:
                print(f"[FS] Error extracting brix from {file}: {e}")
                continue
            feat = np.load(full_path)
            data.append(feat)
            labels.append(brix_val)
            fnames.append(file)
    return np.array(data), np.array(labels), fnames


def select_features_in_folder(folder_path, selector):
    """
    Loads feature data (X, y) from the given folder, applies feature selection using the provided selector,
    and writes the selected features back to the same .npy files.
    """
    X, y, fnames = load_feature_data(folder_path)
    if X.size == 0:
        print(f"[FS] No features loaded from {folder_path}")
        return

    try:
        # Apply supervised feature selection using the Brix labels.
        X_selected = selector.fit_transform(X, y)
    except Exception as e:
        print(f"[FS] Error during feature selection in {folder_path}: {e}")
        return

    # Write the selected features back to each file.
    for i, fname in enumerate(fnames):
        file_path = os.path.join(folder_path, fname)
        try:
            np.save(file_path, X_selected[i])
            print(f"[FS] Feature selected and saved: {file_path}")
        except Exception as e:
            print(f"[FS] Error saving {file_path}: {e}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the feature extraction directory since all processing occurs there.
    feature_extraction_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction")

    # Define the feature selector. Adjust k (e.g., k=50) as needed.
    selector = SelectKBest(score_func=f_regression, k=50)

    # Iterate over each noise reduction folder in the feature extraction directory.
    for nr_folder in os.listdir(feature_extraction_base_dir):
        nr_folder_path = os.path.join(feature_extraction_base_dir, nr_folder)
        if os.path.isdir(nr_folder_path):
            print(f"[FS] Entering noise reduction folder: {nr_folder_path}")
            # Iterate over each feature extraction technique subfolder.
            for fe_folder in os.listdir(nr_folder_path):
                fe_folder_path = os.path.join(nr_folder_path, fe_folder)
                if os.path.isdir(fe_folder_path):
                    print(f"[FS] Applying feature selection to folder: {fe_folder_path}")
                    select_features_in_folder(fe_folder_path, selector)


if __name__ == "__main__":
    main()
    input("Feature selection complete. Press Enter to exit...")

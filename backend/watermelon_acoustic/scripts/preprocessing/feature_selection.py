import os
import numpy as np
from matplotlib import pyplot as plt
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
                print(f"[FS] Error extracting brix from {os.path.relpath(file, os.getcwd())}: {e}")
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
        print(f"[FS] No features loaded from {os.path.relpath(folder_path, os.getcwd())}")
        return

    try:
        # Apply supervised feature selection using the Brix labels.
        X_selected = selector.fit_transform(X, y)

        scores = selector.scores_
        mask = selector.get_support()  # Boolean mask of selected features

        # Get indices of the selected top k features
        selected_indices = np.where(mask)[0]
        selected_scores = scores[mask]

        top_10_idx = np.argsort(selected_scores)[-10:][::-1]
        top_10_scores = selected_scores[top_10_idx]
        top_10_features = selected_indices[top_10_idx]  # actual column indices

        feature_names = [f"f{i}" for i in range(X.shape[1])]  # or load from CSV
        top_10_names = [feature_names[i] for i in top_10_features]

        plt.figure(figsize=(10, 6))
        plt.barh(top_10_names[::-1], top_10_scores[::-1], color='darkgreen')
        plt.xlabel("F-score")
        plt.title("Top 10 Selected Features by SelectKBest")
        plt.tight_layout()
        plt.close()

        # Wrap only scores and angles, not labels
        angles = np.linspace(0, 2 * np.pi, len(top_10_scores), endpoint=False).tolist()
        scores = top_10_scores.tolist()
        scores += scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, scores, linewidth=2, color='teal')
        ax.fill(angles, scores, alpha=0.3)

        labels = top_10_names  # Do NOT wrap labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)

        plt.title("Top 10 Feature Importance (SelectKBest)", y=1.1)
        plt.tight_layout()

        best_feat_dir = os.path.join(os.getcwd(), "best_feature_charts")
        os.makedirs(best_feat_dir, exist_ok=True)

        chart_path = os.path.join(best_feat_dir, os.path.basename(folder_path) + ".png")

        plt.savefig(chart_path)
        plt.close()

    except Exception as e:
        print(f"[FS] Error during feature selection in {os.path.relpath(folder_path, os.getcwd())}: {e}")
        return

    # Write the selected features back to each file.
    for i, fname in enumerate(fnames):
        file_path = os.path.join(folder_path, fname)
        try:
            np.save(file_path, X_selected[i])
            print(f"[FS] Feature selected and saved: {os.path.relpath(file_path, os.getcwd())}")
        except Exception as e:
            print(f"[FS] Error saving {os.path.relpath(file_path, os.getcwd())}: {e}")


def select_features(feature_extraction_base_dir, k=50):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the feature selector. Adjust k (e.g., k=50) as needed.
    selector = SelectKBest(score_func=f_regression, k=k)

    # Iterate over each noise reduction folder in the feature extraction directory.
    for nr_folder in os.listdir(feature_extraction_base_dir):
        nr_folder_path = os.path.join(feature_extraction_base_dir, nr_folder)
        if os.path.isdir(nr_folder_path):
            print(f"[FS] Entering noise reduction folder: {os.path.relpath(nr_folder_path, os.getcwd())}")
            # Iterate over each feature extraction technique subfolder.
            for fe_folder in os.listdir(nr_folder_path):
                fe_folder_path = os.path.join(nr_folder_path, fe_folder)
                if os.path.isdir(fe_folder_path):
                    print(f"[FS] Applying feature selection to folder: {os.path.relpath(fe_folder_path, base_dir)}")
                    select_features_in_folder(fe_folder_path, selector)


if __name__ == "__main__":
    select_features()
    input("[FS] Feature selection complete. Press Enter to exit...")

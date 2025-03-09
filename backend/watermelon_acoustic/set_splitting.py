import os
import shutil

import numpy as np


def load_feature_data(folder):
    """
    Loads .npy feature files from folder.
    Assumes naming format: <watermelonID>_<brix>_<index>.npy
    Returns X, y, fnames.
    (This is a local copy to use in pre-splitting holdouts.)
    """
    data, labels, fnames = [], [], []
    for file in os.listdir(folder):
        if file.lower().endswith(".npy"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            try:
                brix_val = float(parts[1])
            except:
                continue
            file_path = os.path.join(folder, file)
            feat = np.load(file_path)
            data.append(feat)
            labels.append(brix_val)
            fnames.append(file)
    return np.array(data), np.array(labels), fnames


def pre_split_holdouts(feature_extraction_base_dir, holdout_ratio=0.15):
    """
    For each feature folder (combination of noise reduction and feature extraction),
    check if a "train" and "test" subfolder exist. If not, perform one hold-out split
    using train_test_split and copy the original .npy files into the "train" and "test" subfolders.
    """
    from sklearn.model_selection import train_test_split
    for nr in os.listdir(feature_extraction_base_dir):
        nr_path = os.path.join(feature_extraction_base_dir, nr)
        if not os.path.isdir(nr_path):
            continue
        for feat in os.listdir(nr_path):
            feat_folder = os.path.join(nr_path, feat)
            train_folder = os.path.join(feat_folder, "train")
            test_folder = os.path.join(feat_folder, "test")

            rel_feat_folder = os.path.relpath(feat_folder, os.getcwd())

            # If both exist and have files, assume split already exists.
            if (os.path.exists(train_folder) and os.path.exists(test_folder) and
                    os.listdir(train_folder) and os.listdir(test_folder)):
                print(f"[SS] Hold-out split already exists in {rel_feat_folder}")
                continue

            data, labels, fnames = load_feature_data(feat_folder)
            if len(data) == 0:
                continue

            X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
                data, labels, fnames, test_size=holdout_ratio, random_state=42
            )

            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
            for file in fn_train:
                src = os.path.join(feat_folder, file)
                dst = os.path.join(train_folder, file)
                shutil.copy(src, dst)
            for file in fn_test:
                src = os.path.join(feat_folder, file)
                dst = os.path.join(test_folder, file)
                shutil.copy(src, dst)
            print(f"[SS] In {rel_feat_folder}: {len(X_train)} train, {len(X_test)} test samples.")

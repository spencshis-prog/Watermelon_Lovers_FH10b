import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


def transform_features_in_folder(folder_path, pipeline):
    """
    Reads all .npy files in folder_path, stacks them into a matrix,
    fits the transformation pipeline, transforms them, and writes the transformed
    features back to the same files (overwriting).

    Args:
        folder_path (str): Directory containing the extracted feature .npy files.
        pipeline (Pipeline): A scikit-learn pipeline to transform the features.
    """
    # Collect all .npy files in the folder
    npy_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith('.npy')]
    if not npy_files:
        print(f"[FG] No .npy files found in {folder_path}")
        return

    features_list = []
    file_names = []
    for file in npy_files:
        try:
            features = np.load(file)
            # Ensure each file's features are a 2D array (each row is one sample)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features_list.append(features)
            file_names.append(file)
        except Exception as e:
            print(f"[FG] Error loading {file}: {e}")

    try:
        # Stack all features vertically
        X = np.vstack(features_list)
    except Exception as e:
        print(f"[FG] Error stacking features in {folder_path}: {e}")
        return

    # Fit and transform the features using the provided pipeline
    try:
        X_transformed = pipeline.fit_transform(X)
    except Exception as e:
        print(f"[FG] Error during transformation in {folder_path}: {e}")
        return

    # Split the transformed data back into the original file groupings and overwrite each file.
    start = 0
    for file, orig_features in zip(file_names, features_list):
        n_samples = orig_features.shape[0]
        transformed_subset = X_transformed[start:start + n_samples, :]
        start += n_samples
        # If the original file contained a single sample, save as a 1D array.
        if n_samples == 1:
            transformed_subset = transformed_subset.flatten()
        try:
            np.save(file, transformed_subset)
            print(f"[FG] Transformed and saved: {file}")
        except Exception as e:
            print(f"[FG] Error saving {file}: {e}")


def generate_features():
    # Path to the feature extraction output directory (adjust as needed)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_extraction_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction")

    # Define the transformation pipeline:
    # 1. Yeo-Johnson power transform to reduce skewness.
    # 2. RobustScaler to mitigate the influence of outliers.
    # 3. PolynomialFeatures (degree=2) to generate interaction and squared features.
    transformation_pipeline = Pipeline([
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
    ])

    # Iterate over each noise reduction folder (NR folder)
    for nr_folder in os.listdir(feature_extraction_base_dir):
        nr_folder_path = os.path.join(feature_extraction_base_dir, nr_folder)
        if os.path.isdir(nr_folder_path):
            print(f"[FG] Entering noise reduction folder: {nr_folder_path}")
            # Iterate over each feature extraction folder (FE folder) inside the NR folder
            for fe_folder in os.listdir(nr_folder_path):
                fe_folder_path = os.path.join(nr_folder_path, fe_folder)
                if os.path.isdir(fe_folder_path):
                    print(f"[FG] Applying feature transformation to folder: {fe_folder_path}")
                    transform_features_in_folder(fe_folder_path, transformation_pipeline)


if __name__ == "__main__":
    generate_features()

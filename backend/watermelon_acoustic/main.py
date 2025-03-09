import os
import shutil

import pipeline_lr

# Configuration booleans
USE_QILIN = True
USE_LAB = False

USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
# VAL_SPLIT_RATIO = 0.15  # fraction for validation (e.g. 15%), set to 0 for final model
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model

# major pipeline stages
PREPROCESS = True
LINEAR_REGRESSION = True
RANDOM_FOREST = False
XGBOOST = False
NEURAL_NETWORK = False


# TODO: combined testing will try every dataset-noise reduction combination perhaps eliminates dataset-
# TODO: dimension below unless one dataset strictly worsens the medley
# TODO: sort both kfold and holdout report entry orders so they are easily comparable
# perhaps add something that exclusively pulls validation and test from lab dataset
# perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset)

def clear_output_directory(output_dir):
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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step i: Reformat Qilin dataset (m4a to .wav)
    qilin_dataset_dir = os.path.join(base_dir, "input", "qilin_dataset", "19_datasets")
    qilin_preprocess_dir = os.path.join(base_dir, "input", "wav_qilin")
    if USE_QILIN:
        print("Starting Qilin dataset preprocessing...")
        from wav_file_converter import convert_qilin_file_formats_to_wav
        convert_qilin_file_formats_to_wav(qilin_dataset_dir, qilin_preprocess_dir)
    else:
        print("Skipping Qilin dataset preprocessing.")

    # --- Step ii: Standardize .wav files ---
    from standardize_wav import standardize_wav_files
    qilin_standard_dir = os.path.join(base_dir, "intermediate", "standard_qilin")
    if USE_QILIN:
        print("Standardizing Qilin dataset...")
        standardize_wav_files(qilin_preprocess_dir, qilin_standard_dir)
    if USE_LAB:
        lab_dataset_dir = os.path.join(base_dir, "input", "wav_lab")
        lab_standard_dir = os.path.join(base_dir, "intermediate", "standard_lab")
        print("Standardizing Lab dataset...")
        standardize_wav_files(lab_dataset_dir, lab_standard_dir)

    # --- Step iii: Combine datasets if needed ---
    combined_standard_dir = os.path.join(base_dir, "intermediate", "combined_standard")
    if USE_QILIN and USE_LAB:
        combine_folders(qilin_standard_dir, lab_standard_dir, combined_standard_dir)
    elif USE_QILIN:
        combined_standard_dir = qilin_standard_dir
    elif USE_LAB:
        combined_standard_dir = lab_standard_dir
    else:
        print("No dataset selected for training. Please select one.")
        return

    # --- Step iv: Noise Reduction ---
    from noise_reduction import apply_noise_reduction
    noise_reduction_dir = os.path.join(base_dir, "intermediate", "noise_reduction")

    clear_output_directory(noise_reduction_dir)

    apply_noise_reduction(combined_standard_dir, noise_reduction_dir)

    # --- Step v: Feature Extraction ---
    # For each noise reduction technique folder, apply feature extraction.
    from feature_extraction import apply_feature_extraction
    feature_extraction_base_dir = os.path.join(base_dir, "intermediate", "feature_extraction")
    clear_output_directory(feature_extraction_base_dir)
    # Iterate over each technique folder (e.g., technique1, technique2, etc.)
    for technique in os.listdir(noise_reduction_dir):
        technique_path = os.path.join(noise_reduction_dir, technique)
        if os.path.isdir(technique_path):
            # Create a corresponding folder for feature extraction output for this technique.
            output_feat_dir = os.path.join(feature_extraction_base_dir, technique)
            clear_output_directory(output_feat_dir)
            # Apply feature extraction (this function should create subfolders for each feature method)
            apply_feature_extraction(technique_path, output_feat_dir)

    # --- Step vi: Set Splitting ---
    from set_splitting import pre_split_holdouts
    pre_split_holdouts(feature_extraction_base_dir, holdout_ratio=TEST_SPLIT_RATIO)

    # Initiate linear regression pipeline - train and tests LR model for each regularization technique
    if LINEAR_REGRESSION:
        pipeline_lr.main(feature_extraction_base_dir=feature_extraction_base_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import os

import functions


def proceed(USE_QILIN=True, USE_LAB=False, USE_SEPARATE_TEST=False, TEST_SPLIT_RATIO=0.15):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step i: Reformat Qilin dataset (e.g., m4a to .wav)
    qilin_dataset_dir = os.path.join(base_dir, "../../input", "qilin_dataset", "19_datasets")
    qilin_preprocess_dir = os.path.join(base_dir, "../../input", "wav_qilin")
    if USE_QILIN:
        print("Starting Qilin dataset preprocessing...")
        from scripts.preprocessing.wav_file_converter import convert_qilin_file_formats_to_wav
        convert_qilin_file_formats_to_wav(qilin_dataset_dir, qilin_preprocess_dir)
    else:
        print("Skipping Qilin dataset preprocessing.")

    # Step ii: Standardize .wav files
    from scripts.preprocessing.standardize_wav import standardize_wav_files
    qilin_standard_dir = os.path.join(base_dir, "../../intermediate", "standard_qilin")
    if USE_QILIN:
        print("Standardizing Qilin dataset...")
        standardize_wav_files(qilin_preprocess_dir, qilin_standard_dir)
    if USE_LAB:
        lab_dataset_dir = os.path.join(base_dir, "../../input", "wav_lab")
        lab_standard_dir = os.path.join(base_dir, "../../intermediate", "standard_lab")
        print("Standardizing Lab dataset...")
        standardize_wav_files(lab_dataset_dir, lab_standard_dir)

    # Step iii: Combine datasets (if using both)
    combined_standard_dir = os.path.join(base_dir, "../../intermediate", "combined_standard")
    if USE_QILIN and USE_LAB:
        functions.combine_folders(qilin_standard_dir, lab_standard_dir, combined_standard_dir)
    elif USE_QILIN:
        combined_standard_dir = qilin_standard_dir
    elif USE_LAB:
        combined_standard_dir = lab_standard_dir
    else:
        print("No dataset selected for training. Please select at least one.")
        return

    # Step iv: Noise Reduction
    from scripts.preprocessing.noise_reduction import apply_noise_reduction
    noise_reduction_dir = os.path.join(base_dir, "../../intermediate", "noise_reduction")
    functions.clear_output_directory(noise_reduction_dir)
    apply_noise_reduction(combined_standard_dir, noise_reduction_dir)

    # Step v: Feature Extraction
    from scripts.preprocessing.feature_extraction import apply_feature_extraction
    feature_extraction_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction")
    functions.clear_output_directory(feature_extraction_base_dir)
    # Process each noise reduction technique folder separately
    for technique in os.listdir(noise_reduction_dir):
        technique_path = os.path.join(noise_reduction_dir, technique)
        if os.path.isdir(technique_path):
            output_feat_dir = os.path.join(feature_extraction_base_dir, technique)
            functions.clear_output_directory(output_feat_dir)
            apply_feature_extraction(technique_path, output_feat_dir)

    # Step vi: Set Splitting (e.g., creating a holdout/test set)
    from scripts.preprocessing.set_splitting import pre_split_holdouts
    pre_split_holdouts(feature_extraction_base_dir, holdout_ratio=TEST_SPLIT_RATIO)

    print("Preprocessing complete. Ready for the model training.")


if __name__ == "__main__":
    proceed()

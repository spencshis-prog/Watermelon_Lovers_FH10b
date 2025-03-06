import os
import shutil

# Configuration booleans
USE_QILIN = True
USE_LAB = False

USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
VAL_SPLIT_RATIO = 0.15   # fraction for validation (e.g. 15%)
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%)
# TODO: fix test train split when USE_SEPARATE_TEST_SET is flagged False
# TODO: combined testing also compares models trained on individual datasets for reference
# perhaps add something that exclusively pulls validation and test from lab dataset
# perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset)


def clear_output_directory(output_dir):
    """
    Deletes the entire output_dir folder (if it exists)
    and recreates it empty.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the folder and all its contents
    os.makedirs(output_dir)  # Recreate the empty directory


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

    # Define paths for Qilin dataset processing
    qilin_dataset_dir = os.path.join(base_dir, "qilin_dataset", "19_datasets")
    qilin_preprocess_dir = os.path.join(base_dir, "output", "qilin_wav")
    qilin_standard_dir = os.path.join(base_dir, "output", "qilin_standard")

    # Define paths for Lab dataset (which we assume to already be in .wav format)
    lab_dataset_dir = os.path.join(base_dir, "lab_dataset")  # Place your lab files here
    lab_standard_dir = os.path.join(base_dir, "output", "lab_standard")

    # Step i: Reformat Qilin dataset (m4a to .wav)
    if USE_QILIN:
        print("Starting Qilin dataset preprocessing...")
        from wav_file_converter import convert_qilin_file_formats_to_wav
        convert_qilin_file_formats_to_wav(qilin_dataset_dir, qilin_preprocess_dir)
    else:
        print("Skipping Qilin dataset preprocessing.")

    # Step ii: Standardizing .wav file properties to meet Qilin conditions
    from standardize_wav import standardize_wav_files
    if USE_QILIN:
        print("Standardizing Qilin dataset...")
        standardize_wav_files(qilin_preprocess_dir, qilin_standard_dir)
    if USE_LAB:
        print("Standardizing Lab dataset...")
        standardize_wav_files(lab_dataset_dir, lab_standard_dir)

    # Combine datasets if both are used (for later noise reduction)
    combined_standard_dir = os.path.join(base_dir, "output", "combined_standard")
    if USE_QILIN and USE_LAB:
        combine_folders(qilin_standard_dir, lab_standard_dir, combined_standard_dir)
    elif USE_QILIN:
        combined_standard_dir = qilin_standard_dir
    elif USE_LAB:
        combined_standard_dir = lab_standard_dir
    else:
        print("No dataset selected for training. Please select one.")
        return

    # Step iii: Noise reduction – apply techniques to the combined standardized data
    from noise_reduction import apply_noise_reduction
    noise_reduction_base = os.path.join(base_dir, "output", "noise_reduction", "combined")

    clear_output_directory(noise_reduction_base)

    apply_noise_reduction(combined_standard_dir, noise_reduction_base)

    # Step iv: Model training – train a model for each noise reduction technique grouping
    from model_training import train_models
    models_output_dir = os.path.join(base_dir, "output", "models")
    train_models(
        noise_reduction_base,
        models_output_dir,
        use_separate_test_set=USE_SEPARATE_TEST_SET,
        val_split=VAL_SPLIT_RATIO,
        test_split=TEST_SPLIT_RATIO
    )

    # Step v: Testing – run tests on each model, output metrics and charts
    from model_testing import run_tests
    testing_output_dir = os.path.join(base_dir, "output", "testing")
    clear_output_directory(testing_output_dir)
    report_file = os.path.join(testing_output_dir, "test_report.txt")
    run_tests(models_output_dir, noise_reduction_base, report_file)


if __name__ == "__main__":
    main()

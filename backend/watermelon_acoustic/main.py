import os
import shutil

# Configuration booleans
USE_QILIN = True
USE_LAB = False

USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
VAL_SPLIT_RATIO = 0.15  # fraction for validation (e.g. 15%), set to 0 for final model
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model
EPOCHS = 20  # 20 is recommended, use less for model_testing debugging


# TODO: combined testing will try every dataset-noise reduction combination perhaps eliminates dataset
# TODO: dimension below unless one dataset strictly worsens the medley
# perhaps add something that exclusively pulls validation and test from lab dataset
# perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset
# TODO: implement the following facets to be tested on lab condition datasets
'''
dimensions = {
    "dataset": ["qilin", "lab", "combined"],
    "noise_reduction": ["raw", "bandpass", "spectral sub", "wavelet"],
    "model": {
        "LinearRegression": {
            "hp_tuning": ["default"]  # Linear regression might not need extensive tuning
        },
        "RandomForest": {
            "hp_tuning": ["default", "grid search", "random search"]
        },
        "XGBoost": {
            "hp_tuning": ["default", "grid search", "random search", "bayesian"]
        },
        "NeuralNetwork": {
            "hp_tuning": ["default", "grid search", "random search", "bayesian"]
        }
    }
}
'''


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
    noise_reduction_dir = os.path.join(base_dir, "output", "noise_reduction", "combined")

    clear_output_directory(noise_reduction_dir)

    apply_noise_reduction(combined_standard_dir, noise_reduction_dir)

    # Step iv: Model training – train a model for each noise reduction technique grouping
    from model_training import train_all_techniques
    models_output_dir = os.path.join(base_dir, "output", "models")
    # clear_output_directory(models_output_dir)
    train_all_techniques(
        noise_reduction_base_dir=noise_reduction_dir,
        models_output_dir=models_output_dir,
        test_ratio=TEST_SPLIT_RATIO,
        val_ratio=VAL_SPLIT_RATIO,
        epochs=EPOCHS,
        batch_size=16
    )

    # Step v: Testing – run tests on each model, output metrics and charts
    from model_testing import test_all_techniques
    testing_output_dir = os.path.join(base_dir, "output", "testing")
    clear_output_directory(testing_output_dir)
    report_file = os.path.join(testing_output_dir, "test_report.txt")
    test_all_techniques(models_output_dir, noise_reduction_dir, report_file)


if __name__ == "__main__":
    main()

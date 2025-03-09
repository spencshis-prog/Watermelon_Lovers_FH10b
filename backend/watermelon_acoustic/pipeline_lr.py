import os
import shutil

# Configuration booleans and parameters
USE_QILIN = True
USE_LAB = False
EPOCHS = 20
BATCH_SIZE = 16
HOLDOUT_RATIO = 0.15
N_SPLITS = 5

REG_OPTIONS = ["none", "lasso", "ridge", "ElasticNet"]


def clear_output_directory(output_dir):
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Error clearing {output_dir}: {e}")
    os.makedirs(output_dir, exist_ok=True)


def combine_folders(folder1, folder2, output_folder):
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

    # Assume Steps i-v (preprocessing, standardizing, noise reduction, feature extraction)
    # have already been executed and the results are in the following directories:
    qilin_standard_dir = os.path.join(base_dir, "output", "qilin_standard")
    lab_standard_dir = os.path.join(base_dir, "output", "lab_standard")
    combined_standard_dir = os.path.join(base_dir, "output", "combined_standard")
    noise_reduction_dir = os.path.join(base_dir, "output", "noise_reduction", "combined")
    feature_extraction_base_dir = os.path.join(base_dir, "output", "feature_extraction")

    # (You can call your preprocessing modules here if needed)

    # --- Step vi: LR Model Training ---
    from lr_training import kfold_train_all_feature_models
    models_output_dir = os.path.join(base_dir, "output", "models_lr")
    clear_output_directory(models_output_dir)
    # For each regularization option, run the training pipeline.
    for reg in REG_OPTIONS:
        print(f"\n=== Training LR models with regularization: {reg} ===")
        kfold_train_all_feature_models(
            feature_extraction_base_dir=feature_extraction_base_dir,
            models_output_dir=models_output_dir,
            holdout_ratio=HOLDOUT_RATIO,
            n_splits=N_SPLITS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            regularization=reg
        )

    # --- Step vii: LR Model Testing ---
    from lr_testing import test_all_feature_models
    testing_output_dir = os.path.join(base_dir, "output", "testing_lr")
    clear_output_directory(testing_output_dir)
    report_path = os.path.join(testing_output_dir, "lr_report.txt")
    print("\n=== Testing LR models on hold-out test sets ===")
    test_all_feature_models(models_output_dir, feature_extraction_base_dir, report_path)


if __name__ == "__main__":
    main()

import os
import shutil

from lr_training import kfold_train_all_feature_models
from lr_testing import test_all_lr_models

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


def main(feature_extraction_base_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if feature_extraction_base_dir is None:
        feature_extraction_base_dir = os.path.join(base_dir, "intermediate", "feature_extraction")

    models_output_dir = os.path.join(base_dir, "output", "models_lr")
    testing_output_dir = os.path.join(base_dir, "output", "testing_lr")
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")

    open(report_kfold_path, "w").close()  # clearing the report_kfold.txt
    open(report_holdout_path, "w").close()

    # Clear model and testing directories.
    if os.path.exists(models_output_dir):
        shutil.rmtree(models_output_dir)
    os.makedirs(models_output_dir, exist_ok=True)
    if os.path.exists(testing_output_dir):
        shutil.rmtree(testing_output_dir)
    os.makedirs(testing_output_dir, exist_ok=True)

    # Train final models for each (NR, FE) combination for each regularization option.
    for reg in sorted(REG_OPTIONS):
        print(f"\n=== Training LR models with regularization: {reg} ===")
        kfold_train_all_feature_models(
            feature_extraction_base_dir=feature_extraction_base_dir,
            models_output_dir=models_output_dir,
            report_kfold_path=report_kfold_path,
            holdout_ratio=0.15,
            n_splits=5,
            epochs=20,
            batch_size=16,
            regularization=reg
        )

    # Test models on the hold-out test sets and produce visualizations.
    print("\n=== Testing LR models on hold-out test sets ===")
    test_all_lr_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    print("LR pipeline completed.")


if __name__ == "__main__":
    main()

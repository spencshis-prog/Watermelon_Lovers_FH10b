import os
import shutil

import functions
from scripts_linear_regression.lr_training import kfold_train_all_feature_models
from scripts_linear_regression.lr_testing import test_all_lr_models

# Configuration booleans and parameters

REG_OPTIONS = ["none", "lasso", "ridge", "ElasticNet"]


def proceed(TRAIN_NEW_MODELS=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_extraction_base_dir = os.path.join(base_dir, "../intermediate", "feature_extraction")

    models_output_dir = os.path.join(base_dir, "../output", "models_lr")
    testing_output_dir = os.path.join(base_dir, "../output", "testing_lr")
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")

    if TRAIN_NEW_MODELS:
        # clear the report .txt files
        open(report_kfold_path, "w").close()
        # Clear the model directory
        functions.clear_output_directory(models_output_dir)

        # Train final models for each (NR, FE) combination for each regularization option.
        for reg in sorted(REG_OPTIONS):
            print(f"\n=== Training LR models with regularization: {reg} ===")
            kfold_train_all_feature_models(
                feature_extraction_base_dir=feature_extraction_base_dir,
                models_output_dir=models_output_dir,
                report_kfold_path=report_kfold_path,
                holdout_ratio=0.15,  # already only splits a test when not pre-split from preprocessing
                n_splits=5,
                epochs=20,
                batch_size=16,
                regularization=reg
            )

    open(report_holdout_path, "w").close()

    functions.clear_output_directory(os.path.join(testing_output_dir, "heatmaps"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "predicted_vs_actual"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "residual_plots"))

    # Test models on the hold-out test sets and produce visualizations.
    print("\n=== Testing LR models on hold-out test sets ===")
    test_all_lr_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    print("LR pipeline completed.")


if __name__ == "__main__":
    proceed()

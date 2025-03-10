import os
import shutil

import functions
# Import the RF training and testing functions from your RF modules.
from scripts_random_forest.rf_training import kfold_train_all_feature_models_rf
from scripts_random_forest.rf_testing import test_all_rf_models

# Configuration parameters: Hyperparameter tuning options for RF.
HT_OPTIONS = ["default", "grid", "random"]


def proceed(TRAIN_NEW_MODELS=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The feature extraction outputs (with holdout sets already created)
    feature_extraction_base_dir = os.path.join(base_dir, "../intermediate", "feature_extraction")

    # Directories where RF models and test outputs will be stored.
    models_output_dir = os.path.join(base_dir, "../output", "models_rf")
    testing_output_dir = os.path.join(base_dir, "../output", "testing_rf")

    # Report files for K-Fold and hold-out evaluations.
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")

    if TRAIN_NEW_MODELS:
        open(report_kfold_path, "w").close()  # Clear K-Fold report.

        functions.clear_output_directory(models_output_dir)

        # Train final RF models for each (NR, FE) combination for each hyperparameter tuning option.
        for ht in sorted(HT_OPTIONS):
            print(f"\n=== Training RF models with hyperparameter tuning: {ht} ===")
            kfold_train_all_feature_models_rf(
                feature_extraction_base_dir=feature_extraction_base_dir,
                models_output_dir=models_output_dir,
                report_kfold_path=report_kfold_path,
                holdout_ratio=0.15,  # This splits off a hold-out test set if not already pre-split.
                n_splits=5,
                epochs=20,
                batch_size=16,
                hyper_tuning=ht  # Pass the current hyperparameter tuning option.
            )

    open(report_holdout_path, "w").close()  # Clear hold-out report.

    # clear the testing directory
    functions.clear_output_directory(os.path.join(testing_output_dir, "heatmaps"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "predicted_vs_actual"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "residual_plots"))

    # Test the trained RF models on the hold-out test sets and produce visualizations.
    print("\n=== Testing RF models on hold-out test sets ===")
    test_all_rf_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    print("RF pipeline completed.")


if __name__ == "__main__":
    proceed()

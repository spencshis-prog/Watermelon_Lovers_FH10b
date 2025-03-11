import os
import shutil
import functions
from scripts_extra_trees.et_training import kfold_train_all_feature_models_et
from scripts_extra_trees.et_testing import test_all_et_models

# Configuration parameters: Hyperparameter tuning options for ExtraTrees.
HT_OPTIONS = ["default", "grid", "random"]


def proceed(TRAIN_NEW_MODELS=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Feature extraction outputs (with hold-out sets already created)
    feature_extraction_base_dir = os.path.join(base_dir, "../intermediate", "feature_extraction")

    # Directories for ExtraTrees models and testing outputs.
    models_output_dir = os.path.join(base_dir, "../output", "models_et")
    testing_output_dir = os.path.join(base_dir, "../output", "testing_et")

    # Report files for K-Fold and hold-out evaluations.
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")

    if TRAIN_NEW_MODELS:
        open(report_kfold_path, "w").close()  # Clear K-Fold report.
        functions.clear_output_directory(models_output_dir)

        # Train final ExtraTrees models for each (NR, FE) combination for each hyperparameter tuning option.
        for ht in sorted(HT_OPTIONS):
            functions.green_print(f"\n=== Training ExtraTrees models with hyperparameter tuning: {ht} ===")
            kfold_train_all_feature_models_et(
                feature_extraction_base_dir=feature_extraction_base_dir,
                models_output_dir=models_output_dir,
                report_kfold_path=report_kfold_path,
                holdout_ratio=0.15,  # Splits off a hold-out test set if not already pre-split.
                n_splits=5,
                epochs=20,
                batch_size=16,
                hyper_tuning=ht  # Pass the current hyperparameter tuning option.
            )

    open(report_holdout_path, "w").close()  # Clear hold-out report.

    # Clear the testing output subdirectories.
    functions.clear_output_directory(os.path.join(testing_output_dir, "heatmaps"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "predicted_vs_actual"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "residual_plots"))

    # Test the trained ExtraTrees models on the hold-out test sets and produce visualizations.
    print("\n=== Testing ExtraTrees models on hold-out test sets ===")
    test_all_et_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    print("ExtraTrees pipeline completed.")


if __name__ == "__main__":
    proceed()

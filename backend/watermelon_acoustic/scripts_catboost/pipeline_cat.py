import os
import shutil
import functions
# Import the CatBoost training and testing functions from your CatBoost modules.
from scripts_catboost.cat_training import kfold_train_all_feature_models_cat
from scripts_catboost.cat_testing import test_all_cat_models

# Configuration parameters: Hyperparameter tuning options for CatBoost.
HT_OPTIONS = ["default", "grid", "random", "bayesian"]


def proceed(TRAIN_NEW_MODELS=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The feature extraction outputs (with holdout sets already created)
    feature_extraction_base_dir = os.path.join(base_dir, "../intermediate", "feature_extraction")

    # Directories where CatBoost models and test outputs will be stored.
    models_output_dir = os.path.join(base_dir, "../output", "models_cat")
    testing_output_dir = os.path.join(base_dir, "../output", "testing_cat")

    # Report files for K-Fold and hold-out evaluations.
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")
    report_params_path = os.path.join(testing_output_dir, "report_params.txt")

    if TRAIN_NEW_MODELS:
        # Clear the old K-Fold report
        open(report_kfold_path, "w").close()

        # Clear models directory
        functions.clear_output_directory(models_output_dir)

        # Train final CatBoost models for each (NR, FE) combination for each hyperparameter tuning option.
        for ht in sorted(HT_OPTIONS):
            functions.green_print(f"\n=== Training CatBoost models with hyperparameter tuning: {ht} ===")
            kfold_train_all_feature_models_cat(
                feature_extraction_base_dir=feature_extraction_base_dir,
                models_output_dir=models_output_dir,
                report_kfold_path=report_kfold_path,
                holdout_ratio=0.15,  # Splits off a hold-out test set if not already pre-split.
                n_splits=5,
                epochs=20,  # Not used internally (for a uniform interface).
                batch_size=16,  # Not used internally.
                hyper_tuning=ht  # Pass the current hyperparameter tuning option.
            )

    # Clear (or create) hold-out report.
    open(report_holdout_path, "w").close()
    open(report_params_path, "w").close()

    functions.generate_report_params(models_output_dir, report_params_path)

    # Clear the testing subdirectories.
    functions.clear_output_directory(os.path.join(testing_output_dir, "heatmaps"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "predicted_vs_actual"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "residual_plots"))

    # Test the trained CatBoost models on the hold-out test sets and produce visualizations.
    print("\n=== Testing CatBoost models on hold-out test sets ===")
    test_all_cat_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    print("CatBoost pipeline completed.")


if __name__ == "__main__":
    proceed()

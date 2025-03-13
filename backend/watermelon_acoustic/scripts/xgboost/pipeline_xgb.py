import os

import functions
from scripts.xgboost.xgb_optuna import kfold_train_feature_set_xgb_optuna
# Import the XGB training and testing functions from your XGB modules
from scripts.xgboost.xgb_training import kfold_train_all_feature_models_xgb
from scripts.xgboost.xgb_testing import test_all_xgb_models

# Configuration parameters: Hyperparameter tuning options for XGB
HT_OPTIONS = ["default", "grid", "random", "bayesian", "optuna"]


def proceed(TRAIN_NEW_MODELS=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The feature extraction outputs (with holdout sets already created)
    feature_extraction_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction")

    # Directories where XGB models and test outputs will be stored.
    models_output_dir = os.path.join(base_dir, "../../output", "models_xgb")
    testing_output_dir = os.path.join(base_dir, "../../output", "testing_xgb")

    # Report files for K-Fold and hold-out evaluations.
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")
    report_params_path = os.path.join(testing_output_dir, "report_params.txt")

    if TRAIN_NEW_MODELS:
        # Clear the old K-Fold report
        open(report_kfold_path, "w").close()

        # Clear models directory
        functions.clear_output_directory(models_output_dir)

        # Train final XGB models for each (NR, FE) combination for each hyperparameter tuning option
        for ht in sorted(HT_OPTIONS):
            functions.green_print(f"\n=== Training XGBoost models with hyperparameter tuning: {ht} ===")
            if ht == "optuna":
                # For Optuna, call the specialized function with an additional parameter for n_trials.
                kfold_train_feature_set_xgb_optuna(
                    feature_folder=feature_extraction_base_dir,
                    models_output_dir=models_output_dir,
                    holdout_ratio=0.15,
                    n_splits=5,
                    n_trials=50  # Adjust number of Optuna trials as needed
                )
            else:
                # Call your existing kfold_train_all_feature_models_xgb for other hyper_tuning options.
                from xgb_training import kfold_train_all_feature_models_xgb
                kfold_train_all_feature_models_xgb(
                    feature_extraction_base_dir=feature_extraction_base_dir,
                    models_output_dir=models_output_dir,
                    report_kfold_path=report_kfold_path,
                    holdout_ratio=0.15,
                    n_splits=5,
                    epochs=20,
                    batch_size=16,
                    hyper_tuning=ht
                )

    # Clear or create holdout report
    open(report_holdout_path, "w").close()
    open(report_params_path, "w").close()

    functions.generate_report_params(models_output_dir, report_params_path)

    # Clear the testing subdirectories
    functions.clear_output_directory(os.path.join(testing_output_dir, "heatmaps"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "predicted_vs_actual"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "residual_plots"))

    # Test the trained XGB models on the hold-out test sets and produce visualizations
    print("\n=== Testing XGBoost models on hold-out test sets ===")
    test_all_xgb_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    print("XGBoost pipeline completed.")


if __name__ == "__main__":
    proceed()

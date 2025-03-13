import os
import functions
from lgbm_training import kfold_train_all_feature_models_lgbm
from lgbm_testing import test_all_lgbm_models
from scripts.lightgbm.lgbm_optuna import kfold_train_feature_set_lgbm_optuna

# Hyperparameter tuning options for LightGBM
HT_OPTIONS = ["default", "grid", "random", "bayesian", "optuna"]


def proceed(TRAIN_NEW_MODELS=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Feature extraction outputs (with hold-out sets already created)
    feature_extraction_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction")

    # Directories where LightGBM models and test outputs will be stored.
    models_output_dir = os.path.join(base_dir, "../../output", "models_lgbm")
    testing_output_dir = os.path.join(base_dir, "../../output", "testing_lgbm")

    # Report files for K-Fold and hold-out evaluations.
    report_kfold_path = os.path.join(testing_output_dir, "report_kfold.txt")
    report_holdout_path = os.path.join(testing_output_dir, "report_holdout.txt")
    report_params_path = os.path.join(testing_output_dir, "report_params.txt")

    if TRAIN_NEW_MODELS:
        # Clear the old K-Fold report and models directory.
        open(report_kfold_path, "w").close()
        functions.clear_output_directory(models_output_dir)

        # Train final LightGBM models for each (NR, FE) combination for each hyperparameter tuning option.
        for ht in sorted(HT_OPTIONS):
            functions.green_print(f"\n=== Training LightGBM models with hyperparameter tuning: {ht} ===")
            if ht == "optuna":
                # Call the dedicated Optuna routine.
                kfold_train_feature_set_lgbm_optuna(
                    feature_folder=feature_extraction_base_dir,
                    models_output_dir=models_output_dir,
                    holdout_ratio=0.15,  # Splits off a hold-out test set if not already pre-split.
                    n_splits=5,
                    n_trials=50  # Adjust the number of Optuna trials as needed.
                )
            else:
                kfold_train_all_feature_models_lgbm(
                    feature_extraction_base_dir=feature_extraction_base_dir,
                    models_output_dir=models_output_dir,
                    report_kfold_path=report_kfold_path,
                    holdout_ratio=0.15,  # Splits off a hold-out test set if not already pre-split.
                    n_splits=5,
                    hyper_tuning=ht
                )

    # Clear or create hold-out and parameter reports.
    open(report_holdout_path, "w").close()
    open(report_params_path, "w").close()
    functions.generate_report_params(models_output_dir, report_params_path)

    # Clear testing output subdirectories.
    functions.clear_output_directory(os.path.join(testing_output_dir, "heatmaps"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "predicted_vs_actual"))
    functions.clear_output_directory(os.path.join(testing_output_dir, "residual_plots"))

    # Test the trained LightGBM models on the hold-out test sets and produce visualizations.
    functions.green_print("\n=== Testing LightGBM models on hold-out test sets ===")
    test_all_lgbm_models(models_output_dir, feature_extraction_base_dir, report_holdout_path)
    functions.green_print("LightGBM pipeline completed.")


if __name__ == "__main__":
    proceed()

# xgb_optuna.py
import os
import numpy as np
import math
import shutil
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna
import joblib

import functions  # your shared helper functions (e.g., load_feature_data, clear_output_directory)


# Optionally: import params if you want to reference default ranges

def objective_xgb(trial, X, y):
    """
    Optuna objective function for tuning an XGBRegressor.
    """
    # Define the search space
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
    gamma = trial.suggest_float("gamma", 0, 0.5)
    reg_alpha = trial.suggest_float("reg_alpha", 0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1, 2.0)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)

    # Build model with suggested hyperparameters
    model = XGBRegressor(random_state=42, eval_metric='rmse',
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         subsample=subsample)

    # 3-fold cross-validation for the current trial
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        cv_scores.append(mse)
    return np.mean(cv_scores)


def build_xgb_optuna_model(X, y, n_trials=50):
    """
    Runs an Optuna study to find the best hyperparameters for XGBRegressor,
    trains a model on all provided data, and returns the trained model and best parameters.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    print("[XGB-Optuna] Best hyperparameters:", best_params)
    best_model = XGBRegressor(random_state=42, eval_metric='rmse', **best_params)
    best_model.fit(X, y, verbose=False)
    return best_model, best_params


def kfold_train_and_evaluate_xgb_optuna(X, y, n_splits=5, n_trials=50):
    """
    Performs K-fold cross-validation using an Optuna-tuned XGBRegressor.
    Returns average metrics (MAE, MSE, RMSE, R2) across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model, _ = build_xgb_optuna_model(X_train, y_train, n_trials=n_trials)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[XGB-Optuna] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_feature_set_xgb_optuna(feature_folder, models_output_dir, holdout_ratio=0.15,
                                       n_splits=5, n_trials=50):
    """
    Loads feature data from feature_folder, splits off a hold-out test set (saved in a 'test' subfolder),
    performs K-fold evaluation using Optuna-based tuning on the training+validation data,
    then trains a final XGB model on all available training data.
    Saves the final model to models_output_dir and returns the robust K-fold metrics.
    """
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    functions.green_print(f"\n[XGB-Optuna] K-fold training for [NR: '{nr_method}', FE: '{feat_method}']")

    X, y, fnames = functions.load_feature_data(feature_folder)
    if len(X) == 0:
        print(f"[XGB-Optuna] No data in {feature_folder}. Skipping.")
        return None

    # Check if a hold-out test set already exists.
    test_folder = os.path.join(feature_folder, "test")
    if os.path.exists(test_folder) and os.listdir(test_folder):
        fnames_test = os.listdir(test_folder)
        X_train_val, y_train_val = [], []
        for x, label, f in zip(X, y, fnames):
            if f not in fnames_test:
                X_train_val.append(x)
                y_train_val.append(label)
        X_train_val = np.array(X_train_val)
        y_train_val = np.array(y_train_val)
        print(
            f"[XGB-Optuna] Using pre-split hold-out: {len(fnames_test)} test samples, {len(X_train_val)} training samples")
    else:
        X_train_val, X_test, y_train_val, y_test, fn_train_val, fn_test = train_test_split(
            X, y, fnames, test_size=holdout_ratio, random_state=42)
        os.makedirs(test_folder, exist_ok=True)
        for file in fn_test:
            shutil.copy(os.path.join(feature_folder, file), os.path.join(test_folder, file))
        print(f"[XGB-Optuna] Hold-out test set created with {len(fn_test)} samples")

    robust_metrics = kfold_train_and_evaluate_xgb_optuna(X_train_val, y_train_val, n_splits=n_splits, n_trials=n_trials)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[XGB-Optuna] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    best_model, best_params = build_xgb_optuna_model(X_train_val, y_train_val, n_trials=n_trials)
    print("[XGB-Optuna] Best hyperparameters:", best_params)

    model_name = f"xgb_{nr_method}_{feat_method}_optuna.pkl"
    model_path = os.path.join(models_output_dir, model_name)
    os.makedirs(models_output_dir, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"[XGB-Optuna] Final model saved to {os.path.relpath(model_path, os.getcwd())}")
    return mae_avg, mse_avg, rmse_avg, r2_avg

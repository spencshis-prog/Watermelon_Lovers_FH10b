import os
import numpy as np
import math
import shutil
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import optuna
import joblib

import functions  # your shared helper functions, e.g., load_feature_data, clear_output_directory


def objective_cat(trial, X, y):
    """
    Optuna objective function for tuning a CatBoostRegressor.
    """
    iterations = trial.suggest_int("iterations", 100, 300)
    depth = trial.suggest_int("depth", 4, 10)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)

    model = CatBoostRegressor(
        random_state=42,
        verbose=0,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg
    )

    # Use 3-fold CV on the trial
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        cv_scores.append(mse)
    return np.mean(cv_scores)


def build_cat_optuna_model(X, y, n_trials=50):
    """
    Runs an Optuna study to find the best hyperparameters for a CatBoostRegressor,
    trains the model on all provided data, and returns the trained model and best parameters.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_cat(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    print("[Cat-Optuna] Best hyperparameters:", best_params)
    best_model = CatBoostRegressor(
        random_state=42,
        verbose=0,
        **best_params
    )
    best_model.fit(X, y)
    return best_model, best_params


def kfold_train_and_evaluate_cat_optuna(X, y, n_splits=5, n_trials=50):
    """
    Performs K-fold cross-validation using an Optuna-tuned CatBoostRegressor.
    Returns average metrics (MAE, MSE, RMSE, R2) across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model, _ = build_cat_optuna_model(X_train, y_train, n_trials=n_trials)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[Cat-Optuna] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_feature_set_cat_optuna(feature_folder, models_output_dir,
                                       holdout_ratio=0.15, n_splits=5, n_trials=50):
    """
    Loads feature data from feature_folder, splits off a hold-out test set (stored in 'test'),
    performs K-fold evaluation using Optuna-based tuning on the training+validation data,
    then trains a final CatBoost model on all available training data.
    Saves the final model to models_output_dir and returns the robust K-fold metrics.
    """
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    functions.green_print(f"\n[Cat-Optuna] K-fold training for [NR: '{nr_method}', FE: '{feat_method}']")

    X, y, fnames = functions.load_feature_data(feature_folder)
    if len(X) == 0:
        print(f"[Cat-Optuna] No data in {feature_folder}. Skipping.")
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
            f"[Cat-Optuna] Using pre-split hold-out: {len(fnames_test)} test samples, {len(X_train_val)} training samples")
    else:
        X_train_val, X_test, y_train_val, y_test, _, fn_test = train_test_split(
            X, y, fnames, test_size=holdout_ratio, random_state=42)
        os.makedirs(test_folder, exist_ok=True)
        for file in fn_test:
            shutil.copy(os.path.join(feature_folder, file), os.path.join(test_folder, file))
        print(f"[Cat-Optuna] Hold-out test set created with {len(fn_test)} samples")

    robust_metrics = kfold_train_and_evaluate_cat_optuna(X_train_val, y_train_val,
                                                         n_splits=n_splits, n_trials=n_trials)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[Cat-Optuna] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    best_model, best_params = build_cat_optuna_model(X_train_val, y_train_val, n_trials=n_trials)
    print("[Cat-Optuna] Best hyperparameters:", best_params)

    model_name = f"cat_optuna_{nr_method}_{feat_method}.pkl"
    model_path = os.path.join(models_output_dir, model_name)
    os.makedirs(models_output_dir, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"[Cat-Optuna] Final model saved to {os.path.relpath(model_path, os.getcwd())}")
    return mae_avg, mse_avg, rmse_avg, r2_avg

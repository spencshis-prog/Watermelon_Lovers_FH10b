import os
import numpy as np
import math
import shutil
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import optuna
import joblib

import functions  # your helper functions (e.g. load_feature_data, clear_output_directory)
import params


def suggest_params(trial, search_space):
    """
    Given an Optuna trial and a search space dictionary, this function returns
    a dictionary of hyperparameters by calling the appropriate trial.suggest_* methods.

    Parameters:
        trial (optuna.trial.Trial): The current trial object.
        search_space (dict): Dictionary defining the search space for hyperparameters.
            Examples:
              - For numeric parameters with distribution:
                  'learning_rate': (1e-3, 1e-1, 'log-uniform')
              - For integer parameters:
                  'n_estimators': (50, 300)
              - For categorical parameters:
                  'max_features': ['sqrt', 'log2', None]

    Returns:
        dict: A dictionary of suggested hyperparameters.
    """
    suggested = {}
    for key, value in search_space.items():
        if isinstance(value, tuple):
            if len(value) == 3:
                low, high, dist = value
                if dist == 'log-uniform':
                    suggested[key] = trial.suggest_loguniform(key, low, high)
                elif dist == 'uniform':
                    suggested[key] = trial.suggest_float(key, low, high)
                else:
                    raise ValueError(f"Unknown distribution type '{dist}' for parameter '{key}'")
            elif len(value) == 2:
                low, high = value
                # If one of the bounds is float, use float suggestion.
                if isinstance(low, float) or isinstance(high, float):
                    suggested[key] = trial.suggest_float(key, low, high)
                else:
                    suggested[key] = trial.suggest_int(key, low, high)
            else:
                raise ValueError(f"Tuple for parameter '{key}' must have length 2 or 3. Got: {value}")
        elif isinstance(value, list):
            suggested[key] = trial.suggest_categorical(key, value)
        else:
            raise ValueError(f"Unsupported type for parameter '{key}': {type(value)}. Expected tuple or list.")
    return suggested

def objective_rf(trial, X, y):
    """
    Objective function for RandomForestRegressor hyperparameter tuning with Optuna.
    """
    search_space = params.rf_optuna_search_spaces
    hyperparams = suggest_params(trial, search_space)
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **hyperparams)

    # Perform a 3-fold CV for the trial.
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


def build_rf_optuna_model(X, y, n_trials=50):
    """
    Runs an Optuna study to find the best hyperparameters for a RandomForestRegressor,
    then trains a model on all provided data.
    Returns the trained model and the best hyperparameters.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_rf(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    print("[RF-Optuna] Best hyperparameters:", best_params)
    best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X, y)
    return best_model, best_params


def kfold_train_and_evaluate_rf_optuna(X, y, n_splits=5, n_trials=50):
    """
    Performs K-fold cross-validation using an Optuna-tuned RandomForestRegressor.
    Returns average metrics (MAE, MSE, RMSE, R2) across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model, _ = build_rf_optuna_model(X_train, y_train, n_trials=n_trials)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[RF-Optuna] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_feature_set_rf_optuna(feature_folder, models_output_dir, holdout_ratio=0.15,
                                      n_splits=5, n_trials=50):
    """
    Loads feature data from feature_folder, splits off a hold-out test set (in 'test' subfolder),
    performs K-fold evaluation using Optuna-based tuning on the remaining data,
    then trains a final RandomForestRegressor on the full training+validation set.
    Saves the final model to models_output_dir and returns the robust K-fold metrics.
    """
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    functions.green_print(f"\n[RF-Optuna] K-fold training for [NR: '{nr_method}', FE: '{feat_method}']")

    X, y, fnames = functions.load_feature_data(feature_folder)
    if len(X) == 0:
        print(f"[RF-Optuna] No data in {feature_folder}. Skipping.")
        return None

    # Check for existing hold-out test set.
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
            f"[RF-Optuna] Using pre-split hold-out: {len(fnames_test)} test samples, {len(X_train_val)} training samples")
    else:
        X_train_val, X_test, y_train_val, y_test, fn_train_val, fn_test = train_test_split(
            X, y, fnames, test_size=holdout_ratio, random_state=42)
        os.makedirs(test_folder, exist_ok=True)
        for file in fn_test:
            shutil.copy(os.path.join(feature_folder, file), os.path.join(test_folder, file))
        print(f"[RF-Optuna] Hold-out test set created with {len(fn_test)} samples")

    robust_metrics = kfold_train_and_evaluate_rf_optuna(X_train_val, y_train_val, n_splits=n_splits, n_trials=n_trials)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[RF-Optuna] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    best_model, best_params = build_rf_optuna_model(X_train_val, y_train_val, n_trials=n_trials)
    print("[RF-Optuna] Best hyperparameters:", best_params)

    model_name = f"rf_optuna_{nr_method}_{feat_method}.pkl"
    model_path = os.path.join(models_output_dir, model_name)
    os.makedirs(models_output_dir, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"[RF-Optuna] Final model saved to {os.path.relpath(model_path, os.getcwd())}")
    return mae_avg, mse_avg, rmse_avg, r2_avg

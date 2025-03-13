import os
import numpy as np
import math
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import optuna
import joblib
import functions  # your helper functions (load_feature_data, clear_output_directory, etc.)


def objective_lgbm(trial, X, y):
    # Suggest hyperparameters using Optuna's API:
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    num_leaves = trial.suggest_int('num_leaves', 15, 127)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 20)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)

    # Build the LightGBM regressor with these hyperparameters.
    model = lgb.LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    # Perform 3-fold cross-validation on the current trial.
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


def build_lgbm_optuna_model(X, y, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    print("[LGBM-Optuna] Best hyperparameters:", best_params)
    best_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X, y, verbose=False)
    return best_model, best_params


def kfold_train_and_evaluate_lgbm_optuna(X, y, n_splits=5, n_trials=50):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        best_model, _ = build_lgbm_optuna_model(X_train, y_train, n_trials=n_trials)
        y_pred = best_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[LGBM-Optuna] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_feature_set_lgbm_optuna(feature_folder, models_output_dir,
                                        holdout_ratio=0.15, n_splits=5, n_trials=50):
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    functions.green_print(f"\n[LGBM-Optuna] K-fold training for [NR: '{nr_method}', FE: '{feat_method}']")
    X, y, fnames = functions.load_feature_data(feature_folder)
    if len(X) == 0:
        print(f"[LGBM-Optuna] No data in {feature_folder}. Skipping.")
        return None
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
            f"[LGBM-Optuna] Using pre-split hold-out: {len(fnames_test)} test samples, {len(X_train_val)} training samples")
    else:
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test, fn_train_val, fn_test = train_test_split(
            X, y, fnames, test_size=holdout_ratio, random_state=42)
        import shutil
        os.makedirs(test_folder, exist_ok=True)
        for file in fn_test:
            shutil.copy(os.path.join(feature_folder, file), os.path.join(test_folder, file))
        print(f"[LGBM-Optuna] Hold-out test set created with {len(fn_test)} samples")

    robust_metrics = kfold_train_and_evaluate_lgbm_optuna(X_train_val, y_train_val, n_splits=n_splits,
                                                          n_trials=n_trials)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[LGBM-Optuna] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    best_model, best_params = build_lgbm_optuna_model(X_train_val, y_train_val, n_trials=n_trials)
    model_name = f"lgbm_{nr_method}_{feat_method}_optuna.pkl"
    model_path = os.path.join(models_output_dir, model_name)
    os.makedirs(models_output_dir, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"[LGBM-Optuna] Final model saved to {os.path.relpath(model_path, os.getcwd())}")
    return mae_avg, mse_avg, rmse_avg, r2_avg

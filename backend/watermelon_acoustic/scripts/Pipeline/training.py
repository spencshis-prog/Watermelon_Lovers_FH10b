import math
import os.path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


import functions


def maybe_convert_features(X, pipeline):
    """
    Convert features to a pandas DataFrame with dummy feature names if using models
    that expect feature names (e.g. LGBM or XGBoost).
    """
    if pipeline.model_tag.lower() in ["lgbm", "xgb"]:
        # Only convert if X is not already a DataFrame.
        if not isinstance(X, pd.DataFrame):
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=feature_names)
    return X


def build_model(pipeline, ht):
    params_dict = pipeline.get_params_dict(ht)

    if ht == "default":
        return pipeline.model_cls
    elif ht == "grid":
        if params_dict is None:
            raise ValueError("Grid search selected but no parameter grid provided.")
        return GridSearchCV(pipeline.model_cls, params_dict, cv=pipeline.inner_folds, scoring='neg_mean_squared_error',
                            verbose=0)
    elif ht == "random":
        if params_dict is None:
            raise ValueError("Random search selected but no parameter distribution provided.")
        return RandomizedSearchCV(pipeline.model_cls, param_distributions=params_dict, n_iter=10,
                                  cv=pipeline.inner_folds,
                                  scoring='neg_mean_squared_error', random_state=42, verbose=0)
    elif ht == "bayesian":
        if params_dict is None:
            raise ValueError("Bayesian search selected but no parameter dictionary provided.")
        return BayesSearchCV(pipeline.model_cls, params_dict, n_iter=16, cv=pipeline.inner_folds,
                             scoring='neg_mean_squared_error', random_state=42, verbose=0)
    else:
        raise ValueError(f"Unknown hyper-tuning method: {ht}")


def outer_kfolding(pipeline, ht, X, y):
    kf = KFold(n_splits=pipeline.outer_folds, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_conv = maybe_convert_features(X_train, pipeline)
        X_val_conv = maybe_convert_features(X_val, pipeline)

        model = build_model(pipeline, ht)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if pipeline.model_tag.lower() in ["lgbm", "xgb"]:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        fit_kwargs = {}
        if pipeline.early_stopping_rounds is not None:
            if pipeline.model_tag.lower() == "lgbm":
                fit_kwargs["callbacks"] = [early_stopping(pipeline.early_stopping_rounds)]  # , log_evaluation(0)
            else:
                fit_kwargs["early_stopping_rounds"] = pipeline.early_stopping_rounds
        if pipeline.use_eval_set is True:
            fit_kwargs["eval_set"] = [(X_val_conv, y_val)]
        model.fit(X_train_conv, y_train, **fit_kwargs)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        y_pred = model.predict(X_val_conv)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[{pipeline.model_tag.upper()}] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")

    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def train_for_combination_set(pipeline, ht, combination_dir):
    nr_method = os.path.basename(os.path.dirname(combination_dir))
    fe_method = os.path.basename(combination_dir)
    functions.green_print(
        f"\n[{pipeline.model_tag.upper()}] Outer K-fold training for [NR: '{nr_method}', FE: '{fe_method}', HT: '{ht}']")

    data = functions.load_feature_data_wrapped(combination_dir)
    if len(data["X"]) == 0:
        print(f"[{pipeline.model_tag.upper()} No data in {combination_dir}. Skipping.")
        return None

    X, y, fnames, transformers = pipeline.prime(data["X"], data["y"], data["fnames"])

    test_set_dir = os.path.join(combination_dir, "test")
    if os.path.exists(test_set_dir) and os.listdir(test_set_dir):
        fnames_test = os.listdir(test_set_dir)
        X_train_val, y_train_val, fnames_train_val = [], [], []
        for x, label, f in zip(X, y, fnames):
            if f not in fnames_test:
                X_train_val.append(x)
                y_train_val.append(label)
                fnames_train_val.append(f)
        X_train_val = np.array(X_train_val)
        y_train_val = np.array(y_train_val)
        print(
            f"[{pipeline.model_tag.upper()}] Using pre-split hold-out: {len(fnames_test)} test samples, {len(fnames_train_val)} training samples")
    else:
        X_train_val, X_test, y_train_val, y_test, fn_train_val, fn_test = train_test_split(
            X, y, fnames, test_size=pipeline.holdout_ratio, random_state=42
        )
        os.makedirs(test_set_dir, exist_ok=True)
        for file in fn_test:
            shutil.copy(os.path.join(combination_dir, file), os.path.join(test_set_dir, file))
        print(f"[{pipeline.model_tag.upper()}] Hold-out test set created with {len(fn_test)} samples")

    robust_metrics = outer_kfolding(pipeline, ht, X_train_val, y_train_val)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[{pipeline.model_tag.upper()}] [{nr_method} - {fe_method} - {ht}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    X_train_val_conv = maybe_convert_features(X_train_val, pipeline)
    final_model = build_model(pipeline, ht)

    fit_kwargs = {}
    if pipeline.early_stopping_rounds is not None:
        if pipeline.model_tag.lower() == "lgbm":
            from lightgbm import early_stopping, log_evaluation
            fit_kwargs["callbacks"] = [early_stopping(pipeline.early_stopping_rounds)]
            # Optionally, you can add log_evaluation(0) if you want to suppress logging:
            # fit_kwargs["callbacks"].append(log_evaluation(0))
        elif pipeline.model_tag.lower() == "xgb":
            # For XGBoost, skip early_stopping_rounds if unsupported.
            pass
        else:
            fit_kwargs["early_stopping_rounds"] = pipeline.early_stopping_rounds

    if pipeline.use_eval_set:
        # Split off 10% for evaluation
        X_train_sub, X_eval, y_train_sub, y_eval = train_test_split(
            X_train_val_conv, y_train_val, test_size=0.1, random_state=42
        )

        fit_kwargs["eval_set"] = [(X_eval, y_eval)]

        # For LGBM and XGB, you might want to suppress excessive output:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if pipeline.model_tag.lower() in ["lgbm", "xgb"]:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        final_model.fit(X_train_sub, y_train_sub, **fit_kwargs)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
    else:
        final_model.fit(X_train_val_conv, y_train_val)

    if hasattr(final_model, 'best_estimator_'):
        final_model = final_model.best_estimator_
    print(f"[{pipeline.model_tag.upper()}] Final model parameters:",
          functions.relevant_params(final_model, pipeline.model_tag.lower(), ht))

    model_filename = f"{pipeline.model_tag.lower()}_{nr_method}_{fe_method}_{ht}.pkl"
    model_path = os.path.join(pipeline.models_output_dir, model_filename)
    os.makedirs(pipeline.models_output_dir, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"[{pipeline.model_tag.upper()}] Final model saved to {os.path.relpath(model_path, os.getcwd())}")
    return mae_avg, mse_avg, rmse_avg, r2_avg


def train_all(pipeline, ht):
    if not os.path.exists(pipeline.models_output_dir):
        os.makedirs(pipeline.models_output_dir)
    results = {}
    nr_folders = sorted(
        [d for d in os.listdir(pipeline.fe_base_dir) if os.path.isdir(os.path.join(pipeline.fe_base_dir, d))])
    for nr in nr_folders:
        nr_path = os.path.join(pipeline.fe_base_dir, nr)
        fe_folders = sorted([d for d in os.listdir(nr_path) if os.path.isdir(os.path.join(nr_path, d))])
        for fe in fe_folders:
            combination_dir = os.path.join(nr_path, fe)
            metrics = train_for_combination_set(pipeline, ht, combination_dir)
            if metrics is not None:
                results[(nr, fe, ht)] = metrics

    functions.green_print("\n=== Printing all average K-fold error metrics ===")
    for (nr, fe, ht), (mae_avg, mse_avg, rmse_avg, r2_avg) in results.items():
        print(
            f"[{pipeline.model_tag.upper()}] {nr}-{fe} ({ht}) => MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")
    print(f"[{pipeline.model_tag.upper()}] K-fold training completed for all combination sets.")

    with open(pipeline.report_kfold_path, "a") as f:
        for key in sorted(results.keys()):
            nr, fe, ht = key
            mae, mse, rmse, r2 = results[key]
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {fe}, Hyper-tuning: {ht}\n")
            f.write(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\n\n")
    print(f"[{pipeline.model_tag.upper()}] Report saved to {os.path.relpath(pipeline.report_kfold_path, os.getcwd())}")

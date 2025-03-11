import os
import numpy as np
import shutil
import math
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import functions


def build_et_model(hyper_tuning="default"):
    """
    Returns an ExtraTreesRegressor configured according to the hyper_tuning strategy.
    """
    if hyper_tuning == "default":
        return ExtraTreesRegressor(random_state=42)
    elif hyper_tuning == "grid":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            # 'min_samples_leaf': [1, 2, 4],  # added
            # 'max_features': ['sqrt', 'log2', None]  # added
        }
        et = ExtraTreesRegressor(random_state=42)
        model = GridSearchCV(et, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
        return model
    elif hyper_tuning == "random":
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 6],  # added
            'max_features': ['sqrt', 'log2', None]  # added
        }
        et = ExtraTreesRegressor(random_state=42)
        model = RandomizedSearchCV(et, param_distributions=param_dist, n_iter=10, cv=3,
                                   scoring='neg_mean_squared_error', random_state=42, verbose=0)
        return model
    else:
        return ExtraTreesRegressor(random_state=42)


def kfold_train_and_evaluate_et(X, y, n_splits=5, epochs=20, batch_size=16, hyper_tuning="default"):
    """
    Performs K-fold cross-validation on (X, y) using an ExtraTrees model.
    Returns average metrics (MAE, MSE, RMSE, R2) across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_et_model(hyper_tuning)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[ET] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_feature_set_et(feature_folder, models_output_dir,
                               holdout_ratio=0.15, n_splits=5, epochs=20,
                               batch_size=16, hyper_tuning="default"):
    """
    Loads feature data from feature_folder, splits off a hold-out test set (saved in a 'test' subfolder),
    performs K-fold evaluation on the remaining data using the given hyper_tuning strategy,
    then trains a final ExtraTrees model on the remaining data.
    Saves the final model to models_output_dir and returns the robust k-fold metrics.
    """
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    functions.green_print(f"\n[ET] K-fold training for [NR: '{nr_method}', FE: '{feat_method}', HT: '{hyper_tuning}']")

    X, y, fnames = functions.load_feature_data(feature_folder)
    if len(X) == 0:
        print(f"[ET] No data in {feature_folder}. Skipping.")
        return None

    # Check if a hold-out test set already exists.
    test_folder = os.path.join(feature_folder, "test")
    if os.path.exists(test_folder) and os.listdir(test_folder):
        fnames_test = os.listdir(test_folder)
        X_train_val, y_train_val, fnames_train_val = [], [], []
        for x, label, f in zip(X, y, fnames):
            if f not in fnames_test:
                X_train_val.append(x)
                y_train_val.append(label)
                fnames_train_val.append(f)
        X_train_val = np.array(X_train_val)
        y_train_val = np.array(y_train_val)
        print(
            f"[ET] Using pre-split hold-out: {len(fnames_test)} test samples, {len(fnames_train_val)} training samples")
    else:
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test, fn_train_val, fn_test = train_test_split(
            X, y, fnames, test_size=holdout_ratio, random_state=42
        )
        os.makedirs(test_folder, exist_ok=True)
        for file in fn_test:
            shutil.copy(os.path.join(feature_folder, file), os.path.join(test_folder, file))
        print(f"[ET] Hold-out test set created with {len(fn_test)} samples")

    # K-fold evaluation on the training+validation data.
    robust_metrics = kfold_train_and_evaluate_et(X_train_val, y_train_val, n_splits=n_splits,
                                                 epochs=epochs, batch_size=batch_size,
                                                 hyper_tuning=hyper_tuning)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[ET] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    # Train final ExtraTrees model on all training+validation data.
    final_model = build_et_model(hyper_tuning)
    final_model.fit(X_train_val, y_train_val)
    print("[ET] Final model parameters:", functions.relevant_params(
        final_model.get_params() if hasattr(final_model, 'get_params') else {},
        "cat", hyper_tuning
    ))

    model_name = f"et_{nr_method}_{feat_method}_{hyper_tuning}.pkl"
    model_path = os.path.join(models_output_dir, model_name)
    os.makedirs(models_output_dir, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"[ET] Final model saved to {os.path.relpath(model_path, os.getcwd())}")
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_all_feature_models_et(feature_extraction_base_dir, models_output_dir, report_kfold_path,
                                      holdout_ratio=0.15, n_splits=5, epochs=20, batch_size=16,
                                      hyper_tuning="default"):
    """
    Iterates over each noise reduction technique folder and each feature extraction method folder,
    performs K-fold training (with hold-out test sets saved) using the specified hyperparameter tuning option,
    saves the final models, and writes consolidated metrics to report_kfold_path.
    """
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)
    results = {}
    nr_folders = sorted([d for d in os.listdir(feature_extraction_base_dir)
                         if os.path.isdir(os.path.join(feature_extraction_base_dir, d))])
    for nr in nr_folders:
        nr_path = os.path.join(feature_extraction_base_dir, nr)
        feat_folders = sorted([d for d in os.listdir(nr_path)
                               if os.path.isdir(os.path.join(nr_path, d))])
        for feat in feat_folders:
            feat_folder = os.path.join(nr_path, feat)
            metrics = kfold_train_feature_set_et(feat_folder, models_output_dir,
                                                 holdout_ratio=holdout_ratio,
                                                 n_splits=n_splits,
                                                 epochs=epochs,
                                                 batch_size=batch_size,
                                                 hyper_tuning=hyper_tuning)
            if metrics is not None:
                results[(nr, feat, hyper_tuning)] = metrics

    for (nr, feat, ht), (mae_avg, mse_avg, rmse_avg, r2_avg) in results.items():
        print(f"[ET] {nr}-{feat} ({ht}) => MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")
    print("[ET] K-fold training completed for all feature sets.")

    with open(report_kfold_path, "a") as f:
        for key in sorted(results.keys()):
            nr, feat, ht = key
            mae, mse, rmse, r2 = results[key]
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {feat}, Hyper-tuning: {ht}\n")
            f.write(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\n\n")
    print(f"[ET] Report saved to {os.path.relpath(report_kfold_path, os.getcwd())}")

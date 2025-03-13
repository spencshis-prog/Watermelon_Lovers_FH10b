import os
import numpy as np
import shutil
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

import functions
from scripts.linear_regression.lr_regularization import build_lr_model  # Use our LR builder with regularization


def load_feature_data(folder):
    """
    Loads .npy feature files from folder (only files directly in folder, not in subfolders).
    Assumes naming format: <watermelonID>_<brix>_<index>.npy.
    Returns X, y, fnames.
    """
    data, labels, fnames = [], [], []
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path) and file.lower().endswith(".npy"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            try:
                brix_val = float(parts[1])
            except:
                continue
            feat = np.load(full_path)
            data.append(feat)
            labels.append(brix_val)
            fnames.append(file)
    return np.array(data), np.array(labels), fnames


def kfold_train_and_evaluate(X, y, n_splits=5, epochs=20, batch_size=16, regularization="none"):
    """
    Performs K-fold cross-validation on (X, y) using a linear regression model with the specified regularization.
    Returns average metrics (MAE, MSE, RMSE, R2) across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        input_shape = (X_train.shape[1],)
        model = build_lr_model(input_shape, regularization=regularization)
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0)
        y_pred = model.predict(X_val).flatten()
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        fold_metrics.append((mae, mse, rmse, r2))
        print(f"[LR] [Fold {fold_idx + 1}] MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    mae_avg = np.mean([m[0] for m in fold_metrics])
    mse_avg = np.mean([m[1] for m in fold_metrics])
    rmse_avg = np.mean([m[2] for m in fold_metrics])
    r2_avg = np.mean([m[3] for m in fold_metrics])
    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_feature_set(feature_folder, models_output_dir,
                            holdout_ratio=0.15, n_splits=5,
                            epochs=20, batch_size=16, regularization="none"):
    """
    Loads feature data from feature_folder, then checks if a hold-out test set exists in a subfolder "test".
    If so, excludes those files from the training data.
    Otherwise, splits off a hold-out test set, saves it into "test" subfolder.
    Then performs K-fold evaluation on the remaining data and trains a final model.
    Saves the final model to models_output_dir.
    Returns the robust k-fold metrics.
    """
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    functions.green_print(f"\n[LR] K-fold training for [NR: '{nr_method}', FE: '{feat_method}', REG: '{regularization}']")

    # Load data from the top level of the feature folder (excluding subfolders)
    X_all, y_all, fnames_all = load_feature_data(feature_folder)
    if len(X_all) == 0:
        print(f"[LR] No data in {feature_folder}. Skipping.")
        return None

    # Check if hold-out test set already exists:
    test_folder = os.path.join(feature_folder, "test")
    if os.path.exists(test_folder) and os.listdir(test_folder):
        # If it exists, load the list of test filenames from the test folder.
        fnames_test = os.listdir(test_folder)
        # Filter out test files from our full list.
        X_train_val, y_train_val, fnames_train_val = [], [], []
        for x, y, f in zip(X_all, y_all, fnames_all):
            if f not in fnames_test:
                X_train_val.append(x)
                y_train_val.append(y)
                fnames_train_val.append(f)
        X_train_val = np.array(X_train_val)
        y_train_val = np.array(y_train_val)
        rel_feature_folder = os.path.relpath(feature_folder, os.getcwd())
        print(
            f"[LR] Using pre-split hold-out: {len(fnames_test)} test samples, {len(fnames_train_val)} training samples from {rel_feature_folder}")
    else:
        # Otherwise, perform a split.
        X_train_val, X_test, y_train_val, y_test, fn_train_val, fn_test = train_test_split(
            X_all, y_all, fnames_all, test_size=holdout_ratio, random_state=42
        )
        os.makedirs(test_folder, exist_ok=True)
        for file in fn_test:
            src = os.path.join(feature_folder, file)
            dst = os.path.join(test_folder, file)
            shutil.copy(src, dst)
        print(
            f"[LR] Hold-out test set created with {len(fn_test)} samples in {os.path.relpath(test_folder, os.getcwd())}")

    # Perform K-fold evaluation on the training+validation data.
    mae_avg, mse_avg, rmse_avg, r2_avg = kfold_train_and_evaluate(X_train_val, y_train_val,
                                                                  n_splits=n_splits,
                                                                  epochs=epochs,
                                                                  batch_size=batch_size,
                                                                  regularization=regularization)
    print(
        f"[LR] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    # Train final model on the entire remaining training+validation data.
    input_shape = (X_train_val.shape[1],)
    final_model = build_lr_model(input_shape, regularization=regularization)
    final_model.fit(X_train_val, y_train_val, epochs=epochs, batch_size=batch_size, verbose=0)

    model_name = f"lr_{nr_method}_{feat_method}_{regularization}.keras"
    model_path = os.path.join(models_output_dir, model_name)
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)
    final_model.save(model_path)
    print(f"[LR] Final model saved to {os.path.relpath(model_path, os.getcwd())}")

    return mae_avg, mse_avg, rmse_avg, r2_avg


def kfold_train_all_feature_models(feature_extraction_base_dir, models_output_dir, report_kfold_path,
                                   holdout_ratio=0.15, n_splits=5,
                                   epochs=20, batch_size=16, regularization="none"):
    """
    Iterates over each noise reduction technique folder, then each feature extraction method folder,
    and performs K-fold training (with a hold-out test set saved) using the specified regularization.
    Saves the final models and collects the robust metrics for each combination.
    Finally, writes all metrics to a single report file named "report_kfold.txt".
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
            metrics = kfold_train_feature_set(feat_folder, models_output_dir,
                                              holdout_ratio=holdout_ratio,
                                              n_splits=n_splits,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              regularization=regularization)
            if metrics is not None:
                # Key now includes the regularization type.
                results[(nr, feat, regularization)] = metrics

    for (nr, feat, reg), (mae_avg, mse_avg, rmse_avg, r2_avg) in results.items():
        print(f"[LR] {nr}-{feat} ({reg}) => MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")
    functions.green_print("[LR] K-fold training completed for all feature sets.")

    # Write the consolidated report to report_kfold.txt
    with open(report_kfold_path, "a") as f:
        for key in sorted(results.keys()):
            nr, feat, reg = key
            mae, mse, rmse, r2 = results[key]
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {feat}, Regularization: {reg}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2: {r2:.4f}\n\n")
    print(f"[LR] Report saved to {os.path.relpath(report_kfold_path, os.getcwd())}")

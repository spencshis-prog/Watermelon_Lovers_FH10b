import os
import numpy as np
import shutil
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from lr_regularization import build_lr_model  # Use our LR builder with regularization


def green_print(message):
    print("\033[92m" + message + "\033[0m")


def load_feature_data(folder):
    """
    Loads .npy feature files from folder.
    Assumes naming format: <watermelonID>_<brix>_<index>.npy
    Returns X, y, fnames
    """
    data, labels, fnames = [], [], []
    for file in os.listdir(folder):
        if file.lower().endswith(".npy"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            try:
                brix_val = float(parts[1])
            except:
                continue
            file_path = os.path.join(folder, file)
            feat = np.load(file_path)
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
    Loads feature data from feature_folder, splits off a hold-out test set,
    saves the hold-out test data to a 'test' subfolder,
    performs K-fold evaluation on the remaining data using the specified regularization,
    writes the k-fold metrics to a summary file,
    and then trains a final model on the remaining data.
    Saves the final model to models_output_dir.
    Returns the robust k-fold metrics.
    """
    nr_method = os.path.basename(os.path.dirname(feature_folder))
    feat_method = os.path.basename(feature_folder)
    green_print(f"\n[LR] K-fold training for Linear Regression with "
                f"[NR: '{nr_method}', FE: '{feat_method}', REG: '{regularization}']")

    X, y, fnames = load_feature_data(feature_folder)
    if len(X) == 0:
        print(f"[LR] No data in {feature_folder}. Skipping.")
        return None

    # Split off a hold-out test set
    X_remaining, X_test, y_remaining, y_test, fn_remaining, fn_test = train_test_split(
        X, y, fnames, test_size=holdout_ratio, random_state=42
    )
    test_folder = os.path.join(feature_folder, "test")
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for file in fn_test:
        src = os.path.join(feature_folder, file)
        dst = os.path.join(test_folder, file)
        shutil.copy(src, dst)
    print(
        f"[LR] Hold-out test set saved with {len(X_test)} samples in {os.path.relpath(test_folder, os.getcwd())}")

    robust_metrics = kfold_train_and_evaluate(X_remaining, y_remaining,
                                              n_splits=n_splits,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              regularization=regularization)
    mae_avg, mse_avg, rmse_avg, r2_avg = robust_metrics
    print(
        f"[LR] [{nr_method} - {feat_method}] K-fold average -> MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")

    summary_path = os.path.join(feature_folder, "kfold_metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"MAE: {mae_avg:.4f}\nMSE: {mse_avg:.4f}\nRMSE: {rmse_avg:.4f}\nR2: {r2_avg:.4f}\n")
    print(f"[LR] K-fold summary saved to {os.path.relpath(summary_path, os.getcwd())}")

    input_shape = (X_remaining.shape[1],)
    final_model = build_lr_model(input_shape, regularization=regularization)
    final_model.fit(X_remaining, y_remaining, epochs=epochs, batch_size=batch_size, verbose=0)

    model_name = f"lr_{nr_method}_{feat_method}_{regularization}.keras"
    model_path = os.path.join(models_output_dir, model_name)
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)
    final_model.save(model_path)
    print(f"[LR] Final model saved to {os.path.relpath(model_path, os.getcwd())}")

    return (mae_avg, mse_avg, rmse_avg, r2_avg)


def kfold_train_all_feature_models(feature_extraction_base_dir, models_output_dir,
                                   holdout_ratio=0.15, n_splits=5,
                                   epochs=20, batch_size=16, regularization="none"):
    """
    Iterates over each noise reduction technique folder,
    then each feature extraction method folder, does K-fold training
    (with a hold-out test set saved) using the specified regularization,
    and saves the final model.
    Prints out the robust metrics for each combination.
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
                results[(nr, feat)] = metrics

    for (nr, feat), (mae_avg, mse_avg, rmse_avg, r2_avg) in results.items():
        print(f"[LR] {nr}-{feat} ({regularization}) => MAE={mae_avg:.2f}, RMSE={rmse_avg:.2f}, R2={r2_avg:.2f}")
    green_print("[LR] K-fold training completed for all feature sets.")

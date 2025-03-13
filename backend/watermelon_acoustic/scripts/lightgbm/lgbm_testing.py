import os
import math
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_feature_data(folder):
    """
    Loads feature data from .npy files in a folder.
    Assumes naming format: <watermelonID>_<brix>_<index>.npy.
    Returns X, y, and file names.
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


def test_model_on_holdout_lgbm(model_path, test_folder):
    """
    Loads a saved LightGBM model and evaluates it on the hold-out test set.
    Returns a dictionary of metrics and predictions.
    """
    if not os.path.exists(test_folder):
        print(f"[LGBM-TE] Test folder {test_folder} does not exist. Skipping.")
        return None
    X, y, _ = load_feature_data(test_folder)
    if len(X) == 0:
        print("[LGBM-TE] No test data found.")
        return None
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "y_true": y, "y_pred": y_pred}


def test_all_lgbm_models(models_dir, feature_extraction_base_dir, report_path):
    """
    Iterates over each (NR, FE, hyper-tuning) combination in the feature extraction base directory,
    loads the corresponding final LightGBM model (named lgbm_<NR>_<FE>_<ht>.pkl),
    evaluates it on the hold-out test set (located in the 'test' subfolder),
    and writes a consolidated report.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    results = {}
    nr_list = sorted([d for d in os.listdir(feature_extraction_base_dir)
                      if os.path.isdir(os.path.join(feature_extraction_base_dir, d))])
    for nr in nr_list:
        nr_path = os.path.join(feature_extraction_base_dir, nr)
        fe_list = sorted([d for d in os.listdir(nr_path)
                          if os.path.isdir(os.path.join(nr_path, d))])
        for fe in fe_list:
            feat_folder = os.path.join(nr_path, fe)
            test_folder = os.path.join(feat_folder, "test")
            for ht in ["default", "grid", "random", "bayesian", "optuna"]:
                model_name = f"lgbm_{nr}_{fe}_{ht}.pkl"
                model_path = os.path.join(models_dir, model_name)
                rel_model_path = os.path.relpath(model_path, project_root)
                if not os.path.exists(model_path):
                    print(f"[LGBM-TE] No model found for {nr}, {fe}, HT={ht} at {rel_model_path}. Skipping.")
                    continue
                print(
                    f"[LGBM-TE] Evaluating model {model_name} on test set from {os.path.relpath(test_folder, project_root)}")
                metrics = test_model_on_holdout_lgbm(model_path, test_folder)
                if metrics is not None:
                    results[(nr, fe, ht)] = metrics

    # Write consolidated report.
    with open(report_path, "w") as f:
        for (nr, fe, ht), m in sorted(results.items()):
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {fe}, Hyper-tuning: {ht}\n")
            f.write(f"MAE: {m['MAE']:.4f}\n")
            f.write(f"MSE: {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2: {m['R2']:.4f}\n\n")
    rel_report_path = os.path.relpath(report_path, project_root)
    print(f"[LGBM-TE] Report saved to {rel_report_path}")

    # (Optional) You can add plotting routines here similar to your XGB testing code.


import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_feature_data(folder):
    """
    Loads feature data from .npy files in a folder.
    Assumes naming format: <watermelonID>_<brix>_<index>.npy.
    Returns (X, y, fnames) where X is an array of feature vectors.
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

def test_model_on_holdout_cat(model_path, test_folder):
    """
    Loads a final CatBoost model and evaluates it on the hold-out test set.
    Expects test_folder to contain the held-out .npy feature files.
    Returns a dictionary of test metrics and predictions.
    """
    if not os.path.exists(test_folder):
        print(f"[CAT-TE] Test folder {test_folder} does not exist. Skipping.")
        return None

    X, y, _ = load_feature_data(test_folder)
    if len(X) == 0:
        print("[CAT-TE] No test data found.")
        return None

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "y_true": y, "y_pred": y_pred}

def test_all_cat_models(models_dir, feature_extraction_base_dir, report_path):
    """
    Iterates over each (NR, FE, hyper-tuning) combination in feature_extraction_base_dir,
    loads the corresponding final model (named cat_<NR>_<FE>_<ht>.pkl),
    and evaluates it on the hold-out test set stored in the 'test' subfolder of the FE folder.
    Writes a report and produces grid plots:
      1. Heatmaps for hold-out test metrics.
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
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
            for ht in ["default", "grid", "random", "bayesian"]:
                model_name = f"cat_{nr}_{fe}_{ht}.pkl"
                model_path = os.path.join(models_dir, model_name)
                rel_model_path = os.path.relpath(model_path, project_root)
                if not os.path.exists(model_path):
                    print(f"[CAT-TE] No model found for {nr}, {fe}, ht={ht} at {rel_model_path}. Skipping.")
                    continue
                print(f"[CAT-TE] Evaluating model {model_name} on test set from {os.path.relpath(test_folder, project_root)}")
                metrics = test_model_on_holdout_cat(model_path, test_folder)
                if metrics is not None:
                    results[(nr, fe, ht)] = metrics

    # Write a consolidated report.
    with open(report_path, "w") as f:
        for (nr, fe, ht), m in sorted(results.items()):
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {fe}, Hyper-tuning: {ht}\n")
            f.write(f"MAE: {m['MAE']:.4f}\n")
            f.write(f"MSE: {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2: {m['R2']:.4f}\n\n")
    rel_report_path = os.path.relpath(report_path, project_root)
    print(f"[CAT-TE] Report saved to {rel_report_path}")

    plot_results_grid_cat(results, project_root)

def parse_report(report_path):
    """
    Parses a report file with blocks like:
      Noise Reduction: <nr>, Feature Extraction: <fe>, Hyper-tuning: <ht>
      MAE: <value>
      MSE: <value>
      RMSE: <value>
      R2: <value>
    Returns a dictionary with keys (nr, fe, ht) mapping to metrics.
    """
    results = {}
    if not os.path.exists(report_path):
        print(f"[CAT-Parse] Report file {report_path} not found.")
        return results

    with open(report_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    while idx < len(lines):
        header = lines[idx]
        try:
            parts = header.split(',')
            nr = parts[0].split("Noise Reduction:")[-1].strip()
            fe = parts[1].split("Feature Extraction:")[-1].strip()
            ht = parts[2].split("Hyper-tuning:")[-1].strip()
        except Exception:
            idx += 1
            continue

        try:
            mae = float(lines[idx + 1].split("MAE:")[-1].strip())
            mse = float(lines[idx + 2].split("MSE:")[-1].strip())
            rmse = float(lines[idx + 3].split("RMSE:")[-1].strip())
            r2 = float(lines[idx + 4].split("R2:")[-1].strip())
            results[(nr, fe, ht)] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        except Exception:
            pass
        idx += 5
    return results

def plot_results_grid_cat(results, project_root):
    """
    Produces grid plots for:
      1. Combined heatmaps for hold-out test metrics and (if available) K-Fold metrics.
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
    Figures are saved and displayed.
    """
    base_save_dir = os.path.join(project_root, "../output", "testing_cat")
    heatmaps_dir = os.path.join(base_save_dir, "heatmaps")
    pred_vs_actual_dir = os.path.join(base_save_dir, "predicted_vs_actual")
    residual_plots_dir = os.path.join(base_save_dir, "residual_plots")
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(pred_vs_actual_dir, exist_ok=True)
    os.makedirs(residual_plots_dir, exist_ok=True)

    report_kfold_path = os.path.join(project_root, "../output", "testing_cat", "report_kfold.txt")
    kfold_results = parse_report(report_kfold_path)

    cat_ht_options = sorted(list(set([ht for (_, _, ht) in results.keys()])))
    for ht in cat_ht_options:
        holdout_filtered = {(nr, fe): results[(nr, fe, ht)]
                            for (nr, fe, tuning) in results.keys() if tuning == ht}
        kfold_filtered = {}
        if kfold_results:
            for (nr, fe, tuning) in kfold_results.keys():
                if tuning == ht:
                    kfold_filtered[(nr, fe)] = kfold_results[(nr, fe, tuning)]

        all_keys = set(holdout_filtered.keys()).union(set(kfold_filtered.keys()))
        all_nr = sorted({nr for (nr, fe) in all_keys})
        all_fe = sorted({fe for (nr, fe) in all_keys})
        n_nr = len(all_nr)
        n_fe = len(all_fe)

        mae_grid_hold = np.full((n_nr, n_fe), np.nan)
        mse_grid_hold = np.full((n_nr, n_fe), np.nan)
        rmse_grid_hold = np.full((n_nr, n_fe), np.nan)
        r2_grid_hold = np.full((n_nr, n_fe), np.nan)

        mae_grid_kfold = np.full((n_nr, n_fe), np.nan)
        mse_grid_kfold = np.full((n_nr, n_fe), np.nan)
        rmse_grid_kfold = np.full((n_nr, n_fe), np.nan)
        r2_grid_kfold = np.full((n_nr, n_fe), np.nan)

        for i, nr in enumerate(all_nr):
            for j, fe in enumerate(all_fe):
                key = (nr, fe)
                if key in holdout_filtered:
                    m = holdout_filtered[key]
                    mae_grid_hold[i, j] = m["MAE"]
                    mse_grid_hold[i, j] = m["MSE"]
                    rmse_grid_hold[i, j] = m["RMSE"]
                    r2_grid_hold[i, j] = m["R2"]
                if key in kfold_filtered:
                    m = kfold_filtered[key]
                    mae_grid_kfold[i, j] = m["MAE"]
                    mse_grid_kfold[i, j] = m["MSE"]
                    rmse_grid_kfold[i, j] = m["RMSE"]
                    r2_grid_kfold[i, j] = m["R2"]

        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        metrics_titles = ["MAE", "MSE", "RMSE", "R²"]
        holdout_grids = [mae_grid_hold, mse_grid_hold, rmse_grid_hold, r2_grid_hold]
        kfold_grids = [mae_grid_kfold, mse_grid_kfold, rmse_grid_kfold, r2_grid_kfold]

        for idx, (grid, title) in enumerate(zip(holdout_grids, metrics_titles)):
            ax = axes1[0, idx]
            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            ax.set_title(f"{title} (Hold-out) - {ht}", fontsize=10)
            ax.set_xticks(np.arange(n_fe))
            ax.set_yticks(np.arange(n_nr))
            ax.set_xticklabels(all_fe, fontsize=8, rotation=45, ha="right")
            ax.set_yticklabels(all_nr, fontsize=8)
            fig1.colorbar(im, ax=ax)

        for idx, (grid, title) in enumerate(zip(kfold_grids, metrics_titles)):
            ax = axes1[1, idx]
            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            ax.set_title(f"{title} (K-Fold) - {ht}", fontsize=10)
            ax.set_xticks(np.arange(n_fe))
            ax.set_yticks(np.arange(n_nr))
            ax.set_xticklabels(all_fe, fontsize=8, rotation=45, ha="right")
            ax.set_yticklabels(all_nr, fontsize=8)
            fig1.colorbar(im, ax=ax)

        fig1.suptitle(f"CatBoost Test Metrics Heatmaps (HT: {ht})", fontsize=12)
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        heatmap_path = os.path.join(heatmaps_dir, f"cat_{ht}_heatmap.png")
        fig1.savefig(heatmap_path, dpi=300)
        print(f"[CAT-TE] Saved combined heatmap to {os.path.relpath(heatmap_path, project_root)}")
        plt.close(fig1)

        scatter_nr = sorted(list(set([nr for (nr, fe) in holdout_filtered.keys()])))
        scatter_fe = sorted(list(set([fe for (nr, fe) in holdout_filtered.keys()])))
        scatter_data = {}
        residual_data = {}
        for i, nr in enumerate(scatter_nr):
            for j, fe in enumerate(scatter_fe):
                key = (nr, fe)
                if key in holdout_filtered:
                    m = holdout_filtered[key]
                    scatter_data[(i, j)] = (m["y_true"], m["y_pred"])
                    residual_data[(i, j)] = (m["y_pred"], m["y_true"] - m["y_pred"])

        n_nr_scatter = len(scatter_nr)
        n_fe_scatter = len(scatter_fe)

        fig2, axes2 = plt.subplots(n_nr_scatter, n_fe_scatter,
                                   figsize=(4 * n_fe_scatter, 4 * n_nr_scatter),
                                   squeeze=False)
        for i in range(n_nr_scatter):
            for j in range(n_fe_scatter):
                ax = axes2[i][j]
                if (i, j) in scatter_data:
                    y_true, y_pred = scatter_data[(i, j)]
                    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
                    mae_val = holdout_filtered[(scatter_nr[i], scatter_fe[j])]["MAE"]
                    r2_val = holdout_filtered[(scatter_nr[i], scatter_fe[j])]["R2"]
                    ax.set_title(f"{scatter_nr[i]} | {scatter_fe[j]}\nMAE={mae_val:.2f}, R²={r2_val:.2f}", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=8)
                ax.set_xlabel("Actual", fontsize=8)
                ax.set_ylabel("Predicted", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)

        fig2.suptitle(f"CatBoost Predicted vs. Actual (Hold-out) (HT: {ht})", fontsize=10)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        pred_vs_actual_path = os.path.join(pred_vs_actual_dir, f"cat_{ht}_predicted_vs_actual.png")
        fig2.savefig(pred_vs_actual_path, dpi=300)
        print(f"[CAT-TE] Saved predicted vs. actual plot to {os.path.relpath(pred_vs_actual_path, project_root)}")
        plt.close(fig2)

        fig3, axes3 = plt.subplots(n_nr_scatter, n_fe_scatter,
                                   figsize=(4 * n_fe_scatter, 4 * n_nr_scatter),
                                   squeeze=False)
        for i in range(n_nr_scatter):
            for j in range(n_fe_scatter):
                ax = axes3[i][j]
                if (i, j) in residual_data:
                    y_pred, residuals = residual_data[(i, j)]
                    ax.scatter(y_pred, residuals, alpha=0.6, s=20)
                    ax.axhline(0, color='red', linestyle='--', linewidth=1)
                    ax.set_title(f"{scatter_nr[i]} | {scatter_fe[j]}", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=8)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Residual", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)

        fig3.suptitle(f"CatBoost Residual Plots (Hold-out) (HT: {ht})", fontsize=10)
        fig3.tight_layout(rect=[0, 0, 1, 0.95])
        residual_plot_path = os.path.join(residual_plots_dir, f"cat_{ht}_residual_plot.png")
        fig3.savefig(residual_plot_path, dpi=300)
        print(f"[CAT-TE] Saved residual plot to {os.path.relpath(residual_plot_path, project_root)}")
        plt.close(fig3)

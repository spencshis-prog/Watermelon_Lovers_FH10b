import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def test_model_on_holdout(model_path, test_folder):
    """
    Loads a final model and evaluates it on the hold-out test set.
    Expects test_folder to contain the held-out .npy feature files.
    Returns a dict of test metrics and predictions.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    rel_test_folder = os.path.relpath(test_folder, project_root)

    if not os.path.exists(test_folder):
        print(f"[TE] Test folder {rel_test_folder} does not exist. Skipping.")
        return None

    X, y, _ = load_feature_data(test_folder)
    if len(X) == 0:
        print("[TE] No test data found.")
        return None

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X).flatten()
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "y_true": y, "y_pred": y_pred}


def parse_report(report_path):
    """
    Parses a report file with blocks like:

      Noise Reduction: <nr>, Feature Extraction: <fe>, Regularization: <reg>
      MAE: <value>
      MSE: <value>
      RMSE: <value>
      R2: <value>

    Returns a dictionary with keys (nr, fe, reg) mapping to metrics.
    """
    results = {}
    if not os.path.exists(report_path):
        print(f"[Parse] Report file {report_path} not found.")
        return results

    with open(report_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    while idx < len(lines):
        # First line: contains NR, FE, and reg info.
        header = lines[idx]
        try:
            # Expecting header like "Noise Reduction: {nr}, Feature Extraction: {fe}, Regularization: {reg}"
            parts = header.split(',')
            nr = parts[0].split("Noise Reduction:")[-1].strip()
            fe = parts[1].split("Feature Extraction:")[-1].strip()
            reg = parts[2].split("Regularization:")[-1].strip()
        except Exception as e:
            print(f"[Parse] Error parsing header: {header}")
            idx += 1
            continue

        try:
            mae = float(lines[idx + 1].split("MAE:")[-1].strip())
            mse = float(lines[idx + 2].split("MSE:")[-1].strip())
            rmse = float(lines[idx + 3].split("RMSE:")[-1].strip())
            r2 = float(lines[idx + 4].split("R2:")[-1].strip())
            results[(nr, fe, reg)] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        except Exception as e:
            print(f"[Parse] Error parsing metrics for {header}")
        idx += 5  # Move to next block
    return results


def test_all_lr_models(models_dir, feature_extraction_base_dir, report_path):
    """
    Iterates over each (NR, FE, regularization) combination in feature_extraction_base_dir,
    loads the corresponding final model (named lr_<NR>_<FE>_<reg>.keras),
    and evaluates it on the hold-out test set stored in the 'test' subfolder of the FE folder.
    Writes a report and produces grid plots:
      1. Heatmaps for hold-out test metrics and (if available) K-Fold metrics.
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
    The figures are saved in corresponding directories.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    results = {}
    nr_list = sorted([d for d in os.listdir(feature_extraction_base_dir)
                      if os.path.isdir(os.path.join(feature_extraction_base_dir, d))])
    # Loop through each combination.
    for nr in nr_list:
        nr_path = os.path.join(feature_extraction_base_dir, nr)
        fe_list = sorted([d for d in os.listdir(nr_path)
                          if os.path.isdir(os.path.join(nr_path, d))])
        for fe in fe_list:
            feat_folder = os.path.join(nr_path, fe)
            test_folder = os.path.join(feat_folder, "test")
            rel_test_folder = os.path.relpath(test_folder, project_root)
            # Loop through the regularization options.
            for reg in ["none", "lasso", "ridge", "ElasticNet"]:
                model_name = f"lr_{nr}_{fe}_{reg}.keras"
                model_path = os.path.join(models_dir, model_name)
                rel_model_path = os.path.relpath(model_path, project_root)
                if not os.path.exists(model_path):
                    print(f"[TE] No model found for {nr}, {fe}, reg={reg} at {rel_model_path}. Skipping.")
                    continue
                print(f"[TE] Evaluating model {model_name} on hold-out test set from {rel_test_folder}")
                metrics = test_model_on_holdout(model_path, test_folder)
                if metrics is not None:
                    results[(nr, fe, reg)] = metrics

    # Write hold-out report.
    with open(report_path, "w") as f:
        for key in sorted(results.keys()):
            nr, fe, reg = key
            m = results[key]
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {fe}, Regularization: {reg}\n")
            f.write(f"MAE: {m['MAE']:.4f}\n")
            f.write(f"MSE: {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2: {m['R2']:.4f}\n\n")
    rel_report_path = os.path.relpath(report_path, project_root)
    print(f"[TE] Hold-out report saved to {rel_report_path}")

    plot_results_grid(results, project_root)


def plot_results_grid(results, project_root):
    """
    Produces grid plots for:
      1. Heatmaps showing four metrics (MAE, MSE, RMSE, R²) for hold-out test metrics
         as well as a second layer for K-Fold metrics (from report_kfold.txt).
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
    Figures are saved in separate directories.
    """
    # Create directories for saving images.
    base_save_dir = os.path.join(project_root, "output", "testing_lr")
    heatmaps_dir = os.path.join(base_save_dir, "heatmaps")
    pred_vs_actual_dir = os.path.join(base_save_dir, "predicted_vs_actual")
    residual_plots_dir = os.path.join(base_save_dir, "residual_plots")
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(pred_vs_actual_dir, exist_ok=True)
    os.makedirs(residual_plots_dir, exist_ok=True)

    # Attempt to load the K-Fold report.
    report_kfold_path = os.path.join(project_root, "output", "testing_lr", "report_kfold.txt")
    kfold_results = parse_report(report_kfold_path)
    if kfold_results:
        print(f"[TE] Parsed K-Fold report from {os.path.relpath(report_kfold_path, project_root)}")
    else:
        print("[TE] No valid K-Fold report found; only hold-out heatmaps will be plotted.")

    # Compute the union of regularization options from both hold-out and k-fold.
    holdout_regs = set(r for (_, _, r) in results.keys())
    kfold_regs = set(r for (_, _, r) in kfold_results.keys())
    reg_list = sorted(holdout_regs.union(kfold_regs))

    # For scatter and residual plots, we continue to use hold-out results.
    # (Only the heatmap plots are augmented with K-Fold data.)
    # Gather union keys for NR and FE across both hold-out and k-fold for each reg.
    for reg in reg_list:
        print(f"[TE] Plotting results for Regularization: {reg}")
        holdout_filtered = {(nr, fe): results[(nr, fe, reg)] for (nr, fe, r) in results.keys() if r == reg}
        kfold_filtered = {(nr, fe): kfold_results[(nr, fe, reg)] for (nr, fe, r) in kfold_results.keys() if r == reg}

        all_keys = set(holdout_filtered.keys()).union(set(kfold_filtered.keys()))
        if not all_keys:
            print(f"[TE] No data for reg {reg} in either hold-out or K-Fold; skipping heatmap.")
            continue

        all_nr = sorted({nr for (nr, fe) in all_keys})
        all_fe = sorted({fe for (nr, fe) in all_keys})
        n_nr = len(all_nr)
        n_fe = len(all_fe)

        # Create grids for hold-out metrics.
        mae_grid_hold = np.full((n_nr, n_fe), np.nan)
        mse_grid_hold = np.full((n_nr, n_fe), np.nan)
        rmse_grid_hold = np.full((n_nr, n_fe), np.nan)
        r2_grid_hold = np.full((n_nr, n_fe), np.nan)
        # Create grids for K-Fold metrics.
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

        # Plot a combined heatmap: top row for hold-out and bottom row for K-Fold.
        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        titles = ["MAE", "MSE", "RMSE", "R²"]
        # Top row: hold-out heatmaps.
        hold_grids = [mae_grid_hold, mse_grid_hold, rmse_grid_hold, r2_grid_hold]
        for idx, (grid, title) in enumerate(zip(hold_grids, titles)):
            ax = axes1[0, idx]
            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            ax.set_title(f"{title} (Hold-out)", fontsize=10)
            ax.set_xticks(np.arange(n_fe))
            ax.set_yticks(np.arange(n_nr))
            ax.set_xticklabels(all_fe, fontsize=8, rotation=45, ha="right")
            ax.set_yticklabels(all_nr, fontsize=8)
            fig1.colorbar(im, ax=ax)
        # Bottom row: K-Fold heatmaps.
        kfold_grids = [mae_grid_kfold, mse_grid_kfold, rmse_grid_kfold, r2_grid_kfold]
        for idx, (grid, title) in enumerate(zip(kfold_grids, titles)):
            ax = axes1[1, idx]
            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            ax.set_title(f"{title} (K-Fold)", fontsize=10)
            ax.set_xticks(np.arange(n_fe))
            ax.set_yticks(np.arange(n_nr))
            ax.set_xticklabels(all_fe, fontsize=8, rotation=45, ha="right")
            ax.set_yticklabels(all_nr, fontsize=8)
            fig1.colorbar(im, ax=ax)

        fig1.suptitle(f"Hold-out and K-Fold Test Metrics Heatmaps (Reg: {reg})", fontsize=12)
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        heatmap_path = os.path.join(heatmaps_dir, f"lr_{reg}_heatmap.png")
        rel_heatmap_path = os.path.relpath(heatmap_path, project_root)
        fig1.savefig(heatmap_path, dpi=300)
        print(f"[TE] Saved combined heatmap to {rel_heatmap_path}")
        plt.close(fig1)

    # The scatter and residual plots remain based on hold-out data.
    # Predicted vs. Actual scatter plots.
    # For these plots, we use the hold-out keys from the original results.
    keys = list(results.keys())  # (nr, fe, reg)
    holdout_nr_list = sorted({k[0] for k in results.keys()})
    holdout_fe_list = sorted({k[1] for k in results.keys()})
    n_nr_hold = len(holdout_nr_list)
    n_fe_hold = len(holdout_fe_list)
    for reg in sorted(holdout_regs):
        fig2, axes2 = plt.subplots(n_nr_hold, n_fe_hold, figsize=(4 * n_fe_hold, 4 * n_nr_hold), squeeze=False)
        mae_grid = np.full((n_nr_hold, n_fe_hold), np.nan)
        r2_grid = np.full((n_nr_hold, n_fe_hold), np.nan)
        scatter_data = {}
        for i, nr in enumerate(holdout_nr_list):
            for j, fe in enumerate(holdout_fe_list):
                key = (nr, fe, reg)
                if key in results:
                    m = results[key]
                    mae_grid[i, j] = m["MAE"]
                    r2_grid[i, j] = m["R2"]
                    scatter_data[(i, j)] = (m["y_true"], m["y_pred"])
        for i in range(n_nr_hold):
            for j in range(n_fe_hold):
                ax = axes2[i][j]
                if (i, j) in scatter_data:
                    y_true, y_pred = scatter_data[(i, j)]
                    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
                    ax.set_title(
                        f"{holdout_nr_list[i]} | {holdout_fe_list[j]}\nMAE={mae_grid[i, j]:.2f}, R²={r2_grid[i, j]:.2f}",
                        fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_xlabel("Actual", fontsize=8)
                ax.set_ylabel("Predicted", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
        fig2.suptitle(f"Predicted vs. Actual (Hold-out Test Set) (Reg: {reg})", fontsize=10)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        pred_vs_actual_path = os.path.join(pred_vs_actual_dir, f"lr_{reg}_predicted_vs_actual.png")
        rel_pred_vs_actual_path = os.path.relpath(pred_vs_actual_path, project_root)
        fig2.savefig(pred_vs_actual_path, dpi=300)
        print(f"[TE] Saved predicted vs. actual plot to {rel_pred_vs_actual_path}")
        plt.close(fig2)

        # Residual plots.
        fig3, axes3 = plt.subplots(n_nr_hold, n_fe_hold, figsize=(4 * n_fe_hold, 4 * n_nr_hold), squeeze=False)
        for i in range(n_nr_hold):
            for j in range(n_fe_hold):
                ax = axes3[i][j]
                key = (holdout_nr_list[i], holdout_fe_list[j], reg)
                if key in results:
                    m = results[key]
                    y_pred = m["y_pred"]
                    residuals = m["y_true"] - y_pred
                    ax.scatter(y_pred, residuals, alpha=0.6, s=20)
                    ax.axhline(0, color='red', linestyle='--', linewidth=1)
                    ax.set_title(f"{holdout_nr_list[i]} | {holdout_fe_list[j]}", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Residual", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
        fig3.suptitle(f"Residual Plots (Hold-out Test Set) (Reg: {reg})", fontsize=10)
        fig3.tight_layout(rect=[0, 0, 1, 0.95])
        residual_plot_path = os.path.join(residual_plots_dir, f"lr_{reg}_residual_plot.png")
        rel_residual_plot_path = os.path.relpath(residual_plot_path, project_root)
        fig3.savefig(residual_plot_path, dpi=300)
        print(f"[TE] Saved residual plot to {rel_residual_plot_path}")
        plt.close(fig3)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "output", "models_lr")
    feature_extraction_base_dir = os.path.join(base_dir, "intermediate", "feature_extraction")
    report_path = os.path.join(base_dir, "output", "testing", "feature_test_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    test_all_lr_models(models_dir, feature_extraction_base_dir, report_path)

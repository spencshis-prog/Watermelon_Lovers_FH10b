import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import functions


def test_model_on_holdout(pipeline, model_path, test_folder):
    """
    Loads a final model and evaluates it on the hold-out test set.
    Expects test_folder to contain the held-out .npy feature files.
    Returns a dictionary of test metrics and predictions.
    """
    if not os.path.exists(test_folder):
        print(f"[{pipeline.model_tag.upper()}-TE] Test folder {test_folder} does not exist. Skipping.")
        return None

    data = functions.load_feature_data_wrapped(test_folder)
    if len(data["X"]) == 0:
        print(f"[{pipeline.model_tag.upper()}-TE] No test data found.")
        return None

    X, y, fnames = pipeline.prime(data["X"], data["y"], data["fnames"])

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "y_true": y, "y_pred": y_pred}


def test_all(pipeline):
    """
    Iterates over each (NR, FE, hyper-tuning) combination in feature_extraction_base_dir,
    loads the corresponding final model (named <model_name>_<NR>_<FE>_<ht>.pkl),
    and evaluates it on the hold-out test set stored in the 'test' subfolder of the FE folder.
    Writes a report and produces grid plots:
      1. Heatmaps for hold-out test metrics.
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    results = {}
    # Get sorted list of noise reduction (NR) folders.
    nr_list = sorted([d for d in os.listdir(pipeline.fe_base_dir)
                      if os.path.isdir(os.path.join(pipeline.fe_base_dir, d))])
    # Loop over each NR and then each feature extraction (FE) subfolder.
    for nr in nr_list:
        nr_path = os.path.join(pipeline.fe_base_dir, nr)
        fe_list = sorted([d for d in os.listdir(nr_path)
                          if os.path.isdir(os.path.join(nr_path, d))])
        for fe in fe_list:
            feat_folder = os.path.join(nr_path, fe)
            test_folder = os.path.join(feat_folder, "test")
            if not os.path.exists(test_folder) or not os.listdir(test_folder):
                print(f"[{pipeline.model_tag.upper()}-TE] No test data for {nr}/{fe}. Skipping.")
                continue

            # Loop through hyperparameter tuning options.
            for ht in pipeline.ht_options:
                file_name = f"{pipeline.model_tag.lower()}_{nr}_{fe}_{ht}.pkl"
                model_path = os.path.join(pipeline.models_output_dir, file_name)
                rel_model_path = os.path.relpath(model_path, project_root)
                if not os.path.exists(model_path):
                    print(f"[{pipeline.model_tag.upper()}-TE] No model found for {nr}, {fe}, ht={ht} at {rel_model_path}. Skipping.")
                    continue
                print(
                    f"[{pipeline.model_tag.upper()}-TE] Evaluating model {file_name} on test set from {os.path.relpath(test_folder, project_root)}")
                metrics = test_model_on_holdout(pipeline, model_path, test_folder)
                if metrics is not None:
                    results[(nr, fe, ht)] = metrics

    # Write a consolidated report.
    with open(pipeline.report_holdout_path, "w") as f:
        for (nr, fe, ht), m in sorted(results.items()):
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {fe}, Hyper-tuning: {ht}\n")
            f.write(f"MAE: {m['MAE']:.4f}\n")
            f.write(f"MSE: {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2: {m['R2']:.4f}\n\n")
    rel_report_path = os.path.relpath(pipeline.report_holdout_path, project_root)
    print(f"[{pipeline.model_tag.upper()}-TE] Report saved to {rel_report_path}")

    plot_results_grid(pipeline, results, project_root)


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
            ht = parts[2].split("Hyper-tuning:")[-1].strip()
        except Exception as e:
            print(f"[Parse] Error parsing header: {header}")
            idx += 1
            continue

        try:
            mae = float(lines[idx + 1].split("MAE:")[-1].strip())
            mse = float(lines[idx + 2].split("MSE:")[-1].strip())
            rmse = float(lines[idx + 3].split("RMSE:")[-1].strip())
            r2 = float(lines[idx + 4].split("R2:")[-1].strip())
            results[(nr, fe, ht)] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        except Exception as e:
            print(f"[Parse] Error parsing metrics for {header}")
        idx += 5  # Move to next block
    return results


def plot_results_grid(pipeline, results, project_root):
    """
    Produces grid plots for:
      1. Combined heatmaps for hold-out test metrics and (if available) K-Fold metrics.
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
    Figures are saved and displayed.
    """
    heatmaps_dir = os.path.join(pipeline.testing_output_dir, "heatmaps")
    pred_vs_actual_dir = os.path.join(pipeline.testing_output_dir, "predicted_vs_actual")
    residual_plots_dir = os.path.join(pipeline.testing_output_dir, "residual_plots")
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(pred_vs_actual_dir, exist_ok=True)
    os.makedirs(residual_plots_dir, exist_ok=True)

    # Attempt to load the K-Fold report.
    report_kfold_path = os.path.join(project_root, "../../output", f"testing_{pipeline.model_tag.lower()}", "report_kfold.txt")
    kfold_results = parse_report(report_kfold_path)
    if kfold_results:
        print(f"[{pipeline.model_tag.upper()}-TE] Parsed K-Fold report from {os.path.relpath(report_kfold_path, project_root)}")
    else:
        print(f"[{pipeline.model_tag.upper()}-TE] No valid K-Fold report found; only hold-out heatmaps will be plotted.")

    # Produce separate grids for each hyperparameter tuning option.
    ht_options = sorted(list(set([ht for (_, _, ht) in results.keys()])))
    for ht in ht_options:
        print(f"[{pipeline.model_tag.upper()}-TE] Plotting results for Hyper-tuning: {ht}")
        # Filter hold-out results for current hyper-tuning.
        holdout_filtered = {(nr, fe): results[(nr, fe, ht)]
                            for (nr, fe, tuning) in results.keys() if tuning == ht}
        # Filter K-Fold results (if available) for current hyper-tuning.
        kfold_filtered = {}
        if kfold_results:
            for (nr, fe, tuning) in kfold_results.keys():
                if tuning == ht:
                    kfold_filtered[(nr, fe)] = kfold_results[(nr, fe, tuning)]

        # --- Heatmap Plotting ---
        if kfold_filtered:
            # Use the union of keys for a consistent grid.
            all_keys = set(holdout_filtered.keys()).union(set(kfold_filtered.keys()))
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

            # Plot combined heatmaps: top row (hold-out) and bottom row (K-Fold).
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

            fig1.suptitle(f"Test Metrics Heatmaps (HT: {ht})", fontsize=12)
            fig1.tight_layout(rect=[0, 0, 1, 0.95])
            heatmap_path = os.path.join(heatmaps_dir, f"{pipeline.model_tag.lower()}_{ht}_heatmap.png")
            rel_heatmap_path = os.path.relpath(heatmap_path, project_root)
            fig1.savefig(heatmap_path, dpi=300)
            print(f"[{pipeline.model_tag.upper()}-TE] Saved combined heatmap to {rel_heatmap_path}")
            plt.close(fig1)
        else:
            # No K-Fold data: plot only hold-out heatmaps.
            nr_list = sorted(list(set([key[0] for key in holdout_filtered.keys()])))
            fe_list = sorted(list(set([key[1] for key in holdout_filtered.keys()])))
            n_nr = len(nr_list)
            n_fe = len(fe_list)
            mae_grid = np.full((n_nr, n_fe), np.nan)
            mse_grid = np.full((n_nr, n_fe), np.nan)
            rmse_grid = np.full((n_nr, n_fe), np.nan)
            r2_grid = np.full((n_nr, n_fe), np.nan)
            for i, nr in enumerate(nr_list):
                for j, fe in enumerate(fe_list):
                    key = (nr, fe)
                    if key in holdout_filtered:
                        m = holdout_filtered[key]
                        mae_grid[i, j] = m["MAE"]
                        mse_grid[i, j] = m["MSE"]
                        rmse_grid[i, j] = m["RMSE"]
                        r2_grid[i, j] = m["R2"]
            fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
            metrics = [("MAE", mae_grid), ("MSE", mse_grid), ("RMSE", rmse_grid), ("R²", r2_grid)]
            for idx, (title, grid) in enumerate(metrics):
                ax = axes1[idx]
                im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
                ax.set_title(f"{title} (Hold-out) - {ht}", fontsize=10)
                ax.set_xticks(np.arange(n_fe))
                ax.set_yticks(np.arange(n_nr))
                ax.set_xticklabels(fe_list, fontsize=8, rotation=45, ha="right")
                ax.set_yticklabels(nr_list, fontsize=8)
                fig1.colorbar(im, ax=ax)
            fig1.suptitle(f"Hold-out Test Metrics Heatmaps (HT: {ht})", fontsize=12)
            fig1.tight_layout(rect=[0, 0, 1, 0.95])
            heatmap_path = os.path.join(heatmaps_dir, f"{pipeline.model_tag.lower()}_{ht}_heatmap.png")
            rel_heatmap_path = os.path.relpath(heatmap_path, project_root)
            fig1.savefig(heatmap_path, dpi=300)
            print(f"[{pipeline.model_tag.upper()}-TE] Saved hold-out heatmap to {rel_heatmap_path}")
            plt.close(fig1)

        # --- Scatter and Residual Plots (Hold-out only) ---
        # Use only the hold-out filtered keys for scatter/residual plots.
        scatter_nr = sorted(list(set([nr for (nr, fe) in holdout_filtered.keys()])))
        scatter_fe = sorted(list(set([fe for (nr, fe) in holdout_filtered.keys()])))
        n_nr_scatter = len(scatter_nr)
        n_fe_scatter = len(scatter_fe)
        scatter_data = {}
        residual_data = {}
        for i, nr in enumerate(scatter_nr):
            for j, fe in enumerate(scatter_fe):
                key = (nr, fe)
                if key in holdout_filtered:
                    m = holdout_filtered[key]
                    scatter_data[(i, j)] = (m["y_true"], m["y_pred"])
                    residual_data[(i, j)] = (m["y_pred"], m["y_true"] - m["y_pred"])

        # Predicted vs. Actual Scatter Plots.
        fig2, axes2 = plt.subplots(n_nr_scatter, n_fe_scatter, figsize=(4 * n_fe_scatter, 4 * n_nr_scatter),
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
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_xlabel("Actual", fontsize=8)
                ax.set_ylabel("Predicted", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
        fig2.suptitle(f"Predicted vs. Actual (Hold-out Test Set) (HT: {ht})", fontsize=10)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        pred_vs_actual_path = os.path.join(pred_vs_actual_dir, f"{pipeline.model_tag.lower()}_{ht}_predicted_vs_actual.png")
        rel_pred_vs_actual_path = os.path.relpath(pred_vs_actual_path, project_root)
        fig2.savefig(pred_vs_actual_path, dpi=300)
        print(f"[{pipeline.model_tag.upper()}-TE] Saved predicted vs. actual plot to {rel_pred_vs_actual_path}")
        plt.close(fig2)

        # Residual Plots.
        fig3, axes3 = plt.subplots(n_nr_scatter, n_fe_scatter, figsize=(4 * n_fe_scatter, 4 * n_nr_scatter),
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
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("Residual", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
        fig3.suptitle(f"Residual Plots (Hold-out Test Set) (HT: {ht})", fontsize=10)
        fig3.tight_layout(rect=[0, 0, 1, 0.95])
        residual_plot_path = os.path.join(residual_plots_dir, f"{pipeline.model_tag.lower()}_{ht}_residual_plot.png")
        rel_residual_plot_path = os.path.relpath(residual_plot_path, project_root)
        fig3.savefig(residual_plot_path, dpi=300)
        print(f"[{pipeline.model_tag.upper()}-TE] Saved residual plot to {rel_residual_plot_path}")
        plt.close(fig3)

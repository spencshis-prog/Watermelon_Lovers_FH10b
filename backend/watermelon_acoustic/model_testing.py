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
    if not os.path.exists(test_folder):
        print(f"[TE] Test folder {test_folder} does not exist. Skipping.")
        return None

    X, y, _ = load_feature_data(test_folder)
    if len(X) == 0:
        print(f"[TE] No test data in {test_folder}.")
        return None

    # For an MLP, features are assumed to be 1D vectors (batch, n_features)
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X).flatten()
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        "y_true": y, "y_pred": y_pred
    }


def test_all_feature_models(models_dir, feature_extraction_base_dir, report_path):
    """
    Iterates over each (NR, FE) combination in feature_extraction_base_dir,
    loads the corresponding final model (model_<NR>_<FE>.keras),
    and evaluates it on the hold-out test set stored in the 'test' subfolder of the FE folder.
    Collects metrics in a dictionary and writes a report.
    Then produces grid plots for:
      1. Four heatmaps (MAE, MSE, RMSE, R²) based on the hold-out test set.
      2. Predicted vs. Actual scatter plots.
      3. Residual plots.
    """
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
            model_name = f"model_{nr}_{fe}.keras"
            model_path = os.path.join(models_dir, model_name)
            if not os.path.exists(model_path):
                print(f"[TE] No model found for {nr} and {fe} at {model_path}. Skipping.")
                continue
            print(f"[TE] Evaluating model {model_name} on hold-out test set from {test_folder}")
            metrics = test_model_on_holdout(model_path, test_folder)
            if metrics is not None:
                results[(nr, fe)] = metrics

    # Write report to file.
    with open(report_path, "w") as f:
        for (nr, fe), m in results.items():
            f.write(f"Noise Reduction: {nr}, Feature Extraction: {fe}\n")
            f.write(f"MAE: {m['MAE']:.4f}\n")
            f.write(f"MSE: {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2: {m['R2']:.4f}\n\n")
    print(f"[TE] Report saved to {report_path}")

    # Produce grid plots
    plot_results_grid(results)


def plot_results_grid(results):
    """
    Produces grid plots for:
      1. Four heatmaps (MAE, MSE, RMSE, R²) based on hold-out test metrics.
      2. A grid of Predicted vs. Actual scatter plots.
      3. A grid of Residual plots.
    Font sizes are reduced to avoid overlap.
    """
    nr_list = sorted(list(set([k[0] for k in results.keys()])))
    fe_list = sorted(list(set([k[1] for k in results.keys()])))
    n_nr, n_fe = len(nr_list), len(fe_list)

    mae_grid = np.full((n_nr, n_fe), np.nan)
    mse_grid = np.full((n_nr, n_fe), np.nan)
    rmse_grid = np.full((n_nr, n_fe), np.nan)
    r2_grid = np.full((n_nr, n_fe), np.nan)

    scatter_data = {}
    residual_data = {}

    for i, nr in enumerate(nr_list):
        for j, fe in enumerate(fe_list):
            key = (nr, fe)
            if key in results:
                m = results[key]
                mae_grid[i, j] = m["MAE"]
                mse_grid[i, j] = m["MSE"]
                rmse_grid[i, j] = m["RMSE"]
                r2_grid[i, j] = m["R2"]
                y_true = m["y_true"]
                y_pred = m["y_pred"]
                scatter_data[(i, j)] = (y_true, y_pred)
                residual_data[(i, j)] = (y_pred, y_true - y_pred)

    # Figure 1: Four Heatmaps side by side.
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    grids = [mae_grid, mse_grid, rmse_grid, r2_grid]
    titles = ["MAE", "MSE", "RMSE", "R²"]
    for idx, (grid, title) in enumerate(zip(grids, titles)):
        ax = axes1[idx]
        c = ax.imshow(grid, cmap='viridis', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(n_fe))
        ax.set_yticks(np.arange(n_nr))
        ax.set_xticklabels(fe_list, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(nr_list, fontsize=8)
        fig1.colorbar(c, ax=ax)
    fig1.suptitle("Hold-out Test Metrics Heatmaps", fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # Figure 2: Predicted vs. Actual scatter plots grid.
    fig2, axes2 = plt.subplots(n_nr, n_fe, figsize=(4 * n_fe, 4 * n_nr), squeeze=False)
    for i in range(n_nr):
        for j in range(n_fe):
            ax = axes2[i][j]
            if (i, j) in scatter_data:
                y_true, y_pred = scatter_data[(i, j)]
                ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
                ax.set_title(f"{nr_list[i]} | {fe_list[j]}\nMAE={mae_grid[i, j]:.2f}, R²={r2_grid[i, j]:.2f}",
                             fontsize=8)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            ax.set_xlabel("Actual", fontsize=8)
            ax.set_ylabel("Predicted", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
    fig2.suptitle("Predicted vs. Actual (Hold-out Test Set)", fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # Figure 3: Residual plots grid.
    fig3, axes3 = plt.subplots(n_nr, n_fe, figsize=(4 * n_fe, 4 * n_nr), squeeze=False)
    for i in range(n_nr):
        for j in range(n_fe):
            ax = axes3[i][j]
            if (i, j) in residual_data:
                y_pred, residuals = residual_data[(i, j)]
                ax.scatter(y_pred, residuals, alpha=0.6, s=20)
                ax.axhline(0, color='red', linestyle='--', linewidth=1)
                ax.set_title(f"{nr_list[i]} | {fe_list[j]}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Residual", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
    fig3.suptitle("Residual Plots (Hold-out Test Set)", fontsize=10)
    fig3.tight_layout(rect=[0, 0, 1, 0.95])

    save_dir = os.path.join(os.getcwd(), "output", "testing")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig1.savefig(os.path.join(save_dir, "heatmaps.png"))
    fig2.savefig(os.path.join(save_dir, "predicted_vs_actual.png"))
    fig3.savefig(os.path.join(save_dir, "residual_plots.png"))
    print(f"[TE] Figures saved in {save_dir}")

    # Display all figures concurrently.
    plt.show()
    input("Press Enter to close all figures...")
    plt.close('all')


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "output", "models")
    feature_extraction_base_dir = os.path.join(base_dir, "output", "feature_extraction")
    report_path = os.path.join(base_dir, "output", "testing", "feature_test_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    test_all_feature_models(models_dir, feature_extraction_base_dir, report_path)

import os
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_wav_data(folder):
    """Loads .wav from folder, returns (X, y). Same as in training but ignoring filenames."""
    data, labels = [], []
    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            try:
                brix_val = float(parts[1])
            except:
                continue
            wav_path = os.path.join(folder, file)
            audio_bin = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1).numpy()
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
            else:
                audio = audio[:16000]
            data.append(audio)
            labels.append(brix_val)
    return np.array(data), np.array(labels)


def test_model_on_folder(model_path, folder):
    """Loads model, runs inference on all .wav in `folder`, returns metrics."""
    model = tf.keras.models.load_model(model_path)
    X, y = load_wav_data(folder)
    if len(X) == 0:
        print(f"No .wav files found in {folder} for testing.")
        return None
    X = X.reshape(-1, 16000, 1)
    y_pred = model.predict(X).flatten()

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {
        "folder": folder,
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        "y_true": y, "y_pred": y_pred
    }


def plot_all_results(results):
    """
    Plots multiple comparisons for a list of (technique_name, metrics_dict) pairs.
    metrics_dict should contain:
      "MAE", "MSE", "RMSE", "R2", "y_true", "y_pred"
    This function creates three separate figures:
      1) A grouped bar chart of MAE, MSE, RMSE, R2 for each technique
      2) A multi-subplot figure of predicted vs. actual for each technique
      3) A multi-subplot figure of residuals (actual - predicted) for each technique
    """
    # --------------------------------------------------------------------------
    # 1) GROUPED BAR CHART FOR MAE, MSE, RMSE, R2
    # --------------------------------------------------------------------------
    metrics_to_plot = ["MAE", "MSE", "RMSE", "R2"]
    metrics_labels = ["MAE", "MSE", "RMSE", "R$^2$"]  # what to display on the plot
    techniques = [r[0] for r in results]  # e.g. ["technique1", "technique2", ...]
    n_techs = len(techniques)
    n_metrics = len(metrics_to_plot)

    # Create a 2D array of shape (n_techs, n_metrics)
    values = []
    for _, m in results:
        row = [m[metric] for metric in metrics_to_plot]
        values.append(row)
    values = np.array(values)  # shape: (n_techs, n_metrics)

    x = np.arange(n_metrics)  # for the metric categories
    width = 0.8 / n_techs  # width of each bar group

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for i in range(n_techs):
        ax1.bar(x + i * width, values[i], width, label=techniques[i])
    ax1.set_xticks(x + width * (n_techs - 1) / 2)
    ax1.set_xticklabels(metrics_labels)
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Metrics Comparison Across Techniques")
    ax1.legend()
    fig1.tight_layout()

    # --------------------------------------------------------------------------
    # 2) PREDICTED VS. ACTUAL SCATTER PLOTS (one subplot per technique)
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, n_techs, figsize=(5 * n_techs, 4), sharex=False, sharey=False)
    if n_techs == 1:
        axes2 = [axes2]  # make it iterable if only one technique

    for i, (tech_name, m) in enumerate(results):
        ax = axes2[i]
        y_true = m["y_true"]
        y_pred = m["y_pred"]
        ax.scatter(y_true, y_pred, alpha=0.6)
        # plot diagonal if you want to see ideal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        ax.set_title(f"{tech_name}\nMAE={m['MAE']:.2f}, R$^2$={m['R2']:.2f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    fig2.suptitle("Predicted vs. Actual per Technique")
    fig2.tight_layout()

    # --------------------------------------------------------------------------
    # 3) RESIDUAL PLOTS (one subplot per technique)
    # residual = actual - predicted
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, n_techs, figsize=(5 * n_techs, 4), sharex=False, sharey=False)
    if n_techs == 1:
        axes3 = [axes3]  # make it iterable if only one technique

    for i, (tech_name, m) in enumerate(results):
        ax = axes3[i]
        y_true = m["y_true"]
        y_pred = m["y_pred"]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f"Residuals: {tech_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual - Predicted")

    fig3.suptitle("Residual Plots per Technique")
    fig3.tight_layout()

    # Finally, show all figures
    plt.show()


def test_all_techniques(models_dir, noise_reduction_base_dir, report_path):
    """
    For each technique subfolder (e.g. technique1) in noise_reduction_base_dir,
    find the matching model_{technique}.keras, test it on that technique's 'test/' subfolder,
    and compile results.
    """
    results = []
    for tech_folder in os.listdir(noise_reduction_base_dir):
        full_tech_path = os.path.join(noise_reduction_base_dir, tech_folder)
        if not os.path.isdir(full_tech_path):
            continue

        test_folder = os.path.join(full_tech_path, "test")  # We created a 'test' subfolder
        model_name = f"model_{tech_folder}.keras"
        model_path = os.path.join(models_dir, model_name)

        if not os.path.exists(model_path):
            print(f"No model found for technique '{tech_folder}' at {model_path}. Skipping.")
            continue

        print(f"Testing technique '{tech_folder}' with model {model_path} on {test_folder}")
        metrics = test_model_on_folder(model_path, test_folder)
        if metrics:
            results.append((tech_folder, metrics))

    # Write results to a report
    with open(report_path, "w") as f:
        for tech_folder, m in results:
            f.write(f"Technique: {tech_folder}\n")
            f.write(f"MAE: {m['MAE']:.4f}\n")
            f.write(f"MSE: {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2: {m['R2']:.4f}\n\n")
    print(f"Report saved to {report_path}")

    # Create bar chart for MAE
    if results:
        plot_all_results(results)
    else:
        print("No test results to plot.")


def predict_sweetness(audio_dir, model_path):
    """
    Loads a saved model and predicts brix values for each .wav file in audio_dir.
    Returns a dict {filename: predicted_value}.
    """
    model = tf.keras.models.load_model(model_path)
    predictions = {}
    for file in os.listdir(audio_dir):
        if file.lower().endswith(".wav"):
            wav_path = os.path.join(audio_dir, file)
            audio_bin = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1).numpy()
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), 'constant')
            else:
                audio = audio[:16000]
            audio = audio.reshape(1, 16000, 1)
            pred = model.predict(audio)[0][0]
            predictions[file] = pred
    return predictions


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    models_dir = os.path.join(base_dir, "output", "models")
    noise_reduction_base_dir = os.path.join(base_dir, "output", "noise_reduction", "combined")
    report_path = os.path.join(base_dir, "output", "testing", "test_report.txt")

    if not os.path.exists(os.path.dirname(report_path)):
        os.makedirs(os.path.dirname(report_path))

    test_all_techniques(models_dir, noise_reduction_base_dir, report_path)

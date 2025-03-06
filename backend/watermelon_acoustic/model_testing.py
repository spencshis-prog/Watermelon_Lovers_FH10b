import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
import main

def load_dataset_from_folder(folder):
    """
    Loads .wav files from folder, converts them to numpy arrays,
    and extracts labels from filenames (assumes format: <watermelonID>_<brix>_<recording>.wav).
    """
    data = []
    brix_vals = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            try:
                parts = file.split("_")
                # brix value is the second element.
                brix_val = float(parts[1])
            except Exception as e:
                print(f"Could not extract label from {file}: {e}")
                continue
            wav_path = os.path.join(folder, file)
            audio_binary = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1)
            audio_np = audio.numpy()
            if len(audio_np) < 16000:
                audio_np = np.pad(audio_np, (0, 16000 - len(audio_np)), mode='constant')
            else:
                audio_np = audio_np[:16000]
            data.append(audio_np)
            brix_vals.append(brix_val)
    return np.array(data), np.array(brix_vals)

def test_model(model_path, test_folder):
    """
    Loads a model and tests it on data from test_folder.
    Computes metrics: MAE, MSE, RMSE, R2.
    """
    model = tf.keras.models.load_model(model_path)
    X, y = load_dataset_from_folder(test_folder)
    if len(X) == 0:
        print(f"No test data found in {test_folder}")
        return None
    X = X.reshape(-1, 16000, 1)
    y_pred = model.predict(X).flatten()
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {"model": model_path, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "y_true": y, "y_pred": y_pred}

def run_tests(models_dir, test_folder, report_file):
    """
    Tests every model in models_dir using data from test_folder.
    Outputs metrics to the terminal, writes a report, and generates a bar chart comparing MAE.
    """
    results = []
    for file in os.listdir(models_dir):
        if file.endswith(".keras"):
            model_path = os.path.join(models_dir, file)
            print(f"Testing model: {model_path}")
            metrics = test_model(model_path, test_folder)
            if metrics:
                results.append(metrics)
                print(f"Results for {file}: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, "
                      f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    # Write report to a text file
    with open(report_file, "w") as f:
        for res in results:
            f.write(f"Model: {res['model']}\n")
            f.write(f"MAE: {res['MAE']:.4f}\n")
            f.write(f"MSE: {res['MSE']:.4f}\n")
            f.write(f"RMSE: {res['RMSE']:.4f}\n")
            f.write(f"R2: {res['R2']:.4f}\n")
            f.write("\n")
    print(f"Report saved to {report_file}")

    # Generate a bar chart for MAE comparison
    model_names = [os.path.basename(r["model"]) for r in results]
    mae_values = [r["MAE"] for r in results]
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mae_values)
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Model MAE Comparison")
    chart_path = os.path.join(os.path.dirname(report_file), "mae_comparison.png")
    plt.savefig(chart_path)
    plt.show()
    print(f"Chart saved to {chart_path}")

def predict_sweetness(audio_dir, model_path):
    """
    Iterates over .wav files in audio_dir, predicts sweetness using the provided model,
    and returns a dictionary mapping filename to predicted brix value.
    """
    model = tf.keras.models.load_model(model_path)
    predictions = {}
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            wav_path = os.path.join(audio_dir, file)
            # Preprocess the audio file.
            audio_binary = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1).numpy()
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
            else:
                audio = audio[:16000]
            # Reshape to (1, 16000, 1)
            audio = np.expand_dims(audio, axis=0)
            audio = np.expand_dims(audio, axis=-1)
            pred = model.predict(audio)[0][0]
            predictions[file] = pred
    return predictions

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "output", "models")
    # Use the test_set folder that was physically created during training.
    test_folder = os.path.join(base_dir, "output", "test_set")
    report_file = os.path.join(base_dir, "output", "testing", "test_report.txt")
    main.clear_output_directory(os.path.dirname(report_file))
    run_tests(models_dir, test_folder, report_file)

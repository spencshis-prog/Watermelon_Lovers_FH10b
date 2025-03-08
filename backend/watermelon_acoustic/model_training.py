import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_model(input_shape=(16000, 1)):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def load_wav_data(folder):
    """
    Loads .wav files from a folder, returns (X, y, filenames).
    Assumes naming format: <watermelonID>_<brix>_<index>.wav
    e.g. 1_9.4_0.wav -> brix=9.4
    """
    data, labels, fnames = [], [], []
    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            parts = file.split("_")
            if len(parts) < 3:
                print(f"Skipping {file}, not in the expected format.")
                continue
            try:
                brix_val = float(parts[1])  # second field is the brix
            except Exception as e:
                print(f"Skipping {file}, brix not parseable.", e)
                continue

            wav_path = os.path.join(folder, file)
            audio_binary = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1).numpy()

            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
            else:
                audio = audio[:16000]

            data.append(audio)
            labels.append(brix_val)
            fnames.append(file)
    return np.array(data), np.array(labels), fnames


def copy_files(filenames, src_folder, dst_folder):
    """Utility to copy .wav files from src_folder to dst_folder."""
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for fn in filenames:
        src = os.path.join(src_folder, fn)
        dst = os.path.join(dst_folder, fn)
        shutil.copy(src, dst)


def train_model_for_technique(tech_folder, models_output_dir,
                              test_ratio=0.15, val_ratio=0.15,
                              epochs=20, batch_size=16):
    """
    1) Load all .wav data from tech_folder.
    2) Split into train/val/test in memory (70/15/15).
    3) Copy files into subfolders: tech_folder/train, tech_folder/val, tech_folder/test.
    4) Train an LSTM model on the train set (with val for validation).
    5) Save the model to models_output_dir.
    """
    rel_tech = os.path.relpath(tech_folder, os.getcwd())
    print(f"--- Processing technique folder: {rel_tech} ---")

    # Step A: Load data
    X, y, fnames = load_wav_data(tech_folder)
    if len(X) == 0:
        print(f"No data found in {rel_tech}, skipping.")
        return

    # Make subfolders for train/val/test
    train_subfolder = os.path.join(tech_folder, "train")
    val_subfolder = os.path.join(tech_folder, "val")
    test_subfolder = os.path.join(tech_folder, "test")

    for folder in [train_subfolder, val_subfolder, test_subfolder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Step B: Split out test (15%)
    X_trainval, X_test, y_trainval, y_test, fn_trainval, fn_test = train_test_split(
        X, y, fnames, test_size=test_ratio, random_state=42
    )

    # Step C: Split out val (15% of entire dataset => 0.176 of leftover)
    effective_val_split = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val, fn_train, fn_val = train_test_split(
        X_trainval, y_trainval, fn_trainval,
        test_size=effective_val_split,
        random_state=42
    )

    # Copy files physically
    copy_files(fn_train, tech_folder, train_subfolder)
    copy_files(fn_val, tech_folder, val_subfolder)
    copy_files(fn_test, tech_folder, test_subfolder)

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Step D: Train in memory
    X_train = X_train.reshape(-1, 16000, 1)
    X_val = X_val.reshape(-1, 16000, 1)

    model = build_model(input_shape=(16000, 1))
    tech_name = os.path.basename(tech_folder)
    print(f"Starting training on {tech_name} dataset (LSTM regression model)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Save the model
    model_name = f"model_{tech_name}.keras"
    model_path = os.path.join(models_output_dir, model_name)
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)
    model.save(model_path)
    print(f"Model for technique '{tech_name}' saved to {model_path}")


def train_all_techniques(noise_reduction_base_dir, models_output_dir,
                         test_ratio=0.15, val_ratio=0.15,
                         epochs=20, batch_size=16):
    """
    Iterates over each technique subfolder (e.g. 'technique1', 'technique2')
    under noise_reduction_base_dir, trains a model, and saves it in models_output_dir.
    """
    if not os.path.exists(models_output_dir):
        os.makedirs(models_output_dir)

    for tech_folder in os.listdir(noise_reduction_base_dir):
        full_path = os.path.join(noise_reduction_base_dir, tech_folder)
        if os.path.isdir(full_path):
            train_model_for_technique(
                tech_folder=full_path,
                models_output_dir=models_output_dir,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                epochs=epochs,
                batch_size=batch_size
            )
    print("All techniques have been trained.")

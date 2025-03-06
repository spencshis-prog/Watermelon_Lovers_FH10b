import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

import main


def load_dataset_from_folder(folder):
    """
    Loads .wav files from a folder, converts them to numpy arrays,
    and extracts labels from file names (assumes format: <label>_filename.wav).
    """
    data = []
    brix_vals = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            try:
                brix_str = file.split("_")[1]  # bc files are in {watermelon}_{brix}_{number} format
                brix_val = float(brix_str)
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


def build_model(input_shape=(16000, 1)):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model_for_folder(data_folder, output_model_path,
                           epochs=20, batch_size=16,
                           use_separate_test_set=False,
                           val_split=0.2,
                           test_split=0.0):
    """
    Loads data from data_folder, optionally does a train/val/test split, trains the model, and saves it.
    """
    print(f"Loading dataset from {data_folder}")
    X, y = load_dataset_from_folder(data_folder)
    if len(X) == 0:
        print(f"No data found in {data_folder}. Skipping training.")
        return None

    # If we want a separate test set, do a two-step split:
    if use_separate_test_set and test_split > 0:
        # e.g. 70/15/15 if val_split=0.15 and test_split=0.15
        # First, carve out test from the entire set:
        from sklearn.model_selection import train_test_split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )

        # Now carve out val from trainval
        # e.g. 15% of the 85% leftover => 0.176 (which is ~15% of the original total)
        effective_val_split = val_split / (1.0 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=effective_val_split, random_state=42
        )

        # We could optionally hold onto X_test, y_test for final testing
        # or just skip it if we want to evaluate inside this function.
        # Typically, we'd save it or pass it to a separate testing function.
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    else:
        # If not using a separate test set, do your usual single 80/20 train/val:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Reshape for LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    model = build_model(input_shape=(16000, 1))
    print("Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size
    )

    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")
    return output_model_path


def train_models(noise_reduction_base_dir, output_models_dir,
                 use_separate_test_set=False,
                 val_split=0.2,
                 test_split=0.2):
    """
    Iterates over each noise reduction technique folder under noise_reduction_base_dir,
    trains a model for each grouping, and saves the models in output_models_dir.
    """
    main.clear_output_directory(output_models_dir)

    for tech_folder in os.listdir(noise_reduction_base_dir):
        tech_path = os.path.join(noise_reduction_base_dir, tech_folder)
        if os.path.isdir(tech_path):
            model_path = os.path.join(output_models_dir, f"model_{tech_folder}.keras")
            print(f"Training model for noise reduction technique: {tech_folder}")

            # Pass the config to train_model_for_folder
            train_model_for_folder(
                data_folder=tech_path,
                output_model_path=model_path,
                use_separate_test_set=use_separate_test_set,
                val_split=val_split,
                test_split=test_split
            )
    print("All models trained.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming combined noise reduction folder is here if both datasets were used:
    noise_reduction_base_dir = os.path.join(base_dir, "output", "noise_reduction", "combined")
    output_models_dir = os.path.join(base_dir, "output", "models")
    train_models(noise_reduction_base_dir, output_models_dir)

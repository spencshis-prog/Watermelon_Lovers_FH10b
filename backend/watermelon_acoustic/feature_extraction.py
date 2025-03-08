# feature_extraction.py
import os
import shutil
import numpy as np
import librosa
import pywt  # for wavelet extraction
import main  # to use clear_output_directory


def extract_features_raw(audio, sr=16000):
    """
    Raw features: simply return the audio waveform as is.
    """
    # Optionally, you might want to normalize the audio.
    return audio


def extract_features_mfcc(audio, sr=16000, n_mfcc=13):
    """
    Computes MFCC features and returns the mean coefficients over time.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def extract_features_spectral(audio, sr=16000):
    """
    Computes the spectral centroid and returns its mean value as a 1D feature vector.
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = np.mean(centroid, axis=1)
    return centroid_mean


def extract_features_wavelet(audio, sr=16000):
    """
    Applies a discrete wavelet transform using PyWavelets.
    Here we use a simple strategy: decompose with a Daubechies wavelet,
    then compute the mean absolute value of the detail coefficients.
    """
    wavelet = 'db1'
    # Determine maximum level for decomposition
    max_level = pywt.dwt_max_level(len(audio), wavelet)
    coeffs = pywt.wavedec(audio, wavelet, level=max_level)
    # Skip the approximation coefficients (coeffs[0]) and compute the mean abs value of each detail level.
    features = [np.mean(np.abs(detail)) for detail in coeffs[1:]]
    return np.array(features)


def extract_features_zcr(audio, sr=16000):
    """
    Zero-Crossing Rate: mean of zero-crossing rate across frames.
    """
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=512)
    return np.array([np.mean(zcr)])


def extract_features_rolloff(audio, sr=16000):
    """
    Spectral rolloff: the frequency below which a certain fraction
    (e.g. 85%) of the energy lies. We take the mean across time frames.
    """
    roll = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    return np.array([np.mean(roll)])


def extract_features_rms(audio, sr=16000):
    """
    Short-time RMS energy. We take the mean across frames.
    """
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)
    return np.array([np.mean(rms)])


def apply_feature_extraction(input_dir, output_base_dir):
    """
    Applies various feature extraction methods to every .wav file in input_dir.
    Creates separate subfolders (e.g., "raw", "mfcc", "spectral", "wavelet") under output_base_dir.
    Saves the resulting feature vectors as .npy files.
    """
    # Define the feature extraction methods.
    methods = {
        "raw": extract_features_raw,
        "mfcc": extract_features_mfcc,
        "spectral": extract_features_spectral,
        "wavelet": extract_features_wavelet,
        "zcr": extract_features_zcr,
        "rolloff": extract_features_rolloff,
        "rms": extract_features_rms
    }

    # Print a header with relative paths.
    rel_input = os.path.relpath(input_dir, os.getcwd())
    rel_output = os.path.relpath(output_base_dir, os.getcwd())
    print(f"Extracting features from files in {rel_input}.")
    print(f"Output will be organized under {rel_output} by extraction method.")

    # For each extraction method, create a clean subfolder.
    for method_name, func in methods.items():
        method_dir = os.path.join(output_base_dir, method_name)
        main.clear_output_directory(method_dir)

        # Process each .wav file in the input directory.
        for file in os.listdir(input_dir):
            if file.lower().endswith(".wav"):
                input_path = os.path.join(input_dir, file)
                try:
                    # Load audio using librosa with a target sample rate.
                    audio, sr = librosa.load(input_path, sr=16000)
                    features = func(audio, sr)

                    # Save features as a .npy file using the same basename.
                    base_name = os.path.splitext(file)[0]
                    out_file = base_name + ".npy"
                    output_path = os.path.join(method_dir, out_file)
                    np.save(output_path, features)

                    rel_in = os.path.relpath(input_path, os.getcwd())
                    rel_out = os.path.relpath(output_path, os.getcwd())
                    print(f"Extracted {method_name} features from {rel_in} -> saved to {rel_out}")
                except Exception as e:
                    print(f"Error processing {file} with {method_name}: {e}")
    print("Feature extraction processing complete.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Example: Process feature extraction for one noise reduction technique.
    # Adjust these paths as needed.
    input_dir = os.path.join(base_dir, "output", "noise_reduction", "combined", "technique1")
    output_base_dir = os.path.join(base_dir, "output", "feature_extraction", "technique1")
    apply_feature_extraction(input_dir, output_base_dir)

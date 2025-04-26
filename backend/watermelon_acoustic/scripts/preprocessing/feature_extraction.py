import os
import numpy as np
import librosa
import pywt  # for wavelet extraction

import functions


def ef_raw(audio, sr=16000):
    """
    Raw features: simply return the audio waveform as is.
    """
    return audio


def ef_mel_features_cepstral_coefficients(audio, sr=16000, n_mfcc=13):
    """
    Computes Mel-Frequency Cepstral Coefficients features
    and returns the mean coefficients over time.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def ef_spectral_centroid(audio, sr=16000):
    """
    Computes the spectral centroid and returns its mean value as a 1D feature vector.
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = np.mean(centroid, axis=1)
    return centroid_mean


def ef_db1_wavelet(audio, sr=16000):
    """
    Applies a discrete wavelet transform using PyWavelets.
    Here we use a simple strategy: decompose with a Daubechies wavelet,
    then compute the mean absolute value of the detail coefficients.
    """
    wavelet = 'db1'  # or db8
    # Determine maximum level for decomposition
    max_level = pywt.dwt_max_level(len(audio), wavelet)
    coeffs = pywt.wavedec(audio, wavelet, level=max_level)
    # Skip the approximation coefficients (coeffs[0]) and compute the mean abs value of each detail level.
    features = [np.mean(np.abs(detail)) for detail in coeffs[1:]]
    return np.array(features)


def ef_zero_cross_rate(audio, sr=16000):
    """
    Zero-Crossing Rate: mean of zero-crossing rate across frames.
    """
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=512)
    return np.array([np.mean(zcr)])


def ef_spectral_rolloff(audio, sr=16000):
    """
    Spectral rolloff: the frequency below which a certain fraction
    (e.g. 85%) of the energy lies. We take the mean across time frames.
    """
    roll = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    return np.array([np.mean(roll)])


def ef_rms(audio, sr=16000):
    """
    Short-time RMS energy. We take the mean across frames.
    """
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)
    return np.array([np.mean(rms)])


def ef_chroma(audio, sr=16000):
    """
    Computes chroma features using the Short-Time Fourier Transform (STFT).
    Returns the mean chroma vector over time as a 12-dimensional feature vector.
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    return chroma_mean


def ef_log_mel_spectrogram(audio, sr=16000, n_mels=40):
    """
    Computes a log-Mel spectrogram and returns a feature vector
    composed of the mean and standard deviation over time for each mel band.
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    mean_log_S = np.mean(log_S, axis=1)
    std_log_S = np.std(log_S, axis=1)
    return np.concatenate([mean_log_S, std_log_S])


def ef_delta_mfcc(audio, sr=16000, n_mfcc=13):
    """
    Computes MFCCs and their first and second order derivatives.
    Returns a concatenated feature vector (static + delta + delta-delta).
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    return np.concatenate([mfcc_mean, delta_mean, delta2_mean])


def ef_spectral_contrast(audio, sr=16000):
    """
    Computes spectral contrast and returns the mean contrast across time
    for each frequency band.
    """
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    return contrast_mean


def ef_percussive_mfcc(audio, sr=16000, n_mfcc=13):
    """
    Separates the audio into harmonic and percussive components,
    then computes MFCCs on the percussive part.
    """
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    mfcc = librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def ef_spectral_flux(audio, sr=16000, n_fft=2048, hop_length=512):
    """
    Computes spectral flux: the frame-to-frame change in the magnitude spectrum.
    Returns the mean spectral flux as a one-dimensional feature.
    """
    D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    # Calculate the difference between successive frames and take the norm
    flux = np.sqrt(np.sum(np.diff(D, axis=1) ** 2, axis=0))
    return np.array([np.mean(flux)])


def extract_features(input_dir, output_base_dir):
    """
    Applies various feature extraction methods to every .wav file in input_dir.
    Creates separate subfolders (e.g., "raw", "mfcc", "spectral", "wavelet") under output_base_dir.
    Saves the resulting feature vectors as .npy files.
    """
    methods = {  # as assessed by qilin set
        # "raw": ef_raw,  # scales badly with hypertuning runtimes
        "mfcc": ef_mel_features_cepstral_coefficients,  # high performance
        # "spectral_centroid": ef_spectral_centroid,  # poor performance qilin
        # "db1_wavelet": ef_db1_wavelet,  # mid performance
        # "zcr": ef_zero_cross_rate,  # good performance ONLY with naive spectral subtraction qilin
        # "spectral_rolloff": ef_spectral_rolloff,  # poor performance qilin
        # "rms": ef_rms,  # awful performance qilin
        # "chroma": ef_chroma,  # poor performance qilin
        "log_mel_spec": ef_log_mel_spectrogram,  # untested
        "delta_mfcc": ef_delta_mfcc,  # good performance typically only w steg
        # "spectral_contrast": ef_spectral_contrast,  # poor performance qilin
        # "percussive_mfcc": ef_percussive_mfcc,  # untested
        "spectral_flux": ef_spectral_flux,  # poor performance qilin
    }

    # Print a header with relative paths.
    rel_input = os.path.relpath(input_dir, os.getcwd())
    rel_output = os.path.relpath(output_base_dir, os.getcwd())
    print(f"[FE] Extracting features from files in {rel_input}.")
    print(f"[FE] Output will be organized under {rel_output} by extraction method.")

    # For each extraction method, create a clean subdirectory.
    for method_name, func in methods.items():
        method_dir = os.path.join(output_base_dir, method_name)
        functions.clear_output_directory(method_dir)

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
                    print(f"[FE] Extracted {method_name} features from {rel_in} -> saved to {rel_out}")
                except Exception as e:
                    print(f"[FE] Error processing {file} with {method_name}: {e}")
    print("Feature extraction processing complete.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Example: Process feature extraction for one noise reduction technique.
    # Adjust these paths as needed.
    input_dir = os.path.join(base_dir, "../../intermediate", "noise_reduction", "technique1")
    output_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction", "technique1")
    extract_features(input_dir, output_base_dir)

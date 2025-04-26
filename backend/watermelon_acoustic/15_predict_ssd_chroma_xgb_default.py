import os
import numpy as np
import joblib
from pydub import AudioSegment, effects
import librosa


# --- Step 1: Standardize the WAV file ---
def standardize(input_path, target_duration_ms=1000, target_sample_rate=16000, target_channels=1):
    """
    Loads a WAV file and adjusts its duration, sample rate, and number of channels.
    Pads with silence if too short or trims if too long.
    """
    try:
        audio = AudioSegment.from_wav(input_path)
    except Exception as e:
        raise RuntimeError(f"Could not load {input_path}: {e}")

    # Set sample rate and channels
    audio = audio.set_frame_rate(target_sample_rate).set_channels(target_channels)

    current_duration_ms = len(audio)
    if current_duration_ms < target_duration_ms:
        silence = AudioSegment.silent(duration=target_duration_ms - current_duration_ms,
                                      frame_rate=target_sample_rate)
        audio = audio + silence
    elif current_duration_ms > target_duration_ms:
        audio = audio[:target_duration_ms]
    return audio


# --- Step 2: Noise Reduction ---
def noise_reduce(audio: AudioSegment, frame_length=2048, hop_length=512, noise_frames_percent=0.1):
    """
    Computes the STFT of the input audio and identifies a fraction
    of the frames (determined by noise_frames_percent) with the lowest energy.
    It averages the magnitude of these frames to create a noise profile, and then
    subtracts this profile (scaled by alpha) from all frames.
    """
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # STFT
    D = librosa.stft(samples, n_fft=frame_length, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)

    # Identify frames by energy
    frame_energies = np.sum(magnitude, axis=0)
    num_noise_frames = int(noise_frames_percent * magnitude.shape[1])
    # Indices of frames with the lowest energy
    noise_frame_indices = np.argsort(frame_energies)[:num_noise_frames]

    # Estimate noise magnitude by averaging noise-only frames
    noise_mag = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)

    # Spectral subtraction
    alpha = 1.0
    magnitude_denoised = np.maximum(magnitude - alpha * noise_mag, 0.0)

    # Reconstruct
    D_denoised = magnitude_denoised * phase
    samples_denoised = librosa.istft(D_denoised, hop_length=hop_length)

    # Convert back to 16-bit
    samples_denoised_16bit = (samples_denoised * 32767).astype(np.int16)
    return audio._spawn(samples_denoised_16bit.tobytes())


# --- Step 3: Normalization ---
def normalize(audio: AudioSegment) -> AudioSegment:
    """
    Normalizes the AudioSegment using librosa effects (or any other normalization method).
    """
    normalized = effects.normalize(audio)
    return normalized


# --- Step 4: Feature Extraction ---
def extract_features(audio, sr=16000):
    """
    Computes chroma features using the Short-Time Fourier Transform (STFT).
    Returns the mean chroma vector over time as a 12-dimensional feature vector.
    """
    samples = np.array(audio.get_array_of_samples())

    # PyDub typically stores 16-bit PCM
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0  # scale to [-1, 1] float range

    chroma = librosa.feature.chroma_stft(y=samples, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    return chroma_mean.reshape(1, -1)  # important: reshape to (1, 12) for sklearn


# --- Step 5: Feature Generation (Transformation) ---
def generate_features(features):
    """
    Applies a transformation pipeline to the raw features.
    In production, you would load a pre-fitted pipeline; here we simulate by fitting on the fly.
    """
    from sklearn.preprocessing import PowerTransformer, RobustScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    transformation_pipeline = Pipeline([
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
    ])
    return transformation_pipeline.fit_transform(features)


# --- Step 6: Feature Selection ---
def select_features(features, k=50):
    """
    Selects the k best features from the generated feature set.
    In production, you should use a pre-fitted feature selector.
    Here we simulate fitting by using a dummy target since f_regression requires one.
    """
    from sklearn.feature_selection import SelectKBest, f_regression
    selector = SelectKBest(score_func=f_regression, k=k)
    dummy_target = np.zeros(features.shape[0])
    return selector.fit_transform(features, dummy_target)


# --- Step 7: Prediction Pipeline ---
def predict_brix(input_wav_path, model_pkl_path):
    """
    Runs a streamlined pipeline:
      1. Standardizes the WAV file.
      2. Applies noise reduction.
      3. Normalizes the audio.
      4. Extracts raw features.
      5. Generates new features.
      6. Applies feature selection.
      7. Loads a pretrained regressor from a PKL file.
      8. Outputs a predicted Brix score.
    """
    audio = standardize(input_wav_path)  # assuming 1 sec duration, 16 kHz sample rate, monochannel
    audio = noise_reduce(audio)  # using bandpass filter as example
    audio = normalize(audio)  # may not always be useful, run tests without j in case

    features = extract_features(audio)  # using mfcc extraction as example
    features = generate_features(features)  # y-j power transform, robust scale, 2nd degree PolyFeat gen
    features = select_features(features)  # choose best k features, k is defaulted to 50

    # Load the pretrained regressor (which we don't have yet)
    try:
        model = joblib.load(model_pkl_path)
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_pkl_path}: {e}")

    # Predict Brix score (assuming model.predict returns an array-like)
    prediction = model.predict(features)
    return prediction[0]


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_wav = os.path.join(base_dir, "input", "wav_lab", "1b_10.15_1.wav")  # replace w the actual .wav file
    model_path = os.path.join(base_dir, "15_model_ssd_chroma_xgb_default.pkl")  # replace w the actual .pkl file

    try:
        brix_score = predict_brix(input_wav, model_path)
        print("Predicted Brix: ", brix_score)
    except Exception as e:
        print("Error during prediction: ", e)

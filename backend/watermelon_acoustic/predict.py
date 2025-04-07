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
def noise_reduce(audio: AudioSegment, lowcut=100, highcut=2000) -> AudioSegment:
    """
    Applies a simple bandpass filter by chaining a high-pass and low-pass filter.
    Adjust lowcut and highcut as needed.
    """
    filtered = audio.high_pass_filter(lowcut)
    filtered = filtered.low_pass_filter(highcut)
    return filtered


# --- Step 3: Normalization ---
def normalize(audio: AudioSegment) -> AudioSegment:
    """
    Normalizes the AudioSegment using librosa effects (or any other normalization method).
    """
    normalized = effects.normalize(audio)
    return normalized


# --- Step 4: Feature Extraction ---
def extract_features(audio):
    """
    Converts the AudioSegment into a NumPy array and computes MFCC features using librosa.
    Returns a 2D array of shape (1, n_features) where n_features is the number of MFCC coefficients.
    """
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sr = audio.frame_rate
    # Normalize samples to the range [-1, 1]
    if np.max(np.abs(samples)) != 0:
        samples = samples / np.max(np.abs(samples))
    # Compute 13 MFCCs
    mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
    # Aggregate: take mean of each coefficient across frames
    mfcc_mean = np.mean(mfccs, axis=1)
    return mfcc_mean.reshape(1, -1)


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
    return int(round(prediction[0]))


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_wav = os.path.join(base_dir, "sample.wav")  # replace w the actual .wav file
    model_path = os.path.join(base_dir, "regressor_model.pkl")  # replace w the actual .pkl file

    try:
        brix_score = predict_brix(input_wav, model_path)
        print("Predicted Brix: ", brix_score)
    except Exception as e:
        print("Error during prediction: ", e)

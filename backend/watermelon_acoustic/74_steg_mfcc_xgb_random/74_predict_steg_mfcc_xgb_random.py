import os
import numpy as np
import joblib
from pydub import AudioSegment, effects
import librosa


# --- Step 1: Standardize the WAV file ---
def standardize(input_path, target_duration_ms=1000, target_sample_rate=16000, target_channels=1):
    try:
        audio = AudioSegment.from_wav(input_path)
    except Exception as e:
        raise RuntimeError(f"Could not load {input_path}: {e}")

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
def noise_reduce(audio: AudioSegment, frame_ms=10, gate_threshold=0.01, attenuation=0.1):
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.max(np.abs(samples)) + 1e-9  # guard-rail to keep in [-1,1]
    frame_len = int(sr * frame_ms / 1000)
    num_frames = len(samples) // frame_len
    for i in range(num_frames):
        start, end = i * frame_len, (i + 1) * frame_len
        frame = samples[start:end]
        if np.mean(frame ** 2) < gate_threshold:
            samples[start:end] *= attenuation
    gated_16bit = (samples * 32767).astype(np.int16)
    return audio._spawn(gated_16bit.tobytes())


# --- Step 3: Normalization ---
def normalize(audio: AudioSegment) -> AudioSegment:
    return effects.normalize(audio)


# --- Step 4: Feature Extraction ---
def mfcc(raw_audio, sr=16000, n_mfcc=13):
    """
    Computes MFCCs and returns the mean over time.
    """
    mfcc = librosa.feature.mfcc(y=raw_audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_features(audio: AudioSegment, sr=16000, n_mfcc=13):
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    mfcc_mean = mfcc(samples, sr=sr, n_mfcc=n_mfcc)
    return mfcc_mean.reshape(1, -1)


# --- Step 5: Feature Transformation ---
def apply_transformer(transformer_pkl_path, features):
    try:
        transformer = joblib.load(transformer_pkl_path)
    except Exception as e:
        raise RuntimeError(f"Could not load transformer from {transformer_pkl_path}: {e}")

    return transformer.transform(features)


# --- Step 6: Feature Selection ---
def apply_selector(selector_pkl_path, features):
    try:
        selector = joblib.load(selector_pkl_path)
    except Exception as e:
        raise RuntimeError(f"Could not load selector from {selector_pkl_path}: {e}")

    return selector.transform(features)


# --- Step 7: Prediction Pipeline ---
def predict_brix(input_wav_path, model_pkl_path, transformer_pkl_path, selector_pkl_path):
    audio = standardize(input_wav_path)
    audio = noise_reduce(audio)
    audio = normalize(audio)

    features = extract_features(audio)
    features = apply_transformer(transformer_pkl_path, features)
    features = apply_selector(selector_pkl_path, features)

    try:
        model = joblib.load(model_pkl_path)
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_pkl_path}: {e}")

    prediction = model.predict(features)
    return prediction[0]


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_wav = os.path.join(base_dir, "../input", "wav_lab", "1b_10.15_1.wav")
    model_path = os.path.join(base_dir, "74_model_steg_mfcc_xgb_random.pkl")
    transformer_path = os.path.join(base_dir, "74_transformer_steg_mfcc_xgb_random.pkl")
    selector_path = os.path.join(base_dir, "74_selector_steg_mfcc_xgb_random.pkl")

    try:
        brix_score = predict_brix(input_wav, model_path, transformer_path, selector_path)
        print("Predicted Brix: ", brix_score)
    except Exception as e:
        print("Error during prediction: ", e)

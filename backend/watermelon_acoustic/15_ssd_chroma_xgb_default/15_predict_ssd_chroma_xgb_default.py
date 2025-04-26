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
def noise_reduce(audio: AudioSegment, frame_length=2048, hop_length=512, noise_frames_percent=0.1):
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    D = librosa.stft(samples, n_fft=frame_length, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)

    frame_energies = np.sum(magnitude, axis=0)
    num_noise_frames = int(noise_frames_percent * magnitude.shape[1])
    noise_frame_indices = np.argsort(frame_energies)[:num_noise_frames]

    noise_mag = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)

    alpha = 1.0
    magnitude_denoised = np.maximum(magnitude - alpha * noise_mag, 0.0)

    D_denoised = magnitude_denoised * phase
    samples_denoised = librosa.istft(D_denoised, hop_length=hop_length)

    samples_denoised_16bit = (samples_denoised * 32767).astype(np.int16)
    return audio._spawn(samples_denoised_16bit.tobytes())


# --- Step 3: Normalization ---
def normalize(audio: AudioSegment) -> AudioSegment:
    return effects.normalize(audio)


# --- Step 4: Feature Extraction ---
def extract_features(audio, sr=16000):
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0

    chroma = librosa.feature.chroma_stft(y=samples, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    return chroma_mean.reshape(1, -1)


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
    model_path = os.path.join(base_dir, "15_model_ssd_chroma_xgb_default.pkl")
    transformer_path = os.path.join(base_dir, "15_transformer_ssd_chroma_xgb_default.pkl")
    selector_path = os.path.join(base_dir, "15_selector_ssd_chroma_xgb_default.pkl")

    try:
        brix_score = predict_brix(input_wav, model_path, transformer_path, selector_path)
        print("Predicted Brix: ", brix_score)
    except Exception as e:
        print("Error during prediction: ", e)

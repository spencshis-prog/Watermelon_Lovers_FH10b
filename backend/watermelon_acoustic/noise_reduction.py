import os

from pydub import AudioSegment
import numpy as np

import librosa
import pywt  # only needed for wavelet denoising


def noise_reduction_raw(audio) -> AudioSegment:
    """
    Dummy noise reduction technique 1.
    (For example, applying a slight gain reduction.)
    """
    return audio  # Control


def noise_reduction_bandpass(audio: AudioSegment, lowcut=100, highcut=2000) -> AudioSegment:
    """
    Example: Bandpass filtering by chaining a high-pass and then a low-pass filter.
    Adjust lowcut/highcut for your knocking frequency range.
    """
    filtered = audio.high_pass_filter(lowcut)
    filtered = filtered.low_pass_filter(highcut)
    return filtered


def noise_reduction_spectral_subtraction(audio: AudioSegment) -> AudioSegment:
    """
    Example: Spectral Subtraction using librosa.
    1) Convert AudioSegment to NumPy array.
    2) Compute STFT.
    3) Estimate noise (naively using min across frames).
    4) Subtract noise, clamp to zero.
    5) Inverse STFT.
    6) Convert back to AudioSegment.
    """
    # Ensure we have a NumPy array of floats
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # STFT
    D = librosa.stft(samples, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(D)

    # Naive noise estimate: minimum magnitude across time
    noise_mag = np.min(magnitude, axis=1, keepdims=True)

    # Subtract the noise (alpha scales how aggressively to subtract)
    alpha = 1.0
    magnitude_denoised = np.maximum(magnitude - alpha * noise_mag, 0.0)

    # Reconstruct signal
    D_denoised = magnitude_denoised * phase
    samples_denoised = librosa.istft(D_denoised, hop_length=512)

    # Convert back to 16-bit PCM for pydub
    samples_denoised_16bit = (samples_denoised * 32767).astype(np.int16)

    # Spawn a new AudioSegment with the denoised samples
    denoised_segment = audio._spawn(samples_denoised_16bit.tobytes())
    return denoised_segment


def noise_reduction_wavelet(audio: AudioSegment) -> AudioSegment:
    """
    Example: Wavelet Denoising using PyWavelets.
    1) Convert AudioSegment to NumPy array.
    2) Perform Discrete Wavelet Transform (DWT).
    3) Threshold wavelet coefficients.
    4) Inverse DWT.
    5) Convert back to AudioSegment.
    """
    # Convert to numpy float32
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Choose a wavelet, e.g., 'db1' or 'db8'
    wavelet = 'db1'
    # Determine maximum level for the transform
    max_level = pywt.dwt_max_level(len(samples), wavelet)

    # Perform multi-level wavelet decomposition
    coeffs = pywt.wavedec(samples, wavelet, level=max_level)

    # Simple threshold strategy: we pick a fraction of the max detail coeff
    threshold = 0.04 * np.nanmax(np.abs(coeffs[-1]))

    # Apply soft threshold to each set of coefficients (except the first, which is the approximation)
    new_coeffs = [coeffs[0]]  # keep approximation as is
    for detail_level in coeffs[1:]:
        new_coeffs.append(pywt.threshold(detail_level, threshold, mode='soft'))

    # Reconstruct signal
    denoised_samples = pywt.waverec(new_coeffs, wavelet)

    # Ensure the reconstructed signal matches the original length
    denoised_samples = denoised_samples[:len(samples)]

    # Convert back to 16-bit PCM for pydub
    denoised_samples_16bit = (denoised_samples * 32767).astype(np.int16)

    # Spawn a new AudioSegment with the denoised samples
    denoised_segment = audio._spawn(denoised_samples_16bit.tobytes())
    return denoised_segment


# Append alternative noise reduction technique methods here


def apply_noise_reduction(input_dir, output_base_dir):
    """
    Applies two noise reduction techniques to every .wav file in input_dir.
    Creates separate folders (e.g., "technique1", "technique2") under output_base_dir.
    """
    techniques = {
        "raw": noise_reduction_raw,
        "bandpass": noise_reduction_bandpass,
        "spectral_sub": noise_reduction_spectral_subtraction,
        "wavelet_denoise": noise_reduction_wavelet
        # Append alternative noise reduction technique methods here
    }

    rel_input = os.path.relpath(input_dir, os.getcwd())
    rel_output = os.path.relpath(output_base_dir, os.getcwd())
    print(f"Applying noise reduction to files in {rel_input}.")
    print(f"Output will be organized under {rel_output} by technique.")

    for tech_name, func in techniques.items():
        tech_dir = os.path.join(output_base_dir, tech_name)
        func.clear_output_directory(tech_dir)

        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                input_path = os.path.join(input_dir, file)
                try:
                    audio = AudioSegment.from_wav(input_path)
                    processed_audio = func(audio)
                    output_path = os.path.join(tech_dir, file)

                    processed_audio.export(output_path, format="wav")

                    rel_in = os.path.relpath(input_path, os.getcwd())
                    rel_out = os.path.relpath(output_path, os.getcwd())
                    print(f"[NR] Applied {tech_name} to {rel_in} -> saved to {rel_out}")
                except Exception as e:
                    print(f"Error processing {os.path.basename(input_path)} with {tech_name}: {e}")
    print("[NR] Noise reduction processing complete.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "intermediate", "standard_qilin")
    output_base_dir = os.path.join(base_dir, "intermediate", "noise_reduction", "qilin")
    apply_noise_reduction(input_dir, output_base_dir)

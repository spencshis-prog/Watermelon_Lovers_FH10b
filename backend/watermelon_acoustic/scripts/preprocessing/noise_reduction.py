import os

from pydub import AudioSegment
import numpy as np
import noisereduce as nr

import librosa
import pywt  # only needed for wavelet denoising

import functions


def nr_raw(audio) -> AudioSegment:
    """
    Dummy noise reduction technique 1.
    (For example, applying a slight gain reduction.)
    """
    return audio  # Control


def nr_bandpass(audio: AudioSegment, lowcut=100, highcut=2000) -> AudioSegment:
    """
    Example: Bandpass filtering by chaining a high-pass and then a low-pass filter.
    Adjust lowcut/highcut for your knocking frequency range.
    """
    filtered = audio.high_pass_filter(lowcut)
    filtered = filtered.low_pass_filter(highcut)
    return filtered


def nr_spectral_subtraction_naive(audio: AudioSegment) -> AudioSegment:
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


def nr_db1_wavelet(audio: AudioSegment) -> AudioSegment:
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


def nr_spectral_subtraction_dynamic(audio: AudioSegment,
                                    frame_length=2048,
                                    hop_length=512,
                                    noise_frames_percent=0.1):
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


def nr_wiener(audio: AudioSegment):
    """
    Wiener filtering is a frequency-domain technique that adaptively filters each frequency bin
    based on an estimated signal-to-noise ratio. It’s commonly used for speech denoising but can
    also help with knocks if the noise is relatively stationary.

    Leverages noisereduce.reduce_noise, which implements a
    combination of spectral gating and Wiener filtering to reduce noise.
    """
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Use noisereduce's reduce_noise function (Wiener + spectral gating approach)
    reduced_samples = nr.reduce_noise(y=samples, sr=sr)

    # Convert back
    samples_16bit = (reduced_samples * 32767).astype(np.int16)
    return audio._spawn(samples_16bit.tobytes())


def nr_short_time_energy_gating(audio: AudioSegment, frame_ms=10, gate_threshold=0.01, attenuation=0.1):
    """
    Because knocks are transient, you can apply a noise gate in the time domain that lowers
    (or zeroes out) frames whose energy is below a threshold. Effective if knock louder than noise.
    - Break the audio into short frames (e.g., 10–20 ms).
    - Compute the energy (sum of squares) or RMS of each frame.
    - If energy < threshold, attenuate that frame by some factor.
    - Reconstruct the signal.
    """
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    samples_per_frame = int(sr * (frame_ms / 1000.0))
    num_frames = len(samples) // samples_per_frame

    for i in range(num_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        frame = samples[start:end]
        frame_energy = np.mean(frame ** 2)

        # If energy is below threshold, attenuate
        if frame_energy < gate_threshold:
            samples[start:end] *= attenuation

    # Convert back
    samples_16bit = (samples * 32767).astype(np.int16)
    return audio._spawn(samples_16bit.tobytes())


def nr_db8_wavelet_advanced(audio: AudioSegment, wavelet='db8'):
    """
    Applying a more adaptive threshold (universal threshold, SURE, or level-dependent thresholds).
    """
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    max_level = pywt.dwt_max_level(len(samples), wavelet)
    coeffs = pywt.wavedec(samples, wavelet, level=max_level)

    # coeffs[0] is the approximation; coeffs[1:] are detail coefficients
    new_coeffs = [coeffs[0]]
    for i, detail in enumerate(coeffs[1:], start=1):
        # Estimate noise sigma in this level (median absolute deviation method)
        sigma = np.median(np.abs(detail)) / 0.6745
        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(len(detail)))
        # Soft threshold
        detail_denoised = pywt.threshold(detail, threshold, mode='soft')
        new_coeffs.append(detail_denoised)

    denoised_samples = pywt.waverec(new_coeffs, wavelet)
    denoised_samples = denoised_samples[:len(samples)]  # match original length

    samples_16bit = (denoised_samples * 32767).astype(np.int16)
    return audio._spawn(samples_16bit.tobytes())


def apply_noise_reduction(input_dir, output_base_dir):
    """
    Applies two noise reduction techniques to every .wav file in input_dir.
    Creates separate folders (e.g., "technique1", "technique2") under output_base_dir.
    """
    techniques = {  # as assessed by qilin set
        "raw": nr_raw,  # high performance
        "bandpass": nr_bandpass,  # high performance
        # "spectral_sub_naive": nr_spectral_subtraction_naive,  # mid performance, good only w zcr
        # "db1_wavelet": nr_db1_wavelet  # poor performance
        "spectral_sub_dynamic": nr_spectral_subtraction_dynamic,  # good performance
        "wiener": nr_wiener,  # decent performance
        "steg": nr_short_time_energy_gating,  # good performance
        # "db8_dwt": nr_db8_wavelet_advanced,  # very poor performance
    }

    rel_input = os.path.relpath(input_dir, os.getcwd())
    rel_output = os.path.relpath(output_base_dir, os.getcwd())
    print(f"Applying noise reduction to files in {rel_input}.")
    print(f"Output will be organized under {rel_output} by technique.")

    for tech_name, func in techniques.items():
        tech_dir = os.path.join(output_base_dir, tech_name)
        functions.clear_output_directory(tech_dir)

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
    input_dir = os.path.join(base_dir, "../../intermediate", "standard_qilin")
    output_base_dir = os.path.join(base_dir, "../../intermediate", "noise_reduction", "qilin")
    apply_noise_reduction(input_dir, output_base_dir)

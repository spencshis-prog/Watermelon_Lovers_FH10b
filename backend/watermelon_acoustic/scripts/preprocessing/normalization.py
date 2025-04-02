#!/usr/bin/env python
import os
from pydub import AudioSegment, effects


def normalize_audio_files(input_dir, output_dir):
    """
    Normalizes all .wav files in each subfolder of input_dir and writes them to a corresponding
    subfolder in output_dir.

    Args:
        input_dir (str): Directory containing subfolders for noise-reduced .wav files.
        output_dir (str): Base directory where normalized files will be saved (with same subfolder structure).
    """
    base_dir = os.getcwd()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each noise reduction technique subfolder.
    for technique in os.listdir(input_dir):
        technique_path = os.path.join(input_dir, technique)
        if os.path.isdir(technique_path):
            normalized_technique_dir = os.path.join(output_dir, technique)
            if not os.path.exists(normalized_technique_dir):
                os.makedirs(normalized_technique_dir)
            print(f"[NM] Normalizing files in technique folder: {os.path.relpath(technique_path, base_dir)}")
            # Process each WAV file in the subfolder.
            for file in os.listdir(technique_path):
                if file.lower().endswith(".wav"):
                    input_path = os.path.join(technique_path, file)
                    output_path = os.path.join(normalized_technique_dir, file)
                    try:
                        # Load the audio file.
                        audio = AudioSegment.from_wav(input_path)
                        # Normalize the audio (adjusts volume to a target level).
                        normalized_audio = effects.normalize(audio)
                        # Export the normalized audio.
                        normalized_audio.export(output_path, format="wav")
                        print(f"[NM] Normalized {os.path.relpath(input_path, base_dir)} -> {os.path.relpath(output_path, base_dir)}")
                    except Exception as e:
                        print(f"[NM] Error normalizing {os.path.relpath(input_path, base_dir)}: {e}")


if __name__ == "__main__":
    # Define the input and output directories.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Input: your noise reduction stage output directory.
    noise_reduction_dir = os.path.join(base_dir, "../../intermediate", "noise_reduction")
    # Output: a new directory for normalized files.
    normalized_dir = os.path.join(base_dir, "../../intermediate", "normalized_noise_reduction")

    print(f"[NM] Normalizing files from {os.path.relpath(noise_reduction_dir, base_dir)} and saving to {os.path.relpath(normalized_dir, base_dir)}")
    normalize_audio_files(noise_reduction_dir, normalized_dir)
    print("Normalization complete.")

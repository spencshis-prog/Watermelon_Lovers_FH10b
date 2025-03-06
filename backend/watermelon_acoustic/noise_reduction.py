import os

from pydub import AudioSegment

import main


def noise_reduction_technique1(audio):
    """
    Dummy noise reduction technique 1.
    (For example, applying a slight gain reduction.)
    """
    return audio.apply_gain(-2)  # Simulated reduction


def noise_reduction_technique2(audio):
    """
    Dummy noise reduction technique 2.
    (For example, applying a low-pass filter.)
    """
    return audio.low_pass_filter(3000)

# Append alternative noise reduction technique methods here


def apply_noise_reduction(input_dir, output_base_dir):
    """
    Applies two noise reduction techniques to every .wav file in input_dir.
    Creates separate folders (e.g., "technique1", "technique2") under output_base_dir.
    """
    techniques = {
        "technique1": noise_reduction_technique1,
        "technique2": noise_reduction_technique2
        # Append alternative noise reduction technique methods here
    }
    for tech_name, func in techniques.items():
        output_dir = os.path.join(output_base_dir, tech_name)
        main.clear_output_directory(output_dir)

        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                input_path = os.path.join(input_dir, file)
                try:
                    audio = AudioSegment.from_wav(input_path)
                    processed_audio = func(audio)
                    output_path = os.path.join(output_dir, file)
                    processed_audio.export(output_path, format="wav")
                    print(f"Applied {tech_name} to {file}, saved to {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path} with {tech_name}: {e}")
    print("Noise reduction processing complete.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "output", "qilin_standard")
    output_base_dir = os.path.join(base_dir, "output", "noise_reduction", "qilin")
    apply_noise_reduction(input_dir, output_base_dir)

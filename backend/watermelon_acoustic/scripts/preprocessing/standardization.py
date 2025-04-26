import os

from pydub import AudioSegment
from pydub.utils import which

import functions
import main


# from pydub.generators import Silence


def standardize_wav_file(input_path, output_path, target_duration_ms=1000, target_sample_rate=16000, target_channels=1):
    """
    Reformat a .wav file:
      - Adjust duration (pad with silence or trim)
      - Set sample rate and channels
      - Export with 16-bit PCM
    """
    try:
        AudioSegment.converter = which("ffmpeg") or os.path.abspath("ffmpeg/bin/ffmpeg.exe")

        audio = AudioSegment.from_wav(input_path)
        # Set sample rate and channels
        audio = audio.set_frame_rate(target_sample_rate).set_channels(target_channels)
        current_duration_ms = len(audio)
        if current_duration_ms < target_duration_ms:
            # Pad with silence if too short
            silent_segment = AudioSegment.silent(duration=1000, frame_rate=16000)
            audio = audio + silent_segment

            # Use the following if you can get import Silence to work
            # silence = Silence().to_audio_segment(duration=target_duration_ms - current_duration_ms)
            # audio = audio + silence
        elif current_duration_ms > target_duration_ms:
            audio = audio[:target_duration_ms]
        # Export ensuring 16-bit PCM (using pcm_s16le)
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

        rel_input = os.path.relpath(input_path, os.getcwd())
        rel_output = os.path.relpath(output_path, os.getcwd())
        print(f"[SW] Standardized {rel_input} to {rel_output}")
    except Exception as e:
        print(f"[SW] Error processing {input_path}: {e}")


def standardize_wav_files(input_dir, output_dir):
    """
    Processes all .wav files in input_dir and writes standardized versions to output_dir.
    """
    functions.clear_output_directory(output_dir)

    rel_input_dir = os.path.relpath(input_dir, os.getcwd())
    rel_output_dir = os.path.relpath(output_dir, os.getcwd())
    print(f"[SW] Standardizing all .wav files in {rel_input_dir} into {rel_output_dir}")

    for file in os.listdir(input_dir):
        input_path = os.path.normpath(os.path.join(input_dir, file))
        output_path = os.path.normpath(os.path.join(output_dir, file))

        if not os.path.isfile(input_path):
            continue

        print(f"[SW] Preparing to standardize: {input_path}")
        try:
            standardize_wav_file(input_path, output_path)
        except Exception as e:
            print(f"[SW] ERROR for {file}: {e}")
    print(f"[SW] Standardizing complete. Standardized files are in {rel_output_dir}")


if __name__ == "__main__":
    # if simply called, default to qilin dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "../../input", "wav_qilin")
    output_dir = os.path.join(base_dir, "../../intermediate", "standard_qilin")
    standardize_wav_files(input_dir, output_dir)

import os

from pydub import AudioSegment

import main


def convert_m4a_to_wav(m4a_path, output_dir, label, sweetness, index=0):
    """
    Converts an .m4a file to .wav format and saves it in the output directory.
    """
    wav_filename = f"{label}_{sweetness}_{index}.wav"
    wav_path = os.path.join(output_dir, wav_filename)

    if os.path.exists(wav_path):
        os.remove(wav_path)

    try:
        print(f"Converting {m4a_path} to {wav_path}...")
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"Failed to convert {m4a_path}, invalid m4a: {e}")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return None

    return wav_path


def convert_qilin_file_formats_to_wav(qilin_dataset_dir, output_dir):
    """
    Processes the Qilin dataset by converting .m4a files to .wav and copying existing .wav files.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ffmpeg_path = os.path.join(base_dir, "ffmpeg\\bin\\ffmpeg.exe")
    ffprobe_path = os.path.join(base_dir, "ffmpeg\\bin\\ffprobe.exe")

    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffprobe_path)

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    ffmpeg_available = "ffmpeg" in os.popen("where ffmpeg").read()
    ffprobe_available = "ffprobe" in os.popen("where ffprobe").read()

    if ffmpeg_available and ffprobe_available:
        print("ffmpeg and ffprobe are now available in the PATH.")
    else:
        print("Failed to add ffmpeg or ffprobe to PATH. Check the paths.")

    main.clear_output_directory(output_dir)

    # Iterate over subdirectories (assumes folder names contain an underscore for valid ones)
    for subdir in os.listdir(qilin_dataset_dir):
        subdir_path = os.path.join(qilin_dataset_dir, subdir)
        if not os.path.isdir(subdir_path) or "_" not in subdir:
            print(f"Skipping directory: {subdir}")
            continue

        try:
            label_str, sweetness_str = subdir.split("_")
            label = int(label_str)  # or int(label_str) if these are IDs
            sweetness = float(sweetness_str)
        except ValueError:
            print(f"Skipping invalid folder (no numeric label/sweetness): {subdir}")
            continue

        # Process 'audio' folder (for .m4a files)
        audio_dir = os.path.join(subdir_path, "audio")
        if os.path.isdir(audio_dir):
            m4a_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".m4a")]
            for i, m4a_file in enumerate(m4a_files):
                m4a_path = os.path.join(audio_dir, m4a_file)
                wav_path = convert_m4a_to_wav(m4a_path, output_dir, label, sweetness, index=i)
                if wav_path is None:
                    print(f"Skipping {m4a_path} due to conversion failure.")
                    continue  # Skip further processing for this file if needed

        # Process 'audios' folder (for .wav files)
        audios_dir = os.path.join(subdir_path, "audios")
        if os.path.isdir(audios_dir):
            wav_files = [f for f in os.listdir(audios_dir) if f.lower().endswith(".wav")]
            for i, wav_file in enumerate(wav_files):
                old_path = os.path.join(audios_dir, wav_file)
                # Construct a new name for the .wav
                new_wav_name = f"{label}_{sweetness}_{i}.wav"
                new_path = os.path.join(output_dir, new_wav_name)
                if not os.path.exists(new_path):
                    print(f"Copying {old_path} to {new_path}...")
                    try:
                        audio = AudioSegment.from_wav(old_path)
                        audio.export(new_path, format="wav")
                    except Exception as e:
                        print(f"Failed to copy {old_path}: {e}")
                        continue

    print("Preprocessing of Qilin dataset completed.")


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    ffmpeg_path = os.path.join(base_dir, "ffmpeg\\bin\\ffmpeg.exe")
    ffprobe_path = os.path.join(base_dir, "ffmpeg\\bin\\ffprobe.exe")

    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffprobe_path)

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    ffmpeg_available = "ffmpeg" in os.popen("where ffmpeg").read()
    ffprobe_available = "ffprobe" in os.popen("where ffprobe").read()

    if ffmpeg_available and ffprobe_available:
        print("ffmpeg and ffprobe are now available in the PATH.")
    else:
        print("Failed to add ffmpeg or ffprobe to PATH. Check the paths.")

    # Example usage:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qilin_dataset_dir = os.path.join(base_dir, "qilin_dataset", "19_datasets")
    output_dir = os.path.join(base_dir, "output", "qilin_wav")
    convert_qilin_file_formats_to_wav(qilin_dataset_dir, output_dir)

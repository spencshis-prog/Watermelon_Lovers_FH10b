import serial
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import json
import argparse

# Serial port configuration
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLE_RATE = 16000  # Hz
DURATION = 1  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION
AMPLITUDE = 4095  # 12-bit max

def get_wav_serial():
    """Read audio data from the serial port."""
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    data = []
    print("Collecting data...")

    while len(data) < NUM_SAMPLES:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("Microphone_Output="):
            try:
                value = int(line.split('=')[1].strip())
                value = max(0, min(value, AMPLITUDE))  # Clamp values within 0-4095
                data.append(value)
            except ValueError:
                pass  # Ignore invalid lines

    ser.close()
    print(f"Collected {len(data)} samples.")
    
    # Convert list to NumPy array in 16-bit format
    return ((np.array(data) - 2048) * (32767 / 2048)).astype(np.int16)

def get_wav_file(filename):
    """Load audio data from a WAV file."""
    sample_rate, audio_data = wav.read(filename)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE}, but got {sample_rate}")
    print(f"Loaded WAV file: {filename}")
    return audio_data

def fft(audio_data, watermelon_folder):
    """Perform FFT analysis and save results."""
    N = len(audio_data)
    freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)
    fft_magnitude = np.abs(np.fft.rfft(audio_data))

    resonant_freq = freqs[np.argmax(fft_magnitude)]
    print(f"Resonant Frequency: {resonant_freq:.2f} Hz")

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude, label="FFT Magnitude")
    plt.axvline(resonant_freq, color='r', linestyle='--', label=f"Peak: {resonant_freq:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Watermelon Resonant Frequency Spectrum")
    plt.legend()
    plt.grid()

    fft_plot_file = os.path.join(watermelon_folder, "fft_plot.png")
    plt.savefig(fft_plot_file)
    plt.close()
    print(f"Saved FFT plot to {fft_plot_file}")

    fft_data = {
        "frequencies": freqs.tolist(),  # List of frequencies (Hz)
        "magnitudes": fft_magnitude.tolist(),  # List of magnitudes
        "resonant_frequency": resonant_freq,  # Resonant frequency (Hz)
        "sampling_rate": SAMPLE_RATE,  # Sampling rate (Hz)
        "fft_resolution": N  # FFT resolution (number of bins)
    }

    # Save the FFT data to a JSON file
    fft_data_file = os.path.join(watermelon_folder, "fft_data.json")
    with open(fft_data_file, 'w') as f:
        json.dump(fft_data, f, indent=4)

    print(f"Saved FFT data to {fft_data_file}")

def main():
    parser = argparse.ArgumentParser(description="Watermelon FFT Analysis")
    parser.add_argument("--file", type=str, help="Path to a WAV file for debugging")
    args = parser.parse_args()

    output_dir = "watermelon_data"
    os.makedirs(output_dir, exist_ok=True)

    watermelon_id = 1
    while os.path.exists(os.path.join(output_dir, f"watermelon_{watermelon_id}")):
        watermelon_id += 1

    watermelon_folder = os.path.join(output_dir, f"watermelon_{watermelon_id}")
    os.makedirs(watermelon_folder)

    if args.file:
        audio_data = get_wav_file(args.file)
    else:
        audio_data = get_wav_serial()

    wav_file = os.path.join(watermelon_folder, "watermelon.wav")
    wav.write(wav_file, SAMPLE_RATE, audio_data)
    print(f"Saved WAV file: {wav_file}")

    fft(audio_data, watermelon_folder)

if __name__ == "__main__":
    main()

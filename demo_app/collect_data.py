import serial
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import json
import argparse, time

# Serial port configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
SAMPLE_RATE = 16000  # Hz
DURATION = 1  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION
AMPLITUDE = 4095  # 12-bit max


# Define bandpass filter range (Adjust as needed)
LOWCUT = 50.0   # Hz (Lower bound of expected resonance)
HIGHCUT = 600.0  # Hz (Upper bound of useful frequencies)

def bandpass_filter(data, order=5):
    """Apply a Butterworth bandpass filter to the signal."""
    nyquist = 0.5 * SAMPLE_RATE
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    # Apply filter and handle potential issues
    filtered_data = signal.filtfilt(b, a, data)
    
    # Ensure no NaN or Inf values
    filtered_data = np.nan_to_num(filtered_data)
    
    # Normalize back to 16-bit range
    filtered_data = np.clip(filtered_data, -32768, 32767).astype(np.int16)
    
    return filtered_data

def get_wav_file(filename):
    """Load audio data from a WAV file."""
    sample_rate, audio_data = wav.read(filename)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE}, but got {sample_rate}")
    print(f"Loaded WAV file: {filename}")
    return audio_data

def get_wav_serial():
    """Read audio data from the serial port with timeout for inactivity."""
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    data = []
    print("Waiting for valid input...")

    

    # Wait for first valid input
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                value = int(line)
                print("Received first valid input. Starting recording...")
                break
            except ValueError:
                pass  # Ignore non-numeric lines
        

    data.append(value)
    
    start_time = time.time()
    TIMEOUT_SECONDS = 2  # Stop if no input for this long
    while len(data) < NUM_SAMPLES:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                value = int(line)
                value = max(0, min(value, AMPLITUDE))
                data.append(value)
                last_data_time = time.time()  # Reset timeout
            except ValueError:
                pass
        elif time.time() - start_time > TIMEOUT_SECONDS:
            print("Timeout during recording. Ending early.")
            break

    ser.close()
    print(f"Collected {len(data)} samples.")

    signal_data = ((np.array(data) - 2048) * (32767 / 2048)).astype(np.int16)

    return bandpass_filter(signal_data)


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
    plt.xlim(0, 500) 
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
        "fft_resolution": N,  # FFT resolution (number of bins)
    }

    # Save the FFT data to a JSON file
    fft_data_file = os.path.join(watermelon_folder, "fft_data.json")
    with open(fft_data_file, 'w') as f:
        json.dump(fft_data, f, indent=4)

    print(f"Saved FFT data to {fft_data_file}")

def collect():

    watermelon_id = input("Enter the watermelon ID: ")
    index = input("Which test is this? :")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "watermelon_data")
    output_dir = os.path.join(data_dir, f"{watermelon_id}_{index}")
    os.makedirs(output_dir, exist_ok=True)

    watermelon_folder = os.path.join(output_dir, f"{watermelon_id}_BrixPlaceholder_{index}")
    os.makedirs(watermelon_folder, exist_ok=True)

    
    audio_data = get_wav_file("debug_audio.wav") # Uncomment this line to use a WAV file instead of serial input
    #audio_data = get_wav_serial()


    wav_file = os.path.join(watermelon_folder, "watermelon.wav")
    wav.write(wav_file, SAMPLE_RATE, audio_data)
    print(f"Saved WAV file: {wav_file}")

    fft(audio_data, watermelon_folder)

def collect_api(watermelon_id, index):

    
    output_dir = "watermelon_data/{watermelon_id}_{index}"
    os.makedirs(output_dir, exist_ok=True)

    watermelon_folder = os.path.join(output_dir, f"{watermelon_id}_BrixPlaceholder_{index}")
    os.makedirs(watermelon_folder, exist_ok=True)

    
    audio_data = get_wav_serial()

    wav_file = os.path.join(watermelon_folder, "watermelon.wav")
    wav.write(wav_file, SAMPLE_RATE, audio_data)
    print(f"Saved WAV file: {wav_file}")

    fft(audio_data, watermelon_folder)

if __name__ == "__main__":
    collect()
import serial
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

# Serial port configuration
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
SAMPLE_RATE = 16000  # Hz
DURATION = 1  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION
AMPLITUDE = 4095  # 12-bit max

def get_wav_serial():
    

    
    # Open serial connection
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

    # Close serial port
    ser.close()
    print(f"Collected {len(data)} samples.")

    # Convert list to NumPy array in 16 bit format
    audio_data = ((np.array(data) - 2048) * (32767 / 2048)).astype(np.int16)
    

    return audio_data
    


def fft(audio_data):
    # Perform FFT analysis
    N = len(audio_data)
    freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)
    fft_magnitude = np.abs(np.fft.rfft(audio_data))

    # Find peak frequency (resonant frequency)
    resonant_freq = freqs[np.argmax(fft_magnitude)]
    print(f"Resonant Frequency: {resonant_freq:.2f} Hz")

    # Save FFT plot
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

    # Save FFT data
    fft_data_file = os.path.join(watermelon_folder, "fft_data.npy")
    np.save(fft_data_file, fft_magnitude)
    print(f"Saved FFT data to {fft_data_file}")


def main():
    # Define output directory
    output_dir = "backend/watermelon_data"
    os.makedirs(output_dir, exist_ok=True)

    # Find the next available watermelon folder
    watermelon_id = 1
    while os.path.exists(os.path.join(output_dir, f"watermelon_{watermelon_id}")):
        watermelon_id += 1

    watermelon_folder = os.path.join(output_dir, f"watermelon_{watermelon_id}")
    os.makedirs(watermelon_folder)


    audio_data = get_wav_serial()
    wav_file = os.path.join(dir, "watermelon.wav")
    wav.write(wav_file, SAMPLE_RATE, audio_data)
    print(f"Saved WAV file: {wav_file}")
    

    fft(audio_data)


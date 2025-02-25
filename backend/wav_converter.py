import numpy as np
import scipy.io.wavfile as wav

# Parameters for the .wav file
sample_rate = 16000  # Samples per second (Hz)
duration = 1  # Duration of the sound (seconds)
amplitude = 32767  # Max amplitude for 16-bit audio (int16)

# Generating sample sine wav
#t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#frequency = 440  # Frequency of the sine wave in Hz
#data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)  # Convert to int16

# Write the raw data to a .wav file
wav.write('output.wav', sample_rate, data)

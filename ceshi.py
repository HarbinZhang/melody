import numpy as np
 
import wave
 
import struct
 
import matplotlib.pyplot as plt
 
# frequency is the number of times a wave repeats a second
 
frequency = 1000

noisy_freq = 50
 
num_samples = 48000
 
# The sampling rate of the analog to digital convert
 
sampling_rate = 48000.0
 
amplitude = 16000
 
file = "test.wav"

nframes=num_samples
 
comptype="NONE"
 
compname="not compressed"
 
nchannels=1
 
sampwidth=2

sine_wave = [np.sin(2 * np.pi * frequency * x/sampling_rate) for x in range(num_samples)]
# sine_noise = [np.sin(2 * np.pi * noisy_freq * x1/  sampling_rate) for x1 in range(num_samples)]

sine_wave = np.array(sine_wave)
 # sine_noise = np.array(sine_noise)
 
combined_signal = sine_wave # + sine_noise

wav_file=wave.open(file, 'w')
wav_file.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))
for s in combined_signal:
   wav_file.writeframes(struct.pack('h', int(s*amplitude)))

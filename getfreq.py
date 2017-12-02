import numpy as np
 
import wave
 
import struct
 
import matplotlib.pyplot as plt

frame_rate = 48000.0
 
infile = "test.wav"
 
num_samples = 48000
 
wav_file = wave.open(infile, 'r')
 
data = wav_file.readframes(num_samples)
 
wav_file.close()

data = struct.unpack('{n}h'.format(n=num_samples), data)

data = np.array(data)

data_fft = np.fft.fft(data)

frequencies = np.abs(data_fft)

# filtered_freq
for i in range(len(frequencies)):
	# if 950 < i < 1050 and frequencies[i] > 1 :
	if 950 < i < 1050 :
		frequencies[i] = frequencies[i]
	else:
		frequencies[i] = 0
# filtered_freq = [f if (950 < index < 1050 and f > 1) else 0 for index, f in enumerate(frequencies)]
recovered_signal = np.fft.ifft(frequencies)


plt.subplot(3,1,1)
 
plt.plot(data[:3000])
 
plt.title("Original audio wave")
 
plt.subplots_adjust(hspace=.5) 

plt.subplot(3,1,2)
 
plt.plot(frequencies)
 
plt.title("Frequencies found")
 
plt.xlim(0,1200)

plt.subplot(3,1,3)
 
plt.plot(recovered_signal)

plt.subplots_adjust(hspace=.5)
 
plt.title("Frequencies filtered")
 
plt.xlim(0,3000)

 
plt.show()
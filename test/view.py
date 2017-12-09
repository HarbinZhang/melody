import numpy as np
import os
import matplotlib.pyplot as plt


infile = "data"

infile = open(infile, 'r')

data = infile.read().split('\n')
data = map(float, data)

data = np.array(data)
data_fft = np.fft.fft(data)
freq = np.abs(data_fft)
print data_fft


plt.subplot(3,1,1)
 
plt.plot(data)
 
plt.title("Original audio wave")
 
plt.subplots_adjust(hspace=.5) 

plt.subplot(3,1,2)
 
plt.plot(freq)
 
plt.title("Frequencies found")
 

plt.show()
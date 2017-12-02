import numpy as np
import wave, struct
import matplotlib.pyplot as plt


# filepath = './test.wav'

# read filepath
# display data


infile = "newone.wav"

infile = wave.open(infile, 'r')
sampwidth = infile.getsampwidth()

# ofile = wave.open("part_output.wav", "w")
# ofile.setparams(infile.getparams())


# fmts = (None, "=B", "=h", None, "=l")
# fmt = fmts[sampwidth]
# dcs  = (None, 128, 0, None, 0)
# dc = dcs[sampwidth]

# print infile.getnframes()

# for i in range(44100*10):
#     iframe = infile.readframes(1)

#     iframe = struct.unpack(fmt, iframe)[0]
#     iframe -= dc

#     oframe = iframe / 2;

#     oframe += dc
#     oframe = struct.pack(fmt, oframe)
#     ofile.writeframes(oframe)

# infile.close()
# ofile.close()





num_samples = 44100

data = infile.readframes(num_samples)
 
infile.close()

print data[12000]

data = struct.unpack('{n}h'.format(n=num_samples), data)

print data[12000]

data = np.array(data)

print data[10200]
# data_fft = np.fft.fft(data)

# freq = np.abs(data_fft)

# # print len(data[-4410:])
# part_data = data[-441:]
# part_fft = np.abs(np.fft.fft(part_data))

# # Display


# plt.subplot(4,1,1)
# plt.plot(data)
# plt.title("Original audio wave")
# plt.subplots_adjust(hspace=.5) 

# plt.subplot(4,1,2)
# plt.plot(freq)
# plt.title("Frequencies found")
# plt.xlim(0,1200)

# plt.subplot(4,1,3)
# plt.plot(part_data)
# plt.title("partial audio wave")
# plt.subplots_adjust(hspace=.5) 

# plt.subplot(4,1,4)
# plt.plot(part_fft)
# plt.title("partial Frequencies found")
# plt.xlim(0,1200)
 
# plt.show()
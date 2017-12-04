import numpy as np
import os
import matplotlib.pyplot as plt


infile = "data"

infile = open(infile, 'r')

data = infile.read().split('\n')



plt.plot(data)
plt.xlim(0, 200)
plt.show()
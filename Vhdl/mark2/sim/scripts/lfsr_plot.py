import numpy as np
import matplotlib.pyplot as plt

lfsr_file = "files/lfsr_out.txt"

with open(lfsr_file, 'r') as fp:
	lfsr_data = np.loadtxt(fp)


plt.plot(lfsr_data)
plt.show()

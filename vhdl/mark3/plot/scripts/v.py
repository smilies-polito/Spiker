import numpy as np
import matplotlib.pyplot as plt

dataDir = "../data"
figureDir = "../figures"

v_filename = dataDir + "/v.txt"

outputFilename = figureDir + "v.pdf"

with open(v_filename, 'r') as fp:
	v = np.loadtxt(fp, dtype = 'int')

plt.plot(v)
plt.show()

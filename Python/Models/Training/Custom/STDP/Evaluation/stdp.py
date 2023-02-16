import numpy as np
import matplotlib.pyplot as plt
from files import *

tau = 200

Delta_t = np.linspace(0 ,1000, 10000)

ltp = np.exp(-Delta_t/tau)

Delta_t = np.linspace(1000, 0, 10000)
ltd = - np.exp(- Delta_t/tau)

stdp = np.concatenate((ltd, ltp))


xAxis = np.linspace(-1000, 1000, 20000)
yAxis = np.linspace(-1.2, 1.2, 10000)

plt.plot(xAxis, stdp)

xAxis = np.linspace(-1200, 1200, 20000)

plt.arrow(-1200, 0, 2400, 0, head_width = 0.03, head_length = 30, color = 'k')
plt.arrow(0, -1.2, 0, 2.4, head_width = 20, head_length = 0.05, color = 'k')


plt.savefig(fname = stdpFile, dpi = 1000)

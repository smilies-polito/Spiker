import numpy as np
from files import *

countersParallelism = 16
N_neurons = 400


counters = np.zeros(N_neurons).astype(int)

with open(countersFilename, "r") as fp:
	counters_string = fp.readline()


for i in range(400):

	counters[i] = int(counters_string[i*countersParallelism:
		(i+1)*countersParallelism], 2)


with open(countersFilename, "w") as fp:
	fp.write(str(list(counters.astype(int))).replace(",",
		"").replace(" ", "\n")[1:-1])

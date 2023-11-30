import numpy as np

np.set_printoptions(threshold = np.inf)

target_dir = "./DummyAccelerator"
spikes_file = "in_spike.txt"

spikes_file = target_dir + "/" + spikes_file

p0 = 0
p1 = 1 - p0

spikes = np.random.choice([0, 1], size = (1000, 700), p = [p0, p1])

with open(spikes_file, "w") as fp:
	for t in range(spikes.shape[0]):
		s = str(spikes[t])[1:-1].replace(" ", "").replace("\n", "")

		fp.write(s)
		fp.write("\n")

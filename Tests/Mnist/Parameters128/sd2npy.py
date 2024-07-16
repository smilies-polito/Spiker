import torch
import numpy as np

state_dict = torch.load("state_dict_mnist.pt", map_location = torch.device("cpu"))

print(state_dict)

with open("thresholds1.npy", "wb") as fp:
	np.save(fp, np.array(state_dict["lif1.threshold"])*np.ones(128))

with open("weights1.npy", "wb") as fp:
	np.save(fp, np.array(state_dict["fc1.weight"]))

with open("thresholds2.npy", "wb") as fp:
	np.save(fp, np.array(state_dict["lif2.threshold"])*np.ones(10))
	
with open("weights2.npy", "wb") as fp:
	np.save(fp, np.array(state_dict["fc2.weight"]))

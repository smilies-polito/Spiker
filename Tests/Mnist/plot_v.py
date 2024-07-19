import matplotlib.pyplot as plt

neuron_datapath_v = "/home/alessio/PhD/Progetti/Spiker/VhdlGenerator/"\
	"test_neuron/test_neuron.sim/sim_1/behav/xsim/neuron_datapath_v.txt"

with open(neuron_datapath_v) as fp:
	v = fp.read()


v = [int(i) for i in v.split("\n")[:-1]]
plt.plot(v)
plt.show()

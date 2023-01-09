import numpy as np
import matplotlib.pyplot as plt

def visualizeConnections(synapses):

	Ns = len(synapses.source)
	Nt = len(synapses.target)

	offset = Ns/2 - Nt/2

	plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
	plt.plot(np.ones(Nt), np.arange(Nt) + offset, 'ok', ms=10)

	for i,j in zip(synapses.i, synapses.j):
		plt.plot([0,1], [i, j + offset])

	plt.xticks([0, 1], ['Source', 'Target'])
	plt.ylabel('Neuron index')

	plt.show()

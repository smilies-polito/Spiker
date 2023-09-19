import numpy as np

from network import Network, Network_tb
from layer import Layer

from vhdl import fast_compile, elaborate

parameters_dir = "../Models/Inference/Custom/FullPrecisionSupervised/"\
		"Parameters/"

with open(parameters_dir + "/weights1.npy", "rb") as fp:
	exc_w1 = np.load(fp)

with open(parameters_dir + "/weights2.npy", "rb") as fp:
	exc_w2 = np.load(fp)

with open(parameters_dir + "/thresholds1.npy", "rb") as fp:
	th1 = np.load(fp)[0]

with open(parameters_dir + "/thresholds2.npy", "rb") as fp:
	th2 = np.load(fp)[0]

inh_w1 = np.zeros([exc_w1.shape[0], exc_w1.shape[0]])
inh_w2 = np.zeros([exc_w2.shape[0], exc_w2.shape[0]])

reset1 = np.zeros(th1.shape[0])
reset2 = np.zeros(th2.shape[0])


net = Network(
	n_cycles = 100
)

l1 = Layer(
	w_exc		= exc_w1,
	w_inh		= inh_w1,
	v_th		= th1,
	v_reset		= reset1,
	bitwidth 	= 16,
	fp_decimals	= 3, 
	w_exc_bw	= 6,
	w_inh_bw	= 6,
	shift		= 10
	#reset 		= "subtractive"
)

l2 = Layer(
	w_exc		= exc_w2,
	w_inh		= inh_w2,
	v_th		= th2,
	v_reset		= reset2,
	bitwidth 	= 16,
	fp_decimals	= 3, 
	w_exc_bw	= 6,
	w_inh_bw	= 6,
	shift		= 10
	#reset 		= "subtractive"
)

net.add(l1)
net.add(l2)

tb = Network_tb(net,
	file_input		= True,
	input_signal_list	= [
		"in_spikes"
	],
	file_output		= True
)

tb.write_file_all(rm=True)

fast_compile(net)
elaborate(net)

import torch
import numpy as np

import sys

vhdl_gen_dir = "../../VhdlGenerator"
hdl_dir = "../../HdlTools/hdltools"

if vhdl_gen_dir not in sys.path:
	sys.path.insert(0, vhdl_gen_dir)

if hdl_dir not in sys.path:
	sys.path.insert(0, hdl_dir)

from network import DummyAccelerator
from vhdl import write_file_all, fast_compile, elaborate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

parameters_dir 	= "./TrainedParameters"
spiker_dir	= "VhdlSpiker"

trained_param_dict = parameters_dir + "/state_dict_audiomnist.pt"

state_dict = torch.load(trained_param_dict, map_location = device)

exc_w1 = state_dict["fc1.weight"].cpu().detach().numpy()
inh_w1 = np.zeros((exc_w1.shape[0], exc_w1.shape[0]))
exc_w2 = state_dict["fc2.weight"].cpu().detach().numpy()
inh_w2 = np.zeros((exc_w2.shape[0], exc_w2.shape[0]))

th1 = state_dict["lif1.threshold"].cpu().detach().numpy() * \
	np.ones((exc_w1.shape[0]))
th2 = state_dict["readout.threshold"].cpu().detach().numpy() * \
	np.ones((exc_w2.shape[0]))

fp_dec		= 7
bitwidth1	= 8
bitwidth2	= 8
w_bitwidth1	= 6
w_bitwidth_fb1	= 1
w_bitwidth2 	= 6
w_bitwidth_fb2	= 1
exp_shift	= 4

n_cycles = 93

layer_0 = {
	"label"		: "",
	"w_exc"		: exc_w1,
	"w_inh"		: inh_w1,
	"v_th"		: th1,
	"v_reset"	: None,
	"bitwidth"	: bitwidth1,
	"fp_decimals"	: fp_dec,
	"w_inh_bw"	: w_bitwidth_fb1,
	"w_exc_bw"	: w_bitwidth1,
	"shift"		: exp_shift,
	"reset"		: "subtractive",
	"debug"		: False,
	"debug_list"	: []
}

layer_1 = {
	"label"		: "",
	"w_exc"		: exc_w2,
	"w_inh"		: inh_w2,
	"v_th"		: th2,
	"v_reset"	: None,
	"bitwidth"	: bitwidth2,
	"fp_decimals"	: fp_dec,
	"w_inh_bw"	: w_bitwidth_fb2,
	"w_exc_bw"	: w_bitwidth2,
	"shift"		: exp_shift,
	"reset"		: "subtractive",
	"debug"		: False,
	"debug_list"	: []
}


config_dict = {
	"n_cycles"	: n_cycles,
	"layer_0"	: layer_0,
	"layer_1"	: layer_1
}


spiker = DummyAccelerator(
	config	= config_dict
)

write_file_all(spiker,
	output_dir	= spiker_dir,
	rm		= True
)

fast_compile(spiker, output_dir = spiker_dir)
elaborate(spiker, output_dir = spiker_dir)

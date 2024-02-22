import numpy as np

import path_config

from network import DummyAccelerator
from vhdl import write_file_all, fast_compile, elaborate

import sys

vhdl_gen_dir = "../../VhdlGenerator"
hdl_dir = "../../HdlTools/hdltools"

if vhdl_gen_dir not in sys.path:
	sys.path.insert(0, vhdl_gen_dir)

if hdl_dir not in sys.path:
	sys.path.insert(0, hdl_dir)

parameters_dir = "./Parameters"

# with open(parameters_dir + "/weights1.npy", "rb") as fp:
# 	exc_w1 = np.load(fp)
# 
# with open(parameters_dir + "/weights2.npy", "rb") as fp:
# 	exc_w2 = np.load(fp)
# 
# with open(parameters_dir + "/thresholds1.npy", "rb") as fp:
# 	th1 = np.load(fp)[0]
# 
# with open(parameters_dir + "/thresholds2.npy", "rb") as fp:
# 	th2 = np.load(fp)[0]

exc_w1 = np.random.rand(128, 784)
exc_w2 = np.random.rand(10, 128)
th1 = np.ones(128)
th2 = np.ones(128)

inh_w1 = np.zeros([exc_w1.shape[0], exc_w1.shape[0]])
inh_w2 = np.zeros([exc_w2.shape[0], exc_w2.shape[0]])

reset1 = np.zeros(th1.shape[0])
reset2 = np.zeros(th2.shape[0])

fp_dec		= 4
bitwidth1	= 6
bitwidth2	= 6
w_bitwidth1	= 4
w_bitwidth_fb1	= 1
w_bitwidth2 	= 4
exp_shift	= 10

n_cycles = 100

layer_0 = {
	"label"	: "",
	"w_exc"		: exc_w1,
	"w_inh"		: inh_w1,
	"v_th"		: th1,
	"v_reset"	: reset1,
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
	"label"	: "",
	"w_exc"		: exc_w2,
	"w_inh"		: inh_w2,
	"v_th"		: th2,
	"v_reset"	: reset2,
	"bitwidth"	: bitwidth2,
	"fp_decimals"	: fp_dec,
	"w_inh_bw"	: w_bitwidth_fb1,
	"w_exc_bw"	: w_bitwidth1,
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

spiker_dir = "VhdlSpiker"


spiker = DummyAccelerator(
	config	= config_dict
)

write_file_all(spiker,
	output_dir	= spiker_dir,
	rm		= True
)

fast_compile(spiker, output_dir = spiker_dir)
elaborate(spiker, output_dir = spiker_dir)

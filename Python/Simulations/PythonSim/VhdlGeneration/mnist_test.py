import numpy as np

from config import n_cycles, config_dict
import path_config

from network import DummyAccelerator
from vhdl import write_file_all, fast_compile, elaborate


spiker = DummyAccelerator(
	config	= config_dict
)

spiker.write_file_all(
	output_dir	= "DummyAccelerator",
	rm		= True
)

fast_compile(spiker)
elaborate(spiker)

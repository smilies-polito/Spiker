import numpy as np

from config import n_cycles, config_dict
import path_config

from network import DummyAccelerator
from vhdl import write_file_all, fast_compile, elaborate


spiker = DummyAccelerator(
	config	= config_dict
)

write_file_all(spiker,
	output_dir	= "Prova",
	rm		= True
)

fast_compile(spiker, output_dir = "Prova")
elaborate(spiker, output_dir = "Prova")

import sys

vhdl_gen_dir = "../../../VhdlGeneration"
hdl_dir = "../../../HdlTools/hdltools"

if vhdl_gen_dir not in sys.path:
	sys.path.insert(0, vhdl_gen_dir)

if hdl_dir not in sys.path:
	sys.path.insert(0, hdl_dir)


import sys

vhdl_gen_dir = "../../../VhdlGeneration"
hdl_dir = "../../../HdlTools"

if hdl_dir not in sys.path:
	sys.path.append(hdl_dir)

if vhdl_gen_dir not in sys.path:
	sys.path.append(vhdl_gen_dir)

import subprocess as sp
from os.path import isdir


def write_file(component, output_dir = "output", rm = False):

	if hasattr(component, "entity"):
		output_file_name = output_dir + "/" + component.entity.name + \
					".vhd"

	elif hasattr(component, "name"):
		output_file_name = output_dir + "/" + component.name+".vhd"

	else:
		raise TypeError("Component is not a valid VHDL object")


	if hasattr(component, "code") and callable(component.code):
		hdl_code = component.code()
	else:
		raise TypeError("Component has no code method")

	if rm:
		sp.run("rm -rf " + output_dir, shell = True)

	if not isdir(output_dir):
		sp.run("mkdir " + output_dir, shell = True)
	

	with open(output_file_name, "w+") as fp:

		for line in hdl_code:
			fp.write(line)


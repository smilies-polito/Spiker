import subprocess as sp
from os.path import isfile, isdir

from .utils import obj_types, is_iterable
from .headers import coe_header

from .vhdltools.write_file import write_file


def write_file_all(component, output_dir = "output", rm = False):

	write_file(
		component	= component, 
		output_dir	= output_dir, 
		rm		= rm
	)

	attr_list = [ attr for attr in dir(component) if not
			attr.startswith("__")]

	for attr_name in attr_list:

		sub = getattr(component, attr_name)

		if hasattr(sub, "write_file_all") and \
		callable(sub.write_file_all):
			sub.write_file_all(output_dir = output_dir)
		elif hasattr(sub, "write_file") and \
		callable(sub.write_file):
			sub.write_file(output_dir = output_dir)

	if is_iterable(component) and component.keys():

		for key in component:

			if hasattr(component[key], "write_file_all") and \
			callable(component[key].write_file_all):
				component[key].write_file_all(output_dir
				= output_dir)
			elif hasattr(component[key], "write_file") and \
			callable(component[key].write_file):
				component[key].write_file(output_dir = output_dir)




def vhdl_compile(name, output_dir = "output"):

	command = "cd " + output_dir + "; "
	command = command + "xvhdl --2008 " + name

	sp.run(command, shell = True)

	print("\n")


def vhdl_obj_compile(component, output_dir = "output"):

	attr_list = [ attr for attr in dir(component) if not
			attr.startswith("__")]

	if "entity" in attr_list:
		name = component.entity.name

	elif "name" in attr_list:
		name = component.name

	else:
		raise TypeError("Component cannot be compiled")

	name = name + ".vhd"

	print("\nCompiling " + name + "\n")
	vhdl_compile(name, output_dir = output_dir)



def fast_compile(component, output_dir = "output"):

	if hasattr(component, "components") and component.components:

		filenames = ""

		for name in component.components:
			filenames = filenames + name + ".vhd" + " "

		print("\nCompiling " + filenames + "\n")
		vhdl_compile(filenames, output_dir = output_dir)

	vhdl_obj_compile(component, output_dir = output_dir)


def clear_compile(component, output_dir = "output"):

	if hasattr(component, "components") and component.components:

		filenames = ""

		for name in component.components:
			filename = name + ".vhd"

			if not isfile(output_dir + "/" + filename):
				raise ValueError("File " + filename + 
					" doesn't exist. Create it "
					"first")

			print("\nCompiling " + filename + "\n")
			vhdl_compile(filename, output_dir = output_dir)

	vhdl_obj_compile(component, output_dir = output_dir)


def elaborate(component, output_dir = "output"):

	attr_list = [ attr for attr in dir(component) if not
			attr.startswith("__")]

	if "entity" not in attr_list:
		raise TypeError("Component has no entity to compile")

	name		= component.entity.name

	print("\nElaborating component %s\n" %(name))

	command = "cd " + output_dir + "; "
	command = command + "xelab " + name

	sp.run(command, shell = True)

	print("\n")


def sub_components(component):

	sub_comp = []

	attr_list = [ attr for attr in dir(component) if not
			attr.startswith("__")]

	for attr_name in attr_list:

		sub = getattr(component, attr_name)

		if "VHDLblock" in obj_types(sub):
			sub_comp.append(sub.entity.name)
			
		elif "Package" in obj_types(sub):
			sub_comp.insert(0, sub.name)

		if hasattr(sub, "components"):
			sub_comp += sub.components

	if is_iterable(component) and component.keys():

		for key in component:

			if "VHDLblock" in obj_types(component[key]):
				sub_comp.append(component[key].entity.name)
				
			elif "Package" in obj_types(component[key]):
				sub_comp.insert(0, component[key].name)

			if hasattr(component[key], "components"):
				sub_comp += component[key].components

	return list(dict.fromkeys(sub_comp))



def track_signals(signals_dict, name):

	signals_list = list(signals_dict.keys())

	exit_flag = 0
	tracked = []
	first = True
	invalid_signal = False

	while(exit_flag == 0 and signals_list):

		while first == False and good_answ == False and \
				invalid_signal == False:

			answer = input("Do you want to add others?(y/n) ")

			if answer == "n":

				exit_flag = 1
				good_answ = True

			elif answer == "y":
				good_answ = True

		if exit_flag == 0:

			input_msg = "Component " + name  + ": which signal " \
				"do you want to track(write exit to " \
				"stop)?\n\n" + str(signals_list) + "\n\n"

			signal = input(input_msg)

			if signal == "exit":
				exit_flag = 1
				invalid_signal = False
				
			elif signal not in signals_list:
				print("\nInvalid answer. ")
				invalid_signal = True

			else:
				tracked.append(signal)
				signals_list.remove(signal)
				invalid_signal = False

			good_answ = False
			first = False

	return tracked



def debug_component(component, db_list = []):

	attr_list = [ attr for attr in dir(component) if not 
			attr.startswith("__")]
	
	debug = []

	for attr_name in attr_list:

		sub_component = getattr(component, attr_name)

		if hasattr(sub_component, "debug"):

			debug += sub_component.debug

			for debug_port in sub_component.debug:


				component.entity.port.add(
					name 		=
						debug_port, 
					direction	= "out",
					port_type	= sub_component.entity.\
						port[debug_port].port_type
				)

	if db_list:
		
		debug_list = []

		for signal_name in db_list:

			if component.entity.name in signal_name:

				for internal_signal in \
				component.architecture.signal:

					if component.entity.name + "_" + \
					internal_signal == signal_name:

						debug_list.append(
							internal_signal)

	else:
		debug_list = track_signals(component.architecture.signal, 
				component.entity.name)

	for debug_port in debug_list:

		debug_port_name = component.entity.name + "_" + debug_port

		component.entity.port.add(
			name 		= debug_port_name, 
			direction	= "out",
			port_type	= component.architecture.\
					signal[debug_port].\
					signal_type)

		# Bring the signal out
		connect_string = debug_port_name + " <= " + \
					debug_port + ";"
		component.architecture.bodyCodeHeader.\
				add(connect_string)

		debug.append(debug_port_name)

	setattr(component, "debug", debug)


def coe_file(str_array, out_file, output_dir = "output"):

	with open(output_dir + "/" + out_file, "w") as fp:

		fp.write(coe_header)

		for i in range(len(str_array)):
			if i < len(str_array) - 1:
				fp.write(str_array[i] + ",\n")
			else:
				fp.write(str_array[i] + ";\n")

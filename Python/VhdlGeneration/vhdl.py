import subprocess as sp
from os.path import isfile, isdir

import path_config
from package_vhdl import Package



def write_file_all(component, output_dir = "output", rm = False)

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
	



def vhdl_compile(name, output_dir = "output"):

	file_name	= name + ".vhd"

	if not isfile(output_dir + "/" + file_name):
		raise FileNotFoundError(name + " file doesn't exist, create "
				"it first")

	print("\nCompiling component %s\n" %(name))

	command = "cd " + output_dir + "; "
	command = command + "xvhdl " + file_name + "; "

	sp.run(command, shell = True)

	print("\n")


def vhdl_obj_compile(component, output_dir = "output"):

	attr_list = [ attr for attr in dir(component) if not
			attr.startswith("__")]

	if "entity" not in attr_list or not isinstance(component, Package):
		raise TypeError("Component cannot be compile")

	name		= component.entity.name

	vhdl_compile(name, output_dir = output_dir)



def compile_all(component, output_dir = "output"):

	vhdl_obj_compile(component, output_dir = output_dir)

	if hasattr(component, "components"):
		for name in component.components:
			vhdl_compile(name, output_dir = output_dir)



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

		if hasattr(sub, "entity"):
			sub_comp.append(sub.entity.name)
			
		if isinstance(sub, Package):
			sub_comp.append(sub.name)

		if hasattr(sub, "components"):
			sub_comp += sub.components

		sub_comp = list(set(sub_comp))

	return sub_comp


def track_signals(signals_dict, name):

	signals_list = list(signals_dict.keys())

	exit_flag = 0
	tracked = []
	first = True
	invalid_signal = False

	while(exit_flag == 0):

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

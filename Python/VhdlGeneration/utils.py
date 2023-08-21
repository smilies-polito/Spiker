import subprocess as sp
import numpy as np
from random import randint


def createDir(dirName):

	"""
	Create a new directory. If it already exists it is firstly remove.

	INPUT:

		dirName: string. Name of the directory to create
	"""

	# Check if the directory exists
	cmdString = "if [[ -d " + dirName + " ]]; then "

	# If it exists remove it
	cmdString += "rm -r " + dirName + "; "
	cmdString += "fi; "

	# Create the directory
	cmdString += "mkdir " + dirName + "; "
	
	# Run the complete bash command
	sp.run(cmdString, shell=True, executable="/bin/bash")


def ceil_pow2(x):

	power = 0
	go = True

	while go:
		if x // 2**power == 0 or x // 2**power == 1 and \
			x % 2**power == 0:

			go = False

		else:
			power += 1
			
	return 2**power

def n_bytes(x):

	if x == 0:
		return 1

	return int((x-1) // 8) + 1


def floor_pow2(x):

	power = 0
	go = True

	while go:
		if x // 2**power == 1 or x == 0:
			go = False

		else:
			power += 1
			
	return 2**power


def random_binary(min_value = 0, max_value = 255, bitwidth = 8):

	if max_value > 2**bitwidth - 1 or min_value < 0:
		print("Random number not representable on bitwidth\n")
		exit(-1)

	rand_int = randint(min_value, max_value)

	return "{0:{fill}{width}{base}}".format(rand_int, fill = 0,
		width = bitwidth, base = "b")

def int_to_bin(value, fill = 0, width = 8):
	return "{0:{fill}{width}{base}}".format(value, fill = fill, width =
			width, base = "b")

def int_to_hex(value, fill = 0, width = 8):
	return "{0:{fill}{width}{base}}".format(value, fill = fill, width =
			width, base = "x")


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


def fixed_point_array(numpyArray, bitwidth, fixed_point_decimals = 0,
	conv_type = "signed"):

	'''
	Convert a NumPy array into fixed point notation.

	INPUT:

		1) numpyArray: floating point array to convert.

		2) fixed_point_decimals: number of decimal bits in the fixed
		point representation.

	'''

	numpyArray = numpyArray * 2**fixed_point_decimals
	fp_array = numpyArray.astype(int)

	if conv_type == "signed":
		fp_array[fp_array > 2**(bitwidth-1)-1] = 2**(bitwidth-1)-1
		fp_array[fp_array < -2**(bitwidth-1)] = -2**(bitwidth-1)

	elif conv_type == "unsigned":
		fp_array[fp_array > 2**bitwidth-1] = 2**(bitwidth)-1

	else:
		raise ValueError("Invalid fixed-point convertion type")

	return fp_array

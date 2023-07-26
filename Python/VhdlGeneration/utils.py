import subprocess as sp
import numpy as np


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

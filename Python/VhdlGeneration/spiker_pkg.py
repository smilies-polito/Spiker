import path_config
import subprocess as sp

from package_vhdl import Package

class SpikerPackage(Package):

	def __init__(self):

		Package.__init__(self, "spiker_pkg")


		self.pkg_dec.type_list.add("neuron_states", "Enumeration")
		self.pkg_dec.type_list["neuron_states"].add("reset")
		self.pkg_dec.type_list["neuron_states"].add("load")
		self.pkg_dec.type_list["neuron_states"].add("idle")
		self.pkg_dec.type_list["neuron_states"].add("init")
		self.pkg_dec.type_list["neuron_states"].add("excite")
		self.pkg_dec.type_list["neuron_states"].add("inhibit")
		self.pkg_dec.type_list["neuron_states"].add("fire")
		self.pkg_dec.type_list["neuron_states"].add("leak")


	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")

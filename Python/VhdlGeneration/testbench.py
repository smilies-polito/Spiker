import subprocess as sp
import path_config
from vhdl_block import VHDLblock


class Testbench(VHDLblock):

	def __init__(self, component, clock_period = 20):

		self.dut = component
		self.name = self.dut.entity.name + "_tb"
		self.clk_names = ["clk", "clock", "CLK", "CLOCK"]

		VHDLblock.__init__(self, entity_name = self.name)

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		for name in self.dut.entity.generic:
			if self.dut.entity.generic[name].value:
				self.architecture.constant.add(
					self.dut.entity.generic[name].name,
					self.dut.entity.generic[name].gen_type,
					self.dut.entity.generic[name].value
				)
			else:
				value = input("Choose a value for testbench "
					"constant " + 
					self.dut.entity.generic[name].name +
					": ")
				self.architecture.constant.add(
					self.dut.entity.generic[name].name,
					self.dut.entity.generic[name].gen_type,
					value
				)


		for name in self.dut.entity.port:
			self.architecture.signal.add(
				self.dut.entity.port[name].name,
				self.dut.entity.port[name].port_type,
				self.dut.entity.port[name].value
			)

		self.architecture.component.add(self.dut)

		self.architecture.instances.add(self.dut, "dut")
		self.architecture.instances["dut"].generic_map()
		self.architecture.instances["dut"].port_map()


		for name in self.dut.entity.port:
			if name in self.clk_names:
				self.architecture.processes.add(name + "_gen")
				self.architecture.processes[name +
					"_gen"].body.add(name + " <= '0';")
				self.architecture.processes[name +
					"_gen"].body.add("wait for " +
					str(clock_period//2) + " ns;")
				self.architecture.processes[name +
					"_gen"].body.add(name + " <= '1';")
				self.architecture.processes[name +
					"_gen"].body.add("wait for " +
					str(clock_period//2) + " ns;")

			elif self.dut.entity.port[name].direction == "in":
				self.architecture.processes.add(name = name + "_gen",
						final_wait = True)


	def compile(self, output_dir = "output"):

		print("\nCompiling component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xvhdl --2008 " + self.entity.name + ".vhd"

		sp.run(command, shell = True)

		print("\n")


	def compile_all(self, output_dir = "output"):

		if hasattr(self.dut, "compile_all") and \
			callable(self.dut.compile_all):

			self.dut.compile_all()

		else:
			self.dut.compile()

		self.compile()


	def write_file_all(self, output_dir = "output"):

		if hasattr(self.dut, "write_file_all") and \
			callable(self.dut.write_file_all):

			self.dut.write_file_all(output_dir = output_dir)

		else:

			self.dut.write_file(output_dir = output_dir)

		self.write_file(output_dir = output_dir)


	def elaborate(self, output_dir = "output"):

		print("\nElaborating component %s\n"
				%(self.entity.name))

		command = "cd " + output_dir + "; "
		command = command + "xelab " + self.entity.name

		sp.run(command, shell = True)

		print("\n")

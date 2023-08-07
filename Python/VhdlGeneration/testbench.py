import subprocess as sp
import path_config
from vhdl_block import VHDLblock
from if_statement import If


class Testbench(VHDLblock):

	def __init__(self, component, clock_period = 20, file_output = False):

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
					"_gen"].bodyHeader.add(name + 
					" <= '0';")
				self.architecture.processes[name +
					"_gen"].bodyHeader.add("wait for " +
					str(clock_period//2) + " ns;")
				self.architecture.processes[name +
					"_gen"].bodyHeader.add(name + 
					" <= '1';")
				self.architecture.processes[name +
					"_gen"].bodyHeader.add("wait for " +
					str(clock_period//2) + " ns;")

			elif self.dut.entity.port[name].direction == "in":
				self.architecture.processes.add(name = name +
						"_gen", final_wait = True)

			elif self.dut.entity.port[name].direction == "out" and \
				file_output :

				self.library.add("std")
				self.library["std"].package.add(
					"textio"
				)

				self.library["ieee"].package.add(
					"std_logic_textio"
				)

				process_name = name + "_save"
				en_name	= name + "_w_en"
				out_file = name + "_file"
				out_filename = name + "_filename"
				en_gen = en_name + "_gen"

				self.architecture.constant.add(
					name = out_filename,
					const_type = "string",
					value = "\"" + name + ".txt\""
				)

				self.architecture.signal.add(
					name = en_name,
					signal_type = "std_logic"
				)


				self.architecture.processes.add(en_gen, 
						final_wait = True)
				self.architecture.processes[en_gen].bodyHeader.\
						add(en_name + "<= '1';")


				self.architecture.processes.add(name =
						process_name)

				self.architecture.processes[process_name].\
				     	sensitivity_list.add("clk")
				self.architecture.processes[process_name].\
				     	sensitivity_list.add(en_name)

				self.architecture.processes[process_name].\
				     variables.add(
				     name 		= "row",
				     var_type	= "line"
				)
				self.architecture.processes[process_name].\
				     variables.add(
				     name 		= "write_var",
				     var_type	= "integer"
				)
				self.architecture.processes[process_name].\
					files.add(
					name 		= out_file,
					file_type	= "text",
					mode		= "write_mode",
					filename	= out_filename
				)

				w_en_if = If()
				w_en_if._if_.conditions.add(en_name + " = '1'")

				if self.dut.entity.port[name].port_type == \
					"std_logic":

					w_en_if._if_.body.add(
						"write(row, " + name  + ");")

				else:
					w_en_if._if_.body.add(
						"write_var := to_integer("
						"unsigned(" + name  + "));")

					w_en_if._if_.body.add(
						"write(row, write_var);")
				w_en_if._if_.body.add(
					"writeline(" + out_file  + ", row);")

				clk_if = If()
				clk_if._if_.conditions.add("clk'event")
				clk_if._if_.conditions.add("clk <= '1'", "and")
				clk_if._if_.body.add(w_en_if)

				self.architecture.processes[process_name].\
					bodyHeader.add(clk_if)



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

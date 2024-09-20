from .vhdl import sub_components, write_file_all
from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class Testbench(VHDLblock):

	def __init__(self, dut, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = []):

		self.dut = dut
		self.name = self.dut.entity.name + "_tb"

		self.clk_names = ["clk", "clock", "CLK", "CLOCK"]
		self.clock_period = clock_period

		VHDLblock.__init__(self, entity_name = self.name)
		self.vhdl_template( 
			file_output		= file_output, 
			output_dir		= output_dir, 
			file_input		= file_input, 
			input_dir		= input_dir, 
			input_signal_list	= input_signal_list
		)

		self.components = sub_components(self)


	def vhdl_template(self, file_output = False, output_dir = "output",
			file_input = False, input_dir = "",
			input_signal_list = []):

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
					str(self.clock_period//2) + " ns;")
				self.architecture.processes[name +
					"_gen"].bodyHeader.add(name + 
					" <= '1';")
				self.architecture.processes[name +
					"_gen"].bodyHeader.add("wait for " +
					str(self.clock_period//2) + " ns;")

			elif self.dut.entity.port[name].direction == "in":
				self.architecture.processes.add(name = name +
						"_gen", final_wait = True)

			elif self.dut.entity.port[name].direction == "out" and \
				file_output :

				self.save(name, output_dir)

		if file_input:
			for signal in input_signal_list:
				if signal not in self.architecture.signal:

					raise ValueError("Error, cannot read " +
							signal + " from file. "
							"Signal doesn't exist.")

				self.load(signal, input_dir)


	def save(self, signal_name, output_dir = "output"):

		clk_present = False

		for clk in self.clk_names:
			if clk in self.entity.port:
				clk_present = True
				clk_name = clk

		if not clk_present:
			clk_name = "clk"
			self.architecture.signal.add(
				name = clk_name,
				signal_type = "std_logic"
			)
			self.architecture.processes.add(clk_name + "_gen")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add(clk_name + " <= '0';")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add("wait for " + 
				str(self.clock_period//2) + " ns;")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add(clk_name + " <= '1';")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add("wait for " + 
				str(self.clock_period//2) + " ns;")

		self.library.add("std")

		self.library["std"].package.add("textio")

		self.library["ieee"].package.add("std_logic_textio")

		process_name = signal_name + "_save"
		en_name	= signal_name + "_w_en"
		out_file = signal_name + "_file"
		out_filename = signal_name + "_filename"
		en_gen = en_name + "_gen"

		self.architecture.constant.add(
			name = out_filename,
			const_type = "string",
			value = "\"" + signal_name + ".txt\""
		)

		self.architecture.signal.add(
			name = en_name,
			signal_type = "std_logic"
		)

		self.architecture.processes.add(en_gen, final_wait = True)
		self.architecture.processes[en_gen].bodyHeader.\
				add(en_name + "<= '1';")


		self.architecture.processes.add(name = process_name)

		self.architecture.processes[process_name].\
			sensitivity_list.add(clk_name)
		self.architecture.processes[process_name].\
			sensitivity_list.add(en_name)


		self.architecture.processes[process_name].\
		     variables.add(
		     name 		= "row",
		     var_type		= "line"
		)

		if self.dut.entity.port[signal_name].port_type != "std_logic":

			self.architecture.processes[process_name].\
			     variables.add(
			     name 		= "write_var",
			     var_type		= "integer"
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

		if self.dut.entity.port[signal_name].port_type == \
			"std_logic":

			w_en_if._if_.body.add(
				"write(row, " + signal_name  + ");")

		else:
			w_en_if._if_.body.add(
				"write_var := to_integer("
				"unsigned(" + signal_name  + "));")

			w_en_if._if_.body.add(
				"write(row, write_var);")
		w_en_if._if_.body.add(
			"writeline(" + out_file  + ", row);")

		clk_if = If()
		clk_if._if_.conditions.add(clk_name + "\'event")
		clk_if._if_.conditions.add(clk_name + " = '1'", "and")
		clk_if._if_.body.add(w_en_if)

		self.architecture.processes[process_name].\
			bodyHeader.add(clk_if)


	def load(self, signal_name, input_dir):

		clk_present = False

		for clk in self.clk_names:
			if clk in self.entity.port:
				clk_present = True
				clk_name = clk

		if not clk_present:
			clk_name = "clk"
			self.architecture.signal.add(
				name = clk_name,
				signal_type = "std_logic"
			)
			self.architecture.processes.add(clk_name + "_gen")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add(clk_name + " <= '0';")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add("wait for " + 
				str(self.clock_period//2) + " ns;")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add(clk_name + " <= '1';")
			self.architecture.processes[clk_name + "_gen"].\
				bodyHeader.add("wait for " + 
				str(self.clock_period//2) + " ns;")

		self.library.add("std")

		self.library["std"].package.add("textio")

		self.library["ieee"].package.add("std_logic_textio")

		process_name = signal_name + "_load"
		en_name	= signal_name + "_rd_en"
		in_file = signal_name + "_file"
		in_filename = signal_name + "_filename"
		en_gen = en_name + "_gen"

		if self.architecture.processes[signal_name + "_gen"]:
			del self.architecture.processes[signal_name + "_gen"]

		if input_dir:
			self.architecture.constant.add(
				name = in_filename,
				const_type = "string",
				value = "\"" + input_dir + "/" + signal_name + \
					".txt\""
			)

		else:
			self.architecture.constant.add(
				name = in_filename,
				const_type = "string",
				value = "\"" + signal_name + ".txt\""
			)

		self.architecture.signal.add(
			name = en_name,
			signal_type = "std_logic"
		)

		self.architecture.processes.add(en_gen, final_wait = True)
		self.architecture.processes[en_gen].bodyHeader.\
				add(en_name + "<= '1';")


		self.architecture.processes.add(name = process_name)

		self.architecture.processes[process_name].\
			sensitivity_list.add(clk_name)
		self.architecture.processes[process_name].\
			sensitivity_list.add(en_name)


		self.architecture.processes[process_name].\
		     variables.add(
		     name 		= "row",
		     var_type		= "line"
		)

		self.architecture.processes[process_name].\
		     variables.add(
		     name 		= "read_var",
		     var_type		= self.architecture.
		     			signal[signal_name].signal_type
		)

		self.architecture.processes[process_name].\
			files.add(
			name 		= in_file,
			file_type	= "text",
			mode		= "read_mode",
			filename	= in_filename
		)

		rd_en_if = If()
		rd_en_if._if_.conditions.add(en_name + " = '1'")

		rd_en_if._if_.body.add("readline(" + in_file  + ", row);")
		rd_en_if._if_.body.add("read(row, read_var);")
		rd_en_if._if_.body.add(signal_name + " <= read_var;")

		eof_if = If()
		eof_if._if_.conditions.add("not endfile(" + in_file  + ")")
		eof_if._if_.body.add(rd_en_if)

		clk_if = If()
		clk_if._if_.conditions.add(clk_name + "\'event")
		clk_if._if_.conditions.add(clk_name + " = '1'", "and")
		clk_if._if_.body.add(eof_if)

		self.architecture.processes[process_name].\
			bodyHeader.add(clk_if)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)

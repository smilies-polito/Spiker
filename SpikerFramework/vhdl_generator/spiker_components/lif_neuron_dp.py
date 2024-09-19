from .shifter import Shifter
from .add_sub import AddSub
from .mux import Mux
from .reg import Reg
from .cmp import Cmp

from .testbench import Testbench
from .vhdl import track_signals, debug_component, sub_components, write_file_all

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If

class LIFneuronDP(VHDLblock):

	def __init__(self, bitwidth = 16, w_inh_bw = 16, w_exc_bw = 16,
			shift = 10, reset = "fixed", debug = False, 
			debug_list = []):

		self.reset_types = [
			"fixed",
			"subtractive"
		]
		
		if reset not in self.reset_types:
			raise ValueError(str(reset) + " reset type not "
					"allowed")

		self.reset = reset

		self.name = "neuron_datapath"

		self.shifter			= Shifter(
							bitwidth = bitwidth,
							shift = shift)

		self.mux4to1_signed		= Mux(
							n_in = 4,
							in_type = "signed",
							bitwidth =
							bitwidth)

		self.add_sub			= AddSub(
							bitwidth = bitwidth,
							saturated = True)

		if self.reset == "fixed":
			self.mux2to1_signed	= Mux(
							n_in = 2,
							in_type = "signed",
							bitwidth =
							bitwidth
						)

		self.reg_signed_sync_rst	= Reg(
							bitwidth = bitwidth,
							reg_type = "signed",
							rst = "sync")

		self.cmp_gt			= Cmp(
							bitwidth = bitwidth)

		self.bitwidth = bitwidth
		self.w_exc_bw = w_exc_bw
		self.w_inh_bw = w_inh_bw
		self.shift = shift
		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")


		# Generics
		self.entity.generic.add(
			name		= "neuron_bit_width", 
			gen_type	= "integer",
			value		= str(self.bitwidth))


		if self.w_inh_bw < self.bitwidth:
			self.entity.generic.add(
				name		= "inh_weights_bit_width",
				gen_type	= "integer",
				value		= str(
						self.w_inh_bw))

		if self.w_exc_bw < self.bitwidth:
			self.entity.generic.add(
				name		= "exc_weights_bit_width",
				gen_type	= "integer",
				value		= str(
						self.w_exc_bw))

		self.entity.generic.add(
			name		= "shift",
			gen_type	= "integer",
			value		= str(self.shift))

		# Input parameters
		self.entity.port.add(
			name 		= "v_th", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")

		if self.reset == "fixed":
			self.entity.port.add(
				name 		= "v_reset", 
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
						"downto 0)")

		if self.w_inh_bw < self.bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed("
					"inh_weights_bit_width-1 downto 0)")
		elif self.w_inh_bw == self.bitwidth:
			self.entity.port.add(
				name 		= "inh_weight",
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			raise ValueError("Inhibitory weight bit-width cannot "
					"be larger than the neuron's one")
			

		if self.w_exc_bw < self.bitwidth:
			self.entity.port.add(
				name 		= "exc_weight",
				direction	= "in",
				port_type	= "signed("
					"exc_weights_bit_width-1 downto 0)")

		elif self.w_exc_bw == self.bitwidth:
			self.entity.port.add(
				name 		= "exc_weight",
				direction	= "in",
				port_type	= "signed(neuron_bit_width-1 "
							"downto 0)")
		else:
			raise ValueError("Excitatory weight bit-width cannot "
					"be larger than the neuron's one")

		# Input controls
		self.entity.port.add(
			name 		= "clk", 
			direction	= "in", 
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "update_sel",
			direction	= "in", 
			port_type	= "std_logic_vector(1 downto 0)")
		self.entity.port.add(
			name 		= "add_or_sub", 
			direction	= "in", 
			port_type	= "std_logic")

		if self.reset == "fixed":
			self.entity.port.add(
				name 		= "v_update",
				direction	= "in",
				port_type	= "std_logic")

		self.entity.port.add(
			name 		= "v_en",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "v_rst_n",
			direction	= "in",
			port_type	= "std_logic")


		# Output
		self.entity.port.add(
			name 		= "exceed_v_th",
			direction	= "out",
			port_type	= "std_logic")
		

		# Signals
		self.architecture.signal.add(
			name		= "update",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")

		if self.reset == "fixed":
			self.architecture.signal.add(
				name		= "update_value",
				signal_type	= "signed(neuron_bit_width-1 "
						"downto 0)")

		self.architecture.signal.add(
			name		= "v_value",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")

		self.architecture.signal.add(
			name		= "v",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")

		self.architecture.signal.add(
			name		= "v_shifted",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")


		# Components
		self.architecture.component.add(self.shifter)
		self.architecture.component.add(self.add_sub)
		self.architecture.component.add(self.mux4to1_signed)
		self.architecture.component.add(self.add_sub)

		if self.reset == "fixed":
			self.architecture.component.add(self.mux2to1_signed)

		self.architecture.component.add(self.reg_signed_sync_rst)
		self.architecture.component.add(self.cmp_gt)


		# Shifter
		self.architecture.instances.add(self.shifter, "v_shifter")
		self.architecture.instances["v_shifter"].generic_map()
		self.architecture.instances["v_shifter"].g_map.add("N",
				"neuron_bit_width")
		self.architecture.instances["v_shifter"].port_map("pos", "v",
				"v_shifted")


		# Multiplexer 4 to 1 signed
		self.architecture.instances.add(self.mux4to1_signed,
				"update_mux")
		self.architecture.instances["update_mux"].generic_map()
		self.architecture.instances["update_mux"].g_map.add("bitwidth",
				"neuron_bit_width")
		self.architecture.instances["update_mux"].port_map(mode = "no")
		self.architecture.instances["update_mux"].p_map.add("mux_sel",
				"update_sel")

		if self.reset == "fixed":
			self.architecture.instances["update_mux"].p_map.add(
					"in0", "(others => '0')")

		elif self.reset == "subtractive":
			self.architecture.instances["update_mux"].p_map.add(
					"in0", "v_th")


		self.architecture.instances["update_mux"].p_map.add("in1",
				"v_shifted")

		if self.w_exc_bw < self.bitwidth:

			self.architecture.instances["update_mux"].p_map.add(
					"in2", "(others => "
					"exc_weight(exc_weights_bit_width-1))", 
					conn_range = "(neuron_bit_width-1 "
					"downto exc_weights_bit_width)")
			self.architecture.instances["update_mux"].p_map.add(
					"in2", "exc_weight", 
					conn_range = "(exc_weights_bit_width-1 "
					"downto 0)")

		elif self.w_exc_bw == self.bitwidth:
			self.architecture.instances["update_mux"].p_map.add(
					"in2", "exc_weight")

		else:
			raise ValueError("Excitatory weight bit-width cannot "
					"be larger than the neuron's one")


		if self.w_inh_bw < self.bitwidth:

			self.architecture.instances["update_mux"].p_map.add(
					"in3", "(others => "
					"inh_weight(inh_weights_bit_width-1))", 
					conn_range = "(neuron_bit_width-1 "
					"downto inh_weights_bit_width)")
			self.architecture.instances["update_mux"].p_map.add(
					"in3", "inh_weight", 
					conn_range = "(inh_weights_bit_width-1 "
					"downto 0)")

		elif self.w_inh_bw == self.bitwidth:
			self.architecture.instances["update_mux"].p_map.add(
					"in3", "inh_weight")

		else:
			raise ValueError("Inhibitory weight bit-width cannot "
					"be larger than the neuron's one")


		
		self.architecture.instances["update_mux"].p_map.add("mux_out",
				"update")


		# Adder/subtractor
		self.architecture.instances.add(self.add_sub, "update_add_sub")
		self.architecture.instances["update_add_sub"].generic_map()
		self.architecture.instances["update_add_sub"].g_map.add("N",
				"neuron_bit_width")
		self.architecture.instances["update_add_sub"].port_map()
		self.architecture.instances["update_add_sub"].p_map.add("in0",
				"v")
		self.architecture.instances["update_add_sub"].p_map.add("in1",
				"update")

		if self.reset == "fixed":
			self.architecture.instances["update_add_sub"].p_map.add(
					"add_sub_out", "update_value")

		elif self.reset == "subtractive":
			self.architecture.instances["update_add_sub"].p_map.add(
					"add_sub_out", "v_value")



		# Multiplexer 2 to 1 signed
		if self.reset == "fixed":
			self.architecture.instances.add(self.mux2to1_signed, 
					"v_mux")
			self.architecture.instances["v_mux"].generic_map()
			self.architecture.instances["v_mux"].g_map.add(
					"bitwidth", "neuron_bit_width")
			self.architecture.instances["v_mux"].port_map()
			self.architecture.instances["v_mux"].p_map.add(
					"mux_sel", "v_update")
			self.architecture.instances["v_mux"].p_map.add("in0",
					"v_reset")
			self.architecture.instances["v_mux"].p_map.add("in1",
					"update_value")
			self.architecture.instances["v_mux"].p_map.add(
					"mux_out", "v_value")


		# Signed register with synchronous reset
		self.architecture.instances.add(self.reg_signed_sync_rst,
				"v_reg")
		self.architecture.instances["v_reg"].generic_map()
		self.architecture.instances["v_reg"].g_map.add("N",
				"neuron_bit_width")
		self.architecture.instances["v_reg"].port_map()
		self.architecture.instances["v_reg"].p_map.add("en",
				"v_en")
		self.architecture.instances["v_reg"].p_map.add("rst_n",
				"v_rst_n")
		self.architecture.instances["v_reg"].p_map.add("reg_in",
				"v_value")
		self.architecture.instances["v_reg"].p_map.add("reg_out",
				"v")

		
		# Signed comparator 
		self.architecture.instances.add(self.cmp_gt,
				"fire_cmp")
		self.architecture.instances["fire_cmp"].generic_map()
		self.architecture.instances["fire_cmp"].g_map.add("N",
				"neuron_bit_width")
		self.architecture.instances["fire_cmp"].port_map()
		self.architecture.instances["fire_cmp"].p_map.add("in0",
				"v")
		self.architecture.instances["fire_cmp"].p_map.add("in1",
				"v_th")
		self.architecture.instances["fire_cmp"].p_map.add("cmp_out",
				"exceed_v_th")

		# Debug
		if debug:
			debug_component(self, debug_list)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)



class LIFneuronDP_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = [], bitwidth = 16, w_inh_bw = 16,
			w_exc_bw = 16, shift = 4, reset = "fixed", 
			debug = False, debug_list = []):

		self.dut = LIFneuronDP(
			bitwidth = bitwidth,
			w_inh_bw = w_inh_bw,
			w_exc_bw = w_exc_bw,
			shift = shift,
			reset = reset,
			debug = debug,
			debug_list = debug_list
		)

		self.components = sub_components(self)

		super().__init__(
			dut = self.dut, 
			clock_period = clock_period,
			file_output = file_output,
			output_dir = output_dir,
			file_input = file_input,
			input_dir = input_dir,
			input_signal_list = input_signal_list
		)
		
		self.vhdl(clock_period = clock_period, file_output = file_output)

	def vhdl(self, clock_period = 20, file_output = False):

		# exc_weight
		self.architecture.processes["exc_weight_gen"].bodyHeader.\
				add("exc_weight <= to_signed(500, "
				"exc_weight'length);")

		# inh_weight
		self.architecture.processes["inh_weight_gen"].bodyHeader.\
				add("inh_weight <= to_signed(300, "
				"inh_weight'length);")

		# v_reset
		if self.dut.reset == "fixed":
			self.architecture.processes["v_reset_gen"].bodyHeader.\
					add("v_reset <= to_signed(1000, "
					"v_reset'length);")

		# v_th_value
		self.architecture.processes["v_th_gen"].bodyHeader.\
				add("v_th <= to_signed(3000, "
				"v_th'length);")
		# v_rst_n
		self.architecture.processes["v_rst_n_gen"].bodyHeader.\
				add("v_rst_n <= '1';")
		self.architecture.processes["v_rst_n_gen"].bodyHeader.\
				add("wait for " + str(clock_period) + " ns;")
		self.architecture.processes["v_rst_n_gen"].bodyHeader.\
				add("v_rst_n <= '0';")
		self.architecture.processes["v_rst_n_gen"].bodyHeader.\
				add("wait for " + str(clock_period)  + " ns;")
		self.architecture.processes["v_rst_n_gen"].bodyHeader.\
				add("v_rst_n <= '1';")

		# update_sel
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("update_sel <= \"00\";")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("wait for " + str(2*clock_period) + " ns;")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("update_sel <= \"10\";")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("wait for " + str(4*clock_period) + " ns;")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("update_sel <= \"11\";")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("wait for " + str(clock_period)  + " ns;")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("update_sel <= \"10\";")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("wait for " + str(3*clock_period)  + " ns;")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("update_sel <= \"00\";")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("wait for " + str(3*clock_period)  + " ns;")
		self.architecture.processes["update_sel_gen"].bodyHeader.\
				add("update_sel <= \"01\";")

		# add_or_sub
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("add_or_sub <= '0';")
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("wait for " + str(6*clock_period) + " ns;")
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("add_or_sub <= '1';")
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("wait for " + str(clock_period)  + " ns;")
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("add_or_sub <= '0';")
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("wait for " + str(4*clock_period)  + " ns;")
		self.architecture.processes["add_or_sub_gen"].bodyHeader.\
				add("add_or_sub <= '1';")

		# v_en
		self.architecture.processes["v_en_gen"].bodyHeader.\
				add("v_en <= '0';")
		self.architecture.processes["v_en_gen"].bodyHeader.\
				add("wait for " + str(2*clock_period) + " ns;")
		self.architecture.processes["v_en_gen"].bodyHeader.\
				add("v_en <= '1';")

		# v_update
		if self.dut.reset == "fixed":
			self.architecture.processes["v_update_gen"].bodyHeader.\
					add("v_update <= '1';")
			self.architecture.processes["v_update_gen"].bodyHeader.\
					add("wait for " + str(11*clock_period) 
					+ " ns;")
			self.architecture.processes["v_update_gen"].bodyHeader.\
					add("v_update <= '0';")
			self.architecture.processes["v_update_gen"].bodyHeader.\
					add("wait for " + str(clock_period)
					+ " ns;")
			self.architecture.processes["v_update_gen"].bodyHeader.\
					add("v_update <= '1';")

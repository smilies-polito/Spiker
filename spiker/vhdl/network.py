import numpy as np

from math import log2

from .multi_cycle import MultiCycle
from .layer import Layer
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .decoder import Decoder
from .mux import Mux
from .reg import Reg
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, obj_types, is_iterable

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If, ConditionsList
from .vhdltools.text import SingleCodeLine
from .vhdltools.for_statement import For
from .vhdltools.instance import Instance


class Network(VHDLblock, dict):

	def __init__(self, n_cycles = 10, debug = False, debug_list = []):

		self.layer_index 	= 0
		self.all_ready 		= ConditionsList()
		self.n_cycles		= n_cycles

		self.name = "network"

		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(
			self.n_cycles+1))) + 1


		self.spiker_pkg = SpikerPackage()

		self.multi_cycle = MultiCycle(
			n_cycles = self.n_cycles,
			debug = debug,
			debug_list = debug_list
		)

		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")
				

		# Generics
		self.entity.generic.add(
			name		= "n_cycles", 
			gen_type	= "integer",
			value		= str(self.n_cycles))
		self.entity.generic.add(
			name		= "cycles_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.cycles_cnt_bitwidth))


		# Input controls
		self.entity.port.add(
			name 		= "clk", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "rst_n", 
			direction	= "in", 
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "start", 
			direction	= "in",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "sample_ready", 
			direction	= "in",
			port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "ready", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "sample", 
			direction	= "out",
			port_type	= "std_logic")


		# Components
		self.architecture.component.add(self.multi_cycle)

		self.architecture.signal.add(
			name		= "start_all",
			signal_type	= "std_logic"
		)

		self.architecture.signal.add(
			name		= "all_ready",
			signal_type	= "std_logic"
		)

		self.architecture.signal.add(
			name		= "restart",
			signal_type	= "std_logic"
		)

		self.architecture.bodyCodeHeader.add(
			"sample <= start_all;"
		)

		# Multi-input control
		self.architecture.instances.add(self.multi_cycle,
				"multi_cycle_control")
		self.architecture.instances["multi_cycle_control"].generic_map()
		self.architecture.instances["multi_cycle_control"].port_map()

		self.all_ready.add("sample_ready")
		self.architecture.bodyCodeHeader.add("all_ready <= " +
				self.all_ready.code() + ";\n")

		
		# Debug
		if debug:
			debug_component(self, debug_list)


	def add(self, layer):

		current_layer 	= "layer_" + str(self.layer_index)
		layer_ready	= current_layer + "_ready"

		self[current_layer] = layer

		# Check if component is already declared in the architecture
		declared = False
		for component_name in self.architecture.component:
			if layer.name == component_name:
				declared = True

		if not declared:
			self.architecture.component.add(layer)

		self.components = sub_components(self)

		# Add the ready signal for the layer
		self.architecture.signal.add(
			name		= layer_ready,
			signal_type	= "std_logic"
		)

		self.architecture.signal.add(
			name		= current_layer + "_feedback",
			signal_type	= "std_logic_vector(" +
					str(layer.n_neurons-1)  + " downto 0)"
		)

		# Instantiate the layer
		self.architecture.instances.add(layer, current_layer)
		self.architecture.instances[current_layer].generic_map\
			(mode = "self")
		self.architecture.instances[current_layer].port_map()
		self.architecture.instances[current_layer].p_map.add(
			"start", "start_all")
		self.architecture.instances[current_layer].p_map.add(
			"ready", layer_ready)
		self.architecture.instances[current_layer].p_map.add(
			"out_spikes", current_layer + "_feedback")
		self.architecture.instances[current_layer].p_map.add(
			"inh_spikes", current_layer + "_feedback")


		if self.layer_index == 0:

			self.entity.port.add(
				name		= "in_spikes",
				direction	= "in",
				port_type	= "std_logic_vector(" +
				str(layer.n_exc_inputs-1)  + " downto 0)"
			)

			self.entity.port.add(
				name		= "out_spikes",
				direction	= "out",
				port_type	= "std_logic_vector(" +
				str(layer.n_neurons-1)  + " downto 0)"
			)

			self.architecture.instances[current_layer].p_map.add(
					"exc_spikes", "in_spikes")

			self.architecture.bodyCodeHeader.add("out_spikes <= ",
					current_layer + "_feedback;\n")

		else:

			previous_layer = "layer_" + str(self.layer_index - 1)

			if layer.n_exc_inputs != self[previous_layer].n_neurons:
				raise ValueError("Layer cannot be added to the"
						" network. Incompatile number"
						" of inputs")

			self.entity.port.add(
				name		= "out_spikes",
				direction	= "out",
				port_type	= "std_logic_vector(" +
				str(layer.n_neurons-1)  + " downto 0)")


			exc_spikes_internal = "exc_spikes_" + \
				str(self.layer_index-1) + "_to_" + \
				str(self.layer_index)

			self.architecture.signal.add(
				name		= exc_spikes_internal,
				signal_type	= "std_logic_vector(" +
						str(layer.n_exc_inputs - 1) + 
						" downto 0)"
			)

			self.architecture.instances[current_layer].p_map.add(
				"exc_spikes", exc_spikes_internal)

			self.architecture.bodyCodeHeader[2] = SingleCodeLine(
				"out_spikes <= ", current_layer + 
				"_feedback;\n")

			self.architecture.bodyCodeHeader.add(
				exc_spikes_internal + "<= ", previous_layer + 
				"_feedback;\n")


		self.all_ready.add(layer_ready, "and")
		self.architecture.bodyCodeHeader[1] = SingleCodeLine(
				"all_ready <= " + self.all_ready.code()
				+ ";\n")

		self.layer_index += 1


	
	def first_layer(self):

		attr_list = [ attr for attr in dir(self) if not
				attr.startswith("__")]

		for attr_name in attr_list:

			sub = getattr(self, attr_name)

			print(obj_types(sub))

			if "Layer" in obj_types(sub):
				return False

		return True

	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)



class Network_tb(Testbench):

	def __init__(self, network, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = [], 
			debug = False, debug_list = []):


		self.spiker_pkg = SpikerPackage()

		self.dut = network
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
		
		self.vhdl(
			clock_period		= clock_period,
			file_output		= file_output,
			output_dir		= output_dir,
			file_input		= file_input,
			input_dir		= input_dir,
			input_signal_list 	= input_signal_list
			)


	def vhdl(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = []):

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")

		# rst_n
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 15 ns;")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '0';")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 10 ns;")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")

		# Start
		ready_if = If()
		ready_if._if_.conditions.add("ready = '1'")
		ready_if._if_.body.add("start <= '1';")
		ready_if._else_.body.add("start <= '0';")


		self.architecture.processes["start_gen"].final_wait = False
		self.architecture.processes["start_gen"].sensitivity_list.\
			add("clk")
		self.architecture.processes["start_gen"].if_list.add()
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk'event")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk = '1'", "and")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			body.add(ready_if)


		del self.architecture.processes["sample_ready_gen"]
		self.architecture.bodyCodeHeader.add("sample_ready <= '1';")

		del self.architecture.processes["ready_w_en_gen"]
		self.architecture.bodyCodeHeader.add("ready_w_en <= '0';")

		del self.architecture.processes["sample_w_en_gen"]
		self.architecture.bodyCodeHeader.add("sample_w_en <= '0';")

		del self.architecture.processes["out_spikes_w_en_gen"]
		self.architecture.bodyCodeHeader.add("out_spikes_w_en <= sample;")

		del self.architecture.processes["in_spikes_rd_en_gen"]
		self.architecture.bodyCodeHeader.add("in_spikes_rd_en <= sample;")

		if file_input and "in_spike" in input_signal_list:
			del self.architecture.processes["in_spikes_rd_en_gen"]
			self.architecture.bodyCodeHeader.add(
				"in_spike_rd_en <= ready;")
			self.architecture.bodyCodeHeader.add(
				"out_spike_w_en <= ready;")

class DummyAccelerator(VHDLblock):

	def __init__(self, config, debug = False, debug_list = []):

		self.name = "dummy_spiker"

		self.spiker_pkg = SpikerPackage()

		self.net = Network(
				n_cycles	= config["n_cycles"],
				debug		= debug,
				debug_list	= debug_list
		)

		self.layer_sizes = []

		if "layer_0" in config.keys():
			if "w_exc" in config["layer_0"].keys():
				self.input_size = config["layer_0"]["w_exc"].\
							shape[1]
		else:
			raise ValueError("Invalid config dictionary")

		for key in config:
			if "layer" in key:

				new_layer = Layer(
					label		= config[key]["label"],
					w_exc		= config[key]["w_exc"],
					w_inh		= config[key]["w_inh"],
					v_th		= config[key]["v_th"],
					v_reset		= config[key][
								"v_reset"],
					bitwidth	= config[key][
								"bitwidth"],
					fp_decimals	= config[key][
								"fp_decimals"],
					w_inh_bw	= config[key][
								"w_inh_bw"],
					w_exc_bw	= config[key][
								"w_exc_bw"],
					shift		= config[key]["shift"],
					reset		= config[key]["reset"],
					debug		= config[key]["debug"],
					debug_list 	= config[key][
								"debug_list"]
				)

				self.net.add(new_layer)

				self.layer_sizes.append(
					config[key]["w_exc"].shape[0]
				)

		self.in_addr_bw	= int(log2(ceil_pow2(self.input_size)))
		self.out_addr_bw = int(log2(ceil_pow2(self.layer_sizes[-1])))

		self.input_decoder = Decoder(
			bitwidth = self.in_addr_bw
		)

		self.output_mux = Mux(
			n_in		= 2**self.out_addr_bw,
			in_type		= "std_logic",
			bitwidth	= 1,
		)

		self.ff	= Reg(
			bitwidth	= 1,
			reg_type	= "std_logic",
		)

		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)

	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")

		for name in self.net.entity.generic:
			self.entity.generic.add(
				name		= name,
				gen_type	= self.net.entity.generic[
						name].gen_type,
				value		= self.net.entity.generic[
						name].value
			)	

		for name in self.net.entity.port:
			if name != "in_spikes" and name != \
			"out_spikes":

				self.entity.port.add(
					name		= name,
					direction	= self.net.entity.port[
							name].direction,
					port_type	= self.net.entity.port[
							name].port_type
				)	

		self.entity.port.add(
			name		= "in_spike",
			direction	= "in",
			port_type	= "std_logic"
		)	

		self.entity.port.add(
			name		= "in_spike_addr",
			direction	= "in",
			port_type	= "std_logic_vector(" + 
					str(self.in_addr_bw - 1) + " downto 0)"
		)	

		self.entity.port.add(
			name		= "out_spike",
			direction	= "out",
			port_type	= "std_logic"
		)	

		self.entity.port.add(
			name		= "out_spike_addr",
			direction	= "in",
			port_type	= "std_logic_vector(" + 
					str(self.out_addr_bw - 1) + " downto 0)"
		)	

		self.architecture.component.add(self.net)
		self.architecture.component.add(self.input_decoder)
		self.architecture.component.add(self.output_mux)
		self.architecture.component.add(self.ff)

		self.architecture.signal.add(
			name		= "en",
			signal_type	= "std_logic_vector(" +
					str(2**self.in_addr_bw-1)
					+ " downto 0)"
		)

		self.architecture.signal.add(
			name		= "in_spikes",
			signal_type	= "std_logic_vector(" +
					str(2**self.in_addr_bw-1)
					+ " downto 0)"
		)

		self.architecture.signal.add(
			name		= "out_spikes",
			signal_type	= "std_logic_vector(" +
					str(self.layer_sizes[-1]-1)
					+ " downto 0)"
		)

		ff_instance = Instance(self.ff, "spike_reg_i")
		ff_instance.port_map("key", **{
			"clk"		: "clk",
			"reg_in"	: "in_spike",
			"en"		: "en(i)",
			"reg_out"	: "in_spikes(i)"})

		spikes_sample = For(
			name		= "spikes",
			start		= 0,
			stop		= self.input_size-1,
			loop_type	= "generate"
		)

		spikes_sample.body.add(ff_instance)

		self.architecture.bodyCodeHeader.add(spikes_sample)

		self.architecture.instances.add(self.input_decoder,
				"input_decoder")
		self.architecture.instances["input_decoder"].generic_map("key",
			**{"bitwidth"	: str(self.in_addr_bw)})
		self.architecture.instances["input_decoder"].port_map("key", **{
			"encoded_in"	: "in_spike_addr",
			"decoded_out"	: "en"}
		)

		self.architecture.instances.add(self.output_mux,
				"output_mux")
		self.architecture.instances["output_mux"].port_map()

		self.architecture.instances["output_mux"].p_map.add(
			"mux_sel", "out_spike_addr"
		)

		for i in range(self.layer_sizes[-1]):
			self.architecture.instances["output_mux"].p_map.add(
				"in" + str(i), "out_spikes(" + str(i) + ")"
			)

		if self.layer_sizes[-1] < 2**self.out_addr_bw:
			for i in range(self.layer_sizes[-1],
			2**self.out_addr_bw):
				self.architecture.instances["output_mux"].p_map.add(
					"in" + str(i), "\'0\'"
				)


		self.architecture.instances["output_mux"].p_map.add(
			"mux_out", "out_spike"
		)

		self.architecture.instances.add(self.net,
				"snn")
		self.architecture.instances["snn"].generic_map()
		self.architecture.instances["snn"].port_map()

		if self.input_size < 2**self.in_addr_bw:
			self.architecture.instances["snn"].p_map.add(
				"in_spikes", "in_spikes(" +
				str(self.input_size-1)  + " downto 0)"
			)


class DummyAccelerator_tb(Testbench):

	def __init__(self, dummy_accelerator, clock_period = 20, file_output =
			False, output_dir = "output", file_input = False,
			input_dir = "", input_signal_list = [], debug = False,
			debug_list = []):


		self.spiker_pkg = SpikerPackage()

		self.dut = dummy_accelerator
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
		
		self.vhdl(
			clock_period		= clock_period,
			file_output		= file_output,
			output_dir		= output_dir,
			file_input		= file_input,
			input_dir		= input_dir,
			input_signal_list 	= input_signal_list
			)


	def vhdl(self, clock_period = 20, file_output = False, output_dir =
			"output", file_input = False, input_dir = "",
			input_signal_list = []):

		self.library.add("work")
		self.library["work"].package.add("spiker_pkg")

		# rst_n
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 15 ns;")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '0';")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"wait for 10 ns;")
		self.architecture.processes["rst_n_gen"].bodyHeader.add(
				"rst_n <= '1';")

		# Start
		ready_if = If()
		ready_if._if_.conditions.add("ready = '1'")
		ready_if._if_.body.add("start <= '1';")
		ready_if._else_.body.add("start <= '0';")

		self.architecture.processes["start_gen"].final_wait = False
		self.architecture.processes["start_gen"].sensitivity_list.\
			add("clk")
		self.architecture.processes["start_gen"].if_list.add()
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk'event")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			conditions.add("clk = '1'", "and")
		self.architecture.processes["start_gen"].if_list[0]._if_.\
			body.add(ready_if)

		if file_input and "in_spikes" in input_signal_list:
			del self.architecture.processes["in_spikes_rd_en_gen"]

			self.architecture.bodyCodeHeader.add(
				"in_spikes_rd_en <= sample;")
			self.architecture.bodyCodeHeader.add(
				"out_spikes_w_en <= sample;")

			self.architecture.processes["in_spikes_gen"].bodyHeader.add(
				"sample_ready <= '0';")

		del self.architecture.processes["sample_ready_gen"]
		self.architecture.bodyCodeHeader.add("sample_ready <= sample;")

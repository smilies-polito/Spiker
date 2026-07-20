import torch
import numpy as np
from copy import deepcopy
from math import log2
import logging

from .multi_cycle import MultiCycle
from .layer import Layer
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .decoder import Decoder
from .mux import Mux
from .reg import Reg
from .output_voter import OutputVoter
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, obj_types, is_iterable

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If, ConditionsList
from .vhdltools.text import SingleCodeLine
from .vhdltools.for_statement import For
from .vhdltools.instance import Instance

from .vhdl import write_file_all as write_vhdl
from .vhdl import fast_compile as compile_vhdl
from .vhdl import elaborate as elaborate_vhdl
from .vhdl import simulate as simulate_vhdl

class Network(VHDLblock, dict):

	def __init__(self, n_cycles = 10, learning = None, n_classes = None,
			debug = False, debug_list = []):

		self.layer_index 	= 0
		self.all_ready 		= ConditionsList()
		self.n_cycles		= n_cycles

		# Learning configuration. When non-None the network gains the
		# train_mode / target / update_every_n top-level interface and an
		# OutputVoter; the multi_cycle controller is built in learning
		# mode so it emits the update_weights_layer{0,1} strobes.
		self.learning = learning
		self.n_classes = n_classes  # Required when learning is set.

		self.name = "network"

		self.cycles_cnt_bitwidth = int(log2(ceil_pow2(
			self.n_cycles+1))) + 1


		self.spiker_pkg = SpikerPackage()

		self.multi_cycle = MultiCycle(
			n_cycles = self.n_cycles,
			learning = (self.learning is not None),
			debug = debug,
			debug_list = debug_list
		)

		# OutputVoter is built lazily once n_classes is known. For the
		# learning path the caller passes n_classes explicitly.
		if self.learning is not None:
			if self.n_classes is None:
				raise ValueError(
					"Network(learning=...) requires n_classes (the "
					"output-layer neuron count)")
			self.output_voter = OutputVoter(
				n_classes=self.n_classes,
				n_cycles=self.n_cycles,
			)
		else:
			self.output_voter = None

		self.components = sub_components(self)

		super().__init__(entity_name = self.name)
		self.vhdl(debug = debug, debug_list = debug_list)


	def vhdl(self, debug = False, debug_list = []):

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

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

		if self.learning is not None:
			self.entity.port.add(
				name		= "in_valid",
				direction	= "in",
				port_type	= "std_logic")
		else:
			self.entity.port.add(
				name 		= "sample_ready",
				direction	= "in",
				port_type	= "std_logic")

		# Output
		self.entity.port.add(
			name 		= "ready",
			direction	= "out",
			port_type	= "std_logic")

		if self.learning is not None:
			self.entity.port.add(
				name		= "timestep_start",
				direction	= "out",
				port_type	= "std_logic")
			self.entity.port.add(
				name		= "timestep_end",
				direction	= "out",
				port_type	= "std_logic")

		self.entity.port.add(
			name 		= "sample",
			direction	= "out",
			port_type	= "std_logic")

		# Spiker-LL learning-mode top-level ports. out_spikes (the voted
		# class) and in_spikes are added by add()/finalize() once the
		# layer widths are known.
		if self.learning is not None:
			self.entity.port.add(
				name="out_valid", direction="out",
				port_type="std_logic")
			self.entity.port.add(
				name="train_mode", direction="in",
				port_type="std_logic")
			self.entity.port.add(
				name="target", direction="in",
				port_type="std_logic_vector(" +
				str(self.n_classes-1) + " downto 0)")
			self.entity.port.add(
				name="update_every_n", direction="in",
				port_type="std_logic_vector("
				"cycles_cnt_bitwidth-1 downto 0)")


		# Components
		self.architecture.component.add(self.multi_cycle)
		if self.learning is not None:
			self.architecture.component.add(self.output_voter)

			# Update strobes: the multi_cycle emits the raw _s
			# strobes; the network gates them with train_mode
			# (mirrors the hand-coded reference's hierarchy).
			self.architecture.declarationHeader.add(
				"signal update_weights_layer0, "
				"update_weights_layer0_s : std_logic;")
			self.architecture.declarationHeader.add(
				"signal update_weights_layer1, "
				"update_weights_layer1_s : std_logic;")
			self.architecture.signal.add(
				name="output_voter_en", signal_type="std_logic")
			self.architecture.signal.add(
				name="output_voter_vote",
				signal_type="std_logic")
			self.architecture.signal.add(
				name="vote_valid_s", signal_type="std_logic")

		self.architecture.signal.add(
			name		= "start_all",
			signal_type	= "std_logic"
		)

		self.architecture.signal.add(
			name		= ("layers_ready"
					if self.learning is not None
					else "all_ready"),
			signal_type	= "std_logic"
		)

		self.architecture.signal.add(
			name		= "restart",
			signal_type	= "std_logic"
		)

		if self.learning is not None:
			# ``sample`` is driven by the hidden layer's
			# input_sample echo (wired in add()); ``timestep_start``
			# carries what the legacy design calls sample.
			self.architecture.bodyCodeHeader.add(
				"timestep_start <= start_all;")
		else:
			self.architecture.bodyCodeHeader.add(
				"sample <= start_all;"
			)

		# The ready line must sit at bodyCodeHeader[1]: add() rewrites
		# it by index as layers are added.
		if self.learning is not None:
			self.architecture.bodyCodeHeader.add("layers_ready <= "
					+ self.all_ready.code() + ";\n")
		else:
			self.all_ready.add("sample_ready")
			self.architecture.bodyCodeHeader.add("all_ready <= " +
					self.all_ready.code() + ";\n")

		# Multi-input control
		self.architecture.instances.add(self.multi_cycle,
				"multi_cycle_control")
		self.architecture.instances["multi_cycle_control"].generic_map()
		self.architecture.instances["multi_cycle_control"].port_map()

		if self.learning is not None:
			# Forward the learning-mode signals to the multi_cycle.
			# layers_ready / timestep_end / output_voter_* /
			# update_every_n are matched by name via port_map().
			self.architecture.instances["multi_cycle_control"].p_map.add(
				"input_ready", "in_valid")
			self.architecture.instances["multi_cycle_control"].p_map.add(
				"vote_valid", "vote_valid_s")
			self.architecture.instances["multi_cycle_control"].p_map.add(
				"update_weights_layer0",
				"update_weights_layer0_s")
			self.architecture.instances["multi_cycle_control"].p_map.add(
				"update_weights_layer1",
				"update_weights_layer1_s")

			self.architecture.bodyCodeHeader.add(
				"update_weights_layer0 <= "
				"update_weights_layer0_s and train_mode;")
			self.architecture.bodyCodeHeader.add(
				"update_weights_layer1 <= "
				"update_weights_layer1_s and train_mode;")

		
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

		# Recurrent (self-)inhibition. The legacy/inference path feeds a
		# layer's own output spikes back as inh_spikes (real lateral
		# inhibition, matching mnist/original/network.vhd). Every
		# hand-crafted Spiker-LL reference (mnist/rtl, mnist_8bit,
		# digits/rtl, fmnist/rtl, mnist_reduced/rtl) instead ties
		# inh_spikes to a constant zero for trainable layers -- recurrent
		# inhibition was deliberately dropped for the on-chip-learning
		# design. Reproduce that here so trainable layers match; leave
		# inference-only layers on the original self-feedback wiring.
		if getattr(layer, "trainable", False):
			zero_sig = "zero" + str(layer.n_neurons)
			self.architecture.signal.add(
				zero_sig,
				"std_logic_vector(" + str(layer.n_neurons-1) +
				" downto 0)",
				"(others => '0')")
			self.architecture.instances[current_layer].p_map.add(
				"inh_spikes", zero_sig)
		else:
			self.architecture.instances[current_layer].p_map.add(
				"inh_spikes", current_layer + "_feedback")

		# Spiker-LL learning-mode wiring: route update strobe and target
		# into the trainable layer. The hidden layer also needs the
		# output layer's spikes (pred_spikes); since layers are added in
		# order layer_0, layer_1 we deferred the pred_spikes binding to
		# finalize() once layer_1's feedback signal exists.
		if getattr(layer, "trainable", False):
			update_sig = "update_weights_layer" + str(self.layer_index)
			self.architecture.instances[current_layer].p_map.add(
				"update_weights", update_sig)
			self.architecture.instances[current_layer].p_map.add(
				"target", "target")
			if getattr(layer, "role", None) == "hidden":
				# The hidden layer's input_sample echo drives
				# the top-level sample port (mirrors the
				# hand-coded reference).
				self.architecture.instances[current_layer].\
					p_map.add("input_sample", "sample")


		if self.layer_index == 0:

			self.entity.port.add(
				name		= "in_spikes",
				direction	= "in",
				port_type	= "std_logic_vector(" +
				str(layer.n_exc_inputs-1)  + " downto 0)"
			)

			self.architecture.instances[current_layer].p_map.add(
					"exc_spikes", "in_spikes")

			if self.learning is None:
				self.entity.port.add(
					name		= "out_spikes",
					direction	= "out",
					port_type	= "std_logic_vector(" +
					str(layer.n_neurons-1)  + " downto 0)"
				)
				self.architecture.bodyCodeHeader.add(
					"out_spikes <= ",
					current_layer + "_feedback;\n")

		else:

			previous_layer = "layer_" + str(self.layer_index - 1)

			if layer.n_exc_inputs != self[previous_layer].n_neurons:
				raise ValueError("Layer cannot be added to the"
						" network. Incompatile number"
						" of inputs")

			layer_ports = deepcopy(layer.entity.port).items()

			# Add layer's output signals 
			for _, port in layer_ports:

				if port.direction is "out" and port.name is not "ready":

					for _, generic in layer.entity.generic.items():

						if generic.name in port.port_type:

							port.port_type = port.port_type.replace(
									generic.name,
									generic.value
							)

					self.entity.port.add(
						name		= port.name,
						direction	= port.direction,
						port_type	= port.port_type
					)



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
			if self.learning is None:
				self.architecture.bodyCodeHeader[2] = \
					SingleCodeLine("out_spikes <= ",
					current_layer + "_feedback;\n")

			self.architecture.bodyCodeHeader.add(
				exc_spikes_internal + "<= ", previous_layer +
				"_feedback;\n")


		if self.learning is None:
			self.all_ready.add(layer_ready, "and")
			self.architecture.bodyCodeHeader[1] = SingleCodeLine(
					"all_ready <= " + self.all_ready.code()
					+ ";\n")
		else:
			# Learning designs feed the input-stream valid separately
			# into the CU (in_valid); only the layers' ready signals
			# are ANDed here.
			if self.layer_index == 0:
				self.all_ready.add(layer_ready)
			else:
				self.all_ready.add(layer_ready, "and")
			self.architecture.bodyCodeHeader[1] = SingleCodeLine(
					"layers_ready <= " +
					self.all_ready.code() + ";\n")

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

	def finalize(self):
		"""Resolve cross-layer wiring that depends on all layers existing.

		Specifically: in learning mode, the hidden layer (layer_0) needs
		``pred_spikes`` connected to the output layer (layer_1) feedback,
		and the OutputVoter must be wired between layer_1's spikes and
		the top-level voted_class / out_valid ports.

		Idempotent: calling more than once is harmless.
		"""
		if self.learning is None:
			return
		if self.layer_index < 2:
			raise ValueError(
				"Network.finalize() called before both layers were added; "
				f"got layer_index={self.layer_index}, expected 2.")

		# Hidden layer needs the *previous* timestep's output spikes; the
		# multi_cycle update strobe is delayed one clock for layer 1, so
		# wiring directly to layer_1_feedback is the correct alignment.
		if getattr(self["layer_0"], "trainable", False):
			self.architecture.instances["layer_0"].p_map.add(
				"pred_spikes", "layer_1_feedback")

		# OutputVoter wiring, mirroring the hand-coded reference: the
		# CU enables the per-class counters on each timestep
		# (output_voter_en), requests the vote after the last timestep
		# (output_voter_vote), and the voted one-hot class drives the
		# top-level out_spikes.
		self.architecture.instances.add(self.output_voter, "voter")
		self.architecture.instances["voter"].generic_map(mode="self")
		self.architecture.instances["voter"].port_map()
		self.architecture.instances["voter"].p_map.add(
			"clk", "clk")
		self.architecture.instances["voter"].p_map.add(
			"rst_n", "rst_n")
		self.architecture.instances["voter"].p_map.add(
			"count_en", "output_voter_en")
		self.architecture.instances["voter"].p_map.add(
			"count_rst", "restart")
		self.architecture.instances["voter"].p_map.add(
			"spikes_in", "layer_1_feedback")
		self.architecture.instances["voter"].p_map.add(
			"vote", "output_voter_vote")
		self.architecture.instances["voter"].p_map.add(
			"voted_class", "out_spikes")
		self.architecture.instances["voter"].p_map.add(
			"vote_valid", "vote_valid_s")

		self.architecture.bodyCodeFooter.add(
			"out_valid <= vote_valid_s;")

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
		del self.architecture.processes["ready_save"]
		del self.architecture.signal["ready_w_en"]

		del self.architecture.processes["sample_w_en_gen"]
		del self.architecture.processes["sample_save"]
		del self.architecture.signal["sample_w_en"]

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

class FullAccelerator(VHDLblock):

	def __init__(self, net, input_size, output_size, debug = False,
			debug_list = []):

		self.name = "full_accelerator"

		self.spiker_pkg = SpikerPackage()

		self.net = net
		self.input_size = input_size
		self.output_size = output_size

		self.in_addr_bw	= int(log2(ceil_pow2(self.input_size)))
		self.out_addr_bw = int(log2(ceil_pow2(self.output_size)))

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
		self.library["ieee"].package.add("numeric_std")

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
					str(self.output_size-1)
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

		if self.output_size > 2:
			self.architecture.instances["output_mux"].p_map.add(
				"mux_sel", "out_spike_addr"
			)

		elif self.output_size <= 2:
			self.architecture.instances["output_mux"].p_map.add(
				"mux_sel", "out_spike_addr(0)"
			)

		for i in range(self.output_size):
			self.architecture.instances["output_mux"].p_map.add(
				"in" + str(i), "out_spikes(" + str(i) + ")"
			)

		if self.output_size < 2**self.out_addr_bw:
			for i in range(self.output_size,
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


class FullAccelerator_tb(Testbench):

	def __init__(self, full_accelerator, clock_period = 20, file_output =
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



class NetworkSimulator:

	def __init__(self, vhdl_net, clock_period = 20, output_dir = "output",
			readout_type = "mem_avg"): 

		self.supported_readouts = [
			"mem_softmax",
			"mem_max",
			"mem_avg",
			"spk_count"
		]

		if readout_type in self.supported_readouts:
			self.readout_type	= readout_type

		else:
			raise ValueError("Invalid readout type. Choose between " +
					str(self.supported_readouts) + "\n")

		self.testbench = Network_tb(vhdl_net,
			clock_period		= clock_period,
			output_dir			= output_dir,
			file_output			= True,
			file_input			= True,
			input_signal_list	= ["in_spikes"]
		)

		self.output_dir = output_dir
		self.stimuli_file = self.output_dir + "/in_spikes.txt"
		self.readout_file = self.output_dir + "/neuron_dp_none_v.txt"

		if "mem" in readout_type:

			del self.testbench.architecture.processes[
					"neuron_dp_none_v_w_en_gen"
			]

			self.testbench.architecture.bodyCodeHeader.add(
					"neuron_dp_none_v_w_en <= sample;"
			)

		write_vhdl(self.testbench)
		compile_vhdl(self.testbench)
		elaborate_vhdl(self.testbench)


	def simulate(self, dataloader, sim_duration = "10000ns", print_interval = 10):

		torch.set_printoptions(threshold = np.inf)

		acc = 0
		iter_count = 0

		logging.info("Simulating VHDL network")

		# Iterate over the dataloader
		for batch_idx, (data_batch, labels_batch) in enumerate(dataloader):

			for i in range(data_batch.shape[0]):
			
				spike_trains = data_batch[i, :, :].to(int)
				label = labels_batch[i].item()

				classified = self.inference(spike_trains, sim_duration)

				log_message = "Expected: " + str(label)
				log_message = log_message + ". Classified: " + str(classified)
				logging.info(log_message)

				acc += (classified == label)

				if iter_count == (print_interval - 1):

					acc = acc / (iter_count+1) * 100

					log_message = "Accuracy: " + "{:.2f}".format(acc) + "%\n"
					logging.info(log_message)

					acc = 0

				iter_count = (iter_count + 1) % print_interval


	def inference(self, spike_trains, sim_duration):

			self.dump(spike_trains, self.stimuli_file)

			simulate_vhdl(self.testbench, output_dir = self.output_dir,
					sim_duration = sim_duration)

			mem_out = self.load(self.readout_file)

			_, classified = mem_out.mean(dim=0).max(dim=0)

			return classified.item()


	def dump(self, spike_trains, filename):

		if spike_trains.shape[0] != self.testbench.dut.n_cycles:

			log_message = "Number of input timestes differ network's one. "
			log_message += "Expected "
			log_message += str(self.testbench.dut.n_cycles)
			log_message += " but found "
			log_message += str(spike_trains.shape[0])

			logging.warning(log_message)

		with open(filename, "w") as file:

			for timestep in spike_trains:
				file.write("".join(map(str, 
				torch.flip(timestep, dims = (0,)).tolist())) + "\n")


	def load(self, filename):

		last_layer_idx = self.testbench.dut.layer_index - 1
		last_layer_key = "layer_" + str(last_layer_idx)

		bitwidth = self.testbench.dut[last_layer_key].bitwidth

		mem_out = []

		with open(filename, "r") as file:

			for line in file:

				line = line[:-1]

				if set(line) - {'0', '1'}:
					raise ValueError("String must be binary")

				mem_out_t = []

				for i in range(0, len(line), bitwidth):

					mem_binary = line[i : i + bitwidth]

					mem = self.ca2_to_signed(mem_binary, bitwidth)

					mem_out_t.append(mem)

					i += bitwidth

				mem_out.append(mem_out_t)

		mem_out = torch.tensor(mem_out)
		mem_out = mem_out.flip(dims=(1,))

		if mem_out.shape[0] != self.testbench.dut.n_cycles:

			log_message = "Number of output timesteps differs from network's"
			log_message += "one. Expected "
			log_message += str(self.testbench.dut.n_cycles)
			log_message += " but found "
			log_message += str(mem_out.shape[0])

			logging.warning(log_message)

		if mem_out.shape[1] != self.testbench.dut[last_layer_key].n_neurons:

			log_message = "Number of neurons differs from the network's one. "
			log_message += "Expected "
			log_message += str(self.testbench.dut[last_layer_key].n_neurons)
			log_message += " but found "
			log_message += str(mem_out.shape[1])

			logging.warning(log_message)

		return mem_out.to(float)


	def ca2_to_signed(self, binary_string, bitwidth):

		ca2_val = int(binary_string, 2)

		if binary_string[0] == '1':

			ca2_val = ca2_val - (1 << bitwidth) 

		return ca2_val

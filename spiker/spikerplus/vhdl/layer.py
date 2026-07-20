import numpy as np

from math import log2

from ..quantizer import FP_DEC

from .multi_input import MultiInput
from .lif_neuron import LIFneuron
from .rom import Rom
from .addr_converter import AddrConverter
from .barrier import Barrier
from .testbench import Testbench
from .spiker_pkg import SpikerPackage
from .hidden_layer_trainer import HiddenLayerTrainer
from .output_layer_trainer import OutputLayerTrainer
from .vhdl import track_signals, debug_component, sub_components, write_file_all
from .utils import ceil_pow2, random_binary, int_to_hex, int_to_bin, \
	fixed_point_array

from .vhdltools.vhdl_block import VHDLblock
from .vhdltools.if_statement import If


class Layer(VHDLblock):

	def __init__(self, label = "", w_exc = np.array([[0.2,
		0.3]]), w_inh = np.array([[-0.1, -0.2]]), v_th = np.array([8]),
		v_reset = np.array([2]), bitwidth = 16, fp_decimals = 0,
		w_inh_bw = 5, w_exc_bw = 5, shift = 10, reset = "fixed",
		functional = False,
		# --- Spiker-LL learning-mode additions -----------------------------
		# When ``trainable`` is True the layer instantiates an on-chip
		# trainer module and replaces its excitatory weight ROM with a
		# dual-port RAM. ``role`` picks between the two trainer flavours:
		#   "hidden" -> HiddenLayerTrainer (uses sel_idx/neuron_constants)
		#   "output" -> OutputLayerTrainer (uses output_lr/output_loss)
		# These extra kwargs are ignored when ``trainable`` is False, so
		# existing inference-only Layer call sites stay unchanged.
		trainable = False, role = None, sel_idx = None,
		neuron_constants = None, n_output_neurons = None,
		output_lr = 0.01, output_loss_value = 0.2,
		output_const_value = None,
		# hw_fp_dec: the fixed-point decimal bits used during training.
		# Passed to OutputLayerTrainer so its CONST is computed with the
		# correct scale. None falls back to the module default (FP_DEC=8).
		hw_fp_dec = None,
		# -------------------------------------------------------------------
		debug = False, debug_list = []):

		if reset == "none" and functional is True:
			debug = True
			debug_list.append("neuron_dp_none_v")

		# Learning-mode state. Validate aggressively here so a bad config
		# fails at construction, not deep inside the VHDL emission code.
		self.trainable = trainable
		self.role = role
		if trainable:
			if role not in ("hidden", "output"):
				raise ValueError(
					"Layer(trainable=True) requires role='hidden' or 'output'; "
					f"got {role!r}")
			if role == "hidden":
				if sel_idx is None or neuron_constants is None:
					raise ValueError(
						"Hidden-layer trainer needs sel_idx and "
						"neuron_constants from the STSFTrainer")
				if n_output_neurons is None:
					raise ValueError(
						"Hidden-layer trainer needs n_output_neurons "
						"(the next layer's neuron count)")

		self.n_neurons		= w_exc.shape[0]
		self.n_exc_inputs 	= w_exc.shape[1]
		self.n_inh_inputs 	= w_inh.shape[1]

		self.name = "layer_" + str(self.n_neurons) + "_neurons_" + \
			str(self.n_exc_inputs) + "_inputs"

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(self.n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(self.n_inh_inputs)))

		self.bitwidth 		= bitwidth
		self.fp_decimals	= fp_decimals
		self.w_exc_bw		= w_exc_bw
		self.w_inh_bw		= w_inh_bw
		self.shift		= shift

		self.w_exc		= w_exc
		self.w_inh		= w_inh
		self.v_th		= fixed_point_array(v_th, bitwidth,
					fp_decimals)

		self.functional		= functional

		self.spiker_pkg = SpikerPackage()

		self.multi_input = MultiInput(
			n_exc_inputs 	= self.n_exc_inputs,
			n_inh_inputs 	= self.n_inh_inputs,
			trainable	= self.trainable,
			debug 		= debug,
			debug_list 	= debug_list
		)

		self.lif_neuron = LIFneuron(
			bitwidth	= bitwidth,
			w_inh_bw	= w_inh_bw,
			w_exc_bw	= w_exc_bw,
			shift 		= shift,
			reset		= reset,
			debug 		= debug,
			debug_list 	= debug_list
		)

		# Trainable layers use a dual-port RAM so the on-chip trainer can
		# write back updated weights; inference-only layers keep the
		# original read-only ROM.
		self.exc_mem = Rom(
			init_array 	= w_exc,
			bitwidth 	= w_exc_bw,
			fp_decimals	= fp_decimals,
			name_term 	= "_exc" + label,
			functional	= self.functional,
			writable	= self.trainable,
		)

		self.inh_mem = Rom(
			init_array 	= w_inh,
			bitwidth 	= w_inh_bw,
			fp_decimals	= fp_decimals,
			name_term 	= "_inh" + label,
			functional	= self.functional
		)

		# Instantiate the appropriate on-chip trainer (None when inference).
		if self.trainable and self.role == "hidden":
			self.trainer = HiddenLayerTrainer(
				n_hidden=self.n_neurons,
				n_output=n_output_neurons,
				neuron_bw=bitwidth,
				sel_idx=sel_idx,
				neuron_constants=neuron_constants,
			)
			self.n_output_neurons = n_output_neurons
		elif self.trainable and self.role == "output":
			self.trainer = OutputLayerTrainer(
				n_neurons=self.n_neurons,
				neuron_bw=bitwidth,
				lr=output_lr,
				loss_value=output_loss_value,
				fp_dec=(hw_fp_dec if hw_fp_dec is not None else FP_DEC),
				bw=bitwidth,
				const_value=output_const_value,
			)
			self.n_output_neurons = self.n_neurons
		else:
			self.trainer = None
			self.n_output_neurons = None

		self.addr_converter = AddrConverter(
			bitwidth	= self.exc_cnt_bitwidth
		)

		self.barrier = Barrier(
			bitwidth	= self.n_neurons,
			learning	= self.trainable
		)

		if self.lif_neuron.reset == "fixed":
			self.v_reset		= fixed_point_array(v_reset, 
						bitwidth, fp_decimals)


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
			name		= "n_exc_inputs", 
			gen_type	= "integer",
			value		= str(self.n_exc_inputs))
		self.entity.generic.add(
			name		= "n_inh_inputs", 
			gen_type	= "integer",
			value		= str(self.n_inh_inputs))
		self.entity.generic.add(
			name		= "exc_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.exc_cnt_bitwidth))
		self.entity.generic.add(
			name		= "inh_cnt_bitwidth", 
			gen_type	= "integer",
			value		= str(self.inh_cnt_bitwidth))
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
			name 		= "restart", 
			direction	= "in",
			port_type	= "std_logic")

		# Input spikes
		self.entity.port.add(
			name 		= "exc_spikes", 
			direction	= "in",
			port_type	= "std_logic_vector(n_exc_inputs-1 " 
						"downto 0)")
		self.entity.port.add(
			name 		= "inh_spikes", 
			direction	= "in",
			port_type	= "std_logic_vector(n_inh_inputs-1 " 
						"downto 0)")

		# Output
		self.entity.port.add(
			name 		= "ready", 
			direction	= "out",
			port_type	= "std_logic")

		self.entity.port.add(
			name 		= "out_spikes",
			direction	= "out",
			port_type	= "std_logic_vector(" +
					str(self.n_neurons-1) + " downto 0)")

		# Spiker-LL learning-mode top-level ports. Added here so they end
		# up next to the inference ports in the emitted entity, keeping
		# the interface readable.
		if self.trainable:
			self.entity.port.add(
				name="update_weights", direction="in",
				port_type="std_logic")
			# ``target`` and ``pred_spikes`` are needed by the hidden
			# trainer's error calculation; the output trainer reuses
			# ``target`` for its own target vector.
			if self.role == "hidden":
				self.entity.port.add(
					name="target", direction="in",
					port_type="std_logic_vector(" +
					str(self.n_output_neurons-1) + " downto 0)")
				self.entity.port.add(
					name="pred_spikes", direction="in",
					port_type="std_logic_vector(" +
					str(self.n_output_neurons-1) + " downto 0)")
				# Echo of the barrier's sampling strobe, exported
				# to the network top level (mirrors the reference).
				self.entity.port.add(
					name="input_sample", direction="out",
					port_type="std_logic")
			else:  # role == "output"
				self.entity.port.add(
					name="target", direction="in",
					port_type="std_logic_vector(" +
					str(self.n_neurons-1) + " downto 0)")

		hex_width = int(log2(ceil_pow2(self.n_neurons)) // 4)
		if hex_width == 0:
			hex_width = 1

		# Input parameters
		for i in range(self.n_neurons):
			self.architecture.constant.add(
				name 		= "v_th_" + int_to_hex(i,
						hex_width),
				const_type	= "signed(neuron_bit_width-1 "
						"downto 0)",
				value		= "\"" + int_to_bin(
						self.v_th[i],self.bitwidth) +
						"\""
			)

			if self.lif_neuron.reset == "fixed":
				self.architecture.constant.add(
					name 		= "v_reset_" + 
							int_to_hex(i,
							hex_width),
					const_type	= "signed("
							"neuron_bit_width-1 "
							"downto 0)",
					value		= "\"" + int_to_bin(
							self.v_reset[i],
							self.bitwidth) + "\""
				)

		# Signals
		self.architecture.signal.add(
			name 		= "start_neurons", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "neurons_restart", 
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "neurons_ready",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "exc",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "inh",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "exc",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "inh",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "exc_spike",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "inh_spike",
			signal_type	= "std_logic")

		self.architecture.signal.add(
			name 		= "exc_cnt", 
			signal_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)")
		self.architecture.signal.add(
			name 		= "inh_cnt", 
			signal_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)")

		self.architecture.signal.add(
			name		= "exc_addr",
			signal_type	= "std_logic_vector("
						"exc_cnt_bitwidth - 1 "
						"downto 0)"
		)

		self.architecture.signal.add(
			name		= "inh_addr",
			signal_type	= "std_logic_vector("
						"inh_cnt_bitwidth - 1 "
						"downto 0)"
		)
		self.architecture.signal.add(
			name		= "neuron_restart",
			signal_type	= "std_logic"
		)


		self.architecture.signal.add(
			name		= "barrier_ready",
			signal_type	= "std_logic"
		)

		self.architecture.signal.add(
			name 		= "out_spikes_inst",
			signal_type	= "std_logic_vector(" +
					str(self.n_neurons-1) + " downto 0)")

		self.architecture.signal.add(
			name 		= "out_sample",
			signal_type	= "std_logic")

		# Trainable-mode internal signals — the concatenated weight bus
		# the trainer reads (weights_in) and the bus the RAM writes
		# (weights_out / din), plus the RAM write enable produced by the
		# multi_input block.
		if self.trainable:
			self.architecture.signal.add(
				name="trainer_weights_in",
				signal_type="std_logic_vector(" +
				str(self.n_neurons) +
				"*neuron_bit_width-1 downto 0)")
			self.architecture.signal.add(
				name="trainer_weights_out",
				signal_type="std_logic_vector(" +
				str(self.n_neurons) +
				"*neuron_bit_width-1 downto 0)")
			self.architecture.signal.add(
				name="ram_wea", signal_type="std_logic")
			# Registered (post-barrier) copy of the output spikes:
			# the trainers observe these, and the out_spikes port is
			# driven from it (an out port cannot be read back).
			self.architecture.signal.add(
				name="out_spikes_s",
				signal_type="std_logic_vector(" +
				str(self.n_neurons-1) + " downto 0)")




		for i in range(self.n_neurons):

			hex_index = int_to_hex(i, hex_width)

			self.architecture.signal.add(
				name 		= "neuron_ready_" + hex_index,
				signal_type	= "std_logic")


			if self.w_inh_bw < self.bitwidth:
				self.architecture.signal.add(
					name 		= "inh_weight_" +
							int_to_hex(i,
							hex_width),
					signal_type	= "std_logic_vector("
							"inh_weights_bit_width"
							"-1 downto 0)")
			elif self.w_inh_bw == self.bitwidth:
				self.architecture.signal.add(
					name 		= "inh_weight_" +
							int_to_hex(i,
							hex_width),
					signal_type	= "std_logic_vector("
							"neuron_bit_width-1 "
							"downto 0)")
			else:
				raise ValueError("Inhibitory weight bit-width "
					"cannot be larger than the neuron's "
					"one")
			
			if self.w_exc_bw < self.bitwidth:
				self.architecture.signal.add(
					name 		= "exc_weight_"  +
							int_to_hex(i,
							hex_width),
					signal_type	= "std_logic_vector("
							"exc_weights_bit_width"
							"-1 downto 0)")

			elif self.w_exc_bw == self.bitwidth:
				self.architecture.signal.add(
					name 		= "exc_weight_" +
							int_to_hex(i,
							hex_width),
					signal_type	= "std_logic_vector("
							"neuron_bit_width-1 "
							"downto 0)")

			else:
				raise ValueError("Excitatory weight bit-width "
					"cannot be larger than the neuron's "
					"one")

		# Components
		self.architecture.component.add(self.multi_input)
		self.architecture.component.add(self.lif_neuron)
		self.architecture.component.add(self.exc_mem)
		self.architecture.component.add(self.inh_mem)
		self.architecture.component.add(self.addr_converter)
		self.architecture.component.add(self.barrier)

		neurons_ready = "neurons_ready <= "

		for i in range(self.n_neurons):

			hex_index = int_to_hex(i, hex_width)

			neurons_ready += "neuron_ready_" + hex_index + \
				" and "

		neurons_ready += "barrier_ready;"

		self.architecture.bodyCodeHeader.add(neurons_ready)

		# Multi-input control
		self.architecture.instances.add(self.multi_input,
				"multi_input_control")
		self.architecture.instances["multi_input_control"].generic_map()
		self.architecture.instances["multi_input_control"].port_map()

		if self.trainable:
			# multi_input gains an update_weights input + ram_wea output
			# when trainable=True is propagated; wire them here.
			self.architecture.instances["multi_input_control"].p_map.add(
				"update_weights", "update_weights")
			self.architecture.instances["multi_input_control"].p_map.add(
				"ram_wea", "ram_wea")

		# LIF neuron
		for i in range(self.n_neurons):

			neuron_name = "neuron_" + int_to_hex(i, hex_width)
			exc_weight_name = "exc_weight_" + int_to_hex(i, 
						hex_width)
			inh_weight_name = "inh_weight_" + int_to_hex(i, 
						hex_width)
			v_th_name = "v_th_" + int_to_hex(i, hex_width)

			if self.lif_neuron.reset == "fixed":
				v_reset_name = "v_reset_" + int_to_hex(i, 
						hex_width)

			neuron_ready_name = "neuron_ready_" + \
					int_to_hex(i, hex_width)

			self.architecture.instances.add(self.lif_neuron,
					neuron_name)
			self.architecture.instances[neuron_name].generic_map()
			self.architecture.instances[neuron_name].port_map()
			self.architecture.instances[neuron_name].p_map.add(
				"exc_weight", "signed(" + exc_weight_name + ")")
			self.architecture.instances[neuron_name].p_map.add(
				"inh_weight", "signed(" + inh_weight_name + ")")
			self.architecture.instances[neuron_name].p_map.add(
				"v_th", v_th_name)

			if self.lif_neuron.reset == "fixed":
				self.architecture.instances[neuron_name].p_map.\
					add("v_reset", v_reset_name)

			elif self.lif_neuron.reset == "none" and self.functional is True:
				self.architecture.instances[neuron_name].p_map.\
					add("neuron_dp_none_v", 
						"neuron_dp_none_v(" + str(self.bitwidth * (i+1) - 1) +
						" downto " + str(self.bitwidth * i) + ")"
					)


			self.architecture.instances[neuron_name].p_map.add(
				"restart", "neuron_restart")
			self.architecture.instances[neuron_name].p_map.add(
				"neuron_ready", neuron_ready_name)
			self.architecture.instances[neuron_name].p_map.add(
				"out_spike", "out_spikes_inst(" + str(i) + ")")
			

		# Excitatory memory. When trainable=True the underlying entity is a
		# dual-port RAM (ports clk/raddr/wea/waddr/din) instead of a ROM
		# (ports clka/addra); pick the right port names per mode.
		self.architecture.instances.add(self.exc_mem,
				"exc_mem")
		self.architecture.instances["exc_mem"].generic_map()
		self.architecture.instances["exc_mem"].port_map()
		if self.trainable:
			self.architecture.instances["exc_mem"].p_map.add(
					"raddr", "exc_addr")
			self.architecture.instances["exc_mem"].p_map.add(
					"clk", "clk")
			# Write port — wea/waddr/din are driven by the trainer +
			# multi_input ram_wea strobe.
			self.architecture.instances["exc_mem"].p_map.add(
					"wea", "ram_wea")
			# NOTE: waddr must be the raw exc_cnt, not exc_addr. exc_addr is
			# exc_cnt+1 (addr_converter prefetches next cycle's read), so
			# the data on dout_* this cycle -- and therefore the trainer's
			# weights_in/weights_out -- corresponds to the *current* exc_cnt
			# column, not exc_addr. Writing to exc_addr would silently
			# update the wrong (next) column every cycle.
			self.architecture.instances["exc_mem"].p_map.add(
					"waddr", "exc_cnt")
			self.architecture.instances["exc_mem"].p_map.add(
					"din", "trainer_weights_out")
		else:
			self.architecture.instances["exc_mem"].p_map.add(
					"addra", "exc_addr")
			self.architecture.instances["exc_mem"].p_map.add(
					"clka", "clk")
		for i in range(self.n_neurons):
			exc_weight_name = "exc_weight_" + int_to_hex(i,
						hex_width)
			dout_name = "dout_" + int_to_hex(i, hex_width)

			self.architecture.instances["exc_mem"].p_map.add(
					dout_name, exc_weight_name)

		# Excitatory address converter
		self.architecture.instances.add(self.addr_converter,
				"exc_addr_conv")
		self.architecture.instances["exc_addr_conv"].generic_map()
		self.architecture.instances["exc_addr_conv"].g_map.add(
				"N", "exc_cnt_bitwidth")
		self.architecture.instances["exc_addr_conv"].port_map()
		self.architecture.instances["exc_addr_conv"].p_map.add(
				"addr_in", "exc_cnt")
		self.architecture.instances["exc_addr_conv"].p_map.add(
				"addr_out", "exc_addr")
	

		# Inhibitory memory
		self.architecture.instances.add(self.inh_mem,
				"inh_mem")
		self.architecture.instances["inh_mem"].generic_map()
		self.architecture.instances["inh_mem"].port_map()
		self.architecture.instances["inh_mem"].p_map.add(
				"addra", "inh_addr")
		self.architecture.instances["inh_mem"].p_map.add(
				"clka", "clk")
		for i in range(self.n_neurons):
			inh_weight_name = "inh_weight_" + int_to_hex(i, 
						hex_width)
			dout_name = "dout_" + int_to_hex(i, hex_width)

			self.architecture.instances["inh_mem"].p_map.add(
					dout_name, inh_weight_name)

		# Inhibitory address converter
		self.architecture.instances.add(self.addr_converter,
				"inh_addr_conv")
		self.architecture.instances["inh_addr_conv"].generic_map()
		self.architecture.instances["inh_addr_conv"].g_map.add(
				"N", "inh_cnt_bitwidth")
		self.architecture.instances["inh_addr_conv"].port_map()
		self.architecture.instances["inh_addr_conv"].p_map.add(
				"addr_in", "inh_cnt")
		self.architecture.instances["inh_addr_conv"].p_map.add(
				"addr_out", "inh_addr")

		# Barrier
		self.architecture.instances.add(self.barrier,
				"spikes_barrier")
		self.architecture.instances["spikes_barrier"].generic_map()
		self.architecture.instances["spikes_barrier"].g_map.add(
				"N", str(self.n_neurons))
		self.architecture.instances["spikes_barrier"].port_map()
		self.architecture.instances["spikes_barrier"].p_map.add(
				"reg_in", "out_spikes_inst")
		self.architecture.instances["spikes_barrier"].p_map.add(
				"reg_out",
				"out_spikes_s" if self.trainable
				else "out_spikes")
		self.architecture.instances["spikes_barrier"].p_map.add(
				"ready", "barrier_ready")
		if self.trainable:
			self.architecture.bodyCodeHeader.add(
				"out_spikes <= out_spikes_s;")
			if self.role == "hidden":
				self.architecture.instances[
					"spikes_barrier"].p_map.add(
					"input_sample", "input_sample")
			else:
				# The output layer has no input_sample port; the
				# barrier's echo is left open, as in the
				# hand-coded reference.
				del(self.architecture.instances[
					"spikes_barrier"].p_map["input_sample"])

		# Trainer (only present in learning mode).
		if self.trainable:
			self.architecture.component.add(self.trainer)

			# Build the concatenated weights_in bus from the individual
			# per-neuron exc_weight_<hex> signals. The MSB end of the
			# concatenation is neuron N-1, matching the trainer's slicing
			# convention: weights_in((i+1)*BW-1 downto i*BW) = neuron i.
			weight_terms = []
			for i in range(self.n_neurons - 1, -1, -1):
				weight_terms.append(
					"exc_weight_" + int_to_hex(i, hex_width))
			self.architecture.bodyCodeHeader.add(
				"trainer_weights_in <= " + " & ".join(weight_terms) + ";"
			)

			self.architecture.instances.add(self.trainer, "trainer")
			self.architecture.instances["trainer"].generic_map(mode="self")
			# The trainer's bit width tracks the layer's generic, not
			# a baked literal (mirrors the reference).
			self.architecture.instances["trainer"].g_map.add(
				"neuron_bit_width", "neuron_bit_width")
			self.architecture.instances["trainer"].port_map()
			self.architecture.instances["trainer"].p_map.add(
				"update_weights", "update_weights")
			self.architecture.instances["trainer"].p_map.add(
				"weights_in", "trainer_weights_in")
			self.architecture.instances["trainer"].p_map.add(
				"weights_out", "trainer_weights_out")
			self.architecture.instances["trainer"].p_map.add(
				"target", "target")

			if self.role == "hidden":
				# Hidden trainer:
				#   x_pre      <- most recent excitatory input spike
				#   x_post     <- this layer's registered spikes (out_spikes_s)
				#   out_spikes <- next layer's feedback (pred_spikes input)
				self.architecture.instances["trainer"].p_map.add(
					"x_pre", "exc_spike")
				self.architecture.instances["trainer"].p_map.add(
					"x_post", "out_spikes_s")
				self.architecture.instances["trainer"].p_map.add(
					"out_spikes", "pred_spikes")
			else:  # role == "output"
				# Output trainer:
				#   clk, rst_n drive the (currently unused) clocked ports
				#   in_spike   <- last hidden-layer spike (this layer's exc input)
				#   out_spikes <- this layer's registered spikes (out_spikes_s)
				self.architecture.instances["trainer"].p_map.add(
					"clk", "clk")
				self.architecture.instances["trainer"].p_map.add(
					"rst_n", "rst_n")
				self.architecture.instances["trainer"].p_map.add(
					"in_spike", "exc_spike")
				self.architecture.instances["trainer"].p_map.add(
					"out_spikes", "out_spikes_s")

		# Debug
		if debug:
			debug_component(self, debug_list)


		if self.lif_neuron.reset == "none" and self.functional is True:
			self.entity.port.add(
				name 		= "neuron_dp_none_v", 
				direction	= "out", 
				port_type	= "signed(" + \
					str(self.bitwidth*self.n_neurons - 1) + " downto 0)"
			)


	def write_file_all(self, output_dir = "output", rm = False):
		write_file_all(self, output_dir = output_dir, rm = rm)


class Layer_tb(Testbench):

	def __init__(self, clock_period = 20, file_output = False,
			output_dir = "output", file_input = False, 
			input_dir = "", input_signal_list = [], 
			w_exc = np.array([[3, 4]]), w_inh =
			np.array([[-0.5, -0.1]]), v_th = np.array([8]), v_reset
			= np.array([2]), bitwidth = 16, fp_decimals = 0,
			w_inh_bw = 5, w_exc_bw = 5, shift = 10, reset = "fixed", 
			debug = False, debug_list = []):


		self.n_neurons		= w_exc.shape[0]
		self.n_exc_inputs 	= w_exc.shape[1]
		self.n_inh_inputs 	= w_inh.shape[1]

		self.name = "layer_" + str(self.n_neurons)

		self.exc_cnt_bitwidth = int(log2(ceil_pow2(self.n_exc_inputs)))
		self.inh_cnt_bitwidth = int(log2(ceil_pow2(self.n_inh_inputs)))

		self.bitwidth 		= bitwidth
		self.fp_decimals	= fp_decimals
		self.w_exc_bw		= w_exc_bw
		self.w_inh_bw		= w_inh_bw
		self.shift		= shift

		self.w_exc		= w_exc
		self.w_inh		= w_inh
		self.v_th		= fixed_point_array(v_th, bitwidth,
					fp_decimals)


		self.spiker_pkg = SpikerPackage()


		self.dut = Layer(
			w_exc 		= w_exc,
			w_inh 		= w_inh,
			v_th 		= v_th,
			v_reset 	= v_reset,
			bitwidth 	= bitwidth,
			fp_decimals 	= fp_decimals,
			w_inh_bw 	= w_inh_bw,
			w_exc_bw 	= w_exc_bw,
			shift 		= shift,
			reset		= reset,
			debug 		= debug,
			debug_list 	= debug_list
		)

		if self.dut.lif_neuron.reset == "fixed":
			self.v_reset		= fixed_point_array(v_reset,
						bitwidth, fp_decimals)

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

		# Restart
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for 10000 ns;")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '1';")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"wait for " + str(clock_period) + " ns;")
		self.architecture.processes["restart_gen"].bodyHeader.add(
				"restart <= '0';")


		if file_input and "exc_spikes" in input_signal_list:
			del self.architecture.processes["exc_spikes_rd_en_gen"]
			self.architecture.bodyCodeHeader.add(
				"exc_spikes_rd_en <= ready;")

		if file_input and "inh_spikes" in input_signal_list:
			del self.architecture.processes["inh_spikes_rd_en_gen"]
			self.architecture.bodyCodeHeader.add(
				"inh_spikes_rd_en <= ready;")

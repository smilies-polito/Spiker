"""
VHDLblock for the Spiker-LL ``output_layer_trainer`` entity.

This is the simpler of the two trainers — a delta-rule update with a single,
shared learning constant. On every update strobe:

  * if ``in_spike`` is 0 → no update (no pre-synaptic activity)
  * if target=1 and out=0 (false negative) → weight += CONST
  * if target=0 and out=1 (false positive) → weight -= CONST
  * otherwise                                → no update

The constant CONST is computed at generation time from the training
hyperparameters (``lr * loss_value`` passed through the same fixed-point
quantizer the hardware uses), so changing ``lr`` automatically regenerates a
matching VHDL constant.
"""

from ..quantizer import BW, FP_DEC, fixed_point

from .vhdl import sub_components
from .vhdltools.vhdl_block import VHDLblock


class OutputLayerTrainer(VHDLblock):
	"""Spiker-LL on-chip trainer for the output layer."""

	def __init__(self, n_neurons, neuron_bw, lr=0.01, loss_value=0.2,
			fp_dec=FP_DEC, bw=BW,
			const_value=None, debug=False):
		# Allow explicit override of the CONST value (see note below).

		self.name = "output_layer_trainer"
		self.n_neurons = n_neurons
		self.neuron_bw = neuron_bw

		# Derive CONST from the training hyperparameters. The reference fork
		# hard-codes 5 because ``lr * loss_value * 2^FP_DEC`` with
		# (lr=0.01, loss_value=0.2, FP_DEC=8) rounds to 5.
		if const_value is not None:
			const_value = int(const_value)
		else:
			const_value = int(fixed_point(lr * loss_value, fp_dec, bw))
		# When lr * loss_value is too small to survive fixed-point
		# quantisation (e.g. 0.01 * 0.2 * 2^8 = 0.512 → 0), clamp to 1
		# so the trainer always has a non-trivial effect.  The caller can
		# pass an explicit ``const_value`` kwarg to override this
		# entirely (e.g. to reproduce the hand-tuned CONST=5 in the
		# reference RTL for the MNIST configuration).
		if const_value <= 0:
			import warnings
			warnings.warn(
				f"Output trainer CONST computed as {const_value} from "
				f"lr={lr}, loss_value={loss_value}; clamping to 1. "
				"Pass output_const_value=N to the learning dict to "
				"override explicitly.",
				stacklevel=2,
			)
			const_value = 1
		self.const_value = const_value
		self.lr = lr
		self.loss_value = loss_value

		self.components = sub_components(self)

		# The hand-coded Spiker-LL reference names this architecture
		# "rtl"; mirror it.
		super().__init__(entity_name=self.name, architecture_name="rtl")
		self.vhdl(debug=debug)

	def vhdl(self, debug=False):
		# Libraries
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add(
			name="n_neurons", gen_type="integer",
			value=str(self.n_neurons))
		self.entity.generic.add(
			name="neuron_bit_width", gen_type="integer",
			value=str(self.neuron_bw))

		# Ports — note clk/rst_n are unused by the current combinational
		# implementation but kept for interface parity with the fork
		# (and to leave room for a registered variant later).
		self.entity.port.add(
			name="clk", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="rst_n", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="update_weights", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="in_spike", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="target", direction="in",
			port_type="std_logic_vector(n_neurons-1 downto 0)")
		self.entity.port.add(
			name="out_spikes", direction="in",
			port_type="std_logic_vector(n_neurons-1 downto 0)")
		self.entity.port.add(
			name="weights_in", direction="in",
			port_type="std_logic_vector("
			"n_neurons*neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name="weights_out", direction="out",
			port_type="std_logic_vector("
			"n_neurons*neuron_bit_width-1 downto 0)")

		# Constants — the ± step size baked in from training hyperparams.
		self.architecture.constant.add(
			"CONST", "signed(neuron_bit_width-1 downto 0)",
			f"to_signed({self.const_value}, neuron_bit_width)")
		self.architecture.constant.add(
			"NEG_CONST", "signed(neuron_bit_width-1 downto 0)",
			f"to_signed({-self.const_value}, neuron_bit_width)")
		self.architecture.constant.add(
			"ZERO", "signed(neuron_bit_width-1 downto 0)",
			"(others => '0')")

		# Internal signals.
		self.architecture.signal.add(
			name="update_values",
			signal_type="std_logic_vector("
			"n_neurons*neuron_bit_width-1 downto 0)")
		self.architecture.signal.add(
			name="updated_weights",
			signal_type="std_logic_vector("
			"n_neurons*neuron_bit_width-1 downto 0)")

		# Vivado waveform-debug types/signals, mirrored verbatim from
		# the hand-coded reference. Functionally dead.
		self.architecture.customTypes.add(
			"std_logic_vector_array", "Array",
			"natural range <>",
			"std_logic_vector(neuron_bit_width-1 downto 0)")
		self.architecture.signal.add(
			name="dbg_w_in",
			signal_type="std_logic_vector_array(0 to n_neurons-1)")
		self.architecture.signal.add(
			name="dbg_w_upd",
			signal_type="std_logic_vector_array(0 to n_neurons-1)")
		self.architecture.signal.add(
			name="cmp", signal_type="std_logic")
		self.architecture.customTypes.add(
			"weight_array_t", "Array",
			"0 to n_neurons-1",
			"signed(neuron_bit_width-1 downto 0)")
		self.architecture.signal.add(
			name="debug_weights_in", signal_type="weight_array_t")
		self.architecture.signal.add(
			name="debug_weights_out", signal_type="weight_array_t")
		self.architecture.signal.add(
			name="debug_update_values", signal_type="weight_array_t")

		# Update logic — one slice per output neuron. Same combinational
		# pattern as the hidden trainer but with a fixed CONST instead of
		# a per-neuron lookup.
		generate_block = (
			"update_logic_gen: for i in 0 to n_neurons-1 generate\n"
			"        update_values((i+1)*neuron_bit_width-1 downto i*neuron_bit_width) <= \n"
			"            std_logic_vector(ZERO) when in_spike = '0' else\n"
			"            std_logic_vector(ZERO) when (target(i) = '0' and out_spikes(i) = '0') else\n"
			"            std_logic_vector(NEG_CONST) when (target(i) = '0' and out_spikes(i) = '1') else\n"
			"            std_logic_vector(CONST) when (target(i) = '1' and out_spikes(i) = '0') else\n"
			"            std_logic_vector(ZERO);\n"
			"        updated_weights((i+1)*neuron_bit_width-1 downto i*neuron_bit_width) <= \n"
			"            std_logic_vector(\n"
			"                signed(weights_in((i+1)*neuron_bit_width-1 downto i*neuron_bit_width)) + \n"
			"                signed(update_values((i+1)*neuron_bit_width-1 downto i*neuron_bit_width))\n"
			"            );\n"
			"    end generate update_logic_gen;"
		)
		self.architecture.bodyCodeHeader.add(generate_block)

		# Combinational output gating (same rule as the hidden trainer).
		self.architecture.bodyCodeHeader.add(
			"weights_out <= updated_weights when update_weights = '1' "
			"else (others => '0');"
		)

		self.architecture.bodyCodeHeader.add(
			"debug_assign: for i in 0 to n_neurons-1 generate\n"
			"        dbg_w_in(i) <= weights_in((i+1)*neuron_bit_width-1 downto i*neuron_bit_width);\n"
			"        dbg_w_upd(i) <= updated_weights((i+1)*neuron_bit_width-1 downto i*neuron_bit_width);\n"
			"    end generate debug_assign;"
		)
		self.architecture.bodyCodeHeader.add(
			"cmp <= '1' when weights_in /= updated_weights else '0';")
		self.architecture.bodyCodeHeader.add(
			"debug_assign_gen: for i in 0 to n_neurons-1 generate\n"
			"    begin\n"
			"        debug_weights_in(i) <= signed(weights_in((i+1)*neuron_bit_width-1 downto i*neuron_bit_width));\n"
			"        debug_weights_out(i) <= signed(updated_weights((i+1)*neuron_bit_width-1 downto i*neuron_bit_width));\n"
			"        debug_update_values(i) <= signed(update_values((i+1)*neuron_bit_width-1 downto i*neuron_bit_width));\n"
			"    end generate debug_assign_gen;"
		)

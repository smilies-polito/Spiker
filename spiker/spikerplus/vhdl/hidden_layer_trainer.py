"""
VHDLblock for the Spiker-LL ``hidden_layer_trainer`` entity.

This is the on-chip learning module for the hidden layer. It applies a sparse
Direct-Feedback-Alignment style update: each hidden neuron has a hard-wired
output neuron (``SEL_IDX_ARRAY[i]``) and a signed learning constant
(``NEURON_CONSTANTS[i]``); on every update strobe the weight at the active
input column moves by ±NEURON_CONSTANTS[i] depending on the sign of the
mismatch between ``target`` and ``out_spikes`` at the hard-wired output.

The Python class takes the trained ``sel_idx`` and ``neuron_constants``
arrays produced by ``STSFTrainer`` and renders them as VHDL constant arrays
that match the reference fork in mnist/rtl/hidden_layer_trainer.vhd
byte-for-byte (modulo whitespace).
"""

import numpy as np

from .vhdl import sub_components
from .vhdltools.vhdl_block import VHDLblock


# Number of SEL_IDX_ARRAY indices per VHDL source line, purely for diff
# parity with the reference fork (which groups them in tens).
_SEL_IDX_PER_LINE = 10


def _render_neuron_constants(arr, n_per_line=1):
	"""Build the VHDL value literal for ``NEURON_CONSTANTS``.

	One ``to_signed(<int>, neuron_bit_width)`` per element, plus an
	``-- index N`` trailing comment. Returns a multi-line string suitable
	to feed into ``architecture.constant.add(..., value=...)``.
	"""
	lines = ["("]
	for i, v in enumerate(arr):
		sep = "," if i < len(arr) - 1 else ""
		lines.append(
			f"        to_signed({int(v)}, neuron_bit_width){sep} -- index {i}"
		)
	lines.append("    )")
	return "\n".join(lines)


def _render_sel_idx_array(arr, n_per_line=_SEL_IDX_PER_LINE):
	"""Build the VHDL value literal for ``SEL_IDX_ARRAY``.

	Integers grouped ``n_per_line`` per source line with a trailing range
	comment, matching the layout in the reference fork.
	"""
	lines = ["("]
	for chunk_start in range(0, len(arr), n_per_line):
		chunk = arr[chunk_start:chunk_start + n_per_line]
		chunk_str = ", ".join(str(int(x)) for x in chunk)
		last_chunk = chunk_start + n_per_line >= len(arr)
		sep = "" if last_chunk else ","
		comment = f"-- {chunk_start}-{chunk_start + len(chunk) - 1}"
		lines.append(f"        {chunk_str}{sep} {comment}")
	lines.append("    )")
	return "\n".join(lines)


class HiddenLayerTrainer(VHDLblock):
	"""Spiker-LL on-chip trainer for the hidden layer (one entity per design)."""

	def __init__(self, n_hidden, n_output, neuron_bw, sel_idx,
			neuron_constants, debug=False):

		# Defensive: the RTL relies on these being plain int arrays of the
		# right shape. Coerce eagerly so a bad input fails here, not in the
		# emitted VHDL.
		sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
		neuron_constants = np.asarray(neuron_constants, dtype=np.int64).ravel()
		if sel_idx.shape[0] != n_hidden:
			raise ValueError(
				f"sel_idx length {sel_idx.shape[0]} does not match "
				f"n_hidden {n_hidden}")
		if neuron_constants.shape[0] != n_hidden:
			raise ValueError(
				f"neuron_constants length {neuron_constants.shape[0]} "
				f"does not match n_hidden {n_hidden}")
		if np.any(sel_idx < 0) or np.any(sel_idx >= n_output):
			raise ValueError(
				f"sel_idx contains values outside [0, {n_output})")

		self.name = "hidden_layer_trainer"
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.neuron_bw = neuron_bw
		self.sel_idx = sel_idx
		self.neuron_constants = neuron_constants

		# Trainer has no sub-components (everything is in this entity).
		self.components = sub_components(self)

		super().__init__(entity_name=self.name)
		self.vhdl(debug=debug)

	def vhdl(self, debug=False):
		# Libraries
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")

		# Generics
		self.entity.generic.add(
			name="n_hidden_neurons", gen_type="integer",
			value=str(self.n_hidden))
		self.entity.generic.add(
			name="n_output_neurons", gen_type="integer",
			value=str(self.n_output))
		self.entity.generic.add(
			name="neuron_bit_width", gen_type="integer",
			value=str(self.neuron_bw))

		# Ports
		self.entity.port.add(
			name="update_weights", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="x_pre", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="x_post", direction="in",
			port_type="std_logic_vector(n_hidden_neurons-1 downto 0)")
		self.entity.port.add(
			name="target", direction="in",
			port_type="std_logic_vector(n_output_neurons-1 downto 0)")
		self.entity.port.add(
			name="out_spikes", direction="in",
			port_type="std_logic_vector(n_output_neurons-1 downto 0)")
		self.entity.port.add(
			name="weights_in", direction="in",
			port_type="std_logic_vector("
			"n_hidden_neurons*neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name="weights_out", direction="out",
			port_type="std_logic_vector("
			"n_hidden_neurons*neuron_bit_width-1 downto 0)")

		# Custom array types declared inside the architecture.
		self.architecture.customTypes.add(
			"const_array_t", "Array",
			"0 to n_hidden_neurons-1",
			"signed(neuron_bit_width-1 downto 0)")
		self.architecture.customTypes.add(
			"sel_idx_array_t", "Array",
			"0 to n_hidden_neurons-1",
			"integer range 0 to n_output_neurons-1")

		# Constants (the values are trained, baked in at generation time).
		self.architecture.constant.add(
			"NEURON_CONSTANTS", "const_array_t",
			_render_neuron_constants(self.neuron_constants))
		self.architecture.constant.add(
			"ZERO", "signed(neuron_bit_width-1 downto 0)",
			"(others => '0')")
		self.architecture.constant.add(
			"SEL_IDX_ARRAY", "sel_idx_array_t",
			_render_sel_idx_array(self.sel_idx))

		# Internal signals.
		self.architecture.signal.add(
			name="update_values",
			signal_type="std_logic_vector("
			"n_hidden_neurons*neuron_bit_width-1 downto 0)")
		self.architecture.signal.add(
			name="updated_weights",
			signal_type="std_logic_vector("
			"n_hidden_neurons*neuron_bit_width-1 downto 0)")

		# Update logic — one slice per hidden neuron, generated by a for-
		# generate over n_hidden_neurons. The combinational rule is:
		#
		#   if x_pre=0 or x_post(i)=0 or target(SEL(i))==out(SEL(i))
		#       -> no update
		#   elif target(SEL(i))=1 and out(SEL(i))=0   (false negative)
		#       -> weight -= NEURON_CONSTANTS(i)
		#   elif target(SEL(i))=0 and out(SEL(i))=1   (false positive)
		#       -> weight += NEURON_CONSTANTS(i)
		generate_block = (
			"update_logic_gen: for i in 0 to n_hidden_neurons-1 generate\n"
			"    begin\n"
			"        update_values((i+1)*neuron_bit_width-1 downto i*neuron_bit_width) <= \n"
			"            std_logic_vector(ZERO) when (x_pre = '0' or x_post(i) = '0' or target(SEL_IDX_ARRAY(i)) = out_spikes(SEL_IDX_ARRAY(i))) else\n"
			"            std_logic_vector(-NEURON_CONSTANTS(i)) when (target(SEL_IDX_ARRAY(i)) = '1' and out_spikes(SEL_IDX_ARRAY(i)) = '0') else\n"
			"            std_logic_vector(NEURON_CONSTANTS(i)) when (target(SEL_IDX_ARRAY(i)) = '0' and out_spikes(SEL_IDX_ARRAY(i)) = '1') else\n"
			"            std_logic_vector(ZERO);\n"
			"        updated_weights((i+1)*neuron_bit_width-1 downto i*neuron_bit_width) <= \n"
			"            std_logic_vector(\n"
			"                signed(weights_in((i+1)*neuron_bit_width-1 downto i*neuron_bit_width)) + \n"
			"                signed(update_values((i+1)*neuron_bit_width-1 downto i*neuron_bit_width))\n"
			"            );\n"
			"    end generate update_logic_gen;"
		)
		self.architecture.bodyCodeHeader.add(generate_block)

		# Combinational output: hold the bus at zero when learning is off
		# so the layer's RAM never sees an unintended write enable.
		self.architecture.bodyCodeHeader.add(
			"weights_out <= updated_weights when update_weights = '1' "
			"else (others => '0');"
		)

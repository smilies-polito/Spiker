"""
VHDLblock for the Spiker-LL ``output_voter`` entity.

The voter accumulates output-layer spikes over an inference window and emits
a one-hot ``voted_class`` when prompted by ``vote``. It exists so that a
training-mode Spiker-LL can produce a single classification per sample
(needed for both ground-truth comparison and for the output trainer).
"""

from math import ceil, log2

from .vhdl import sub_components
from .vhdltools.vhdl_block import VHDLblock


def _default_count_width(n_cycles):
	"""How many bits are needed to count up to ``n_cycles`` per class."""
	return max(1, int(ceil(log2(max(2, n_cycles + 1)))))


class OutputVoter(VHDLblock):
	"""Argmax over per-class spike counts across an inference window."""

	def __init__(self, n_classes, n_cycles=10, count_width=None, debug=False):

		self.name = "output_voter"
		self.n_classes = n_classes
		# Counter width has to be large enough to hold ``n_cycles``; default
		# falls back to ceil(log2(n_cycles+1)).
		self.count_width = (count_width if count_width is not None
		                    else _default_count_width(n_cycles))

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
			name="N_CLASSES", gen_type="integer",
			value=str(self.n_classes))
		self.entity.generic.add(
			name="COUNT_WIDTH", gen_type="positive",
			value=str(self.count_width))

		# Ports
		self.entity.port.add(
			name="clk", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="rst_n", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="count_en", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="count_rst", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="spikes_in", direction="in",
			port_type="std_logic_vector(N_CLASSES-1 downto 0)")
		self.entity.port.add(
			name="vote", direction="in", port_type="std_logic")
		self.entity.port.add(
			name="voted_class", direction="out",
			port_type="std_logic_vector(N_CLASSES-1 downto 0)")
		self.entity.port.add(
			name="vote_valid", direction="out", port_type="std_logic")

		# The SubType path in the vhdltools CustomTypeList is broken (uses an
		# undefined GenericList). Emit the subtype and counter array type
		# directly as raw declaration lines instead — which is exactly what
		# the reference fork does, and is the safest approach here.
		self.architecture.declarationHeader.add(
			"subtype counter_t is unsigned(COUNT_WIDTH-1 downto 0);")
		self.architecture.declarationHeader.add(
			"type counter_vector_t is array (0 to N_CLASSES-1) of counter_t;")

		# Signals
		self.architecture.signal.add(
			name="spike_counts",
			signal_type="counter_vector_t := "
			"(others => (others => '0'))")

		# Clocked counter process: zero on rst_n=0 or count_rst, otherwise
		# increment per asserted spike when count_en is high.
		counters_proc = (
			"counters_logic : process(clk, rst_n)\n"
			"    begin\n"
			"        if rst_n = '0' then\n"
			"            spike_counts <= (others => (others => '0'));\n"
			"        elsif rising_edge(clk) then\n"
			"            if count_rst = '1' then\n"
			"                spike_counts <= (others => (others => '0'));\n"
			"            elsif count_en = '1' then\n"
			"                for i in 0 to N_CLASSES-1 loop\n"
			"                    if spikes_in(i) = '1' then\n"
			"                        spike_counts(i) <= spike_counts(i) + 1;\n"
			"                    end if;\n"
			"                end loop;\n"
			"            end if;\n"
			"        end if;\n"
			"    end process;"
		)
		self.architecture.bodyCodeHeader.add(counters_proc)

		# Combinational voting process: argmax over the spike counts.
		voting_proc = (
			"voting_logic : process(spike_counts, vote)\n"
			"        variable max_count : counter_t;\n"
			"        variable max_idx   : integer range 0 to N_CLASSES-1;\n"
			"    begin\n"
			"        if vote = '1' then\n"
			"            max_count := (others => '0');\n"
			"            max_idx := 0;\n"
			"            for i in 0 to N_CLASSES-1 loop\n"
			"                if spike_counts(i) > max_count then\n"
			"                    max_count := spike_counts(i);\n"
			"                    max_idx := i;\n"
			"                end if;\n"
			"            end loop;\n"
			"            voted_class <= (others => '0');\n"
			"            voted_class(max_idx) <= '1';\n"
			"            vote_valid <= '1';\n"
			"        else\n"
			"            voted_class <= (others => '0');\n"
			"            vote_valid <= '0';\n"
			"        end if;\n"
			"    end process;"
		)
		self.architecture.bodyCodeHeader.add(voting_proc)

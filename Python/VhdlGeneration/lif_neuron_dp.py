import path_config
from vhdl_block import VHDLblock
from shifter import Shifter
from mux4to1_signed import Mux4to1_signed
from add_sub import AddSub
from mux2to1_signed import Mux2to1_signed
from reg_signed import RegSigned
from reg_signed_sync_rst import RegSignedSyncRst
from cmp_gt import CmpGt


class LIFneuron(VHDLblock):

	def __init__(self, default_bitwidth = 16, default_inh_weights_bitwidth =
			5, default_exc_weights_bitwidth = 5,
			default_shift = 10):

		VHDLblock.__init__(self, entity_name = "neuron_datapath")

		self.shifter			= Shifter(default_bitwidth =
						default_bitwidth,
						default_shift =
						default_shift)

		self.mux4to1_signed		= Mux4to1_signed(
						default_bitwidth = 
						default_bitwidth)

		self.add_sub			= AddSub(
						default_bitwidth = 
						default_bitwidth)

		self.mux2to1_signed		= Mux2to1_signed(
						default_bitwidth = 
						default_bitwidth)

		self.reg_signed			= RegSigned(
						default_bitwidth = 
						default_bitwidth)

		self.reg_signed_sync_rst	= RegSignedSyncRst(
						default_bitwidth = 
						default_bitwidth)

		self.cmp_gt			= CmpGt(
						default_bitwidth = 
						default_bitwidth)

		# Libraries and packages
		self.library.add("ieee")
		self.library["ieee"].package.add("std_logic_1164")
		self.library["ieee"].package.add("numeric_std")
				

		# Generics
		self.entity.generic.add(
			name		= "neuron_bit_width", 
			gen_type	= "integer",
			value		= str(default_bitwidth))
		self.entity.generic.add(
			name		= "inh_weights_bit_width",
			gen_type	= "integer",
			value		= str(default_inh_weights_bitwidth))
		self.entity.generic.add(
			name		= "exc_weights_bit_width",
			gen_type	= "integer",
			value		= str(default_exc_weights_bitwidth))
		self.entity.generic.add(
			name		= "shift",
			gen_type	= "integer",
			value		= str(default_shift))

		# Input parameters
		self.entity.port.add(
			name 		= "v_th_value", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name 		= "v_reset", 
			direction	= "in",
			port_type	= "signed(neuron_bit_width-1 downto 0)")
		self.entity.port.add(
			name 		= "inh_weight",
			direction	= "in",
			port_type	= "signed(inh_weights_bit_width-1 "
						"downto 0)")
		self.entity.port.add(
			name 		= "exc_weight",
			direction	= "in",
			port_type	= "signed(exc_weights_bit_width-1 downto 0)")

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
		self.entity.port.add(
			name 		= "v_update",
			direction	= "in",
			port_type	= "std_logic")
		self.entity.port.add(
			name 		= "v_th_en", 
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
		
		self.entity.port.add(
			name 		= "v_rst_n",
			direction	= "in",
			port_type	= "std_logic")


		# Signals
		self.architecture.signal.add(
			name		= "update",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")

		self.architecture.signal.add(
			name		= "update_value",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")

		self.architecture.signal.add(
			name		= "v_th",
			signal_type	= "signed(neuron_bit_width-1 downto 0)")

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
		self.architecture.component.add(self.mux2to1_signed)
		self.architecture.component.add(self.reg_signed)
		self.architecture.component.add(self.reg_signed_sync_rst)
		self.architecture.component.add(self.cmp_gt)


		# Shifter port map
		self.architecture.instances.add(self.shifter, "v_shifter")
		self.architecture.instances["v_shifter"].generic_map()
		self.architecture.instances["v_shifter"].g_map.add("N",
				"neuron_bit_width")
		self.architecture.instances["v_shifter"].port_map("pos", "v",
				"v_shifted")

		# Adder port map
		self.architecture.instances.add(self.add_sub, "update_add_sub")
		self.architecture.instances["update_add_sub"].generic_map()
		self.architecture.instances["update_add_sub"].g_map.add("N",
				"neuron_bit_width")
		self.architecture.instances["update_add_sub"].port_map()
		self.architecture.instances["update_add_sub"].p_map.add("in0",
				"v")
		self.architecture.instances["update_add_sub"].p_map.add("in1",
				"update")
		self.architecture.instances["update_add_sub"].p_map.add(
				"add_sub_out", "update_value")





a = LIFneuron()
print(a.code())

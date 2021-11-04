library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_datapath is
	
	generic(
		-- parallelism
		N		: integer := 8		
	);

	port(


		-- input parameters
		v_th_0		: in signed(N-1 downto 0);
		v_rest		: in signed(N-1 downto 0);
		v_reset		: in signed(N-1 downto 0);
		v_th_plus	: in signed(N-1 downto 0);
		inh_weight	: in signed(N-1 downto 0);
		exc_weight	: in signed(N-1 downto 0);

		-- input controls
		clk		: in std_logic;
		v_update_sel	: in std_logic;
		v_or_v_th	: in std_logic;
		add_sub		: in std_logic;
		update_or_rest	: in std_logic;
		v_reset_sel	: in std_logic;
		v_th_en		: in std_logic;
		v_en		: in std_logic;
		
		-- output
		exceed_v_th	: out std_logic;	
	);

end entity neuron_datapath;


architecture behvaiour of neuron datapath is
begin

end architecture behaviour;

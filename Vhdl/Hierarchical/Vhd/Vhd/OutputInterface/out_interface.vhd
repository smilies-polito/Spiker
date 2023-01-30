library ieee;
use ieee.std_logic_1164.all;


entity out_interface is

	generic(
		bit_width	: integer := 16;	
		N_neurons	: integer := 400
	);

	port(
		-- control input
		clk		: in std_logic;
		rst_n		: in std_logic;
		start		: in std_logic;
		stop		: in std_logic;

		-- data input
		spikes		: in std_logic_vector(N_neurons-1 downto 0);

		-- output
		cnt_out		: out std_logic_vector(N_neurons*bit_width-1
					downto 0)

	);

end entity out_interface;




architecture behaviour of out_interface is

	-- from cu to datapath
	signal elaborate	: std_logic;
	signal cnt_rst_n	: std_logic;

	component out_interface_cu is

		port(
			-- input
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;

			-- output
			elaborate	: out std_logic;
			cnt_rst_n	: out std_logic
		);

	end component out_interface_cu;

	component out_interface_datapath is

		generic(
			bit_width	: integer := 16;	
			N_neurons	: integer := 400
		);

		port(
			-- control input
			clk		: in std_logic;
			elaborate	: in std_logic;
			cnt_rst_n	: in std_logic;

			-- data input
			spikes		: in std_logic_vector(N_neurons-1 downto 0);

			-- output
			cnt_out		: out std_logic_vector(N_neurons*bit_width-1
						downto 0)
		);

	end component out_interface_datapath;


begin


	
	control_unit	: out_interface_cu 

		port map(
			-- input
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,

			-- output
			elaborate	=> elaborate,
			cnt_rst_n	=> cnt_rst_n
		);

	datapath	: out_interface_datapath 

		generic map(
			bit_width	=> bit_width,
			N_neurons	=> N_neurons
		)

		port map(
			-- control input
			clk		=> clk,
			elaborate	=> elaborate,
			cnt_rst_n	=> cnt_rst_n,

			-- data input
			spikes		=> spikes,

			-- output
			cnt_out		=> cnt_out
					
		);

end architecture behaviour;

library ieee;
use ieee.std_logic_1164.all;

entity out_interface_datapath is

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

end entity out_interface_datapath;

architecture behaviour of out_interface_datapath is


	signal cnt_en		: std_logic_vector(N_neurons-1 downto 0);

	component cnt is

		generic(
			bit_width		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(bit_width-1 
						downto 0)
		);

	end component cnt;

begin

	enable_counters	: process(elaborate, spikes)
	begin
		for i in 0 to N_neurons-1
		loop
			cnt_en(i) <= elaborate and spikes(i);
		end loop;
	end process enable_counters;



	cnt_layer	: for i in 0 to N_neurons-1
	generate

		neuron_counter	: cnt 

			generic map(
				bit_width	=> bit_width
			)

			port map(
				-- input
				clk		=> clk,
				cnt_en		=> cnt_en(i),
				cnt_rst_n	=> cnt_rst_n,

				-- output
				cnt_out		=> cnt_out((i+1)*bit_width-1
							downto i*bit_width)
			);

	end generate cnt_layer;

end architecture behaviour;

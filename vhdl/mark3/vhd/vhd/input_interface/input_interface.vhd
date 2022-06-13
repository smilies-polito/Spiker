library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity input_interface is

	generic(
		bit_width	: integer := 16;
		N_inputs	: integer := 784
	);

	port(
		-- control input
		clk		: in std_logic;
		load		: in std_logic;
		update		: in std_logic;

		-- data input
		seed		: in unsigned(bit_width-1 downto 0);
		input_data	: in unsigned(N_inputs*bit_width-1 downto 0);

		-- output
		output_spikes	: out std_logic_vector(N_inputs-1 downto 0)
	);

end entity input_interface;

architecture behaviour of input_interface is

	signal pseudo_random	: unsigned(bit_width-1 downto 0);
	
	component lfsr is

		generic(
			bit_width	: integer	:= 16
		);

		port(
			-- control input
			clk		: in std_logic;
			load		: in std_logic;
			update		: in std_logic;

			-- data input
			seed		: in unsigned(bit_width-1 downto 0);

			-- output
			pseudo_random	: out unsigned(bit_width-1 downto 0)
		);

	end component lfsr;


	component unsigned_cmp_gt is
		
		generic(
			bit_width	: integer := 8		
		);


		port(
			-- input
			in0	: in unsigned(bit_width-1 downto 0);
			in1	: in unsigned(bit_width-1 downto 0);

			-- output		
			cmp_out	: out std_logic
		);

	end component unsigned_cmp_gt;

begin


	pseudo_random_generator	: lfsr 

		generic map(
			bit_width	=> bit_width
		)

		port map(
			-- control input
			clk		=> clk,
			load		=> load,
			update		=> update,

			-- data input
			seed		=> seed,

			-- output
			pseudo_random	=> pseudo_random
		);

	cmp_layer	: for i in 0 to N_inputs
	generate	

		cmp_i	: unsigned_cmp_gt 
			
			generic map(
				bit_width	=> bit_width
			)

			port map(
				-- input
				in0		=> pseudo_random,
				in1		=> input_data((i+1)*bit_width-1
							downto i*bit_width),

				-- output		
				cmp_out		=> output_spikes(i)	
			);

	end generate cmp_layer;
	
end architecture behaviour;
